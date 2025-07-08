use burn::prelude::*;
use burn::tensor::backend::Backend;

/// Core D-LinOSS mathematical operations following ArXiv:2505.12171
/// This implements the correct damped linear oscillatory state-space model

/// Binary operator for parallel scan of linear recurrence
/// Implements the associative operation for D-LinOSS state transitions
pub fn binary_operator<B: Backend>(
    q_i: (Tensor<B, 1>, Tensor<B, 1>), 
    q_j: (Tensor<B, 1>, Tensor<B, 1>)
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let (a_i, b_i) = q_i;
    let (a_j, b_j) = q_j;
    
    let n = a_i.dims()[0] / 4;
    
    // Extract block components from flattened representation
    let ia = a_i.clone().slice([0..n]);
    let ib = a_i.clone().slice([n..2*n]);
    let ic = a_i.clone().slice([2*n..3*n]);
    let id = a_i.clone().slice([3*n..4*n]);
    
    let ja = a_j.clone().slice([0..n]);
    let jb = a_j.clone().slice([n..2*n]);
    let jc = a_j.clone().slice([2*n..3*n]);
    let jd = a_j.clone().slice([3*n..4*n]);
    
    // Matrix multiplication for 2x2 block matrices - clone to avoid move
    let ra = ia.clone() * ja.clone() + ib.clone() * jc.clone();
    let rb = ia.clone() * jb.clone() + ib.clone() * jd.clone();
    let rc = ic.clone() * ja + id.clone() * jc;
    let rd = ic.clone() * jb + id.clone() * jd;
    
    // Flatten back to vector
    let a_out = Tensor::cat(vec![ra, rb, rc, rd], 0);
    
    // State update
    let jy = b_j.clone().slice([0..n]);
    let jdy = b_j.slice([n..2*n]);
    
    let y_out = ia * jy.clone() + ib * jdy.clone() + b_i.clone().slice([0..n]);
    let dy_out = ic * jy + id * jdy + b_i.slice([n..2*n]);
    
    let b_out = Tensor::cat(vec![y_out, dy_out], 0);
    
    (a_out, b_out)
}

/// Initialize the oscillatory matrix A (real part from discretization)
pub fn init_oscillatory_a_matrix<B: Backend>(
    ssm_size: usize,
    min_period: f32,
    max_period: f32,
    device: &B::Device,
) -> Tensor<B, 1> {
    let freq_fn = |k: f32| -> f32 {
        let norm_k = k / (ssm_size as f32);
        let log_min = (2.0 * std::f32::consts::PI / max_period).ln();
        let log_max = (2.0 * std::f32::consts::PI / min_period).ln();
        (log_min + norm_k * (log_max - log_min)).exp()
    };
    
    let data: Vec<f32> = (0..ssm_size)
        .map(|k| freq_fn(k as f32))
        .collect();
    
    Tensor::from_floats(data.as_slice(), device)
}

/// Initialize the damping matrix G (purely diagonal damping)
pub fn init_damping_g_matrix<B: Backend>(
    ssm_size: usize,
    r_min: f32,
    r_max: f32,
    device: &B::Device,
) -> Tensor<B, 1> {
    let damping_fn = |k: f32| -> f32 {
        let norm_k = k / (ssm_size as f32);
        let log_min = r_min.max(1e-8).ln();
        let log_max = r_max.ln();
        (log_min + norm_k * (log_max - log_min)).exp()
    };
    
    let data: Vec<f32> = (0..ssm_size)
        .map(|k| -damping_fn(k as f32))
        .collect();
    
    Tensor::from_floats(data.as_slice(), device)
}

/// Apply the damped LinOSS model using IMEX discretization
/// This is the mathematically correct implementation from the paper
pub fn apply_damped_linoss_imex<B: Backend>(
    a_diag: Tensor<B, 1>,      // Diagonal of A matrix (oscillatory frequencies)
    g_diag: Tensor<B, 1>,      // Diagonal of G matrix (damping coefficients)
    b_matrix: Tensor<B, 2>,    // Input projection matrix B
    input_sequence: Tensor<B, 3>,  // Input sequence [batch, seq_len, input_dim]
    step: f32,                  // Time step Δt
    device: &B::Device,
) -> Tensor<B, 3> {  // Output [batch, seq_len, ssm_size]
    let [batch_size, seq_len, input_dim] = input_sequence.dims();
    let ssm_size = a_diag.dims()[0];
    
    // Project inputs through B matrix: [batch, seq_len, ssm_size]
    let bu_elements = input_sequence.reshape([batch_size * seq_len, input_dim])
        .matmul(b_matrix.clone().transpose())
        .reshape([batch_size, seq_len, ssm_size]);
    
    // Initialize state transitions using IMEX discretization
    // s = I + Δt * G (IMEX formula)
    let s = Tensor::ones([ssm_size], device) + g_diag.clone() * step;
    
    // Complex frequencies for oscillations
    let omega = a_diag.clone();
    
    // Initialize state transition matrices for sequential scan
    let mut a_elements: Vec<Tensor<B, 1>> = Vec::new();
    let mut b_elements: Vec<Tensor<B, 3>> = Vec::new();
    
    for l in 0..seq_len {
        // Extract slice for current timestep
        let bu_l: Tensor<B, 2> = bu_elements.clone().slice([0..batch_size, l..l+1, 0..ssm_size])
            .squeeze::<2>(1);  // [batch_size, ssm_size]
        
        // Compute transition coefficients with corrected discretization
        let f1_coeff = s.clone() * (omega.clone() * step).cos() - (omega.clone() * step).sin();
        let f2_coeff = s.clone() * (omega.clone() * step).sin() + (omega.clone() * step).cos();
        
        // State transition for sequential application
        // Using real arithmetic for stability - explicit type annotation
        let a_l: Tensor<B, 1> = Tensor::cat(vec![
            f1_coeff.clone(),
            f2_coeff.clone(),
            -f2_coeff,
            f1_coeff,
        ], 0);
        
        // Input contribution with Δt scaling as per IMEX method - explicit type
        let b_l: Tensor<B, 3> = Tensor::stack(vec![
            bu_l.clone() * step,
            Tensor::zeros_like(&bu_l),
        ], 0);  // This creates [2, batch_size, ssm_size]
        
        a_elements.push(a_l);
        b_elements.push(b_l);
    }
    
    // Sequential scan (parallel scan requires more complex tensor operations)
    let mut outputs = Vec::new();
    let mut state: Tensor<B, 3> = Tensor::zeros([2, batch_size, ssm_size], device);
    
    for l in 0..seq_len {
        // Apply state transition
        let y = state.clone().slice([0..1, 0..batch_size, 0..ssm_size]).squeeze(0);
        let dy = state.clone().slice([1..2, 0..batch_size, 0..ssm_size]).squeeze(0);
        
        let a_l = &a_elements[l];
        let b_l = &b_elements[l];
        
        // Extract components for block matrix multiplication
        let a11 = a_l.clone().slice([0..ssm_size]);
        let a12 = a_l.clone().slice([ssm_size..2*ssm_size]);
        let a21 = a_l.clone().slice([2*ssm_size..3*ssm_size]);
        let a22 = a_l.clone().slice([3*ssm_size..4*ssm_size]);
        
        // New state computation: [y', dy'] = A[y, dy] + B*u
        let new_y = y.clone() * a11.clone().unsqueeze::<2>() + dy.clone() * a12.clone().unsqueeze::<2>() 
            + b_l.clone().slice([0..1, 0..batch_size, 0..ssm_size]).squeeze(0);
        let new_dy = y * a21.clone().unsqueeze::<2>() + dy * a22.clone().unsqueeze::<2>()
            + b_l.clone().slice([1..2, 0..batch_size, 0..ssm_size]).squeeze(0);
        
        state = Tensor::stack::<3>(vec![new_y.clone(), new_dy], 0)
            .reshape([2, batch_size, ssm_size]);
        outputs.push(new_y);
    }
    
    // Stack outputs to form final sequence
    Tensor::stack(outputs, 1)  // [batch_size, seq_len, ssm_size]
}
