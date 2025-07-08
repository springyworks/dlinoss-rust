use burn::prelude::*;
use burn::tensor::backend::Backend;
use crate::burnadd::parallel_scan::*;

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
    
    // Block matrix multiplication: [ja jb; jc jd] * [ia ib; ic id]
    let a_new = ja.clone() * ia.clone() + jb.clone() * ic.clone();
    let b_new = ja.clone() * ib.clone() + jb.clone() * id.clone();
    let c_new = jc.clone() * ia.clone() + jd.clone() * ic.clone();
    let d_new = jc.clone() * ib.clone() + jd.clone() * id.clone();
    
    let a_result = Tensor::cat(vec![a_new, b_new, c_new, d_new], 0);
    
    // Vector operations for b
    let b_i1 = b_i.clone().slice([0..n]);
    let b_i2 = b_i.clone().slice([n..2*n]);
    
    let new_b1 = ja.clone() * b_i1.clone() + jb.clone() * b_i2.clone();
    let new_b2 = jc * b_i1 + jd * b_i2;
    
    let b_result = Tensor::cat(vec![new_b1, new_b2], 0) + b_j;
    
    (a_result, b_result)
}

/// Create the recurrent matrix M for Damped LinOSS-IMEX discretization
/// Following equation (5) in the paper
pub fn make_damped_linoss_imex_recurrence<B: Backend>(
    a_diag: Tensor<B, 1>,
    g_diag: Tensor<B, 1>, 
    step: f64,
    device: &B::Device
) -> Tensor<B, 2> {
    let step_tensor = Tensor::from_floats([step], device);
    let identity = Tensor::ones_like(&a_diag);
    
    // Schur complement: S = I + Δt * G
    let s = identity.clone() + step_tensor.clone() * g_diag.clone();
    
    // M_11 = S^(-1)
    let m_11 = Tensor::ones_like(&s) / s.clone();
    
    // M_12 = -Δt * S^(-1) * A
    let m_12 = -step_tensor.clone() * (Tensor::ones_like(&s) / s.clone()) * a_diag.clone();
    
    // M_21 = Δt * S^(-1)
    let m_21 = step_tensor.clone() * (Tensor::ones_like(&s) / s.clone());
    
    // M_22 = I - Δt² * S^(-1) * A
    let m_22 = identity - (step_tensor.clone() * step_tensor) * (Tensor::ones_like(&s) / s) * a_diag.clone();
    
    // Create block matrix structure  
    let n = a_diag.dims()[0];
    let mut m_data = vec![0.0; 4 * n * n];
    
    // Fill diagonal blocks
    for i in 0..n {
        // M_11 block
        m_data[i * (2 * n) + i] = m_11.clone().slice([i..i+1]).into_scalar().elem::<f32>();
        // M_12 block  
        m_data[i * (2 * n) + (i + n)] = m_12.clone().slice([i..i+1]).into_scalar().elem::<f32>();
        // M_21 block
        m_data[(i + n) * (2 * n) + i] = m_21.clone().slice([i..i+1]).into_scalar().elem::<f32>();
        // M_22 block
        m_data[(i + n) * (2 * n) + (i + n)] = m_22.clone().slice([i..i+1]).into_scalar().elem::<f32>();
    }
    
    Tensor::<B, 1>::from_floats(m_data.as_slice(), device).reshape([2 * n, 2 * n])
}

/// Apply Damped LinOSS-IMEX to input sequence using PROPER parallel scan
/// This is the mathematically correct implementation of D-LinOSS
pub fn apply_damped_linoss_imex<B: Backend>(
    a_diag: Tensor<B, 1>,
    g_diag: Tensor<B, 1>,
    b_matrix: Tensor<B, 2>, // Shape: [ssm_size, input_dim] 
    input_sequence: Tensor<B, 3>, // Shape: [batch_size, seq_len, input_dim]
    step: f64,
    device: &B::Device
) -> Tensor<B, 3> {
    let [batch_size, seq_len, input_dim] = input_sequence.dims();
    let ssm_size = a_diag.dims()[0];
    
    // Project input through B matrix: Bu_t for each timestep
    let bu_elements: Tensor<B, 3> = {
        let mut bu_timesteps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let input_t = input_sequence.clone().slice([0..batch_size, t..t+1, 0..input_dim]).squeeze_dims(&[1]);
            let bu_t = input_t.matmul(b_matrix.clone().transpose());
            bu_timesteps.push(bu_t.unsqueeze_dim(1));
        }
        Tensor::cat(bu_timesteps, 1)
    };
    
    let identity = Tensor::ones_like(&a_diag);
    let s = identity.clone() + Tensor::from_floats([step], device) * g_diag.clone();
    let step_tensor = Tensor::from_floats([step], device);
    
    // Compute IMEX discretization matrices
    let m_11 = Tensor::ones_like(&s) / s.clone();
    let m_12 = -step_tensor.clone() * (Tensor::ones_like(&s) / s.clone()) * a_diag.clone();
    let m_21 = step_tensor.clone() * (Tensor::ones_like(&s) / s.clone());
    let m_22 = identity - (step_tensor.clone() * step_tensor.clone()) * (Tensor::ones_like(&s) / s.clone()) * a_diag;
    
    // Create transition matrices for parallel scan
    let m_elements: Tensor<B, 2> = {
        let m_flat = Tensor::cat(vec![m_11.clone(), m_12.clone(), m_21.clone(), m_22.clone()], 0);
        let m_repeated = m_flat.unsqueeze_dim(0).repeat_dim(0, seq_len);
        m_repeated
    };
    
    // Create input projection matrices for parallel scan
    let f_elements: Tensor<B, 3> = {
        let f1_coeff = step_tensor.clone() * (Tensor::ones_like(&s) / s.clone());
        let f2_coeff = (step_tensor.clone() * step_tensor) * (Tensor::ones_like(&s) / s);
        
        let mut f_timesteps = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let bu_t: Tensor<B, 2> = bu_elements.clone().slice([0..batch_size, t..t+1, 0..ssm_size]).squeeze_dims(&[1]);
            let f1_t: Tensor<B, 2> = bu_t.clone() * f1_coeff.clone().unsqueeze_dim::<2>(0);
            let f2_t: Tensor<B, 2> = bu_t * f2_coeff.clone().unsqueeze_dim::<2>(0);
            let f_t: Tensor<B, 2> = Tensor::cat(vec![f1_t, f2_t], 1);
            f_timesteps.push(f_t.unsqueeze_dim::<3>(1));
        }
        Tensor::cat(f_timesteps, 1)
    };
    
    // Apply the REAL parallel scan using our custom implementation
    let scan_result = dlinoss_parallel_scan(m_elements, f_elements);
    
    // Extract position components (second half of state vector)
    scan_result.slice([0..batch_size, 0..seq_len, ssm_size..2*ssm_size])
}

/// Apply block matrix multiplication for D-LinOSS state transition
fn apply_block_matrix_multiply<B: Backend>(
    state: Tensor<B, 2>, 
    m_elements: Tensor<B, 1>, 
    n: usize
) -> Tensor<B, 2> {
    let [batch_size, _] = state.dims();
    
    let m_11 = m_elements.clone().slice([0..n]);
    let m_12 = m_elements.clone().slice([n..2*n]);
    let m_21 = m_elements.clone().slice([2*n..3*n]);
    let m_22 = m_elements.clone().slice([3*n..4*n]);
    
    let x1 = state.clone().slice([0..batch_size, 0..n]);
    let x2 = state.clone().slice([0..batch_size, n..2*n]);
    
    let new_x1 = x1.clone() * m_11.unsqueeze_dim(0) + x2.clone() * m_12.unsqueeze_dim(0);
    let new_x2 = x1 * m_21.unsqueeze_dim(0) + x2 * m_22.unsqueeze_dim(0);
    
    Tensor::cat(vec![new_x1, new_x2], 1)
}

/// Initialize A matrix with proper oscillatory frequencies
pub fn init_oscillatory_a_matrix<B: Backend>(
    num_oscillators: usize,
    r_min: f64,
    r_max: f64,
    device: &B::Device
) -> Tensor<B, 1> {
    let mut a_values = Vec::new();
    
    for i in 0..num_oscillators {
        // Frequency distribution from r_min to r_max
        let freq = r_min + (i as f64 / num_oscillators as f64) * (r_max - r_min);
        a_values.push(freq * freq); // A represents ω²
    }
    
    Tensor::<B, 1>::from_floats(a_values.as_slice(), device)
}

/// Initialize G matrix with proper damping coefficients  
pub fn init_damping_g_matrix<B: Backend>(
    num_oscillators: usize,
    damping_min: f64,
    damping_max: f64,
    device: &B::Device
) -> Tensor<B, 1> {
    let mut g_values = Vec::new();
    
    for i in 0..num_oscillators {
        // Damping distribution from damping_min to damping_max
        let damping = damping_min + (i as f64 / num_oscillators as f64) * (damping_max - damping_min);
        g_values.push(damping);
    }
    
    Tensor::<B, 1>::from_floats(g_values.as_slice(), device)
}
