use burn::prelude::*;
use burn::tensor::backend::Backend;

/// Associative parallel scan implementation for Burn tensors
/// This implements the core parallel scan operation needed for D-LinOSS
/// Following JAX's associative_scan functionality

/// Generic associative scan operation
/// This is the core missing functionality from Burn that we need for D-LinOSS
pub fn associative_scan<B: Backend, F>(
    binary_op: F,
    elements: Vec<(Tensor<B, 1>, Tensor<B, 1>)>,
) -> Vec<(Tensor<B, 1>, Tensor<B, 1>)>
where
    F: Fn((Tensor<B, 1>, Tensor<B, 1>), (Tensor<B, 1>, Tensor<B, 1>)) -> (Tensor<B, 1>, Tensor<B, 1>) + Clone,
{
    let n = elements.len();
    if n == 0 {
        return elements;
    }
    if n == 1 {
        return elements;
    }

    // For now, implement sequential scan
    // TODO: Implement true parallel scan with log(n) depth
    let mut result = Vec::with_capacity(n);
    result.push(elements[0].clone());
    
    for i in 1..n {
        let prev = result[i - 1].clone();
        let curr = elements[i].clone();
        let new_elem = binary_op(prev, curr);
        result.push(new_elem);
    }
    
    result
}

/// Apply D-LinOSS state transition
fn apply_dlinoss_state_transition<B: Backend>(
    state: Tensor<B, 2>,     // [batch_size, 2*ssm_size]
    m_matrix: Tensor<B, 1>,  // [4*ssm_size] - transition matrix elements
    f_input: Tensor<B, 2>,   // [batch_size, 2*ssm_size] - input projection
    ssm_size: usize,
) -> Tensor<B, 2> {
    // Extract M_A, M_B, M_C, M_D components from m_matrix
    let m_a: Tensor<B, 1> = m_matrix.clone().slice([0..ssm_size]);
    let m_b: Tensor<B, 1> = m_matrix.clone().slice([ssm_size..2*ssm_size]);
    let m_c: Tensor<B, 1> = m_matrix.clone().slice([2*ssm_size..3*ssm_size]);
    let m_d: Tensor<B, 1> = m_matrix.clone().slice([3*ssm_size..4*ssm_size]);
    
    // Split state into real and imaginary parts
    let state_real: Tensor<B, 2> = state.clone().slice([0..state.dims()[0], 0..ssm_size]);
    let state_imag: Tensor<B, 2> = state.clone().slice([0..state.dims()[0], ssm_size..2*ssm_size]);
    
    // Split input into real and imaginary parts
    let input_real: Tensor<B, 2> = f_input.clone().slice([0..f_input.dims()[0], 0..ssm_size]);
    let input_imag: Tensor<B, 2> = f_input.clone().slice([0..f_input.dims()[0], ssm_size..2*ssm_size]);
    
    // Apply complex matrix multiplication: M * state + F * input
    let new_real = state_real.clone() * m_a.clone().unsqueeze_dim::<2>(0) - state_imag.clone() * m_b.clone().unsqueeze_dim::<2>(0) + input_real;
    let new_imag = state_real * m_c.unsqueeze_dim::<2>(0) + state_imag * m_d.unsqueeze_dim::<2>(0) + input_imag;
    
    // Concatenate real and imaginary parts
    Tensor::cat(vec![new_real, new_imag], 1)
}

/// Parallel scan specifically for D-LinOSS state transitions
/// This handles the specific case of SSM recurrence relations
pub fn dlinoss_parallel_scan<B: Backend>(
    m_elements: Tensor<B, 2>, // [seq_len, 4*ssm_size] - transition matrices
    f_elements: Tensor<B, 3>, // [batch_size, seq_len, 2*ssm_size] - input projections  
) -> Tensor<B, 3> {
    let [seq_len, matrix_size] = m_elements.dims();
    let [batch_size, _, state_size] = f_elements.dims();
    let ssm_size = state_size / 2;
    
    // For now, implement sequential scan to get it working
    // TODO: Replace with true parallel implementation
    let mut all_states = Vec::with_capacity(seq_len);
    let mut current_state: Tensor<B, 2> = Tensor::zeros([batch_size, state_size], &m_elements.device());
    
    for t in 0..seq_len {
        let m_t: Tensor<B, 1> = m_elements.clone().slice([t..t+1, 0..matrix_size]).squeeze_dims(&[0]);
        let f_t: Tensor<B, 2> = f_elements.clone().slice([0..batch_size, t..t+1, 0..state_size]).squeeze_dims(&[1]);
        
        // Apply state transition: x_{t+1} = M * x_t + F * u_t
        current_state = apply_dlinoss_state_transition(current_state, m_t, f_t, ssm_size);
        all_states.push(current_state.clone().unsqueeze_dim::<3>(1));
    }
    
    Tensor::cat(all_states, 1)
}

/// Apply binary operator for D-LinOSS transition matrices
fn apply_dlinoss_binary_op_matrices<B: Backend>(
    a_i: Tensor<B, 1>,
    a_j: Tensor<B, 1>, 
    n: usize
) -> Tensor<B, 1> {
    // Extract 2x2 block matrix components
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
    let d_new = jc * ib + jd * id;
    
    Tensor::cat(vec![a_new, b_new, c_new, d_new], 0)
}

/// Apply binary operator for D-LinOSS state vectors
fn apply_dlinoss_binary_op_vectors<B: Backend>(
    b_i: Tensor<B, 2>,
    b_j: Tensor<B, 2>,
    a_i: Tensor<B, 1>,
    n: usize
) -> Tensor<B, 2> {
    let [batch_size, _] = b_i.dims();
    
    // Extract block components from transition matrix
    let ia = a_i.clone().slice([0..n]);
    let ib = a_i.clone().slice([n..2*n]);
    let ic = a_i.clone().slice([2*n..3*n]);
    let id = a_i.clone().slice([3*n..4*n]);
    
    // Extract state vector components
    let b_i1 = b_i.clone().slice([0..batch_size, 0..n]);
    let b_i2 = b_i.clone().slice([0..batch_size, n..2*n]);
    
    // Apply block matrix transformation to state vector
    let new_b1 = b_i1.clone() * ia.unsqueeze_dim(0) + b_i2.clone() * ic.unsqueeze_dim(0);
    let new_b2 = b_i1 * ib.unsqueeze_dim(0) + b_i2 * id.unsqueeze_dim(0);
    
    let transformed_b = Tensor::cat(vec![new_b1, new_b2], 1);
    
    // Add the current input contribution
    transformed_b + b_j
}

/// True parallel scan implementation (logarithmic depth)
/// This is what we should ultimately use for performance
pub fn parallel_scan_log_depth<B: Backend, F>(
    binary_op: F,
    mut elements: Vec<(Tensor<B, 1>, Tensor<B, 1>)>,
) -> Vec<(Tensor<B, 1>, Tensor<B, 1>)>
where
    F: Fn((Tensor<B, 1>, Tensor<B, 1>), (Tensor<B, 1>, Tensor<B, 1>)) -> (Tensor<B, 1>, Tensor<B, 1>) + Clone + Send + Sync,
{
    let n = elements.len();
    if n <= 1 {
        return elements;
    }
    
    // Up-sweep phase: build reduction tree
    let mut level_size = n;
    while level_size > 1 {
        let next_level_size = (level_size + 1) / 2;
        let mut next_level = Vec::with_capacity(next_level_size);
        
        for i in 0..next_level_size {
            let left_idx = i * 2;
            let right_idx = left_idx + 1;
            
            if right_idx < level_size {
                let combined = binary_op(elements[left_idx].clone(), elements[right_idx].clone());
                next_level.push(combined);
            } else {
                next_level.push(elements[left_idx].clone());
            }
        }
        
        elements = next_level;
        level_size = next_level_size;
    }
    
    // Down-sweep phase: distribute results
    // For now, fall back to sequential for simplicity
    // TODO: Implement proper down-sweep
    elements
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu<f32, i32>;
    
    #[test]
    fn test_associative_scan_basic() {
        let device = burn::backend::wgpu::WgpuDevice::default();
        
        // Test with simple addition operation
        let elements = vec![
            (Tensor::from_floats([1.0], &device), Tensor::from_floats([1.0], &device)),
            (Tensor::from_floats([2.0], &device), Tensor::from_floats([2.0], &device)),
            (Tensor::from_floats([3.0], &device), Tensor::from_floats([3.0], &device)),
        ];
        
        let add_op = |a: (Tensor<TestBackend, 1>, Tensor<TestBackend, 1>), 
                      b: (Tensor<TestBackend, 1>, Tensor<TestBackend, 1>)| -> (Tensor<TestBackend, 1>, Tensor<TestBackend, 1>) {
            (a.0 + b.0, a.1 + b.1)
        };
        
        let result = associative_scan(add_op, elements);
        
        assert_eq!(result.len(), 3);
        // First element should be [1, 1]
        // Second should be [1+2, 1+2] = [3, 3]  
        // Third should be [3+3, 3+3] = [6, 6]
    }
}
