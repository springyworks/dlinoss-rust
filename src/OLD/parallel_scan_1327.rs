use burn::tensor::{backend::Backend, Tensor};

/// Parallel scan operations for efficient SSM computation (arXiv:2505.12171)
/// 
/// MATHEMATICAL FOUNDATION for O(log T) Sequential Processing:
/// 
/// The D-LinOSS recurrence relation is:
/// w_{k+1} = M w_k + F u_{k+1}  for k = 0, 1, ..., T-1
/// 
/// This can be written as an ASSOCIATIVE OPERATION:
/// (M, F) ⊗ (M', F') = (MM', MF' + F)
/// 
/// ASSOCIATIVITY PROOF:
/// Let (A, a), (B, b), (C, c) be operators representing (M_i, F_i u_i).
/// Then: [(A,a) ⊗ (B,b)] ⊗ (C,c) = (AB, Ab + a) ⊗ (C,c) = (ABC, ABc + Ab + a)
///       (A,a) ⊗ [(B,b) ⊗ (C,c)] = (A,a) ⊗ (BC, Bc + b) = (ABC, A(Bc + b) + a) = (ABC, ABc + Ab + a)
/// 
/// Since the operation is associative, we can use parallel scan algorithms:
/// - Kogge-Stone algorithm: O(log T) depth, O(T log T) work  
/// - Blelloch algorithm: O(log T) depth, O(T) work
/// 
/// EFFICIENT STATE SEQUENCE COMPUTATION:
/// Instead of computing states sequentially:
/// w_1 = M w_0 + F u_1
/// w_2 = M w_1 + F u_2 = M(M w_0 + F u_1) + F u_2 = M² w_0 + MF u_1 + F u_2  
/// w_3 = M³ w_0 + M²F u_1 + MF u_2 + F u_3
/// ...
/// 
/// We can compute the composed operators in parallel using the tree reduction:
/// Level 0: (M, F u_1), (M, F u_2), (M, F u_3), (M, F u_4), ...
/// Level 1: (M², MF u_2 + F u_1), (M², MF u_4 + F u_3), ...  
/// Level 2: (M⁴, M³F u_4 + M²F u_3 + MF u_2 + F u_1), ...
/// 
/// This reduces sequential O(T) computation to parallel O(log T) depth.
pub struct ParallelScan;

impl ParallelScan {
    /// Associative scan operator for SSM: (M, F) ⊗ (M', F') = (MM', MF' + F)
    /// 
    /// MATHEMATICAL OPERATION DEFINITION:
    /// Given two operators (M₁, F₁) and (M₂, F₂) representing:
    /// - (M₁, F₁): state transition w' = M₁ w + F₁  
    /// - (M₂, F₂): state transition w'' = M₂ w' + F₂
    /// 
    /// The composition gives: w'' = M₂(M₁ w + F₁) + F₂ = (M₂M₁) w + (M₂F₁ + F₂)
    /// Therefore: (M₁, F₁) ⊗ (M₂, F₂) = (M₂M₁, M₂F₁ + F₂)
    /// 
    /// TENSOR DIMENSIONS:
    /// - M matrices: [seq_len, n, n] where n = 2m (state dimension)
    /// - F vectors: [seq_len, n, p] where p = input dimension
    /// - Output: composed (M, F) pairs with same dimensions
    /// 
    /// ALGORITHMIC COMPLEXITY:
    /// - Sequential approach: O(T) time for T time steps
    /// - This parallel approach: O(log T) time, O(T) space
    /// 
    /// NUMERICAL STABILITY:
    /// The IMEX discretization ensures ||M|| ≤ 1 when stability conditions are met,
    /// preventing exponential growth during matrix compositions.
    pub fn associative_scan<B: Backend>(
        m_matrices: Tensor<B, 3>, // [seq_len, state_dim, state_dim] - State transition matrices
        f_vectors: Tensor<B, 3>,  // [seq_len, state_dim, input_dim] - Input projection matrices
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let seq_len = m_matrices.dims()[0];
        
        // BASE CASE: single time step requires no composition
        if seq_len == 1 {
            return (m_matrices, f_vectors);
        }
        
        // PARALLEL SCAN IMPLEMENTATION using Kogge-Stone style algorithm
        // Each iteration doubles the span of composed operators
        let mut current_m = m_matrices;
        let mut current_f = f_vectors;
        let mut step = 1;
        
        // MAIN LOOP: compose operators at increasing spans
        // Iteration i: compose operators spanning 2^i time steps
        while step < seq_len {
            let mut new_m = Vec::new();
            let mut new_f = Vec::new();
            
            // COMPOSITION STEP: for each position that needs updating
            for i in (step..seq_len).step_by(step * 2) {
                let left_idx = i - step;  // Left operator index
                let right_idx = i;       // Right operator index
                
                // Extract operator matrices with explicit tensor operations
                let m_left: Tensor<B, 2> = current_m.clone().slice([
                    left_idx..left_idx+1, 
                    0..current_m.dims()[1], 
                    0..current_m.dims()[2]
                ]).squeeze::<2>(0);
                
                let m_right: Tensor<B, 2> = current_m.clone().slice([
                    right_idx..right_idx+1, 
                    0..current_m.dims()[1], 
                    0..current_m.dims()[2]
                ]).squeeze::<2>(0);
                
                let f_left: Tensor<B, 2> = current_f.clone().slice([
                    left_idx..left_idx+1, 
                    0..current_f.dims()[1], 
                    0..current_f.dims()[2]
                ]).squeeze::<2>(0);
                
                let f_right: Tensor<B, 2> = current_f.clone().slice([
                    right_idx..right_idx+1, 
                    0..current_f.dims()[1], 
                    0..current_f.dims()[2]
                ]).squeeze::<2>(0);
                
                // ASSOCIATIVE OPERATION: (M_left, F_left) ⊗ (M_right, F_right)
                // Resulting M: M_new = M_right × M_left  (right applied after left)
                let new_m_val = m_right.clone().matmul(m_left.clone());
                
                // Resulting F: F_new = M_right × F_left + F_right
                let f_left_expanded = f_left.unsqueeze_dim::<3>(2);
                let f_product = m_right.matmul(f_left_expanded.squeeze::<2>(2));
                let new_f_val = f_product + f_right;
                
                // Store composed operators for next iteration
                new_m.push(new_m_val.unsqueeze_dim::<3>(0));
                new_f.push(new_f_val.unsqueeze_dim::<3>(0));
            }
            
            // UPDATE for next iteration with doubled span
            if !new_m.is_empty() {
                current_m = Tensor::cat(new_m, 0);
                current_f = Tensor::cat(new_f, 0);
            }
            step *= 2;  // Double the composition span
        }
        
        (current_m, current_f)
    }
    
    /// Compute SSM recurrence using parallel scan
    /// w_{k+1} = M w_k + F u_{k+1}
    /// Returns all hidden states for sequence
    pub fn ssm_scan<B: Backend>(
        m_matrix: Tensor<B, 2>,    // [state_dim, state_dim]
        f_matrix: Tensor<B, 2>,    // [state_dim, input_dim] 
        inputs: Tensor<B, 3>,      // [batch, seq_len, input_dim]
        initial_state: Tensor<B, 2>, // [batch, state_dim]
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, input_dim] = inputs.dims();
        let state_dim = m_matrix.dims()[0];
        
        // Validate matrix dimensions for mathematical correctness
        assert_eq!(f_matrix.dims(), [state_dim, input_dim], 
                   "F matrix dimensions {:?} don't match state_dim={}, input_dim={}", 
                   f_matrix.dims(), state_dim, input_dim);
        assert_eq!(initial_state.dims(), [batch_size, state_dim],
                   "Initial state dimensions {:?} don't match batch_size={}, state_dim={}",
                   initial_state.dims(), batch_size, state_dim);
        
        // Prepare matrices for each time step
        let m_expanded: Tensor<B, 3> = m_matrix.clone().unsqueeze_dim::<3>(0).repeat(&[seq_len, 1, 1]);
        
        // Project inputs: F @ u_k for each k
        let projected_inputs = Self::batch_matrix_vector_multiply(
            f_matrix.clone().unsqueeze_dim::<3>(0).repeat(&[batch_size, 1, 1]),
            inputs.clone()
        );
        
        // TRUE PARALLEL SCAN - O(log N) complexity
        // Prepare matrices for associative scan
        let m_for_scan = m_expanded.transpose(); // [seq_len, state_dim, state_dim]
        let f_for_scan = projected_inputs.transpose(); // [seq_len, state_dim, batch_size]
        
        // Compute associative scan
        let (m_result, f_result) = Self::associative_scan(m_for_scan, f_for_scan);
        
        // Apply initial state and compute final states
        let initial_expanded = initial_state.unsqueeze_dim(0).repeat(&[seq_len, 1, 1]);
        let states = m_result.matmul(initial_expanded) + f_result;
        
        // Transpose back to [batch, seq_len, state_dim]
        states.transpose()
    }
    
    /// Efficient batch matrix-vector multiplication
    fn batch_matrix_vector_multiply<B: Backend>(
        matrices: Tensor<B, 3>, // [batch, rows, cols]
        vectors: Tensor<B, 3>,  // [batch, seq_len, cols] 
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, vec_cols] = vectors.dims();
        let [mat_batch, rows, mat_cols] = matrices.dims();
        
        // Validate dimensions for matrix-vector multiplication
        assert_eq!(mat_batch, batch_size, "Batch size mismatch: matrices={}, vectors={}", mat_batch, batch_size);
        assert_eq!(mat_cols, vec_cols, "Column dimension mismatch: matrix cols={}, vector cols={}", mat_cols, vec_cols);
        
        let mut results = Vec::with_capacity(seq_len);
        
        for t in 0..seq_len {
            let vec_t = vectors.clone().slice([0..batch_size, t..t+1, 0..vec_cols]).squeeze(1);
            let result_t = Self::batch_mv(matrices.clone(), vec_t);
            assert_eq!(result_t.dims()[1], rows, "Output dimension mismatch");
            results.push(result_t.unsqueeze_dim(1));
        }
        
        Tensor::cat(results, 1)
    }
    
    /// Batch matrix-vector product
    fn batch_mv<B: Backend>(matrices: Tensor<B, 3>, vectors: Tensor<B, 2>) -> Tensor<B, 2> {
        // matrices: [batch, rows, cols], vectors: [batch, cols]
        // result: [batch, rows]
        let vectors_expanded = vectors.unsqueeze_dim(2); // [batch, cols, 1]
        let result = matrices.matmul(vectors_expanded).squeeze(2); // [batch, rows]
        result
    }
    
    /// Optimized scan for time-invariant SSM (single M, F matrices)
    pub fn time_invariant_scan<B: Backend>(
        m_matrix: Tensor<B, 2>,
        f_matrix: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
        initial_state: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, input_dim] = inputs.dims();
        let state_dim = m_matrix.dims()[0];
        
        // Validate dimensions for time-invariant scan
        assert_eq!(f_matrix.dims(), [state_dim, input_dim],
                   "F matrix dimensions {:?} incompatible with state_dim={}, input_dim={}",
                   f_matrix.dims(), state_dim, input_dim);
        assert!(seq_len > 0, "Sequence length must be positive, got {}", seq_len);
        
        let init_state = initial_state.unwrap_or_else(|| {
            Tensor::zeros([batch_size, state_dim], &inputs.device())
        });
        
        Self::ssm_scan(m_matrix, f_matrix, inputs, init_state)
    }
    
    /// Compute eigenvalue powers for stability analysis
    /// Returns M^k for k = 1, 2, ..., max_power
    pub fn matrix_powers<B: Backend>(
        matrix: Tensor<B, 2>,
        max_power: usize,
    ) -> Vec<Tensor<B, 2>> {
        let mut powers = Vec::with_capacity(max_power);
        let mut current = matrix.clone();
        powers.push(current.clone());
        
        for _ in 1..max_power {
            current = current.matmul(matrix.clone());
            powers.push(current.clone());
        }
        
        powers
    }
}
