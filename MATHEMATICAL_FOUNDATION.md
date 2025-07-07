# D-LinOSS Mathematical Foundation

**Damped Linear Oscillatory State-Space Models** - Comprehensive Mathematical Documentation

Based on arXiv:2505.12171: "Learning to Dissipate Energy in Oscillatory State-Space Models"  
Authors: Jared Boyer, T. Konstantin Rusch, Daniela Rus

---

## 📐 **Core Mathematical Formulation**

### **1. Continuous-Time Second-Order ODE System**

The fundamental D-LinOSS equation models a system of coupled damped harmonic oscillators:

```
x''(t) = -A x(t) - G x'(t) + B u(t)    [Equation 1]
y(t) = C x(t) + D u(t)                  [Output equation]
```

**where:**
- `x(t) ∈ ℝᵐ`: oscillator positions (physical: particle positions)
- `x'(t) ∈ ℝᵐ`: oscillator velocities (physical: particle velocities)
- `u(t) ∈ ℝᵖ`: external input signal
- `y(t) ∈ ℝᵠ`: system output

### **2. Matrix Parameters and Dimensions**

Following the paper's notation conventions:

| Symbol | Dimension | Role | Physical Interpretation |
|--------|-----------|------|------------------------|
| `p` | scalar | Input dimension | Number of external driving signals |
| `m` | scalar | Number of oscillators | Independent harmonic oscillators |
| `n = 2m` | scalar | Total state dimension | Position + velocity for each oscillator |
| `q` | scalar | Output dimension | Number of observable outputs |

### **3. Trainable Matrices**

#### **A ∈ ℝᵐˣᵐ (Diagonal)**: Natural Frequency Matrix
```
A = diag(A₁, A₂, ..., Aₘ)
```
- `Aᵢ > 0`: Natural frequency squared (ωᵢ²) for oscillator i
- Controls oscillation frequency of each harmonic oscillator
- Physical constraint: Must be positive for oscillatory behavior

#### **G ∈ ℝᵐˣᵐ (Diagonal)**: Damping Matrix ⭐**KEY INNOVATION**
```
G = diag(G₁, G₂, ..., Gₘ)
```
- `Gᵢ ≥ 0`: Damping coefficient (γᵢ) for oscillator i
- **LEARNABLE** energy dissipation rates - this is the paper's main contribution
- Physical constraint: Non-negative for energy dissipation (no energy injection)
- Enables adaptive learning of multiple timescales

#### **B ∈ ℝᵐˣᵖ**: Input Projection Matrix
```
B = [B₁₁ B₁₂ ... B₁ₚ]
    [B₂₁ B₂₂ ... B₂ₚ]
    [ ⋮   ⋮  ⋱   ⋮ ]
    [Bₘ₁ Bₘ₂ ... Bₘₚ]
```
- Maps external input `u(t) ∈ ℝᵖ` to driving forces on oscillators
- `Bᵢⱼ`: influence of input component j on oscillator i

#### **C ∈ ℝᵠˣᵐ**: Output Projection Matrix
```
C = [C₁₁ C₁₂ ... C₁ₘ]
    [C₂₁ C₂₂ ... C₂ₘ]
    [ ⋮   ⋮  ⋱   ⋮ ]
    [Cq₁ Cq₂ ... Cqₘ]
```
- Maps oscillator positions to observable outputs
- `Cᵢⱼ`: contribution of oscillator j position to output component i

#### **D ∈ ℝᵠˣᵖ**: Feedthrough Matrix
```
D = [D₁₁ D₁₂ ... D₁ₚ]
    [D₂₁ D₂₂ ... D₂ₚ]
    [ ⋮   ⋮  ⋱   ⋮ ]
    [Dq₁ Dq₂ ... Dqₚ]
```
- Direct input-to-output connection (bypassing oscillator dynamics)
- `Dᵢⱼ`: instantaneous influence of input j on output i

---

## 🔄 **State-Space Reformulation**

### **1. Augmented State Vector**

Define the augmented state `w(t) ∈ ℝⁿ` where `n = 2m`:

```
w(t) = [z(t); x(t)]ᵀ ∈ ℝ²ᵐ
```

where:
- `z(t) = x'(t) ∈ ℝᵐ`: velocity variables
- `x(t) ∈ ℝᵐ`: position variables

### **2. First-Order State-Space System**

The second-order ODE transforms to:

```
w'(t) = [z'(t); x'(t)]ᵀ = [x''(t); z(t)]ᵀ
```

Substituting the original ODE:

```
w'(t) = [-A x(t) - G z(t) + B u(t); z(t)]
```

In matrix form:

```
w'(t) = [[-G, -A]; [I, 0]] w(t) + [[B]; [0]] u(t)    [Equation 2]
y(t) = [0, C] w(t) + D u(t)
```

---

## ⚡ **IMEX Discretization Scheme**

### **1. Motivation for IMEX**

The **Implicit-Explicit (IMEX)** scheme treats:
- **Damping term** `G z(t)`: **IMPLICITLY** (for numerical stability)
- **Frequency term** `A x(t)`: **EXPLICITLY** (for computational efficiency)

This prevents numerical instabilities that occur with explicit treatment of stiff damping.

### **2. IMEX Discretization Equations**

```
z_{k+1} = z_k + Δt(-A x_k - G z_{k+1} + B u_{k+1})    [Implicit in z_{k+1}]
x_{k+1} = x_k + Δt z_{k+1}                            [Explicit in x_k]
```

### **3. Algebraic Solution**

**Step 1**: Solve for `z_{k+1}` from the implicit equation:

```
z_{k+1} + Δt G z_{k+1} = z_k + Δt(-A x_k + B u_{k+1})
(I + Δt G) z_{k+1} = z_k + Δt(-A x_k + B u_{k+1})
```

**Step 2**: Define Schur complement `S = I + Δt G`:

```
z_{k+1} = S⁻¹[z_k + Δt(-A x_k + B u_{k+1})]
```

**Step 3**: Substitute into position equation:

```
x_{k+1} = x_k + Δt S⁻¹[z_k + Δt(-A x_k + B u_{k+1})]
        = (I - Δt² S⁻¹ A) x_k + Δt S⁻¹ z_k + Δt² S⁻¹ B u_{k+1}
```

### **4. Discretized State-Space Matrices**

The final discrete-time system:

```
w_{k+1} = M w_k + F u_{k+1}    [State equation]
y_k = H w_k + D u_k            [Output equation]
```

**State Transition Matrix** `M ∈ ℝ²ᵐˣ²ᵐ`:
```
M = [[S⁻¹,           -Δt S⁻¹ A    ];
     [Δt S⁻¹,        I - Δt² S⁻¹ A]]
```

**Input Matrix** `F ∈ ℝ²ᵐˣᵖ`:
```
F = [[Δt S⁻¹ B];
     [Δt² S⁻¹ B]]
```

**Output Matrix** `H ∈ ℝᵠˣ²ᵐ`:
```
H = [0, C]
```

where `S = I + Δt G` is the Schur complement.

---

## 📊 **Stability Analysis**

### **1. Stability Theorem (Proposition 3.1)**

**Theorem**: The discretized D-LinOSS system `w_{k+1} = M w_k + F u_{k+1}` is exponentially stable if and only if the spectral radius `ρ(M) < 1`.

For the IMEX discretization, this is equivalent to:

```
(Gᵢ - Δt Aᵢ)² ≤ 4Aᵢ  for all oscillators i = 1, 2, ..., m
```

### **2. Eigenvalue Analysis**

The eigenvalues of the 2×2 block corresponding to oscillator i are:

```
λᵢ,₁,₂ = [1 + Δt Gᵢ/2 - Δt² Aᵢ/2 ± √Δᵢ] / (1 + Δt Gᵢ)
```

where the discriminant is:

```
Δᵢ = (1 + Δt Gᵢ/2 - Δt² Aᵢ/2)² - (1 - Δt² Aᵢ/2)²
```

### **3. Physical Interpretation of Stability**

- **Underdamped** (`Gᵢ` small): Risk of numerical instability if condition violated
- **Overdamped** (`Gᵢ` large): Always stable but potentially slow dynamics
- **Critical damping**: Optimal balance between stability and responsiveness

The stability condition ensures that learned damping parameters maintain system stability while allowing flexible temporal modeling.

---

## 🔄 **Parallel Scan Algorithm**

### **1. Associative Operation**

The D-LinOSS recurrence `w_{k+1} = M w_k + F u_{k+1}` can be computed efficiently using parallel scan with the associative operation:

```
(M₁, F₁) ⊗ (M₂, F₂) = (M₂M₁, M₂F₁ + F₂)
```

### **2. Associativity Proof**

For operators `(A, a)`, `(B, b)`, `(C, c)`:

```
[(A,a) ⊗ (B,b)] ⊗ (C,c) = (AB, Ab + a) ⊗ (C,c) = (ABC, ABc + Ab + a)
(A,a) ⊗ [(B,b) ⊗ (C,c)] = (A,a) ⊗ (BC, Bc + b) = (ABC, A(Bc + b) + a) = (ABC, ABc + Ab + a)
```

Since both expressions equal `(ABC, ABc + Ab + a)`, the operation is associative.

### **3. Complexity Analysis**

- **Sequential**: O(T) time for T time steps
- **Parallel Scan**: O(log T) time with O(T) processors
- **Space Complexity**: O(T) for storing intermediate results

This enables efficient processing of long sequences, crucial for modeling long-range dependencies.

---

## 🏗️ **Multi-Layer Architecture**

### **1. Hierarchical Composition**

A D-LinOSS block with L layers computes:

```
h₀ = input ∈ ℝᵖ
h₁ = DLinOSS₁(h₀; A₁, G₁, B₁, C₁, D₁) ∈ ℝᵠ
h₂ = DLinOSS₂(h₁; A₂, G₂, B₂, C₂, D₂) ∈ ℝᵠ
⋮
y = DLinOSSₗ(hₗ₋₁; Aₗ, Gₗ, Bₗ, Cₗ, Dₗ) ∈ ℝᵠ
```

### **2. Multi-Scale Learning**

Different layers can specialize in different temporal scales through their damping matrices:

- **Layer 1**: Fast dynamics, fine-grained patterns (small `G₁`)
- **Layer L**: Slow dynamics, coarse-grained patterns (large `Gₗ`)
- **Within Layer**: Different oscillators learn different rates

### **3. Representation Learning**

Each layer transforms oscillator states into higher-level features:

```
Positions → Features → Positions → Features → ... → Output
x¹ → h₁ → x² → h₂ → ... → y
```

This creates rich hierarchical representations while maintaining the physical interpretability of oscillatory dynamics.

---

## 🧪 **Experimental Validation**

### **1. Exponential Decay Benchmark**

**Test Setup**: Learn to reproduce exponential decay from impulse input:

```
Input: u(t) = δ(t) * A    (impulse at t=0)
Target: y(t) = A * exp(-γt)    (exponential decay)
```

**Mathematical Expectation**: The system should learn damping parameters `Gᵢ ≈ γ` to match the target decay rate.

### **2. Success Criteria**

- **Low MSE**: Between predicted and target exponential curves
- **Parameter Correlation**: Learned `Gᵢ` should correlate with target decay rates `γ`
- **Generalization**: System should work for unseen decay rates

### **3. Physical Validation**

For a critically damped oscillator:
```
x''(t) + 2γx'(t) + γ²x(t) = 0
Solution: x(t) ≈ A exp(-γt)  for appropriate initial conditions
```

---

## 💡 **Key Innovations**

### **1. Learnable Damping**
- Traditional SSMs: Fixed energy dissipation mechanisms
- D-LinOSS: **Adaptive damping matrices** learn optimal energy dissipation for each task

### **2. Multi-Timescale Modeling**
- Single model captures both fast and slow dynamics
- Different oscillators/layers specialize in different temporal scales
- No manual hyperparameter tuning for timescales

### **3. Stability Guarantees**
- Theoretical stability conditions ensure robust training
- IMEX discretization prevents numerical instabilities
- Maintains linear complexity while enabling rich dynamics

### **4. Physical Interpretability**
- Clear physical meaning: damped harmonic oscillators
- Damping parameters directly interpretable as energy dissipation rates
- Bridging physics and machine learning

---

## 📚 **Implementation Notes**

### **1. Numerical Considerations**
- Use IMEX scheme for stability
- Monitor stability condition during training
- Layer normalization for gradient stability

### **2. Optimization**
- Parallel scan enables O(log T) sequence processing
- GPU-friendly matrix operations
- Efficient discretization updates

### **3. Hyperparameter Guidelines**
- `Δt`: Time step, affects stability condition
- `m`: Number of oscillators, controls model capacity
- `L`: Number of layers, enables hierarchical learning
- Initialization: Small `G`, moderate `A` for stable start

---

*This mathematical foundation provides the theoretical basis for the D-LinOSS Rust implementation, ensuring both mathematical rigor and practical applicability.*
