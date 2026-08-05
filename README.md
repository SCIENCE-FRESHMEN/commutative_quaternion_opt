# Commutative Quaternion Polynomial Optimization
This repository provides a fully modular, production-ready Python implementation of randomized polynomial-time approximation algorithms for **sphere-constrained homogeneous polynomial optimization over commutative quaternion space $\mathbb{H}$**, whose special case covers the best rank-one commutative quaternion tensor approximation problem (NP-hard). All modules strictly map to the mathematical statements, probability lemmas, and algorithms proposed in He et al. (2025) arXiv:2512.00779.

## 1. Theoretical Background & Core Definitions
### 1.1 Commutative Quaternion Algebra
A commutative quaternion scalar is defined as
$$
\boldsymbol{q} = q_0 + q_1 i + q_2 j + q_3 k,\quad q_0,q_1,q_2,q_3\in\mathbb{R},
$$
with Segre commutative multiplication identities:
$$
i^2 = k^2 = -1,\quad j^2 = 1,\quad ij=ji=k,\quad jk=kj=i,\quad ki=ik=-j,\quad ijk=-1.
$$
Conjugation of $\boldsymbol{q}$:
$$
\overline{\boldsymbol{q}} = q_0 - q_1 i + q_2 j - q_3 k,
$$
modulus of commutative quaternion:
$$
|\boldsymbol{q}| = \sqrt{\operatorname{Re}(\boldsymbol{q}\cdot\overline{\boldsymbol{q}})} = \sqrt{q_0^2+q_1^2+q_2^2+q_3^2}.
$$

For vector $\boldsymbol{x}\in\mathbb{H}^n$, decompose into real vector components:
$$
\boldsymbol{x} = \boldsymbol{x}_0 + \boldsymbol{x}_1 i + \boldsymbol{x}_2 j + \boldsymbol{x}_3 k,\quad \boldsymbol{x}_0,\boldsymbol{x}_1,\boldsymbol{x}_2,\boldsymbol{x}_3\in\mathbb{R}^n.
$$
Conjugate transpose $\boldsymbol{x}^H = \overline{\boldsymbol{x}}^\top$, inner product and spherical constraint:
$$
\langle \boldsymbol{x},\boldsymbol{y} \rangle = \operatorname{Re}\big(\boldsymbol{x}^H \boldsymbol{y}\big) = \boldsymbol{x}_0^\top\boldsymbol{y}_0 + \boldsymbol{x}_1^\top\boldsymbol{y}_1 + \boldsymbol{x}_2^\top\boldsymbol{y}_2 + \boldsymbol{x}_3^\top\boldsymbol{y}_3,\quad
\|\boldsymbol{x}\| = \sqrt{\langle \boldsymbol{x},\boldsymbol{x} \rangle}.
$$
Unit commutative quaternion sphere:
$$
S^n = \big\{\boldsymbol{x}\in\mathbb{H}^n \,\big|\, \|\boldsymbol{x}\| = 1\big\}.
$$

### 1.2 Homogeneous Polynomial & Multilinear Form Optimization
The target homogeneous polynomial maximization problem $(P)$:
$$
\begin{aligned}
(P)\quad \max_{\boldsymbol{x}\in S^n}\quad &\operatorname{Re}\, H(\boldsymbol{x}) \\
\text{s.t.}\quad &\|\boldsymbol{x}\| = 1,
\end{aligned}
$$
where $H(\boldsymbol{x})$ is a $d$-th degree homogeneous commutative quaternion polynomial:
$$
H(\boldsymbol{x}) = \sum_{1\leq i_1\leq i_2\leq\cdots\leq i_d\leq n} a_{i_1i_2\cdots i_d}\boldsymbol{x}_{i_1}\boldsymbol{x}_{i_2}\cdots\boldsymbol{x}_{i_d},\quad a_{i_1\cdots i_d}\in\mathbb{H}.
$$

The multilinear relaxation counterpart problem $(F)$:
$$
\begin{aligned}
(F)\quad \max\quad &\operatorname{Re}\, \mathcal{F}(\boldsymbol{x}^1,\boldsymbol{x}^2,\dots,\boldsymbol{x}^d) \\
\text{s.t.}\quad &\boldsymbol{x}^k \in S^{n_k},\quad k=1,2,\dots,d,
\end{aligned}
$$
with multilinear form induced by commutative quaternion tensor $\mathcal{F}\in\mathbb{H}^{n_1\times n_2\times\cdots\times n_d}$:
$$
\mathcal{F}(\boldsymbol{x}^1,\dots,\boldsymbol{x}^d) = \sum_{i_1=1}^{n_1}\cdots\sum_{i_d=1}^{n_d} \mathcal{F}_{i_1\cdots i_d} \boldsymbol{x}^1_{i_1}\boldsymbol{x}^2_{i_2}\cdots\boldsymbol{x}^d_{i_d}.
$$

### 1.3 Best Rank-One Tensor Approximation (Key Application)
The low-rank recovery task for commutative quaternion tensor $\mathcal{F}$:
$$
\min_{\boldsymbol{x}^k\in\mathbb{H}^{n_k}} \frac12\big\|\boldsymbol{x}^1\otimes\boldsymbol{x}^2\otimes\cdots\otimes\boldsymbol{x}^d - \mathcal{F}\big\|^2,
$$
equivalently reduces to solving problem $(F)$ via the identity
$$
\min_{\lambda,\|\boldsymbol{x}^k\|=1} \frac12\big\|\lambda\big(\otimes_{k=1}^d\boldsymbol{x}^k\big)-\mathcal{F}\big\|^2
= \|\mathcal{F}\|^2 - \Big(\max_{\|\boldsymbol{x}^k\|=1}\operatorname{Re}\mathcal{F}(\boldsymbol{x}^1,\dots,\boldsymbol{x}^d)\Big)^2.
$$
This equivalence validates our framework for color image denoising and low-rank recovery demos.

## 2. Theoretical Performance Bound & Approximation Ratio
### 2.1 Random Quaternion Spherical Probability Inequality
Let $\xi,\boldsymbol{a}\sim \text{Uniform}(S^n)$, for any $\gamma>0,\gamma\ln n < n$, there exists constant $c(\gamma)>0$ such that
$$
\mathbb{P}\left\{\operatorname{Re}\big(\boldsymbol{a}^\top\xi\big) \geq \sqrt{\frac{\gamma\ln n}{n}}\right\} \geq \frac{c(\gamma)}{n^{4.5\gamma}\sqrt{\ln n}}.
$$
An improved bound with arbitrary small $\delta>0$:
$$
\mathbb{P}\left\{\operatorname{Re}\big(\boldsymbol{a}^\top\xi\big) \geq \sqrt{\frac{\gamma\ln n}{n}}\|\boldsymbol{a}\|\right\} \geq \frac{c(\gamma,\delta)}{n^{(2+\delta+\delta^2/2)\gamma}\sqrt{\ln n}}.
$$

### 2.2 Approximation Ratio for Multilinear Algorithm 1
Assume $n_1\leq n_2\leq\cdots\leq n_d$. Algorithm 1 returns feasible solutions to $(F)$ satisfying, with probability at least $1-\epsilon$ after sufficient random trials:
$$
\operatorname{Re}\mathcal{F}\big(\hat{\boldsymbol{x}}^1,\dots,\hat{\boldsymbol{x}}^d\big) \geq \gamma^{\frac{d-2}{2}} \left(\prod_{k=1}^{d-2}\sqrt{\frac{\ln n_k}{n_k}}\right) v^*(F),
$$
where $v^*(F) = \max_{\boldsymbol{x}^k\in S^{n_k}}\operatorname{Re}\mathcal{F}(\boldsymbol{x}^1,\dots,\boldsymbol{x}^d)$ denotes the optimal objective value of $(F)$.

### 2.3 Approximation Ratio for Homogeneous Polynomial Algorithm 2
Define global approximation coefficient
$$
\tau(P) = \frac{d!}{d^d} \left(\gamma\cdot\frac{\ln n}{n}\right)^{\frac{d-2}{2}}.
$$
1. When polynomial degree $d$ is odd:
$$
\operatorname{Re}\,H(\hat{\boldsymbol{x}}) \geq \tau(P)\cdot v^*(P),
$$
$v^*(P)=\max_{\boldsymbol{x}\in S^n}\operatorname{Re}H(\boldsymbol{x})$.
2. When $d$ is even: let $\underline{v}(P)=\min_{\boldsymbol{x}\in S^n}\operatorname{Re}H(\boldsymbol{x})$, then
$$
\operatorname{Re}\,H(\hat{\boldsymbol{x}}) - \underline{v}(P) \geq 2\tau(P)\cdot v^*(P).
$$

### 2.4 Canonical Worst-Case Test Instance Bound (Proposition 6.1)
For 3rd-order equal-dimension tensor $\mathcal{F}\in\mathbb{H}^{n\times n\times n}$ with real component $\mathcal{F}_0 = \mathbf{1}_{n\times n\times n}$ (all-ones tensor), all imaginary blocks zero:
$$
v^*(F) \leq v_{\text{upper}} = 2\sqrt{n_1n_2n_3}.
$$
This closed-form upper bound serves as ground truth reference for all numerical validation and benchmarking experiments in this repo.

## 3. Repository Key Features
1. **Full Mathematical Traceability**
Every core function is annotated with matching paper equations, lemmas, theorems, and algorithm indices; all tensor relaxations and probability bounds are implemented exactly as derived in the original manuscript.

2. **Strictly Modular Decoupled Architecture (SRP Compliant)**
- `src/algebra`: Segre commutative quaternion arithmetic, conjugate, norm, inner product
- `src/core`: Real block matrix conversion & SVD bilinear subproblem solver (Lemma 2.1)
- `src/data`: Generate canonical all-ones tensor test instance from Proposition 6.1
- `src/algorithms`: Algorithm 1 (Multilinear Form) & Algorithm 2 (Homogeneous Polynomial)
- `src/utils`: Logging, ratio metric calculation, statistical aggregation
- `src/config`: Dataclass hyperparameter management for sampling count, $\gamma$, $\epsilon$, tensor dimension $n$

3. **Type-Safe Numerical Implementation**
Full static type hints with `numpy.typing`, strict dimension assertion checks to eliminate silent tensor shape mismatches during large-scale random sampling.

4. **End-to-End Benchmark Pipeline for Bound Verification**
Automated parameter sweep over $n\in\{2,3,4,5,6,7\}$, trial counts $\{1,5,10,20,50,100,500,1000,10000\}$. Metrics computed: average approximation ratio, worst-case ratio, standard deviation, best-case ratio relative to theoretical upper bound $v_{\text{upper}}=2\sqrt{n^3}$. All statistics exported as structured CSV files in `./reports/`.

5. **Publication-Quality Visualization Tools**
Native export of high-DPI vector plots (PDF/PNG) directly from benchmark CSV outputs for theoretical bound validation figures, showing empirical ratio convergence against $\sqrt{\ln n/n}$ asymptotic guarantee.

6. **Practical Computer Vision Demo: Color Image Low-Rank Recovery**
Map RGB color channels to 4-channel commutative quaternion components, formulate image denoising as best rank-one commutative quaternion tensor approximation. Script `demo_image_recovery.py` outputs original noisy tensor, rank-one reconstructed image to `./demo_output/`.

## 4. Project Directory Structure
