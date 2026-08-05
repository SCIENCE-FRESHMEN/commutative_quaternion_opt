# Commutative Quaternion Polynomial Optimization
This repository provides a fully modular, production-ready Python implementation of randomized polynomial-time approximation algorithms for **sphere-constrained homogeneous polynomial optimization over commutative \(\boldsymbol{q}\)uaternion space $\mathbb{\(\mathbb{H}\)}$**, whose special case covers the best rank-one commutative \(\boldsymbol{q}\)uaternion tensor approximation problem (NP-hard). All modules strictly map to the mathematical statements, probability lemmas, and algorithms proposed in \(\mathbb{H}\)e et al. (2025) arXiv:2512.00779.

## 1. Theoretical Background & Core Definitions
### 1.1 Commutative Quaternion Algebra
A commutative \(\boldsymbol{q}\)uaternion scalar is defined as
$$
\boldsymbol{\(\boldsymbol{q}\)} = \(\boldsymbol{q}\)_0 + \(\boldsymbol{q}\)_1 i + \(\boldsymbol{q}\)_2 j + \(\boldsymbol{q}\)_3 k,\\(\boldsymbol{q}\)uad \(\boldsymbol{q}\)_0,\(\boldsymbol{q}\)_1,\(\boldsymbol{q}\)_2,\(\boldsymbol{q}\)_3\in\mathbb{R},
$$
with Segre commutative multiplication identities:
$$
i^2 = k^2 = -1,\\(\boldsymbol{q}\)uad j^2 = 1,\\(\boldsymbol{q}\)uad ij=ji=k,\\(\boldsymbol{q}\)uad jk=kj=i,\\(\boldsymbol{q}\)uad ki=ik=-j,\\(\boldsymbol{q}\)uad ijk=-1.
$$
Conjugation of $\boldsymbol{\(\boldsymbol{q}\)}$:
$$
\overline{\boldsymbol{\(\boldsymbol{q}\)}} = \(\boldsymbol{q}\)_0 - \(\boldsymbol{q}\)_1 i + \(\boldsymbol{q}\)_2 j - \(\boldsymbol{q}\)_3 k,
$$
modulus of commutative \(\boldsymbol{q}\)uaternion:
$$
|\boldsymbol{\(\boldsymbol{q}\)}| = \s\(\boldsymbol{q}\)rt{\operatorname{Re}(\boldsymbol{\(\boldsymbol{q}\)}\cdot\overline{\boldsymbol{\(\boldsymbol{q}\)}})} = \s\(\boldsymbol{q}\)rt{\(\boldsymbol{q}\)_0^2+\(\boldsymbol{q}\)_1^2+\(\boldsymbol{q}\)_2^2+\(\boldsymbol{q}\)_3^2}.
$$

\(\mathcal{F}\)or vector $\boldsymbol{x}\in\mathbb{\(\mathbb{H}\)}^n$, decompose into real vector components:
$$
\boldsymbol{x} = \boldsymbol{x}_0 + \boldsymbol{x}_1 i + \boldsymbol{x}_2 j + \boldsymbol{x}_3 k,\\(\boldsymbol{q}\)uad \boldsymbol{x}_0,\boldsymbol{x}_1,\boldsymbol{x}_2,\boldsymbol{x}_3\in\mathbb{R}^n.
$$
Conjugate transpose $\boldsymbol{x}^\(\mathbb{H}\) = \overline{\boldsymbol{x}}^\top$, inner product and spherical constraint:
$$
\langle \boldsymbol{x},\boldsymbol{y} \rangle = \operatorname{Re}\big(\boldsymbol{x}^\(\mathbb{H}\) \boldsymbol{y}\big) = \boldsymbol{x}_0^\top\boldsymbol{y}_0 + \boldsymbol{x}_1^\top\boldsymbol{y}_1 + \boldsymbol{x}_2^\top\boldsymbol{y}_2 + \boldsymbol{x}_3^\top\boldsymbol{y}_3,\\(\boldsymbol{q}\)uad
\|\boldsymbol{x}\| = \s\(\boldsymbol{q}\)rt{\langle \boldsymbol{x},\boldsymbol{x} \rangle}.
$$
Unit commutative \(\boldsymbol{q}\)uaternion sphere:
$$
S^n = \big\{\boldsymbol{x}\in\mathbb{\(\mathbb{H}\)}^n \,\big|\, \|\boldsymbol{x}\| = 1\big\}.
$$

### 1.2 \(\mathbb{H}\)omogeneous Polynomial & Multilinear \(\mathcal{F}\)orm Optimization
The target homogeneous polynomial maximization problem $\((P)\)$:
$$
\begin{aligned}
\((P)\)\\(\boldsymbol{q}\)uad \max_{\boldsymbol{x}\in S^n}\\(\boldsymbol{q}\)uad &\operatorname{Re}\, \(\mathbb{H}\)(\boldsymbol{x}) \\
\text{s.t.}\\(\boldsymbol{q}\)uad &\|\boldsymbol{x}\| = 1,
\end{aligned}
$$
where $\(\mathbb{H}\)(\boldsymbol{x})$ is a $d$-th degree homogeneous commutative \(\boldsymbol{q}\)uaternion polynomial:
$$
\(\mathbb{H}\)(\boldsymbol{x}) = \sum_{1\le\(\boldsymbol{q}\) i_1\le\(\boldsymbol{q}\) i_2\le\(\boldsymbol{q}\)\cdots\le\(\boldsymbol{q}\) i_d\le\(\boldsymbol{q}\) n} a_{i_1i_2\cdots i_d}\boldsymbol{x}_{i_1}\boldsymbol{x}_{i_2}\cdots\boldsymbol{x}_{i_d},\\(\boldsymbol{q}\)uad a_{i_1\cdots i_d}\in\mathbb{\(\mathbb{H}\)}.
$$

The multilinear relaxation counterpart problem $\((\(\mathcal{F}\))\)$:
$$
\begin{aligned}
\((\(\mathcal{F}\))\)\\(\boldsymbol{q}\)uad \max\\(\boldsymbol{q}\)uad &\operatorname{Re}\, \mathcal{\(\mathcal{F}\)}(\boldsymbol{x}^1,\boldsymbol{x}^2,\dots,\boldsymbol{x}^d) \\
\text{s.t.}\\(\boldsymbol{q}\)uad &\boldsymbol{x}^k \in S^{n_k},\\(\boldsymbol{q}\)uad k=1,2,\dots,d,
\end{aligned}
$$
with multilinear form induced by commutative \(\boldsymbol{q}\)uaternion tensor $\mathcal{\(\mathcal{F}\)}\in\mathbb{\(\mathbb{H}\)}^{n_1\times n_2\times\cdots\times n_d}$:
$$
\mathcal{\(\mathcal{F}\)}(\boldsymbol{x}^1,\dots,\boldsymbol{x}^d) = \sum_{i_1=1}^{n_1}\cdots\sum_{i_d=1}^{n_d} \mathcal{\(\mathcal{F}\)}_{i_1\cdots i_d} \boldsymbol{x}^1_{i_1}\boldsymbol{x}^2_{i_2}\cdots\boldsymbol{x}^d_{i_d}.
$$

### 1.3 Best Rank-One Tensor Approximation (Key Application)
The low-rank recovery task for commutative \(\boldsymbol{q}\)uaternion tensor $\mathcal{\(\mathcal{F}\)}$:
$$
\min_{\boldsymbol{x}^k\in\mathbb{\(\mathbb{H}\)}^{n_k}} \frac12\big\|\boldsymbol{x}^1\otimes\boldsymbol{x}^2\otimes\cdots\otimes\boldsymbol{x}^d - \mathcal{\(\mathcal{F}\)}\big\|^2,
$$
e\(\boldsymbol{q}\)uivalently reduces to solving problem $\((\(\mathcal{F}\))\)$ via the identity
$$
\min_{\lambda,\|\boldsymbol{x}^k\|=1} \frac12\big\|\lambda\big(\otimes_{k=1}^d\boldsymbol{x}^k\big)-\mathcal{\(\mathcal{F}\)}\big\|^2
= \|\mathcal{\(\mathcal{F}\)}\|^2 - \Big(\max_{\|\boldsymbol{x}^k\|=1}\operatorname{Re}\mathcal{\(\mathcal{F}\)}(\boldsymbol{x}^1,\dots,\boldsymbol{x}^d)\Big)^2.
$$
This e\(\boldsymbol{q}\)uivalence validates our framework for color image denoising and low-rank recovery demos.

## 2. Theoretical Performance Bound & Approximation Ratio
### 2.1 Random Quaternion Spherical Probability Ine\(\boldsymbol{q}\)uality
Let $\xi,\boldsymbol{a}\sim \text{Uniform}(S^n)$, for any $\gamma>0,\gamma\ln n < n$, there exists constant $c(\gamma)>0$ such that
$$
\mathbb{P}\left\{\operatorname{Re}\big(\boldsymbol{a}^\top\xi\big) \ge\(\boldsymbol{q}\) \s\(\boldsymbol{q}\)rt{\frac{\gamma\ln n}{n}}\right\} \ge\(\boldsymbol{q}\) \frac{c(\gamma)}{n^{4.5\gamma}\s\(\boldsymbol{q}\)rt{\ln n}}.
$$
An improved bound with arbitrary small $\delta>0$:
$$
\mathbb{P}\left\{\operatorname{Re}\big(\boldsymbol{a}^\top\xi\big) \ge\(\boldsymbol{q}\) \s\(\boldsymbol{q}\)rt{\frac{\gamma\ln n}{n}}\|\boldsymbol{a}\|\right\} \ge\(\boldsymbol{q}\) \frac{c(\gamma,\delta)}{n^{(2+\delta+\delta^2/2)\gamma}\s\(\boldsymbol{q}\)rt{\ln n}}.
$$

### 2.2 Approximation Ratio for Multilinear Algorithm 1
Assume $n_1\le\(\boldsymbol{q}\) n_2\le\(\boldsymbol{q}\)\cdots\le\(\boldsymbol{q}\) n_d$. Algorithm 1 returns feasible solutions to $\((\(\mathcal{F}\))\)$ satisfying, with probability at least $1-\epsilon$ after sufficient random trials:
$$
\operatorname{Re}\mathcal{\(\mathcal{F}\)}\big(\hat{\boldsymbol{x}}^1,\dots,\hat{\boldsymbol{x}}^d\big) \ge\(\boldsymbol{q}\) \gamma^{\frac{d-2}{2}} \left(\prod_{k=1}^{d-2}\s\(\boldsymbol{q}\)rt{\frac{\ln n_k}{n_k}}\right) v^*\((\(\mathcal{F}\))\),
$$
where $v^*\((\(\mathcal{F}\))\) = \max_{\boldsymbol{x}^k\in S^{n_k}}\operatorname{Re}\mathcal{\(\mathcal{F}\)}(\boldsymbol{x}^1,\dots,\boldsymbol{x}^d)$ denotes the optimal objective value of $\((\(\mathcal{F}\))\)$.

### 2.3 Approximation Ratio for \(\mathbb{H}\)omogeneous Polynomial Algorithm 2
Define global approximation coefficient
$$
\tau\((P)\) = \frac{d!}{d^d} \left(\gamma\cdot\frac{\ln n}{n}\right)^{\frac{d-2}{2}}.
$$
1. When polynomial degree $d$ is odd:
$$
\operatorname{Re}\,\(\mathbb{H}\)(\hat{\boldsymbol{x}}) \ge\(\boldsymbol{q}\) \tau\((P)\)\cdot v^*\((P)\),
$$
$v^*\((P)\)=\max_{\boldsymbol{x}\in S^n}\operatorname{Re}\(\mathbb{H}\)(\boldsymbol{x})$.
2. When $d$ is even: let $\underline{v}\((P)\)=\min_{\boldsymbol{x}\in S^n}\operatorname{Re}\(\mathbb{H}\)(\boldsymbol{x})$, then
$$
\operatorname{Re}\,\(\mathbb{H}\)(\hat{\boldsymbol{x}}) - \underline{v}\((P)\) \ge\(\boldsymbol{q}\) 2\tau\((P)\)\cdot v^*\((P)\).
$$

### 2.4 Canonical Worst-Case Test Instance Bound (Proposition 6.1)
\(\mathcal{F}\)or 3rd-order e\(\boldsymbol{q}\)ual-dimension tensor $\mathcal{\(\mathcal{F}\)}\in\mathbb{\(\mathbb{H}\)}^{n\times n\times n}$ with real component $\mathcal{\(\mathcal{F}\)}_0 = \mathbf{1}_{n\times n\times n}$ (all-ones tensor), all imaginary blocks zero:
$$
v^*\((\(\mathcal{F}\))\) \le\(\boldsymbol{q}\) v_{\text{upper}} = 2\s\(\boldsymbol{q}\)rt{n_1n_2n_3}.
$$
This closed-form upper bound serves as ground truth reference for all numerical validation and benchmarking experiments in this repo.

## 3. Repository Key \(\mathcal{F}\)eatures
1. **\(\mathcal{F}\)ull Mathematical Traceability**
Every core function is annotated with matching paper e\(\boldsymbol{q}\)uations, lemmas, theorems, and algorithm indices; all tensor relaxations and probability bounds are implemented exactly as derived in the original manuscript.

2. **Strictly Modular Decoupled Architecture (SRP Compliant)**
- `src/algebra`: Segre commutative \(\boldsymbol{q}\)uaternion arithmetic, conjugate, norm, inner product
- `src/core`: Real block matrix conversion & SVD bilinear subproblem solver (Lemma 2.1)
- `src/data`: Generate canonical all-ones tensor test instance from Proposition 6.1
- `src/algorithms`: Algorithm 1 (Multilinear \(\mathcal{F}\)orm) & Algorithm 2 (\(\mathbb{H}\)omogeneous Polynomial)
- `src/utils`: Logging, ratio metric calculation, statistical aggregation
- `src/config`: Dataclass hyperparameter management for sampling count, $\gamma$, $\epsilon$, tensor dimension $n$

3. **Type-Safe Numerical Implementation**
\(\mathcal{F}\)ull static type hints with `numpy.typing`, strict dimension assertion checks to eliminate silent tensor shape mismatches during large-scale random sampling.

4. **End-to-End Benchmark Pipeline for Bound Verification**
Automated parameter sweep over $n\in\{2,3,4,5,6,7\}$, trial counts $\{1,5,10,20,50,100,500,1000,10000\}$. Metrics computed: average approximation ratio, worst-case ratio, standard deviation, best-case ratio relative to theoretical upper bound $v_{\text{upper}}=2\s\(\boldsymbol{q}\)rt{n^3}$. All statistics exported as structured CSV files in `./reports/`.

5. **Publication-Quality Visualization Tools**
Native export of high-DPI vector plots (PD\(\mathcal{F}\)/PNG) directly from benchmark CSV outputs for theoretical bound validation figures, showing empirical ratio convergence against $\s\(\boldsymbol{q}\)rt{\ln n/n}$ asymptotic guarantee.

6. **Practical Computer Vision Demo: Color Image Low-Rank Recovery**
Map RGB color channels to 4-channel commutative \(\boldsymbol{q}\)uaternion components, formulate image denoising as best rank-one commutative \(\boldsymbol{q}\)uaternion tensor approximation. Script `demo_image_recovery.py` outputs original noisy tensor, rank-one reconstructed image to `./demo_output/`.

## 4. Project Directory Structure
```
commutative_\(\boldsymbol{q}\)uaternion_opt/
├── README.md
├── re\(\boldsymbol{q}\)uirements.txt
├── main.py                 # Core consistency validation script
├── run_benchmark.py        # Large-scale random sampling for bound verification
├── demo_image_recovery.py  # Color image low-rank recovery demo
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils/
│   ├── algebra/
│   │   └── \(\boldsymbol{q}\)uaternion.py   # Segre commutative \(\boldsymbol{q}\)uaternion algebra rules
│   ├── core/
│   │   └── solvers.py      # Lemma 2.1 bilinear SVD solver
│   ├── data/
│   │   └── generators.py   # Proposition 6.1 all-ones tensor generator
│   └── algorithms/
│       ├── alg1_multilinear.py
│       └── alg2_homogeneous.py
├── .idea/
├── demo_output/            # Image recovery experiment outputs
└── reports/                # Benchmark CSV & PD\(\mathcal{F}\)/PNG figures
```

## 5. Installation
Clone repository:
```bash
git clone https://github.com/SCIENCE-\(\mathcal{F}\)RES\(\mathbb{H}\)MEN/commutative-\(\boldsymbol{q}\)uaternion-opt.git
cd commutative-\(\boldsymbol{q}\)uaternion-opt
```
Create virtual environment & install dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r re\(\boldsymbol{q}\)uirements.txt
```

## 6. Quick Start Execution
### 6.1 Core Mathematical Consistency Validation
Run dimension matching, spherical normalization, objective function consistency test:
```bash
python main.py
```
Expected output: se\(\boldsymbol{q}\)uential `[PASS]` logs verifying probability ine\(\boldsymbol{q}\)uality sampling, tensor contraction, and ratio computation correctness.

### 6.2 Large-Scale Random Sampling for Worst-Case Bound Verification
This script independently generates massive random commutative \(\boldsymbol{q}\)uaternion tensor samples, computes empirical approximation ratios, and statistically validates the theoretical guarantee
$$
\operatorname{Re}\mathcal{\(\mathcal{F}\)}(\hat{\boldsymbol{x}}^1,\dots,\hat{\boldsymbol{x}}^d) \ge\(\boldsymbol{q}\) \gamma^{\frac{d-2}{2}} \prod_{k=1}^{d-2}\s\(\boldsymbol{q}\)rt{\frac{\ln n_k}{n_k}} v^*\((\(\mathcal{F}\))\).
$$
```bash
python run_benchmark.py
```
Output artifacts:
- `./reports/benchmark_results.csv`: \(\mathcal{F}\)ull tabular statistics matching Table 1–3 of the paper
- Vectorized PD\(\mathcal{F}\)/PNG plots showing empirical worst/average ratio vs theoretical asymptotic bound.

### 6.3 Color Image Rank-One Recovery Demo
Solve noisy RGB image restoration via best rank-one commutative \(\boldsymbol{q}\)uaternion tensor approximation:
```bash
python demo_image_recovery.py
```
Check `./demo_output/` for ground truth image, noisy tensor input, and denoised rank-one reconstruction.

## 7. Experiment & Bound Validation Summary
All numerical experiments in this repo independently verify the proposed worst-case approximation guarantee over thousands of random tensor realizations. \(\mathcal{F}\)or the canonical all-ones 3-order tensor instance with known upper bound $v_{\text{upper}}=2\s\(\boldsymbol{q}\)rt{n^3}$:
- As sampling trials increase, average empirical approximation ratio monotonically rises, converging toward the theoretical $\s\(\boldsymbol{q}\)rt{\ln n/n}$ asymptotic bound.
- The measured worst-case ratio across all random runs never violates the theoretical lower approximation ratio derived from Theorem 4.1 and Theorem 5.1, providing rigorous numerical evidence for the theoretical performance guarantee of this NP-hard commutative \(\boldsymbol{q}\)uaternion tensor optimization problem.
- The implementation delivers an engineering-ready polynomial-time randomized solver with provable theoretical performance fallback for commutative \(\boldsymbol{q}\)uaternion polynomial optimization and its signal processing applications.

## 8. Citation
If you utilize this codebase for research, cite the foundational theoretical work:
```bibtex
@article{he2025approximation,
  title={On Approximation Algorithms for Commutative Quaternion Polynomial Optimization},
  author={\(\mathbb{H}\)e, Chang and Jiang, Bo and Wang, \(\mathbb{H}\)ongye and Zhu, Xihua},
  journal={arXiv preprint arXiv:2512.00779},
  year={2025}
}
```

## 9. License
MIT License, see LICENSE file for full terms.

## Developer Notes
Current backend is fully NumPy/SciPy compatible for full mathematical verification. Due to fully decoupled modular design, GPU acceleration via PyTorch/JAX only re\(\boldsymbol{q}\)uires rewriting tensor arithmetic inside `src/algebra/` and bilinear solvers inside `src/core/`, with all high-level randomized algorithm logic unchanged.

