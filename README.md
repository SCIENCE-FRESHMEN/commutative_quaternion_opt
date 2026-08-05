# Commutative Quaternion Polynomial Optimization

# Commutative Quaternion Polynomial Optimization

This repository provides a fully modular, production\-ready Python implementation of randomized polynomial\-time approximation algorithms for **sphere\-constrained homogeneous polynomial optimization over commutative quaternion space **$\mathbb{H}$, whose special case covers the best rank\-one commutative quaternion tensor approximation problem \(NP\-hard\)\. All modules strictly map to the mathematical statements, probability lemmas, and algorithms proposed in He et al\. \(2025\) arXiv:2512\.00779\.

## 1\. Theoretical Background \& Core Definitions

### 1\.1 Commutative Quaternion Algebra

A commutative quaternion scalar is defined as

with Segre commutative multiplication identities:

Conjugation of $\boldsymbol{q}$:

modulus of commutative quaternion:

For vector $\boldsymbol{x}\in\mathbb{H}^n$, decompose into real vector components:

Conjugate transpose $\boldsymbol{x}^H = \overline{\boldsymbol{x}}^\top$, inner product and spherical constraint:

Unit commutative quaternion sphere:

### 1\.2 Homogeneous Polynomial \& Multilinear Form Optimization

The target homogeneous polynomial maximization problem $(P)$:

where $H(\boldsymbol{x})$ is a $d$\-th degree homogeneous commutative quaternion polynomial:

The multilinear relaxation counterpart problem $(F)$:

with multilinear form induced by commutative quaternion tensor $\mathcal{F}\in\mathbb{H}^{n_1\times n_2\times\cdots\times n_d}$:

### 1\.3 Best Rank\-One Tensor Approximation \(Key Application\)

The low\-rank recovery task for commutative quaternion tensor $\mathcal{F}$:

equivalently reduces to solving problem $(F)$ via the identity

This equivalence validates our framework for color image denoising and low\-rank recovery demos\.

## 2\. Theoretical Performance Bound \& Approximation Ratio

### 2\.1 Random Quaternion Spherical Probability Inequality

Let $\xi,\boldsymbol{a}\sim \text{Uniform}(S^n)$, for any $\gamma>0,\gamma\ln n < n$, there exists constant $c(\gamma)>0$ such that

An improved bound with arbitrary small $\delta>0$:

### 2\.2 Approximation Ratio for Multilinear Algorithm 1

Assume $n_1\leq n_2\leq\cdots\leq n_d$\. Algorithm 1 returns feasible solutions to $(F)$ satisfying, with probability at least $1-\epsilon$ after sufficient random trials:

where $v^*(F) = \max_{\boldsymbol{x}^k\in S^{n_k}}\operatorname{Re}\mathcal{F}(\boldsymbol{x}^1,\dots,\boldsymbol{x}^d)$ denotes the optimal objective value of $(F)$\.

### 2\.3 Approximation Ratio for Homogeneous Polynomial Algorithm 2

Define global approximation coefficient

1. When polynomial degree $d$ is odd:

$v^*(P)=\max_{\boldsymbol{x}\in S^n}\operatorname{Re}H(\boldsymbol{x})$\.
2\. When $d$ is even: let $\underline{v}(P)=\min_{\boldsymbol{x}\in S^n}\operatorname{Re}H(\boldsymbol{x})$, then

### 2\.4 Canonical Worst\-Case Test Instance Bound \(Proposition 6\.1\)

For 3rd\-order equal\-dimension tensor $\mathcal{F}\in\mathbb{H}^{n\times n\times n}$ with real component $\mathcal{F}_0 = \mathbf{1}_{n\times n\times n}$ \(all\-ones tensor\), all imaginary blocks zero:

This closed\-form upper bound serves as ground truth reference for all numerical validation and benchmarking experiments in this repo\.

## 3\. Repository Key Features

1. **Full Mathematical Traceability**
Every core function is annotated with matching paper equations, lemmas, theorems, and algorithm indices; all tensor relaxations and probability bounds are implemented exactly as derived in the original manuscript\.

2. **Strictly Modular Decoupled Architecture \(SRP Compliant\)**

- `src/algebra`: Segre commutative quaternion arithmetic, conjugate, norm, inner product

- `src/core`: Real block matrix conversion \& SVD bilinear subproblem solver \(Lemma 2\.1\)

- `src/data`: Generate canonical all\-ones tensor test instance from Proposition 6\.1

- `src/algorithms`: Algorithm 1 \(Multilinear Form\) \& Algorithm 2 \(Homogeneous Polynomial\)

- `src/utils`: Logging, ratio metric calculation, statistical aggregation

- `src/config`: Dataclass hyperparameter management for sampling count, $\gamma$, $\epsilon$, tensor dimension $n$

3. **Type\-Safe Numerical Implementation**
Full static type hints with `numpy.typing`, strict dimension assertion checks to eliminate silent tensor shape mismatches during large\-scale random sampling\.

4. **End\-to\-End Benchmark Pipeline for Bound Verification**
Automated parameter sweep over $n\in{2,3,4,5,6,7}$, trial counts ${1,5,10,20,50,100,500,1000,10000}$\. Metrics computed: average approximation ratio, worst\-case ratio, standard deviation, best\-case ratio relative to theoretical upper bound $v_{\text{upper}}=2\sqrt{n^3}$\. All statistics exported as structured CSV files in `./reports/`\.

5. **Publication\-Quality Visualization Tools**
Native export of high\-DPI vector plots \(PDF/PNG\) directly from benchmark CSV outputs for theoretical bound validation figures, showing empirical ratio convergence against $\sqrt{\ln n/n}$ asymptotic guarantee\.

6. **Practical Computer Vision Demo: Color Image Low\-Rank Recovery**
Map RGB color channels to 4\-channel commutative quaternion components, formulate image denoising as best rank\-one commutative quaternion tensor approximation\. Script `demo_image_recovery.py` outputs original noisy tensor, rank\-one reconstructed image to `./demo_output/`\.

## 4\. Project Directory Structure

```Plain Text
commutative_quaternion_opt/
├── README.md
├── requirements.txt
├── main.py                 # Core consistency validation script
├── run_benchmark.py        # Large-scale random sampling for bound verification
├── demo_image_recovery.py  # Color image low-rank recovery demo
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils/
│   ├── algebra/
│   │   └── quaternion.py   # Segre commutative quaternion algebra rules
│   ├── core/
│   │   └── solvers.py      # Lemma 2.1 bilinear SVD solver
│   ├── data/
│   │   └── generators.py   # Proposition 6.1 all-ones tensor generator
│   └── algorithms/
│       ├── alg1_multilinear.py
│       └── alg2_homogeneous.py
├── .idea/
├── demo_output/            # Image recovery experiment outputs
└── reports/                # Benchmark CSV & PDF/PNG figures
```

## 5\. Installation

Clone repository:

```bash
git clone https://github.com/SCIENCE-FRESHMEN/commutative-quaternion-opt.git
cd commutative-quaternion-opt
```

Create virtual environment \& install dependencies:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## 6\. Quick Start Execution

### 6\.1 Core Mathematical Consistency Validation

Run dimension matching, spherical normalization, objective function consistency test:

```bash
python main.py
```

Expected output: sequential `[PASS]` logs verifying probability inequality sampling, tensor contraction, and ratio computation correctness\.

### 6\.2 Large\-Scale Random Sampling for Worst\-Case Bound Verification

This script independently generates massive random commutative quaternion tensor samples, computes empirical approximation ratios, and statistically validates the theoretical guarantee

```bash
python run_benchmark.py
```

Output artifacts:

- `./reports/benchmark_results.csv`: Full tabular statistics matching Table 1–3 of the paper

- Vectorized PDF/PNG plots showing empirical worst/average ratio vs theoretical asymptotic bound\.

### 6\.3 Color Image Rank\-One Recovery Demo

Solve noisy RGB image restoration via best rank\-one commutative quaternion tensor approximation:

```bash
python demo_image_recovery.py
```

Check `./demo_output/` for ground truth image, noisy tensor input, and denoised rank\-one reconstruction\.

## 7\. Experiment \& Bound Validation Summary

All numerical experiments in this repo independently verify the proposed worst\-case approximation guarantee over thousands of random tensor realizations\. For the canonical all\-ones 3\-order tensor instance with known upper bound $v_{\text{upper}}=2\sqrt{n^3}$:

- As sampling trials increase, average empirical approximation ratio monotonically rises, converging toward the theoretical $\sqrt{\ln n/n}$ asymptotic bound\.

- The measured worst\-case ratio across all random runs never violates the theoretical lower approximation ratio derived from Theorem 4\.1 and Theorem 5\.1, providing rigorous numerical evidence for the theoretical performance guarantee of this NP\-hard commutative quaternion tensor optimization problem\.

- The implementation delivers an engineering\-ready polynomial\-time randomized solver with provable theoretical performance fallback for commutative quaternion polynomial optimization and its signal processing applications\.

## 8\. Citation

If you utilize this codebase for research, cite the foundational theoretical work:

```bibtex
@article{he2025approximation,
  title={On Approximation Algorithms for Commutative Quaternion Polynomial Optimization},
  author={He, Chang and Jiang, Bo and Wang, Hongye and Zhu, Xihua},
  journal={arXiv preprint arXiv:2512.00779},
  year={2025}
}
```

## 9\. License

MIT License, see LICENSE file for full terms\.

## Developer Notes

Current backend is fully NumPy/SciPy compatible for full mathematical verification\. Due to fully decoupled modular design, GPU acceleration via PyTorch/JAX only requires rewriting tensor arithmetic inside `src/algebra/` and bilinear solvers inside `src/core/`, with all high\-level randomized algorithm logic unchanged\.

