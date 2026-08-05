# Commutative Quaternion Polynomial Optimization
This repository contains a fully modular, production-grade Python implementation of randomized polynomial-time approximation algorithms for sphere-constrained homogeneous polynomial optimization over commutative quaternion space. The best rank-one commutative quaternion tensor approximation problem, an NP-hard task widely used in signal processing, is covered as a core special case. Every module strictly aligns with the theorems, lemmas and algorithms proposed in the paper "On Approximation Algorithms for Commutative Quaternion Polynomial Optimization" (arXiv:2512.00779) by He, Jiang, Wang and Zhu.

## 1. Theoretical Overview
### 1.1 Commutative Quaternion Basic Rules
A commutative quaternion number consists of one real component and three imaginary components, expressed as $q = q_0 + q_1i + q_2j + q_3k$ with all coefficients being real numbers. The Segre multiplication rules for imaginary units are fixed:
$i^2 = k^2 = -1,\ j^2 = 1,\ ij=ji=k,\ jk=kj=i,\ ki=ik=-j,\ ijk=-1$.

We define quaternion conjugate, modulus, vector inner product and unit spherical constraint for vectors in commutative quaternion space. All numerical operations in this project strictly follow these algebraic definitions.

### 1.2 Two Core Optimization Problems
1. Homogeneous polynomial optimization problem (P): Maximize the real part of a homogeneous quaternion polynomial under unit spherical constraint for input vectors.
2. Multilinear form relaxation problem (F): Maximize the real part of multilinear mapping defined by high-order commutative quaternion tensors, with each input vector restricted on the unit sphere.

The best rank-one tensor approximation task can be fully transformed into the multilinear optimization problem (F). Minimizing the Frobenius norm error between the original tensor and a rank-one outer product tensor is mathematically equivalent to solving the maximization problem (F). This theoretical equivalence supports the color image recovery demo provided in this repo.

### 1.3 Theoretical Performance Guarantees
1. Spherical random sampling probability inequality: For uniform random vectors on the quaternion unit sphere, we derive a lower bound probability that the inner product exceeds a specific threshold related to tensor dimension and logarithmic term. An improved relaxed bound is also provided for arbitrary small positive offset.
2. Approximation ratio for Algorithm 1 (multilinear solver): After sufficient independent random trials, the algorithm outputs a feasible solution whose objective value reaches a fixed fraction of the global optimum with probability no less than $1-\epsilon$. The fraction is determined by tensor dimensions, logarithmic factor and a controllable constant parameter $\gamma$.
3. Approximation ratio for Algorithm 2 (homogeneous polynomial solver): Built upon Algorithm 1 as a subroutine. Separate guaranteed ratios are derived for odd-degree and even-degree polynomials. For odd-degree cases, the output objective value is bounded below by a fixed multiple of the global optimum. For even-degree cases, the gap between output objective and the global minimum objective satisfies a linear lower bound against the optimal value.
4. Standard test instance upper bound: We construct a benchmark 3-order all-real all-ones tensor with an explicit closed-form upper bound of the optimal objective value. This known theoretical upper limit acts as the ground truth reference for all numerical benchmark experiments.

## 2. Repository Core Features
1. Complete Theoretical Mapping
Each function is labeled with matching equations, lemmas and algorithm serial numbers from the original paper. All tensor relaxation and random sampling logic are implemented without simplification or modification to the theoretical derivation.

2. Modular Single-Responsibility Code Structure
- src/algebra: Realization of full commutative quaternion arithmetic including conjugate, modulus and inner product calculation.
- src/core: Real-domain block matrix transformation and SVD-based bilinear subproblem solver (Lemma 2.1).
- src/data: Generator of standard all-ones tensor test case from Proposition 6.1 for benchmark validation.
- src/algorithms: Independent implementation of Algorithm 1 (multilinear optimization) and Algorithm 2 (homogeneous polynomial optimization).
- src/utils: Tools for log printing, approximation ratio statistics and result aggregation.
- src/config: Dataclass-based centralized hyperparameter management for sampling rounds, $\gamma$, error tolerance $\epsilon$ and tensor dimension settings.

3. Type-Safe Robust Numerical Code
Full static type hints using numpy.typing are added to all scripts. Automatic dimension checking is inserted to avoid silent shape mismatch errors during large-scale random tensor sampling.

4. Automated End-to-End Benchmark Pipeline
Supports automatic parameter sweeping over tensor dimensions $n \in \{2,3,4,5,6,7\}$ and sampling trial counts $\{1,5,10,20,50,100,500,1000,10000\}$. Calculated statistical metrics include average approximation ratio, worst-case ratio, best-case ratio and standard deviation relative to the theoretical upper bound. All statistical results are exported to structured CSV files stored under ./reports.

5. Academic Standard Visualization Module
Built-in plotting tools generate high-resolution vector figures in PDF and PNG formats directly from benchmark CSV outputs. The figures visualize the convergence trend of empirical approximation ratios against the theoretical asymptotic bound.

6. Practical Color Image Recovery Demo
Map RGB three-channel color information into four-component commutative quaternion tensors. Convert color image denoising into a best rank-one tensor approximation task. The demo script outputs original clean images, noisy input tensors and reconstructed denoised images to the ./demo_output folder.

## 3. Project File Tree
```
commutative_quaternion_opt/
├── README.md
├── requirements.txt
├── main.py                 # Script for core mathematical consistency test
├── run_benchmark.py        # Large-scale random sampling for theoretical bound verification
├── demo_image_recovery.py  # End-to-end color image low-rank recovery demo
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils/
│   ├── algebra/
│   │   └── quaternion.py   # Segre commutative quaternion arithmetic implementation
│   ├── core/
│   │   └── solvers.py      # Bilinear optimization solver based on real block SVD
│   ├── data/
│   │   └── generators.py   # Standard all-ones benchmark tensor generator
│   └── algorithms/
│       ├── alg1_multilinear.py
│       └── alg2_homogeneous.py
├── .idea/
├── demo_output/            # Storage for image recovery experimental results
└── reports/                # Storage for benchmark CSV data and exported figures
```

## 4. Installation Guide
Clone the repository locally:
```bash
git clone https://github.com/SCIENCE-FRESHMEN/commutative-quaternion-opt.git
cd commutative-quaternion-opt
```
Create isolated virtual environment and install required dependencies:
```bash
python -m venv venv
# Windows activation command
venv\Scripts\activate
# Linux / macOS activation command
source venv/bin/activate

pip install -r requirements.txt
```

## 5. Quick Start Execution
### 5.1 Core Mathematical Consistency Validation
Run basic self-test to verify tensor dimension matching, spherical normalization and objective function calculation correctness:
```bash
python main.py
```
The terminal will print sequential [PASS] logs if all mathematical modules pass consistency inspection.

### 5.2 Large-Scale Random Sampling for Worst-Case Bound Verification
This script generates massive random commutative quaternion tensor samples independently, computes empirical approximation ratios and statistically validates the theoretical worst-case performance guarantee derived in the paper.
```bash
python run_benchmark.py
```
Output files:
1. ./reports/benchmark_results.csv: Complete tabular statistical data matching Table 1, Table 2 and Table 3 in the original paper.
2. PDF & PNG vector plots showing the comparison between empirical average/worst ratios and the theoretical asymptotic bound.

### 5.3 Color Image Low-Rank Recovery Demo
Test the algorithm on practical computer vision task for noisy RGB image restoration:
```bash
python demo_image_recovery.py
```
Navigate to ./demo_output to view original image, noisy tensor input and rank-one denoised reconstruction result.

## 6. Numerical Experiment Conclusion
All large-scale random numerical experiments fully verify the theoretical worst-case performance bounds proposed in the paper:
1. As the number of random sampling trials increases, the average empirical approximation ratio rises monotonically and gradually converges to the theoretical asymptotic bound determined by tensor dimension and logarithmic term.
2. The worst approximation ratio observed across all random test samples never breaks the lower performance limit proved in Theorem 4.1 and Theorem 5.1, which provides solid numerical evidence for the theoretical guarantee of this NP-hard optimization problem.
3. This codebase delivers a polynomial-time randomized solver with rigorous theoretical fallback guarantee, which can be directly deployed for engineering tasks including color signal processing and tensor low-rank recovery.

## 7. Citation
If you use this implementation in your research work, please cite the original theoretical paper as below:
```bibtex
@article{he2025approximation,
  title={On Approximation Algorithms for Commutative Quaternion Polynomial Optimization},
  author={He, Chang and Jiang, Bo and Wang, Hongye and Zhu, Xihua},
  journal={arXiv preprint arXiv:2512.00779},
  year={2025}
}
```

## 8. License
This project is distributed under the MIT License. Refer to the LICENSE file for full license terms.

## Developer Supplementary Notes
The current numerical backend relies only on NumPy and SciPy for maximum cross-platform compatibility and mathematical verification convenience. Thanks to the fully decoupled modular design, users can migrate to PyTorch or JAX for GPU acceleration of large-scale high-dimensional experiments by only rewriting codes inside src/algebra and src/core folders, without modifying any high-level randomized algorithm logic.
