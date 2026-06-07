
***

# Commutative Quaternion Polynomial Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository provides a production-grade, highly modular Python implementation of approximation algorithms for **sphere-constrained homogeneous polynomial optimization over the commutative quaternion domain**. 

The codebase directly implements the theoretical framework, tensor relaxation techniques, and randomized approximation algorithms proposed in the foundational work by *He, Jiang, Wang, and Zhu*. It encompasses the best rank-one commutative quaternion tensor approximation as a primary special case, offering rigorous worst-case performance guarantees.

---

## 🌟 Key Features

- **Strict Mathematical Mapping**: Every core function is explicitly annotated with the corresponding paper equation, lemma, or algorithm step, ensuring absolute theoretical fidelity.
- **Modular & Decoupled Architecture**: Adheres strictly to the Single Responsibility Principle (SRP). Algebraic rules, tensor operations, solvers, and high-level algorithms are completely isolated.
- **Type-Safe & Robust**: Comprehensive Python Type Hints (`numpy.typing`) are utilized throughout, enabling robust IDE support and preventing silent dimension mismatches.
- **Comprehensive Benchmarking Pipeline**: Includes an automated parameter sweep module that generates statistical metrics (mean, worst-case, best-case, std dev) aligned with the paper's experimental tables.
- **Publication-Ready Visualization**: Built-in tools to generate high-DPI, academic-standard vector plots (PDF/PNG) directly from benchmark CSV data.
- **Real-World Application Demo**: Features an end-to-end color image low-rank recovery demonstration, validating the algorithm's practical utility in signal processing.

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/commutative-quaternion-opt.git
   cd commutative-quaternion-opt
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

### 1. Core Validation & Consistency Check
Run the self-contained validation script to verify dimension alignment, mathematical consistency, and execution flow:
```bash
python main.py
```
*Expected Output: A series of `[PASS]` logs confirming that the objective function evaluation, normalization constraints, and cross-validation are strictly consistent.*

### 2. Systematic Benchmarking
Execute the experimental pipeline to generate statistical reports and academic plots (matches the scope of Tables 1-3 in the paper):
```bash
python run_benchmark.py
```
*This will output a progress bar, followed by the generation of `benchmark_results.csv` and publication-quality `.pdf`/`.png` plots in the `./reports/` directory.*

### 3. Image Low-Rank Recovery Demo
See the algorithm in action on a practical computer vision task (best rank-one tensor approximation for color image denoising):
```bash
python demo_image_recovery.py
```
*Check the `./demo_output/` folder for the original, noisy, and recovered images.*

---

## 🏗️ Project Structure

The codebase is organized to facilitate seamless extension (e.g., migrating from NumPy to PyTorch/JAX for GPU acceleration) without altering high-level logic.

```text
commutative_quaternion_opt/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── main.py                       # Entry point for core validation
├── run_benchmark.py              # Entry point for systematic benchmarking
├── demo_image_recovery.py        # Practical application demonstration
└── src/
    ├── __init__.py
    ├── config.py                 # Strict dataclass-based configuration management
    ├── utils/                    # Logging and metric evaluation utilities
    ├── algebra/                  # Commutative quaternion algebra (Segre rules)
    ├── core/                     # Tensor contraction & SVD-based bilinear solvers
    ├── data/                     # Benchmark instance generators (e.g., Proposition 6.1)
    └── algorithms/               # Algorithm 1 (Multilinear) & Algorithm 2 (Homogeneous)
```

---

## 📚 Theoretical Foundations & Code Mapping

The implementation is rigorously aligned with the mathematical formulations in the reference paper:

| Code Module | Paper Reference | Description |
| :--- | :--- | :--- |
| `src/algebra/quaternion.py` | Section 1, Eq. (1) | Implements Segre's commutative multiplication rules ($i^2=k^2=-1, j^2=1, ij=ji=k$, etc.). |
| `src/core/solvers.py` | Lemma 2.1 | Constructs the $4n \times 4n$ real block matrix and solves the bilinear subproblem via SVD in polynomial time. |
| `src/algorithms/alg1_multilinear.py` | Algorithm 1 | Randomized algorithm for Multilinear Form Optimization (Problem F) using random spherical sampling. |
| `src/algorithms/alg2_homogeneous.py` | Algorithm 2 & Lemma 5.1 | Bridges multilinear forms to homogeneous polynomials (Problem P) via symmetric Bernoulli sign search. |
| `src/data/generators.py` | Proposition 6.1 | Generates the specific all-ones tensor instance with a known theoretical upper bound $2\sqrt{n_1n_2n_3}$. |

---

## 📖 Citation

If you find this implementation useful in your research, please consider citing the foundational work:

```bibtex
@article{he2025approximation,
  title={On Approximation Algorithms for Commutative Quaternion Polynomial Optimization},
  author={He, Chang and Jiang, Bo and Wang, Hongye and Zhu, Xihua},
  journal={arXiv preprint arXiv:2512.00779},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

> **Note for Developers**: The current backend utilizes `NumPy` and `SciPy` for maximum compatibility and ease of verification. Due to the modular design, transitioning to `PyTorch` or `JAX` for large-scale ($n > 20$) GPU-accelerated experiments requires modifying only the `src/algebra/` and `src/core/` modules, leaving the algorithmic logic entirely untouched.