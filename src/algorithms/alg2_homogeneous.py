"""
Algorithm 2: Randomized Algorithm for Homogeneous Polynomial Optimization (Problem P).
"""
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from itertools import product
from src.algorithms.alg1_multilinear import algorithm_1_multilinear
from src.core.tensor_ops import eval_multilinear_form
from src.config import Algorithm2Config


def algorithm_2_homogeneous(T_sym: NDArray[np.float64], config: Algorithm2Config) -> Tuple[NDArray[np.float64], float]:
    """
    [Paper Algorithm 2] Randomized Algorithm for Homogeneous Polynomial Optimization.
    Assumes T_sym is a super-symmetric tensor of shape (4, n, n, ..., n).
    """
    d = T_sym.ndim - 1
    n = T_sym.shape[1]

    relaxed_vectors, _ = algorithm_1_multilinear(T_sym, config.alg1_config)

    best_x: Optional[NDArray[np.float64]] = None
    best_val = -np.inf

    for betas in product([-1.0, 1.0], repeat=d):
        x_tilde = np.zeros((4, n), dtype=np.float64)
        for k in range(d):
            x_tilde += betas[k] * relaxed_vectors[k]
        x_tilde /= d

        norm = np.linalg.norm(x_tilde)
        if norm < 1e-12:
            continue
        x_cand = x_tilde / norm

        vectors_for_eval = [x_cand] * d
        val = eval_multilinear_form(T_sym, vectors_for_eval)

        if d % 2 == 1:
            if val < 0:
                val = -val
                x_cand = -x_cand

        if val > best_val:
            best_val = val
            best_x = x_cand

    assert best_x is not None, "Algorithm 2 failed to find a solution."
    return best_x, best_val