"""
Algorithm 1: Randomized Algorithm for Multilinear Form Optimization (Problem F).
"""
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional
from src.algebra.quaternion import generate_uniform_quaternion_vector
from src.core.solvers import solve_bilinear_subproblem
from src.core.tensor_ops import eval_multilinear_form
from src.config import Algorithm1Config


def algorithm_1_multilinear(T: NDArray[np.float64], config: Algorithm1Config) -> Tuple[
    List[NDArray[np.float64]], float]:
    """
    [Paper Algorithm 1] Randomized Algorithm for Multilinear Form Optimization.
    """
    d = T.ndim - 1
    dims = T.shape[1:]

    best_val = -np.inf
    best_vectors: Optional[List[NDArray[np.float64]]] = None

    for _ in range(config.num_trials):
        vectors = [generate_uniform_quaternion_vector(dims[k]) for k in range(d - 2)]
        x_prev, x_next = solve_bilinear_subproblem(T, vectors)
        vectors.extend([x_prev, x_next])

        val = eval_multilinear_form(T, vectors)

        if val > best_val:
            best_val = val
            best_vectors = vectors

    assert best_vectors is not None, "Algorithm 1 failed to find a solution."
    return best_vectors, best_val