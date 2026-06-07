"""
Evaluation metrics and constraint validation.
"""
import numpy as np
from numpy.typing import NDArray
from typing import List

def check_quaternion_sphere_constraint(vectors: List[NDArray[np.float64]], atol: float = 1e-5) -> bool:
    """Verifies that all vectors lie on the commutative quaternion unit sphere."""
    for v in vectors:
        norm = np.linalg.norm(v)
        if not np.isclose(norm, 1.0, atol=atol):
            return False
    return True

def compute_approximation_ratio(achieved_val: float, theoretical_upper_bound: float) -> float:
    """Computes the empirical approximation ratio."""
    if theoretical_upper_bound <= 0:
        return 0.0
    return max(0.0, achieved_val / theoretical_upper_bound)