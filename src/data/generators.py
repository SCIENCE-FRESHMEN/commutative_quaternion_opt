"""
Data generators for benchmarking and validation.
"""
import numpy as np
import itertools
from numpy.typing import NDArray
from typing import Tuple


def generate_proposition_6_1_tensor(n1: int, n2: int, n3: int) -> Tuple[NDArray[np.float64], float]:
    """
    [Paper Proposition 6.1] Generates the specific test instance where an upper bound is known.
    F_0 is all-ones, F_1 = F_2 = F_3 = 0.
    """
    T = np.zeros((4, n1, n2, n3), dtype=np.float64)
    T[0, :, :, :] = 1.0  # F_0 is all-ones

    upper_bound = 2.0 * np.sqrt(n1 * n2 * n3)
    return T, upper_bound


def generate_random_supersymmetric_tensor(d: int, n: int) -> NDArray[np.float64]:
    """
    Generates a random super-symmetric commutative quaternion tensor.
    """
    T = np.random.randn(4, *([n] * d))
    # Symmetrize T_0 (real part) across all modes
    T[0] = np.mean([np.transpose(T[0], axes) for axes in itertools.permutations(range(d))], axis=0)
    return T