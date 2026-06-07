"""
Commutative Quaternion Algebra Operations.
Implements Segre's multiplication rules and spherical sampling.
"""
import numpy as np
from numpy.typing import NDArray


def q_mul(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    [Paper Eq. 1, Section 1] Element-wise commutative quaternion multiplication.
    Rules: i^2 = k^2 = -1, j^2 = 1, ij = ji = k, jk = kj = i, ki = ik = -j
    """
    A0, A1, A2, A3 = A[..., 0], A[..., 1], A[..., 2], A[..., 3]
    B0, B1, B2, B3 = B[..., 0], B[..., 1], B[..., 2], B[..., 3]

    C0 = A0 * B0 - A1 * B1 + A2 * B2 - A3 * B3
    C1 = A0 * B1 + A1 * B0 + A2 * B3 + A3 * B2
    C2 = A0 * B2 - A1 * B3 + A2 * B0 - A3 * B1
    C3 = A0 * B3 + A1 * B2 + A2 * B1 + A3 * B0

    return np.stack([C0, C1, C2, C3], axis=-1)


def generate_uniform_quaternion_vector(n: int) -> NDArray[np.float64]:
    """
    [Paper Definition 3.2 & Lemma 3.1]
    Generates a random vector uniformly distributed on the n-dimensional
    commutative quaternion unit sphere S^n.
    """
    vec = np.random.randn(4, n)
    vec /= np.linalg.norm(vec)
    return vec