"""
Bilinear Subproblem Solver via Real Matrix SVD.
"""
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from scipy.linalg import svd
from src.core.tensor_ops import contract_quaternion_mode


def solve_bilinear_subproblem(T: NDArray[np.float64], fixed_vectors: List[NDArray[np.float64]]) -> Tuple[
    NDArray[np.float64], NDArray[np.float64]]:
    """
    [Paper Lemma 2.1] Solves the bilinear subproblem in polynomial time via real matrix SVD.
    """
    d = T.ndim - 1
    assert len(fixed_vectors) == d - 2, "Must provide exactly d-2 fixed vectors."

    A = T.copy()
    for v in fixed_vectors:
        A = contract_quaternion_mode(A, v, mode_idx=0)

    n_prev, n_next = A.shape[1], A.shape[2]
    A0, A1, A2, A3 = A[0], A[1], A[2], A[3]

    # [Paper Eq. 2, Page 5-6] Construct the 4x4 block real matrix M_real
    M_real = np.zeros((4 * n_prev, 4 * n_next), dtype=np.float64)

    M_real[0 * n_prev:1 * n_prev, 0 * n_next:1 * n_next] = A0
    M_real[0 * n_prev:1 * n_prev, 1 * n_next:2 * n_next] = -A1
    M_real[0 * n_prev:1 * n_prev, 2 * n_next:3 * n_next] = A2
    M_real[0 * n_prev:1 * n_prev, 3 * n_next:4 * n_next] = -A3

    M_real[1 * n_prev:2 * n_prev, 0 * n_next:1 * n_next] = -A1
    M_real[1 * n_prev:2 * n_prev, 1 * n_next:2 * n_next] = -A0
    M_real[1 * n_prev:2 * n_prev, 2 * n_next:3 * n_next] = -A3
    M_real[1 * n_prev:2 * n_prev, 3 * n_next:4 * n_next] = -A2

    M_real[2 * n_prev:3 * n_prev, 0 * n_next:1 * n_next] = A2
    M_real[2 * n_prev:3 * n_prev, 1 * n_next:2 * n_next] = -A3
    M_real[2 * n_prev:3 * n_prev, 2 * n_next:3 * n_next] = A0
    M_real[2 * n_prev:3 * n_prev, 3 * n_next:4 * n_next] = -A1

    M_real[3 * n_prev:4 * n_prev, 0 * n_next:1 * n_next] = -A3
    M_real[3 * n_prev:4 * n_prev, 1 * n_next:2 * n_next] = -A2
    M_real[3 * n_prev:4 * n_prev, 2 * n_next:3 * n_next] = -A1
    M_real[3 * n_prev:4 * n_prev, 3 * n_next:4 * n_next] = -A0

    U, S, Vt = svd(M_real)

    u_opt = U[:, 0].reshape(4, n_prev)
    v_opt = Vt[0, :].reshape(4, n_next)

    u_opt /= np.linalg.norm(u_opt)
    v_opt /= np.linalg.norm(v_opt)

    return u_opt, v_opt