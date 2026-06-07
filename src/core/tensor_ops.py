"""
Tensor Operations and Multilinear Form Evaluation.
"""
import numpy as np
from numpy.typing import NDArray
from typing import List
from src.algebra.quaternion import q_mul

def contract_quaternion_mode(T: NDArray[np.float64], x: NDArray[np.float64], mode_idx: int) -> NDArray[np.float64]:
    """
    Contracts a commutative quaternion tensor T with a quaternion vector x
    along a specific spatial mode, using quaternion multiplication.

    Args:
        T: Tensor of shape (4, n_1, n_2, ..., n_d)
        x: Vector of shape (4, n_mode)
        mode_idx: The index of the spatial dimension to contract (0-based).
    Returns:
        Res: Contracted tensor of shape (4, n_1, ..., n_{mode-1}, n_{mode+1}, ..., n_d)
    """
    # Step 1: Permute T so the contracted spatial axis is right after the quaternion axis (axis 0)
    axes_order = [0, mode_idx + 1] + [i for i in range(1, T.ndim) if i != mode_idx + 1]
    T_permuted = np.transpose(T, axes_order)
    # T_permuted shape: (4, n_mode, N_rest...)

    # Step 2: Dynamically expand x to match the spatial dimensions of T_permuted for broadcasting
    num_spatial_dims = T.ndim - 1
    # Create shape: [4, n_mode, 1, 1, ..., 1] where the number of 1s is (num_spatial_dims - 1)
    x_shape = [4, x.shape[1]] + [1] * (num_spatial_dims - 1)
    x_expanded = x.reshape(x_shape)

    # Step 3: Move quaternion axis (0) to the last axis (-1) for q_mul compatibility
    T_for_mul = np.moveaxis(T_permuted, 0, -1)      # shape: (n_mode, N_rest..., 4)
    x_for_mul = np.moveaxis(x_expanded, 0, -1)      # shape: (n_mode, 1, 1, ..., 4)

    # Step 4: Broadcast and multiply (x_for_mul automatically broadcasts to T_for_mul.shape)
    prod = q_mul(T_for_mul, x_for_mul)

    # Step 5: Sum over the contracted mode (which is now axis 0)
    res = np.sum(prod, axis=0)                      # shape: (N_rest..., 4)

    # Step 6: Move the quaternion axis back to the front (axis 0)
    return np.moveaxis(res, -1, 0)                  # shape: (4, N_rest...)


def eval_multilinear_form(T: NDArray[np.float64], vectors: List[NDArray[np.float64]]) -> float:
    """
    Evaluates Re F(x_1, x_2, ..., x_d) for a d-th order commutative quaternion tensor.

    Args:
        T: Tensor of shape (4, n_1, n_2, ..., n_d)
        vectors: List of d vectors, each of shape (4, n_k)
    Returns:
        float: The real part of the multilinear form evaluation.
    """
    res = T.copy()
    # After each contraction, the next spatial mode shifts to index 0
    for v in vectors:
        res = contract_quaternion_mode(res, v, mode_idx=0)

    # After d contractions, res is shape (4,). The real part is at index 0.
    return float(res[0])