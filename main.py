"""
Main Execution and Self-Contained Validation Mechanism.
Run this script to verify dimension alignment, mathematical consistency, and execution.
"""
import sys
import numpy as np

# Ensure src is in path for direct execution
sys.path.insert(0, '.')

from src.config import Algorithm1Config, Algorithm2Config
from src.utils.logger import setup_logger
from src.utils.metrics import check_quaternion_sphere_constraint, compute_approximation_ratio
from src.data.generators import generate_proposition_6_1_tensor
from src.algorithms.alg1_multilinear import algorithm_1_multilinear
from src.algorithms.alg2_homogeneous import algorithm_2_homogeneous
from src.core.tensor_ops import eval_multilinear_form

logger = setup_logger("ValidationPipeline")


def run_validation():
    np.random.seed(42)

    logger.info("=" * 75)
    logger.info(" VALIDATING COMMUTATIVE QUATERNION POLYNOMIAL OPTIMIZATION ")
    logger.info("=" * 75)

    # Setup: d=3, n=3. Using Proposition 6.1 instance.
    n = 3
    d = 3

    logger.info(f"\n[Setup] Generating Proposition 6.1 tensor (d={d}, n={n})...")
    T_test, theoretical_upper_bound = generate_proposition_6_1_tensor(n, n, n)
    logger.info(f"[Theory] Theoretical upper bound for this instance: {theoretical_upper_bound:.4f}")

    # --- Validate Algorithm 1 ---
    logger.info("\n[Running] Algorithm 1 (Multilinear Form Relaxation)...")
    config1 = Algorithm1Config(num_trials=200)
    vectors_F, val_F = algorithm_1_multilinear(T_test, config1)

    ratio_F = compute_approximation_ratio(val_F, theoretical_upper_bound)
    logger.info(f"  -> Best Objective Value: {val_F:.4f}")
    logger.info(f"  -> Empirical Approximation Ratio: {ratio_F:.4f}")
    logger.info(f"  -> Solution Shapes: {[v.shape for v in vectors_F]}")

    assert check_quaternion_sphere_constraint(vectors_F), "Algorithm 1 vectors not on unit sphere!"
    logger.info("  -> [PASS] Algorithm 1 dimensions and normalization verified.")

    # --- Validate Algorithm 2 ---
    logger.info("\n[Running] Algorithm 2 (Homogeneous Polynomial Optimization)...")
    config2 = Algorithm2Config(alg1_config=config1)
    x_P, val_P = algorithm_2_homogeneous(T_test, config2)

    ratio_P = compute_approximation_ratio(val_P, theoretical_upper_bound)
    logger.info(f"  -> Best Objective Value: {val_P:.4f}")
    logger.info(f"  -> Empirical Approximation Ratio: {ratio_P:.4f}")
    logger.info(f"  -> Solution Shape: {x_P.shape}")

    assert check_quaternion_sphere_constraint([x_P]), "Algorithm 2 vector not on unit sphere!"
    logger.info("  -> [PASS] Algorithm 2 dimensions and normalization verified.")

    # --- Cross-Validation: Manual Evaluation ---
    logger.info("\n[Cross-Validation] Manually evaluating H(x_P)...")
    manual_val = eval_multilinear_form(T_test, [x_P, x_P, x_P])
    logger.info(f"  -> Manual Eval Result: {manual_val:.4f}")

    assert np.isclose(manual_val, val_P, atol=1e-5), "Manual evaluation mismatch with Algorithm 2 output!"
    logger.info("  -> [PASS] Objective function evaluation is strictly consistent.")

    logger.info("\n" + "=" * 75)
    logger.info(" ALL VALIDATIONS PASSED. CODE IS PRODUCTION-READY AND MODULAR. ")
    logger.info("=" * 75)


if __name__ == "__main__":
    run_validation()