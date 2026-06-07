"""
Systematic Benchmark Runner for Approximation Algorithms.
Executes parameter sweeps and collects statistical metrics.
"""
import numpy as np
import sys
import time
from typing import List, Dict, Any
from src.config import Algorithm1Config, Algorithm2Config
from src.data.generators import generate_proposition_6_1_tensor
from src.algorithms.alg1_multilinear import algorithm_1_multilinear
from src.algorithms.alg2_homogeneous import algorithm_2_homogeneous
from src.utils.metrics import compute_approximation_ratio


def _print_progress(current: int, total: int, desc: str = "Progress"):
    """Simple native progress indicator (zero external dependencies)."""
    percent = (current / total) * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r{desc}: |{bar}| {percent:.1f}% ({current}/{total})")
    sys.stdout.flush()


def run_parameter_sweep(
        n_values: List[int],
        trials_values: List[int],
        num_runs: int = 20,
        verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Runs a systematic benchmark sweep over n and num_trials.
    Matches the experimental setup in the paper (Table 1-3).
    """
    results = []
    total_iters = len(n_values) * len(trials_values) * num_runs
    current_iter = 0

    for n in n_values:
        for trials in trials_values:
            alg1_ratios, alg2_ratios = [], []

            # Generate the fixed Proposition 6.1 instance for this n
            T_test, upper_bound = generate_proposition_6_1_tensor(n, n, n)

            config1 = Algorithm1Config(num_trials=trials)
            config2 = Algorithm2Config(alg1_config=config1)

            for seed in range(num_runs):
                # Deterministic seeding for reproducibility across runs
                np.random.seed(seed + 42)

                _, val_F = algorithm_1_multilinear(T_test, config1)
                _, val_P = algorithm_2_homogeneous(T_test, config2)

                alg1_ratios.append(compute_approximation_ratio(val_F, upper_bound))
                alg2_ratios.append(compute_approximation_ratio(val_P, upper_bound))

                current_iter += 1
                if verbose:
                    _print_progress(current_iter, total_iters, desc="Benchmarking")

            # Aggregate statistics
            results.append({
                "n": n,
                "trials": trials,
                "num_runs": num_runs,
                "alg1_mean": float(np.mean(alg1_ratios)),
                "alg1_std": float(np.std(alg1_ratios)),
                "alg1_worst": float(np.min(alg1_ratios)),
                "alg1_best": float(np.max(alg1_ratios)),
                "alg2_mean": float(np.mean(alg2_ratios)),
                "alg2_std": float(np.std(alg2_ratios)),
                "alg2_worst": float(np.min(alg2_ratios)),
                "alg2_best": float(np.max(alg2_ratios)),
            })

    if verbose:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return results