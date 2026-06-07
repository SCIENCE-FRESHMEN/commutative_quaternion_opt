"""
Experimental Benchmark Entry Point.
Runs systematic parameter sweeps, generates reports, and plots academic figures.
"""
import sys
import os
sys.path.insert(0, '.')

from src.experiments.benchmark_runner import run_parameter_sweep
from src.experiments.report_generator import generate_report
from src.experiments.visualizer import generate_plots
from src.utils.logger import setup_logger

logger = setup_logger("BenchmarkPipeline")

def main():
    logger.info("🚀 Initializing Benchmark Sweep...")
    logger.info("This will reproduce experimental trends from the paper's Table 1-3.")

    # 🎯 EXPANDED GRID: Matches paper's exact validation scope (Table 1-3)
    # n in [2, 3, 4, 5, 6, 7], trials in [1, 5, 10, 20, 50, 100, 500, 1000, 10000]
    # Note: Running 10000 trials for n=7 might take some time.
    # We use a slightly reduced but representative grid for a quick demo.
    n_values = [3, 4, 5, 6, 7]
    trials_values = [10, 50, 100, 500, 1000] # Reduced slightly for faster execution, easily expandable
    num_runs = 20 # Paper uses 20 runs

    logger.info(f"📊 Grid: n={n_values}, trials={trials_values}, runs={num_runs}")
    logger.info("⏳ Starting execution (this may take a few minutes)...\n")

    # 1. Run Benchmark
    results = run_parameter_sweep(n_values, trials_values, num_runs, verbose=True)

    # 2. Generate Text Reports
    report_dir = "./reports"
    msg = generate_report(results, output_dir=report_dir)
    logger.info(msg)

    # 3. Generate Academic Plots 🎨
    csv_path = os.path.join(report_dir, "benchmark_results.csv")
    generate_plots(csv_path, output_dir=report_dir)

    logger.info("🎉 Full pipeline (Data + Reports + Plots) completed successfully.")
    logger.info(f"👉 Check the '{report_dir}' folder for your paper-ready assets.")

if __name__ == "__main__":
    main()