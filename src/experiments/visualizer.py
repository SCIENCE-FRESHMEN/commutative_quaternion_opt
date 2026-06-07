"""
Academic Visualizer for Benchmark Results.
Generates publication-quality plots matching the paper's experimental trends.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(csv_path: str, output_dir: str):
    """
    Reads the benchmark CSV and generates high-quality academic plots.
    """
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Set academic style (compatible with newer matplotlib versions)
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid')  # Fallback for older versions

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 10,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300
    })

    unique_ns = sorted(df['n'].unique())
    # Color palette for different 'n'
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(unique_ns)))

    # ==================================================================
    # Plot 1: Algorithm 1 (Multilinear Form Relaxation)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, n in enumerate(unique_ns):
        subset = df[df['n'] == n]
        # Plot Average Ratio (Solid line with markers)
        ax.plot(subset['trials'], subset['alg1_mean'],
                marker='o', linestyle='-', color=colors[i], linewidth=2, markersize=8,
                label=f'$n={n}$ (Avg)')
        # Plot Worst-case Ratio (Dashed line)
        ax.plot(subset['trials'], subset['alg1_worst'],
                marker='x', linestyle='--', color=colors[i], linewidth=1.5, markersize=8, alpha=0.7,
                label=f'$n={n}$ (Worst)')

    ax.set_xscale('log')  # Log scale for trials to match paper's wide range
    ax.set_xlabel('Number of Trials (Log Scale)')
    ax.set_ylabel('Approximation Ratio (Objective / Upper Bound)')
    ax.set_title('Algorithm 1: Multilinear Form Optimization (Problem F)\nReproducing Table 1-3 Trends')

    # Custom legend handling to avoid clutter
    handles, labels = ax.get_legend_handles_labels()
    # We only need to show 'n' in legend, maybe combine avg/worst conceptually or just show all
    ax.legend(ncol=2, loc='lower right', framealpha=0.9)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    alg1_png = os.path.join(output_dir, "Alg1_Performance.png")
    alg1_pdf = os.path.join(output_dir, "Alg1_Performance.pdf")
    plt.savefig(alg1_png, dpi=300)
    plt.savefig(alg1_pdf, format='pdf', bbox_inches='tight')  # Vector format for paper
    plt.close()

    # ==================================================================
    # Plot 2: Algorithm 2 (Homogeneous Polynomial Optimization)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, n in enumerate(unique_ns):
        subset = df[df['n'] == n]
        ax.plot(subset['trials'], subset['alg2_mean'],
                marker='s', linestyle='-', color=colors[i], linewidth=2, markersize=8,
                label=f'$n={n}$ (Avg)')
        ax.plot(subset['trials'], subset['alg2_worst'],
                marker='+', linestyle='--', color=colors[i], linewidth=1.5, markersize=8, alpha=0.7,
                label=f'$n={n}$ (Worst)')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Trials (Log Scale)')
    ax.set_ylabel('Approximation Ratio (Objective / Upper Bound)')
    ax.set_title('Algorithm 2: Homogeneous Polynomial Optimization (Problem P)\nCombining Relaxation & Sign Search')
    ax.legend(ncol=2, loc='lower right', framealpha=0.9)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    alg2_png = os.path.join(output_dir, "Alg2_Performance.png")
    alg2_pdf = os.path.join(output_dir, "Alg2_Performance.pdf")
    plt.savefig(alg2_png, dpi=300)
    plt.savefig(alg2_pdf, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"✅ Publication-quality plots generated:")
    print(f"   📊 Alg1: {alg1_png} & {alg1_pdf}")
    print(f"   📊 Alg2: {alg2_png} & {alg2_pdf}")