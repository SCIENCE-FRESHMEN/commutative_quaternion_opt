"""
Academic Report Generator for Benchmark Results.
Outputs CSV for archival and Markdown for paper integration.
"""
import csv
import os
from typing import List, Dict, Any

def generate_report(results: List[Dict[str, Any]], output_dir: str = "./reports") -> str:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    md_path = os.path.join(output_dir, "benchmark_report.md")

    if not results:
        return "No results to report."

    # 1. Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # 2. Generate Markdown Table (Strictly aligned with Paper Table 3 format)
    md_lines = [
        "# Approximation Ratios Benchmark Report\n",
        "## Algorithm 1: Multilinear Form Relaxation (Problem F)\n",
        "| $n$ | Trials | Average Ratio | Worst-case Ratio | Best-case Ratio | Std Dev |\n",
        "|:---:|:------:|:-------------:|:----------------:|:---------------:|:-------:|"
    ]
    for r in results:
        md_lines.append(
            f"| {r['n']} | {r['trials']} | {r['alg1_mean']:.4f} | "
            f"{r['alg1_worst']:.4f} | {r['alg1_best']:.4f} | {r['alg1_std']:.4f} |"
        )

    md_lines.extend([
        "\n## Algorithm 2: Homogeneous Polynomial Optimization (Problem P)\n",
        "| $n$ | Trials | Average Ratio | Worst-case Ratio | Best-case Ratio | Std Dev |\n",
        "|:---:|:------:|:-------------:|:----------------:|:---------------:|:-------:|"
    ])
    for r in results:
        md_lines.append(
            f"| {r['n']} | {r['trials']} | {r['alg2_mean']:.4f} | "
            f"{r['alg2_worst']:.4f} | {r['alg2_best']:.4f} | {r['alg2_std']:.4f} |"
        )

    md_content = "\n".join(md_lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return f"✅ Reports generated successfully:\n   📄 CSV: {csv_path}\n   📝 Markdown: {md_path}"