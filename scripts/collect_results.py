#!/usr/bin/env python3
"""
Collect results from all experiments and generate summary tables.

Usage:
    python scripts/collect_results.py [--output results/summary.json]
    python scripts/collect_results.py --latex  # Generate LaTeX table
"""

import argparse
import json
from pathlib import Path


def find_all_results(results_dir: str = "./results") -> dict:
    """Find all results.json files in subdirectories."""
    results_path = Path(results_dir)
    all_results = {}
    
    # Check for results in subdirectories
    for subdir in results_path.iterdir():
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                with open(results_file, "r", encoding="utf-8") as f:
                    all_results[subdir.name] = json.load(f)
    
    # Also check for root-level results.json
    root_results = results_path / "results.json"
    if root_results.exists() and "root" not in all_results:
        with open(root_results, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Use experiment name from config if available
            name = data.get("config", {}).get("model", "root")
            if name not in all_results:
                all_results[name] = data
    
    return all_results


def generate_markdown_table(all_results: dict) -> str:
    """Generate a Markdown table from results."""
    lines = [
        "| Experiment | SARI | BERTScore F1 | Readability Δ | PHR |",
        "|------------|------|--------------|---------------|-----|",
    ]
    
    for name, results in sorted(all_results.items()):
        metrics = results.get("metrics", {})
        sari = metrics.get("sari", 0)
        bertscore = metrics.get("bertscore", {}).get("f1_mean", 0)
        readability = metrics.get("readability", {}).get("improvement", 0)
        phr = metrics.get("phr", 0)
        
        lines.append(
            f"| {name} | {sari:.4f} | {bertscore:.4f} | {readability:+.2f} | {phr:.4f} |"
        )
    
    return "\n".join(lines)


def generate_latex_table(all_results: dict) -> str:
    """Generate a LaTeX table from results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Evaluation Results}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Experiment & SARI $\uparrow$ & BERTScore F1 $\uparrow$ & Readability $\Delta$ & PHR $\uparrow$ \\",
        r"\midrule",
    ]
    
    for name, results in sorted(all_results.items()):
        metrics = results.get("metrics", {})
        sari = metrics.get("sari", 0)
        bertscore = metrics.get("bertscore", {}).get("f1_mean", 0)
        readability = metrics.get("readability", {}).get("improvement", 0)
        phr = metrics.get("phr", 0)
        
        # Escape underscores for LaTeX
        name_escaped = name.replace("_", r"\_")
        
        lines.append(
            f"{name_escaped} & {sari:.4f} & {bertscore:.4f} & {readability:+.2f} & {phr:.4f} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_csv(all_results: dict) -> str:
    """Generate CSV from results."""
    lines = ["experiment,sari,bertscore_f1,readability_improvement,phr"]
    
    for name, results in sorted(all_results.items()):
        metrics = results.get("metrics", {})
        sari = metrics.get("sari", 0)
        bertscore = metrics.get("bertscore", {}).get("f1_mean", 0)
        readability = metrics.get("readability", {}).get("improvement", 0)
        phr = metrics.get("phr", 0)
        
        lines.append(f"{name},{sari:.4f},{bertscore:.4f},{readability:.2f},{phr:.4f}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect and summarize experiment results")
    parser.add_argument("--results-dir", default="./results",
                        help="Directory containing experiment results")
    parser.add_argument("--output", default="./results/summary.json",
                        help="Output path for summary JSON")
    parser.add_argument("--latex", action="store_true",
                        help="Generate LaTeX table")
    parser.add_argument("--csv", action="store_true",
                        help="Generate CSV output")
    parser.add_argument("--markdown", action="store_true",
                        help="Generate Markdown table")
    args = parser.parse_args()
    
    # Find all results
    all_results = find_all_results(args.results_dir)
    
    if not all_results:
        print(f"No results found in {args.results_dir}")
        return
    
    print(f"Found {len(all_results)} experiment(s): {list(all_results.keys())}")
    
    # Generate requested outputs
    if args.latex:
        print("\n" + "="*60)
        print("LaTeX Table:")
        print("="*60)
        print(generate_latex_table(all_results))
    
    if args.csv:
        print("\n" + "="*60)
        print("CSV Output:")
        print("="*60)
        print(generate_csv(all_results))
    
    if args.markdown or (not args.latex and not args.csv):
        print("\n" + "="*60)
        print("Markdown Table:")
        print("="*60)
        print(generate_markdown_table(all_results))
    
    # Save summary JSON
    summary = {
        "num_experiments": len(all_results),
        "experiments": {}
    }
    
    for name, results in all_results.items():
        metrics = results.get("metrics", {})
        summary["experiments"][name] = {
            "sari": metrics.get("sari", 0),
            "bertscore_f1": metrics.get("bertscore", {}).get("f1_mean", 0),
            "bertscore_precision": metrics.get("bertscore", {}).get("precision_mean", 0),
            "bertscore_recall": metrics.get("bertscore", {}).get("recall_mean", 0),
            "readability_source": metrics.get("readability", {}).get("source_mean", 0),
            "readability_prediction": metrics.get("readability", {}).get("prediction_mean", 0),
            "readability_improvement": metrics.get("readability", {}).get("improvement", 0),
            "phr": metrics.get("phr", 0),
            "phr_violated": metrics.get("phr_details", {}).get("total_violated", 0),
            "phr_active": metrics.get("phr_details", {}).get("total_active", 0),
            "config": results.get("config", {}),
            "num_examples": results.get("num_examples", 0),
        }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
