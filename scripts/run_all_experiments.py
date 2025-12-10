#!/usr/bin/env python3
"""
Run all experiments for the DPGD paper.

This script runs all configured experiments sequentially and collects results.
Usage:
    python scripts/run_all_experiments.py [--device cuda|cpu] [--skip-baseline]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Define all experiments to run
EXPERIMENTS = [
    # Main experiments with new state-of-the-art models
    ("llama2_baseline", "configs/experiments/llama2_baseline.yaml"),
    ("llama2_dpgd", "configs/experiments/llama2_dpgd.yaml"),
    ("gemma_baseline", "configs/experiments/gemma_baseline.yaml"),
    ("gemma_dpgd", "configs/experiments/gemma_dpgd.yaml"),
    ("mistral_baseline", "configs/experiments/mistral_baseline.yaml"),
    ("mistral_dpgd", "configs/experiments/mistral_dpgd.yaml"),
    
    # Legacy experiments (kept for reference)
    # ("llama_baseline", "configs/experiments/llama_baseline.yaml"),
    # ("llama_dpgd", "configs/experiments/llama_dpgd.yaml"),
    # ("guanaco_baseline", "configs/experiments/guanaco_baseline.yaml"),
    # ("guanaco_dpgd", "configs/experiments/guanaco_dpgd.yaml"),
    
    # Ablation studies
    ("ablation_no_sdp", "configs/experiments/ablation_no_sdp.yaml"),
    ("ablation_no_freq", "configs/experiments/ablation_no_freq.yaml"),
    ("ablation_no_morph", "configs/experiments/ablation_no_morph.yaml"),
    ("ablation_sdp_only", "configs/experiments/ablation_sdp_only.yaml"),
]

# Quick test with smaller model
QUICK_TEST = [
    ("gpt2_test", "configs/experiments/placeholder.yaml"),
]


def run_experiment(config_path: str, output_dir: str, device: str = "cuda") -> bool:
    """Run a single experiment."""
    cmd = [
        sys.executable, "-m", "src.dpgd.cli", "eval",
        config_path,
        "-o", output_dir,
        "--device", device,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def collect_results(results_dirs: list) -> dict:
    """Collect results from all experiment directories."""
    all_results = {}
    
    for name, results_dir in results_dirs:
        results_file = Path(results_dir) / "results.json"
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                all_results[name] = json.load(f)
        else:
            print(f"Warning: Results not found for {name} at {results_file}")
    
    return all_results


def print_summary_table(all_results: dict) -> None:
    """Print a summary table of all results."""
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Experiment':<25} {'SARI':>10} {'BERTScore':>12} {'Readability':>12} {'PHR':>10}")
    print("-"*75)
    
    for name, results in all_results.items():
        metrics = results.get("metrics", {})
        sari = metrics.get("sari", 0)
        bertscore = metrics.get("bertscore", {}).get("f1_mean", 0)
        readability = metrics.get("readability", {}).get("improvement", 0)
        phr = metrics.get("phr", 0)
        
        print(f"{name:<25} {sari:>10.4f} {bertscore:>12.4f} {readability:>+12.2f} {phr:>10.4f}")
    
    print("-"*75)
    print("\nMetric interpretation:")
    print("  - SARI: Higher is better (measures word-level operations)")
    print("  - BERTScore: Higher is better (semantic similarity)")
    print("  - Readability: Positive = easier text (Szigriszt-Pazos improvement)")
    print("  - PHR: Higher is better (fewer difficult words in output)")


def save_summary(all_results: dict, output_path: str) -> None:
    """Save summary to JSON file."""
    summary = {
        "experiments": {},
        "comparison": []
    }
    
    for name, results in all_results.items():
        metrics = results.get("metrics", {})
        summary["experiments"][name] = {
            "sari": metrics.get("sari", 0),
            "bertscore_f1": metrics.get("bertscore", {}).get("f1_mean", 0),
            "readability_improvement": metrics.get("readability", {}).get("improvement", 0),
            "phr": metrics.get("phr", 0),
            "config": results.get("config", {}),
        }
        
        summary["comparison"].append({
            "name": name,
            "sari": metrics.get("sari", 0),
            "bertscore_f1": metrics.get("bertscore", {}).get("f1_mean", 0),
            "readability_improvement": metrics.get("readability", {}).get("improvement", 0),
            "phr": metrics.get("phr", 0),
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all DPGD experiments")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run on (default: cuda)")
    parser.add_argument("--quick-test", action="store_true",
                        help="Run quick test with GPT-2 only")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline experiment")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only specific experiment (by name)")
    args = parser.parse_args()
    
    # Select experiments to run
    if args.quick_test:
        experiments = QUICK_TEST
    elif args.only:
        experiments = [(name, path) for name, path in EXPERIMENTS if name == args.only]
        if not experiments:
            print(f"Error: Experiment '{args.only}' not found")
            print(f"Available: {[name for name, _ in EXPERIMENTS]}")
            sys.exit(1)
    else:
        experiments = EXPERIMENTS
        if args.skip_baseline:
            experiments = [(n, p) for n, p in experiments if "baseline" not in n]
    
    # Auto-detect device if not specified or if cuda is requested but unavailable
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"


    print(f"Running {len(experiments)} experiment(s) on {args.device}")
    print(f"Experiments: {[name for name, _ in experiments]}")
    
    # Run experiments
    results_dirs = []
    for name, config_path in experiments:
        output_dir = f"./results/{name}"
        success = run_experiment(config_path, output_dir, args.device)
        if success:
            results_dirs.append((name, output_dir))
        else:
            print(f"Warning: Experiment {name} failed")
    
    # Collect and display results
    if results_dirs:
        all_results = collect_results(results_dirs)
        print_summary_table(all_results)
        save_summary(all_results, "./results/summary.json")
    else:
        print("No experiments completed successfully")


if __name__ == "__main__":
    main()
