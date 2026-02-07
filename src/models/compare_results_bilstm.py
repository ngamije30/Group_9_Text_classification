"""
Compare BiLSTM Results Across All Embeddings
Group 9 - Text Classification (BiLSTM)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script runs
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "member2_bilstm"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"
FIGURES_BILSTM = PROJECT_ROOT / "figures" / "bilstm"


def load_results():
    """Load all BiLSTM result files."""
    results = {}
    for embedding in ["tfidf", "glove", "word2vec"]:
        path = RESULTS_DIR / f"results_bilstm_{embedding}.json"
        try:
            with open(path, "r") as f:
                results[embedding] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {path} not found")
    return results


def create_comparison_table(results):
    """Create and print comparison table, save to CSV."""
    print(f"\n{'='*80}")
    print("BiLSTM - COMPARISON ACROSS EMBEDDINGS")
    print(f"{'='*80}\n")

    data = []
    for embedding, result in results.items():
        m = result.get("metrics", result)
        row = {
            "Embedding": embedding.upper(),
            "Accuracy": f"{m.get('accuracy', 0):.4f}",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
            "F1-Score": f"{m.get('f1_score', 0):.4f}",
            "ROC-AUC": f"{m.get('roc_auc', 0):.4f}",
            "Training Time (s)": f"{m.get('training_time', 0):.2f}",
        }
        data.append(row)

    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"\n{'='*80}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "comparison_table_bilstm.csv", index=False)
    print(f"✓ Comparison table saved to: {RESULTS_DIR / 'comparison_table_bilstm.csv'}\n")
    return df


def plot_metric_comparison(results):
    """Plot comparison of all metrics and training time."""
    metrics_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    embeddings = list(results.keys())
    colors = ["#4ecdc4", "#ff6b6b", "#95e1d3"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        values = [results[e].get("metrics", results[e]).get(metric, 0) for e in embeddings]
        bars = ax.bar(embeddings, values, color=colors, alpha=0.8, edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison", fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(len(embeddings)))
        ax.set_xticklabels([e.upper() for e in embeddings])

    ax = axes[5]
    times = [results[e].get("metrics", results[e]).get("training_time", 0) for e in embeddings]
    bars = ax.bar(embeddings, times, color=colors, alpha=0.8, edgecolor="black")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(embeddings)))
    ax.set_xticklabels([e.upper() for e in embeddings])

    plt.suptitle("BiLSTM - Performance Across Embeddings", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_BILSTM.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "bilstm_all_metrics_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    import shutil
    shutil.copy(path, FIGURES_BILSTM / "bilstm_all_metrics_comparison.png")
    print(f"✓ Metrics comparison saved to: {path}\n")


def main():
    print("\n" + "#" * 80)
    print("  BiLSTM - RESULTS COMPARISON")
    print("#" * 80 + "\n")

    results = load_results()
    if not results:
        print("No BiLSTM results found. Run bilstm_tfidf.py, bilstm_glove.py, bilstm_word2vec.py first.")
        return

    print(f"Loaded results for: {list(results.keys())}\n")
    create_comparison_table(results)
    plot_metric_comparison(results)
    print("#" * 80 + "\n  COMPARISON COMPLETE!\n" + "#" * 80 + "\n")


if __name__ == "__main__":
    main()
