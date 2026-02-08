"""
LSTM Results Comparison Script
Group 9 - Text Classification

Generates comparison tables and visualizations for LSTM experiments
across TF-IDF, GloVe, and Word2Vec embeddings.

Author: Member 2 (Gustav)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "member3_lstm"
FIGURES_DIR = PROJECT_ROOT / "figures" / "lstm"


def load_results():
    """Load all LSTM results from JSON files."""
    results = {}
    
    files = {
        "TF-IDF": RESULTS_DIR / "results_lstm_tfidf.json",
        "GloVe": RESULTS_DIR / "results_lstm_glove.json",
        "Word2Vec": RESULTS_DIR / "results_lstm_word2vec.json",
    }
    
    for name, path in files.items():
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                results[name] = data.get("metrics", data)
        else:
            print(f"Warning: {path} not found")
    
    return results


def create_comparison_table(results):
    """Create a comparison DataFrame."""
    rows = []
    
    for embedding, metrics in results.items():
        rows.append({
            "Embedding": embedding,
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1-Score": metrics.get("f1_score", 0),
            "ROC-AUC": metrics.get("roc_auc", 0),
            "Training Time (s)": metrics.get("training_time", 0),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Accuracy", ascending=False)
    return df


def plot_metrics_comparison(df):
    """Create bar chart comparing all metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    embeddings = df["Embedding"].tolist()
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # Green, Blue, Red
    
    for i, (embedding, color) in enumerate(zip(embeddings, colors)):
        row = df[df["Embedding"] == embedding].iloc[0]
        values = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=embedding, color=color, alpha=0.85)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.2%}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8,
                fontweight='bold'
            )
    
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("LSTM Performance Comparison Across Embeddings", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(title="Embedding", loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lstm_all_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {FIGURES_DIR / 'lstm_all_metrics_comparison.png'}")


def plot_training_time(df):
    """Plot training time comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    bars = ax.bar(df["Embedding"], df["Training Time (s)"], color=colors, alpha=0.85)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height:.1f}s',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_xlabel("Embedding", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("LSTM Training Time by Embedding", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lstm_training_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {FIGURES_DIR / 'lstm_training_time_comparison.png'}")


def plot_radar_chart(df):
    """Create radar chart for metric comparison."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    embeddings = df["Embedding"].tolist()
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    for embedding, color in zip(embeddings, colors):
        row = df[df["Embedding"] == embedding].iloc[0]
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=embedding, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("LSTM Performance Radar Chart", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lstm_radar_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {FIGURES_DIR / 'lstm_radar_comparison.png'}")


def generate_summary_report(df):
    """Generate a text summary report."""
    report_path = RESULTS_DIR / "summary_report.txt"
    
    best_row = df.iloc[0]  # Already sorted by accuracy
    
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("LSTM TEXT CLASSIFICATION - RESULTS SUMMARY\n")
        f.write("Group 9 - Member 2 (Gustav)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 40 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"\n{row['Embedding']}:\n")
            f.write(f"  Accuracy:  {row['Accuracy']:.4f} ({row['Accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {row['Precision']:.4f}\n")
            f.write(f"  Recall:    {row['Recall']:.4f}\n")
            f.write(f"  F1-Score:  {row['F1-Score']:.4f}\n")
            f.write(f"  ROC-AUC:   {row['ROC-AUC']:.4f}\n")
            f.write(f"  Training:  {row['Training Time (s)']:.2f}s\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("BEST PERFORMER\n")
        f.write("-" * 40 + "\n")
        f.write(f"Embedding: {best_row['Embedding']}\n")
        f.write(f"Accuracy:  {best_row['Accuracy']:.4f} ({best_row['Accuracy']*100:.2f}%)\n")
        f.write(f"F1-Score:  {best_row['F1-Score']:.4f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"✓ Saved: {report_path}")


def main():
    print("\n" + "=" * 60)
    print("LSTM Results Comparison")
    print("=" * 60 + "\n")
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No results found. Run the LSTM training scripts first.")
        return
    
    print(f"Loaded results for: {list(results.keys())}\n")
    
    # Create comparison table
    df = create_comparison_table(results)
    print("Performance Comparison:")
    print(df.to_string(index=False))
    
    # Save comparison table
    csv_path = RESULTS_DIR / "comparison_table_lstm.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_metrics_comparison(df)
    plot_training_time(df)
    plot_radar_chart(df)
    
    # Generate summary report
    generate_summary_report(df)
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
