"""
Compare Results Across All Embeddings
Group 9 - Member 1 (Logistic Regression)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
RESULTS_DIR = "results/member1_logistic/"
FIGURES_DIR = "figures/results/"

def load_results():
    """Load all result files"""
    results = {}
    
    embeddings = ['tfidf', 'glove', 'word2vec']
    
    for embedding in embeddings:
        try:
            with open(f'{RESULTS_DIR}results_lr_{embedding}.json', 'r') as f:
                results[embedding] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Results for {embedding} not found")
    
    return results


def create_comparison_table(results):
    """Create comparison table"""
    print(f"\n{'='*80}")
    print("LOGISTIC REGRESSION - COMPARISON ACROSS EMBEDDINGS")
    print(f"{'='*80}\n")
    
    data = []
    for embedding, result in results.items():
        metrics = result['metrics']
        row = {
            'Embedding': embedding.upper(),
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}",
            'Training Time (s)': f"{metrics.get('training_time', 0):.2f}"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    print(f"\n{'='*80}\n")
    
    # Save to CSV
    df.to_csv(f'{RESULTS_DIR}comparison_table.csv', index=False)
    print(f"✓ Comparison table saved to: {RESULTS_DIR}comparison_table.csv\n")
    
    return df


def plot_metric_comparison(results):
    """Plot comparison of all metrics"""
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    embeddings = list(results.keys())
    
    # Prepare data
    data_dict = {metric: [] for metric in metrics_names}
    
    for embedding in embeddings:
        metrics = results[embedding]['metrics']
        for metric in metrics_names:
            data_dict[metric].append(metrics.get(metric, 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#4ecdc4', '#ff6b6b', '#95e1d3']
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        bars = ax.bar(embeddings, data_dict[metric], color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Embedding Type', fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels([e.upper() for e in embeddings])
    
    # Training time comparison
    ax = axes[5]
    training_times = [results[e]['metrics'].get('training_time', 0) for e in embeddings]
    bars = ax.bar(embeddings, training_times, color=colors, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}s',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xlabel('Embedding Type', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels([e.upper() for e in embeddings])
    
    plt.suptitle('Logistic Regression - Performance Across Embeddings', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}lr_all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Metrics comparison plot saved to: {FIGURES_DIR}lr_all_metrics_comparison.png\n")


def plot_radar_chart(results):
    """Create radar chart for comparison"""
    from math import pi
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    embeddings = list(results.keys())
    
    # Number of variables
    num_vars = len(metrics_names)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#4ecdc4', '#ff6b6b', '#95e1d3']
    
    # Plot data for each embedding
    for idx, embedding in enumerate(embeddings):
        metrics = results[embedding]['metrics']
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics.get('roc_auc', 0)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=embedding.upper(), 
                color=colors[idx], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # Fix axis to go from 0 to 1
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, size=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title('Logistic Regression - Performance Radar Chart', 
              size=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}lr_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Radar chart saved to: {FIGURES_DIR}lr_radar_comparison.png\n")


def generate_summary_report(results):
    """Generate text summary report"""
    print(f"\n{'#'*80}")
    print("SUMMARY REPORT - LOGISTIC REGRESSION")
    print(f"{'#'*80}\n")
    
    # Find best performing embedding for each metric
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    print("Best Performing Embedding by Metric:")
    print("-" * 40)
    for metric in metrics_names:
        best_embedding = max(results.keys(), 
                           key=lambda e: results[e]['metrics'].get(metric, 0))
        best_value = results[best_embedding]['metrics'].get(metric, 0)
        print(f"  {metric.replace('_', ' ').title():15s} : {best_embedding.upper()} ({best_value:.4f})")
    
    # Overall best
    print("\n" + "-" * 40)
    best_overall = max(results.keys(), 
                      key=lambda e: results[e]['metrics']['f1_score'])
    print(f"  Overall Best (by F1): {best_overall.upper()}")
    
    # Fastest training
    fastest = min(results.keys(), 
                 key=lambda e: results[e]['metrics'].get('training_time', float('inf')))
    fastest_time = results[fastest]['metrics'].get('training_time', 0)
    print(f"  Fastest Training: {fastest.upper()} ({fastest_time:.2f}s)")
    
    print("\n" + "#" * 80 + "\n")
    
    # Save summary to text file
    with open(f'{RESULTS_DIR}summary_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("LOGISTIC REGRESSION - SUMMARY REPORT\n")
        f.write("Group 9 - Member 1\n")
        f.write("="*80 + "\n\n")
        
        for embedding, result in results.items():
            f.write(f"\n{embedding.upper()}\n")
            f.write("-" * 40 + "\n")
            metrics = result['metrics']
            for key, value in metrics.items():
                f.write(f"  {key:20s} : {value}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Best Overall (F1-Score): {best_overall.upper()}\n")
        f.write(f"Fastest Training: {fastest.upper()}\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Summary report saved to: {RESULTS_DIR}summary_report.txt\n")


def main():
    """Main function"""
    print("\n" + "#"*80)
    print("  LOGISTIC REGRESSION - RESULTS COMPARISON")
    print("#"*80 + "\n")
    
    # Load all results
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("Error: No results found. Please run the training scripts first.")
        return
    
    print(f"✓ Loaded results for {len(results)} embeddings\n")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Plot comparisons
    print("Generating visualizations...")
    plot_metric_comparison(results)
    plot_radar_chart(results)
    
    # Generate summary report
    generate_summary_report(results)
    
    print("#"*80)
    print("  COMPARISON COMPLETE!")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
