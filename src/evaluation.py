"""
Model Evaluation Utilities
Group 9 - Text Classification Project
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import pandas as pd
from typing import Dict, Tuple
import json
import time


def evaluate_model(y_true: np.ndarray, 
                   y_pred: np.ndarray, 
                   y_prob: np.ndarray = None) -> Dict:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    
    # Add ROC-AUC if probabilities provided
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['roc_auc'] = auc(fpr, tpr)
    
    return metrics


def print_evaluation_report(metrics: Dict, model_name: str = "Model"):
    """
    Print formatted evaluation metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    if 'training_time' in metrics:
        print(f"  Training Time: {metrics['training_time']:.2f} seconds")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         save_path: str = None,
                         title: str = "Confusion Matrix"):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure (optional)
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, 
                  y_prob: np.ndarray,
                  save_path: str = None,
                  title: str = "ROC Curve"):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_path: Path to save figure
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def save_results(metrics: Dict, 
                save_path: str,
                additional_info: Dict = None):
    """
    Save evaluation results to JSON file.
    
    Args:
        metrics: Evaluation metrics dictionary
        save_path: Path to save JSON file
        additional_info: Additional information to save
    """
    results = {
        'metrics': metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if additional_info:
        results.update(additional_info)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {save_path}")


def create_comparison_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table from multiple model results.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with comparison
    """
    data = []
    for model_name, metrics in results_dict.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
        }
        if 'roc_auc' in metrics:
            row['ROC-AUC'] = f"{metrics['roc_auc']:.4f}"
        if 'training_time' in metrics:
            row['Training Time (s)'] = f"{metrics['training_time']:.2f}"
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def plot_metric_comparison(results_dict: Dict[str, Dict],
                          metric: str = 'f1_score',
                          save_path: str = None,
                          title: str = None):
    """
    Plot comparison of a specific metric across models.
    
    Args:
        results_dict: Dictionary mapping model names to metrics
        metric: Metric to compare
        save_path: Path to save figure
        title: Plot title
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title(title or f'{metric.replace("_", " ").title()} Comparison', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation utilities...\n")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_prob = np.random.random(100)
    
    # Evaluate
    metrics = evaluate_model(y_true, y_pred, y_prob)
    print_evaluation_report(metrics, "Sample Model")
    
    # Test comparison table
    results = {
        'Model A': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1_score': 0.85},
        'Model B': {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.90, 'f1_score': 0.88},
    }
    df = create_comparison_table(results)
    print("\nComparison Table:")
    print(df)
