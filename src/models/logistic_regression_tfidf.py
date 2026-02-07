"""
Logistic Regression with TF-IDF Embeddings
Group 9 - Member 1 (Logistic Regression)
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Import custom utilities
from preprocessing import load_imdb_data, preprocess_dataset, split_data
from evaluation import (evaluate_model, print_evaluation_report, 
                            plot_confusion_matrix, plot_roc_curve, save_results)

# Constants
RANDOM_SEED = 42
MODEL_NAME = "Logistic Regression + TF-IDF"
RESULTS_DIR = PROJECT_ROOT / "results" / "member1_logistic"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"


def create_tfidf_features(X_train, X_val, X_test, max_features=5000):
    """
    Create TF-IDF features from text data.
    
    Args:
        X_train: Training text data
        X_val: Validation text data
        X_test: Test text data
        max_features: Maximum number of features to extract
        
    Returns:
        Tuple of (X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer)
    """
    print(f"\n{'='*60}")
    print("Creating TF-IDF Features")
    print(f"{'='*60}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Include unigrams and bigrams
        min_df=5,  # Ignore terms that appear in less than 5 documents
        max_df=0.8,  # Ignore terms that appear in more than 80% of documents
        sublinear_tf=True  # Use logarithmic form for term frequency
    )
    
    # Fit on training data and transform all sets
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"✓ TF-IDF vectorizer created")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Training shape: {X_train_tfidf.shape}")
    print(f"  Validation shape: {X_val_tfidf.shape}")
    print(f"  Test shape: {X_test_tfidf.shape}")
    print(f"  Sparsity: {(1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.4f}")
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        
    Returns:
        Tuple of (best_model, training_time)
    """
    print(f"\n{'='*60}")
    print("Training Logistic Regression Model")
    print(f"{'='*60}")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l2'],  # L2 regularization
        'solver': ['lbfgs'],  # Optimization algorithm
        'max_iter': [1000]
    }
    
    # Initialize base model
    base_model = LogisticRegression(random_state=RANDOM_SEED)
    
    # Perform grid search
    print("Performing hyperparameter tuning...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='f1',
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f} seconds")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV F1-score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    y_val_pred = grid_search.predict(X_val)
    val_metrics = evaluate_model(y_val, y_val_pred)
    print(f"  Validation F1-score: {val_metrics['f1_score']:.4f}")
    
    return grid_search.best_estimator_, training_time


def main():
    """Main execution function"""
    print(f"\n{'#'*60}")
    print(f"  {MODEL_NAME}")
    print(f"{'#'*60}\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    data_path = PROJECT_ROOT / 'data' / 'raw' / 'IMDB Dataset.csv'
    df = load_imdb_data(str(data_path))
    df_clean = preprocess_dataset(df, remove_stopwords=True)
    
    # Step 2: Split data
    print("\nStep 2: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
    
    # Step 3: Create TF-IDF features
    print("\nStep 3: Creating TF-IDF features...")
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(
        X_train, X_val, X_test, max_features=5000
    )
    
    # Step 4: Train model
    print("\nStep 4: Training model...")
    model, training_time = train_logistic_regression(
        X_train_tfidf, y_train, X_val_tfidf, y_val
    )
    
    # Step 5: Evaluate on test set
    print("\nStep 5: Evaluating on test set...")
    y_test_pred = model.predict(X_test_tfidf)
    y_test_prob = model.predict_proba(X_test_tfidf)[:, 1]
    
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)
    test_metrics['training_time'] = training_time
    
    print_evaluation_report(test_metrics, MODEL_NAME)
    
    # Step 6: Generate visualizations
    print("\nStep 6: Generating visualizations...")
    
    # Create directories if they don't exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_test_pred,
        save_path=str(FIGURES_DIR / 'confusion_matrix_lr_tfidf.png'),
        title=f'{MODEL_NAME} - Confusion Matrix'
    )
    
    # ROC curve
    plot_roc_curve(
        y_test, y_test_prob,
        save_path=str(FIGURES_DIR / 'roc_curve_lr_tfidf.png'),
        title=f'{MODEL_NAME} - ROC Curve'
    )
    
    # Step 7: Save results and model
    print("\nStep 7: Saving results and model...")
    
    # Save metrics
    save_results(
        test_metrics,
        f'{RESULTS_DIR}results_lr_tfidf.json',
        additional_info={
            'model': MODEL_NAME,
            'embedding': 'TF-IDF',
            'max_features': 5000,
            'best_params': model.get_params()
        }
    )
    
    # Save model and vectorizer
    joblib.dump(model, f'{RESULTS_DIR}model_lr_tfidf.pkl')
    joblib.dump(vectorizer, f'{RESULTS_DIR}vectorizer_tfidf.pkl')
    print(f"✓ Model saved to: {RESULTS_DIR}model_lr_tfidf.pkl")
    print(f"✓ Vectorizer saved to: {RESULTS_DIR}vectorizer_tfidf.pkl")
    
    # Step 8: Feature importance analysis
    print("\nStep 8: Analyzing feature importance...")
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Top positive features (indicating positive sentiment)
    top_positive_idx = np.argsort(coefficients)[-20:]
    top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_idx]
    
    # Top negative features (indicating negative sentiment)
    top_negative_idx = np.argsort(coefficients)[:20]
    top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_idx]
    
    print("\nTop 10 Positive Sentiment Indicators:")
    for word, coef in reversed(top_positive[-10:]):
        print(f"  {word:20s} : {coef:8.4f}")
    
    print("\nTop 10 Negative Sentiment Indicators:")
    for word, coef in top_negative[:10]:
        print(f"  {word:20s} : {coef:8.4f}")
    
    # Visualize feature importance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Positive features
    words_pos, coefs_pos = zip(*reversed(top_positive[-15:]))
    axes[0].barh(range(len(words_pos)), coefs_pos, color='#4ecdc4')
    axes[0].set_yticks(range(len(words_pos)))
    axes[0].set_yticklabels(words_pos)
    axes[0].set_xlabel('Coefficient Value', fontsize=11)
    axes[0].set_title('Top Features for POSITIVE Sentiment', fontsize=13, fontweight='bold')
    
    # Negative features
    words_neg, coefs_neg = zip(*top_negative[:15])
    axes[1].barh(range(len(words_neg)), coefs_neg, color='#ff6b6b')
    axes[1].set_yticks(range(len(words_neg)))
    axes[1].set_yticklabels(words_neg)
    axes[1].set_xlabel('Coefficient Value', fontsize=11)
    axes[1].set_title('Top Features for NEGATIVE Sentiment', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}feature_importance_lr_tfidf.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Feature importance plot saved")
    
    print(f"\n{'#'*60}")
    print("  TRAINING COMPLETE!")
    print(f"{'#'*60}\n")
    
    return model, vectorizer, test_metrics


if __name__ == "__main__":
    model, vectorizer, metrics = main()
    
    # Example: Make predictions on new reviews
    print("\n" + "="*60)
    print("Example Predictions on New Reviews")
    print("="*60)
    
    new_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Very disappointing."
    ]
    
    from preprocessing import preprocess_text
    
    for review in new_reviews:
        clean_review = preprocess_text(review, remove_stopwords=True)
        review_tfidf = vectorizer.transform([clean_review])
        prediction = model.predict(review_tfidf)[0]
        probability = model.predict_proba(review_tfidf)[0]
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = probability[prediction] * 100
        
        print(f"\nReview: {review}")
        print(f"Prediction: {sentiment} (Confidence: {confidence:.2f}%)")
