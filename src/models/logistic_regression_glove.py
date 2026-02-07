"""
Logistic Regression with GloVe Embeddings
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib

# Import custom utilities
from preprocessing import load_imdb_data, preprocess_dataset, split_data
from evaluation import (evaluate_model, print_evaluation_report, 
                            plot_confusion_matrix, plot_roc_curve, save_results)

# Constants
RANDOM_SEED = 42
MODEL_NAME = "Logistic Regression + GloVe"
RESULTS_DIR = PROJECT_ROOT / "results" / "member1_logistic"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"
GLOVE_PATH = PROJECT_ROOT / "data" / "embeddings" / "glove.6B.100d.txt"


def load_glove_embeddings(glove_path):
    """
    Load pre-trained GloVe embeddings.
    
    Args:
        glove_path: Path to GloVe embeddings file
        
    Returns:
        Dictionary mapping words to their embedding vectors
    """
    print(f"\n{'='*60}")
    print("Loading GloVe Embeddings")
    print(f"{'='*60}")
    
    embeddings_index = {}
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                continue  # Skip malformed lines
    
    embedding_dim = len(next(iter(embeddings_index.values())))
    
    print(f"✓ Loaded {len(embeddings_index):,} word vectors")
    print(f"  Embedding dimension: {embedding_dim}")
    
    return embeddings_index, embedding_dim


def create_glove_features(texts, embeddings_index, embedding_dim):
    """
    Convert texts to document vectors using averaged GloVe embeddings.
    
    Args:
        texts: List of text documents
        embeddings_index: Dictionary of word embeddings
        embedding_dim: Dimension of embeddings
        
    Returns:
        numpy array of document vectors
    """
    document_vectors = []
    oov_count = 0  # Out of vocabulary count
    total_words = 0
    
    for text in texts:
        words = text.split()
        word_vectors = []
        
        for word in words:
            total_words += 1
            if word in embeddings_index:
                word_vectors.append(embeddings_index[word])
            else:
                oov_count += 1
        
        # Average the word vectors to get document vector
        if word_vectors:
            document_vector = np.mean(word_vectors, axis=0)
        else:
            # If no words found, use zero vector
            document_vector = np.zeros(embedding_dim)
        
        document_vectors.append(document_vector)
    
    oov_rate = (oov_count / total_words) * 100 if total_words > 0 else 0
    print(f"  Out-of-vocabulary rate: {oov_rate:.2f}%")
    
    return np.array(document_vectors)


def create_dataset_features(X_train, X_val, X_test, embeddings_index, embedding_dim):
    """
    Create GloVe features for train, validation, and test sets.
    
    Args:
        X_train, X_val, X_test: Text data
        embeddings_index: Dictionary of word embeddings
        embedding_dim: Dimension of embeddings
        
    Returns:
        Tuple of (X_train_glove, X_val_glove, X_test_glove)
    """
    print(f"\n{'='*60}")
    print("Creating GloVe Features")
    print(f"{'='*60}")
    
    print("Processing training set...")
    X_train_glove = create_glove_features(X_train, embeddings_index, embedding_dim)
    
    print("Processing validation set...")
    X_val_glove = create_glove_features(X_val, embeddings_index, embedding_dim)
    
    print("Processing test set...")
    X_test_glove = create_glove_features(X_test, embeddings_index, embedding_dim)
    
    print(f"\n✓ GloVe features created")
    print(f"  Training shape: {X_train_glove.shape}")
    print(f"  Validation shape: {X_val_glove.shape}")
    print(f"  Test shape: {X_test_glove.shape}")
    
    return X_train_glove, X_val_glove, X_test_glove


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
    
    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    }
    
    base_model = LogisticRegression(random_state=RANDOM_SEED)
    
    print("Performing hyperparameter tuning...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
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
    
    # Step 1: Load GloVe embeddings
    print("Step 1: Loading GloVe embeddings...")
    embeddings_index, embedding_dim = load_glove_embeddings(str(GLOVE_PATH))
    
    # Step 2: Load and preprocess data
    print("\nStep 2: Loading and preprocessing data...")
    data_path = PROJECT_ROOT / 'data' / 'raw' / 'IMDB Dataset.csv'
    df = load_imdb_data(str(data_path))
    df_clean = preprocess_dataset(df, remove_stopwords=False)  # Keep stopwords for GloVe
    
    # Step 3: Split data
    print("\nStep 3: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
    
    # Step 4: Create GloVe features
    print("\nStep 4: Creating GloVe features...")
    X_train_glove, X_val_glove, X_test_glove = create_dataset_features(
        X_train, X_val, X_test, embeddings_index, embedding_dim
    )
    
    # Step 5: Train model
    print("\nStep 5: Training model...")
    model, training_time = train_logistic_regression(
        X_train_glove, y_train, X_val_glove, y_val
    )
    
    # Step 6: Evaluate on test set
    print("\nStep 6: Evaluating on test set...")
    y_test_pred = model.predict(X_test_glove)
    y_test_prob = model.predict_proba(X_test_glove)[:, 1]
    
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)
    test_metrics['training_time'] = training_time
    
    print_evaluation_report(test_metrics, MODEL_NAME)
    
    # Step 7: Generate visualizations
    print("\nStep 7: Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_test_pred,
        save_path=str(FIGURES_DIR / 'confusion_matrix_lr_glove.png'),
        title=f'{MODEL_NAME} - Confusion Matrix'
    )
    
    # ROC curve
    plot_roc_curve(
        y_test, y_test_prob,
        save_path=str(FIGURES_DIR / 'roc_curve_lr_glove.png'),
        title=f'{MODEL_NAME} - ROC Curve'
    )
    
    # Step 8: Save results and model
    print("\nStep 8: Saving results and model...")
    
    # Save metrics
    save_results(
        test_metrics,
        str(RESULTS_DIR / 'results_lr_glove.json'),
        additional_info={
            'model': MODEL_NAME,
            'embedding': 'GloVe',
            'embedding_dim': embedding_dim,
            'glove_path': str(GLOVE_PATH),
            'best_params': model.get_params()
        }
    )
    
    # Save model and embeddings
    joblib.dump(model, str(RESULTS_DIR / 'model_lr_glove.pkl'))
    print(f"✓ Model saved to: {RESULTS_DIR / 'model_lr_glove.pkl'}")
    
    print(f"\n{'#'*60}")
    print("  TRAINING COMPLETE!")
    print(f"{'#'*60}\n")
    
    return model, embeddings_index, test_metrics


if __name__ == "__main__":
    model, embeddings_index, metrics = main()
    
    # Example: Make predictions on new reviews
    print("\n" + "="*60)
    print("Example Predictions on New Reviews")
    print("="*60)
    
    new_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Very disappointing."
    ]
    
    from preprocessing import preprocess_text
    
    embedding_dim = len(next(iter(embeddings_index.values())))
    
    for review in new_reviews:
        clean_review = preprocess_text(review, remove_stopwords=False)
        review_glove = create_glove_features([clean_review], embeddings_index, embedding_dim)
        prediction = model.predict(review_glove)[0]
        probability = model.predict_proba(review_glove)[0]
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = probability[prediction] * 100
        
        print(f"\nReview: {review}")
        print(f"Prediction: {sentiment} (Confidence: {confidence:.2f}%)")
