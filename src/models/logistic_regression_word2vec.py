"""
Logistic Regression with Word2Vec (Skip-gram) Embeddings
Group 9 - Davy (Logistic Regression)
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from gensim.models import Word2Vec
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
MODEL_NAME = "Logistic Regression + Word2Vec"
RESULTS_DIR = PROJECT_ROOT / "results" / "member1_logistic"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"


def train_word2vec(texts, vector_size=100, window=5, min_count=2):
    """
    Train Word2Vec model on the corpus.
    
    Args:
        texts: List of text documents
        vector_size: Dimension of word embeddings
        window: Context window size
        min_count: Minimum word frequency
        
    Returns:
        Trained Word2Vec model
    """
    print(f"\n{'='*60}")
    print("Training Word2Vec Model (Skip-gram)")
    print(f"{'='*60}")
    
    # Tokenize texts into list of lists
    sentences = [text.split() for text in texts]
    
    print(f"Training on {len(sentences):,} documents...")
    start_time = time.time()
    
    # Train Word2Vec with Skip-gram (sg=1)
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # Skip-gram (sg=1), CBOW would be sg=0
        seed=RANDOM_SEED,
        epochs=10
    )
    
    training_time = time.time() - start_time
    
    print(f"✓ Word2Vec training completed in {training_time:.2f} seconds")
    print(f"  Vocabulary size: {len(w2v_model.wv):,}")
    print(f"  Vector size: {w2v_model.wv.vector_size}")
    
    return w2v_model


def create_word2vec_features(texts, w2v_model):
    """
    Convert texts to document vectors using averaged Word2Vec embeddings.
    
    Args:
        texts: List of text documents
        w2v_model: Trained Word2Vec model
        
    Returns:
        numpy array of document vectors
    """
    document_vectors = []
    oov_count = 0
    total_words = 0
    
    for text in texts:
        words = text.split()
        word_vectors = []
        
        for word in words:
            total_words += 1
            if word in w2v_model.wv:
                word_vectors.append(w2v_model.wv[word])
            else:
                oov_count += 1
        
        # Average the word vectors
        if word_vectors:
            document_vector = np.mean(word_vectors, axis=0)
        else:
            # If no words found, use zero vector
            document_vector = np.zeros(w2v_model.wv.vector_size)
        
        document_vectors.append(document_vector)
    
    oov_rate = (oov_count / total_words) * 100 if total_words > 0 else 0
    print(f"  Out-of-vocabulary rate: {oov_rate:.2f}%")
    
    return np.array(document_vectors)


def create_dataset_features(X_train, X_val, X_test, w2v_model):
    """
    Create Word2Vec features for all datasets.
    
    Args:
        X_train, X_val, X_test: Text data
        w2v_model: Trained Word2Vec model
        
    Returns:
        Tuple of (X_train_w2v, X_val_w2v, X_test_w2v)
    """
    print(f"\n{'='*60}")
    print("Creating Word2Vec Features")
    print(f"{'='*60}")
    
    print("Processing training set...")
    X_train_w2v = create_word2vec_features(X_train, w2v_model)
    
    print("Processing validation set...")
    X_val_w2v = create_word2vec_features(X_val, w2v_model)
    
    print("Processing test set...")
    X_test_w2v = create_word2vec_features(X_test, w2v_model)
    
    print(f"\n✓ Word2Vec features created")
    print(f"  Training shape: {X_train_w2v.shape}")
    print(f"  Validation shape: {X_val_w2v.shape}")
    print(f"  Test shape: {X_test_w2v.shape}")
    
    return X_train_w2v, X_val_w2v, X_test_w2v


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
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    data_path = PROJECT_ROOT / 'data' / 'raw' / 'IMDB Dataset.csv'
    df = load_imdb_data(str(data_path))
    df_clean = preprocess_dataset(df, remove_stopwords=False)  # Keep stopwords for Word2Vec
    
    # Step 2: Split data
    print("\nStep 2: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
    
    # Step 3: Train Word2Vec model
    print("\nStep 3: Training Word2Vec model...")
    w2v_model = train_word2vec(X_train, vector_size=100, window=5, min_count=2)
    
    # Save Word2Vec model
    w2v_model.save(f'{RESULTS_DIR}word2vec_model.bin')
    print(f"✓ Word2Vec model saved to: {RESULTS_DIR}word2vec_model.bin")
    
    # Step 4: Create Word2Vec features
    print("\nStep 4: Creating Word2Vec features...")
    X_train_w2v, X_val_w2v, X_test_w2v = create_dataset_features(
        X_train, X_val, X_test, w2v_model
    )
    
    # Step 5: Train Logistic Regression
    print("\nStep 5: Training Logistic Regression...")
    model, lr_training_time = train_logistic_regression(
        X_train_w2v, y_train, X_val_w2v, y_val
    )
    
    # Step 6: Evaluate on test set
    print("\nStep 6: Evaluating on test set...")
    y_test_pred = model.predict(X_test_w2v)
    y_test_prob = model.predict_proba(X_test_w2v)[:, 1]
    
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_prob)
    test_metrics['training_time'] = lr_training_time
    
    print_evaluation_report(test_metrics, MODEL_NAME)
    
    # Step 7: Generate visualizations
    print("\nStep 7: Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_test_pred,
        save_path=f'{FIGURES_DIR}confusion_matrix_lr_word2vec.png',
        title=f'{MODEL_NAME} - Confusion Matrix'
    )
    
    # ROC curve
    plot_roc_curve(
        y_test, y_test_prob,
        save_path=f'{FIGURES_DIR}roc_curve_lr_word2vec.png',
        title=f'{MODEL_NAME} - ROC Curve'
    )
    
    # Step 8: Save results and model
    print("\nStep 8: Saving results and model...")
    
    # Save metrics
    save_results(
        test_metrics,
        f'{RESULTS_DIR}results_lr_word2vec.json',
        additional_info={
            'model': MODEL_NAME,
            'embedding': 'Word2Vec (Skip-gram)',
            'vector_size': w2v_model.wv.vector_size,
            'vocabulary_size': len(w2v_model.wv),
            'best_params': model.get_params()
        }
    )
    
    # Save Logistic Regression model
    joblib.dump(model, f'{RESULTS_DIR}model_lr_word2vec.pkl')
    print(f"✓ Logistic Regression model saved to: {RESULTS_DIR}model_lr_word2vec.pkl")
    
    # Step 9: Explore Word2Vec embeddings
    print("\nStep 9: Exploring Word2Vec embeddings...")
    
    # Find similar words
    test_words = ['good', 'bad', 'excellent', 'terrible', 'love', 'hate']
    print("\nSimilar words in learned embeddings:")
    for word in test_words:
        if word in w2v_model.wv:
            similar = w2v_model.wv.most_similar(word, topn=5)
            print(f"\n{word.upper()}:")
            for sim_word, score in similar:
                print(f"  {sim_word:15s} : {score:.4f}")
    
    print(f"\n{'#'*60}")
    print("  TRAINING COMPLETE!")
    print(f"{'#'*60}\n")
    
    return model, w2v_model, test_metrics


if __name__ == "__main__":
    model, w2v_model, metrics = main()
    
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
        clean_review = preprocess_text(review, remove_stopwords=False)
        review_w2v = create_word2vec_features([clean_review], w2v_model)
        prediction = model.predict(review_w2v)[0]
        probability = model.predict_proba(review_w2v)[0]
        
        sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
        confidence = probability[prediction] * 100
        
        print(f"\nReview: {review}")
        print(f"Prediction: {sentiment} (Confidence: {confidence:.2f}%)")
