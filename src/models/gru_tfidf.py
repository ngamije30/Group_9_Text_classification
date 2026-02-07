"""
GRU with TF-IDF Embeddings for IMDB Sentiment Classification
Group 9 - Member 4 (GRU Implementation)
"""

import numpy as np
import pandas as pd
import json
import time
import re
from pathlib import Path

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import set_random_seed

# NLTK
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "member4_gru"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_text(text, remove_stopwords=True):
    """
    Clean and preprocess text.
    
    Args:
        text: Raw text string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Cleaned text string
    """
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords if specified
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text


def load_and_preprocess_data():
    """Load and preprocess IMDB dataset."""
    print("="*80)
    print("Loading and preprocessing data...")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape}")
    
    # Preprocess text (remove stopwords for TF-IDF)
    print("Preprocessing text with stopword removal...")
    df['text_clean'] = df['review'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
    
    # Encode labels
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    
    # Split data: 70% train, 10% val, 20% test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_SEED, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=2/3, random_state=RANDOM_SEED, stratify=temp_df['label']
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def create_tfidf_features(train_df, val_df, test_df):
    """Create TF-IDF feature vectors."""
    print("\n" + "="*80)
    print("Creating TF-IDF features...")
    print("="*80)
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    
    # Fit and transform
    X_train = vectorizer.fit_transform(train_df['text_clean']).toarray()
    X_val = vectorizer.transform(val_df['text_clean']).toarray()
    X_test = vectorizer.transform(test_df['text_clean']).toarray()
    
    # Reshape for GRU: (samples, timesteps=1, features)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    print(f"TF-IDF features created:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_gru_model():
    """Build GRU model for TF-IDF features."""
    print("\n" + "="*80)
    print("Building GRU model...")
    print("="*80)
    
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(1, 5000)),
        GRU(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ], name='GRU_TFIDF')
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the GRU model."""
    print("\n" + "="*80)
    print("Training GRU + TF-IDF model...")
    print("="*80)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ]
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    return history, training_time


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    print("\n" + "="*80)
    print("Evaluating model on test set...")
    print("="*80)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:5d}  |  FP: {cm[0,1]:5d}")
    print(f"  FN: {cm[1,0]:5d}  |  TP: {cm[1,1]:5d}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }


def save_results(model, results, training_time):
    """Save model and results."""
    print("\n" + "="*80)
    print("Saving model and results...")
    print("="*80)
    
    # Save model
    model_path = RESULTS_DIR / 'model_gru_tfidf.keras'
    model.save(model_path)
    print(f"Model saved: {model_path}")
    
    # Save results
    results['model'] = 'GRU'
    results['embedding'] = 'TF-IDF'
    results['training_time'] = float(training_time)
    
    results_path = RESULTS_DIR / 'results_gru_tfidf.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved: {results_path}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("GRU + TF-IDF - IMDB Sentiment Classification")
    print("Group 9 - Member 4")
    print("="*80 + "\n")
    
    # Load and preprocess data
    train_df, val_df, test_df = load_and_preprocess_data()
    
    # Create TF-IDF features
    X_train, X_val, X_test, y_train, y_val, y_test = create_tfidf_features(
        train_df, val_df, test_df
    )
    
    # Build model
    model = build_gru_model()
    
    # Train model
    history, training_time = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Save results
    save_results(model, results, training_time)
    
    print("\n" + "="*80)
    print("GRU + TF-IDF training and evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()