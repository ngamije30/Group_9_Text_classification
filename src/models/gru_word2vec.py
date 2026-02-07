"""
GRU with Word2Vec Embeddings for IMDB Sentiment Classification
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import set_random_seed

# Word2Vec
from gensim.models import Word2Vec

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
MAX_WORDS = 20000
MAX_LEN = 200
EMBEDDING_DIM = 100

np.random.seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "member4_gru"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_text(text, remove_stopwords=False):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text


def load_and_preprocess_data():
    """Load and preprocess IMDB dataset."""
    print("="*80)
    print("Loading and preprocessing data...")
    print("="*80)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape}")
    
    # Preprocess WITHOUT stopword removal for Word2Vec
    print("Preprocessing text (keeping stopwords for embeddings)...")
    df['text_clean'] = df['review'].apply(lambda x: preprocess_text(x, remove_stopwords=False))
    
    # Encode labels
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    
    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=RANDOM_SEED, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=2/3, random_state=RANDOM_SEED, stratify=temp_df['label']
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    return train_df, val_df, test_df


def train_word2vec(train_df):
    """Train Word2Vec model on training data."""
    print("\n" + "="*80)
    print("Training Word2Vec model...")
    print("="*80)
    
    # Tokenize texts
    tokenized_texts = [text.split() for text in train_df['text_clean']]
    
    # Train Word2Vec
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=EMBEDDING_DIM,
        window=5,
        min_count=2,
        epochs=10,
        sg=1,  # Skip-gram
        workers=4,
        seed=RANDOM_SEED
    )
    
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
    print(f"Vector dimension: {w2v_model.vector_size}")
    
    # Test similarity
    try:
        similar = w2v_model.wv.most_similar('good', topn=3)
        print(f"\nMost similar to 'good': {[w for w, _ in similar]}")
    except:
        print("\n'good' not in vocabulary")
    
    return w2v_model


def create_sequences(train_df, val_df, test_df):
    """Tokenize and create padded sequences."""
    print("\n" + "="*80)
    print("Creating sequences...")
    print("="*80)
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_df['text_clean'])
    
    # Convert to sequences
    X_train_seq = tokenizer.texts_to_sequences(train_df['text_clean'])
    X_val_seq = tokenizer.texts_to_sequences(val_df['text_clean'])
    X_test_seq = tokenizer.texts_to_sequences(test_df['text_clean'])
    
    # Pad sequences
    X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values
    
    print(f"Sequences created:")
    print(f"  Vocabulary size: {len(tokenizer.word_index)}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer


def create_embedding_matrix(tokenizer, w2v_model):
    """Create embedding matrix from Word2Vec."""
    print("\n" + "="*80)
    print("Creating embedding matrix...")
    print("="*80)
    
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    
    oov_count = 0
    for word, i in tokenizer.word_index.items():
        if i >= MAX_WORDS:
            continue
        try:
            embedding_vector = w2v_model.wv[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            oov_count += 1
            embedding_matrix[i] = np.random.uniform(-0.1, 0.1, EMBEDDING_DIM)
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Out-of-vocabulary words: {oov_count} ({oov_count/len(tokenizer.word_index)*100:.2f}%)")
    
    return embedding_matrix, vocab_size


def build_gru_model(embedding_matrix, vocab_size):
    """Build GRU model with Word2Vec embeddings."""
    print("\n" + "="*80)
    print("Building GRU model with Word2Vec...")
    print("="*80)
    
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_LEN,
            trainable=True,  # Fine-tune Word2Vec embeddings
            name='word2vec_embedding'
        ),
        GRU(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ], name='GRU_Word2Vec')
    
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
    print("Training GRU + Word2Vec model...")
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
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
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
    
    model_path = RESULTS_DIR / 'model_gru_word2vec.keras'
    model.save(model_path)
    print(f"Model saved: {model_path}")
    
    results['model'] = 'GRU'
    results['embedding'] = 'Word2Vec'
    results['training_time'] = float(training_time)
    
    results_path = RESULTS_DIR / 'results_gru_word2vec.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved: {results_path}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("GRU + Word2Vec - IMDB Sentiment Classification")
    print("Group 9 - Member 4")
    print("="*80 + "\n")
    
    # Load data
    train_df, val_df, test_df = load_and_preprocess_data()
    
    # Train Word2Vec
    w2v_model = train_word2vec(train_df)
    
    # Create sequences
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = create_sequences(
        train_df, val_df, test_df
    )
    
    # Create embedding matrix
    embedding_matrix, vocab_size = create_embedding_matrix(tokenizer, w2v_model)
    
    # Build model
    model = build_gru_model(embedding_matrix, vocab_size)
    
    # Train model
    history, training_time = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Save results
    save_results(model, results, training_time)
    
    print("\n" + "="*80)
    print("GRU + Word2Vec training and evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()