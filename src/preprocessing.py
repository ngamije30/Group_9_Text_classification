"""
Text Preprocessing Utilities for IMDB Sentiment Classification
Group 9 - Davy (Logistic Regression)
"""

import re
import numpy as np
import pandas as pd
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Constants
RANDOM_SEED = 42
STOP_WORDS = set(stopwords.words('english'))


def load_imdb_data(filepath: str) -> pd.DataFrame:
    """
    Load IMDB dataset from CSV file.
    
    Args:
        filepath: Path to IMDB Dataset.csv
        
    Returns:
        DataFrame with 'review' and 'sentiment' columns
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} reviews")
    print(f"Columns: {df.columns.tolist()}")
    return df


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean and preprocess a single text document.
    
    Args:
        text: Raw text string
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords if specified
    if remove_stopwords:
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOP_WORDS and len(word) > 2]
        text = ' '.join(tokens)
    
    return text


def preprocess_dataset(df: pd.DataFrame, remove_stopwords: bool = True) -> pd.DataFrame:
    """
    Preprocess entire IMDB dataset.
    
    Args:
        df: DataFrame with 'review' and 'sentiment' columns
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        DataFrame with cleaned reviews and binary labels
    """
    print("Preprocessing dataset...")
    
    # Create a copy
    df_clean = df.copy()
    
    # Convert sentiment to binary labels (0: negative, 1: positive)
    df_clean['label'] = df_clean['sentiment'].map({'negative': 0, 'positive': 1})
    
    # Clean reviews
    df_clean['review_clean'] = df_clean['review'].apply(
        lambda x: preprocess_text(x, remove_stopwords)
    )
    
    # Calculate review lengths
    df_clean['review_length'] = df_clean['review'].apply(lambda x: len(x.split()))
    df_clean['clean_length'] = df_clean['review_clean'].apply(lambda x: len(x.split()))
    
    print(f"Preprocessing complete. {len(df_clean)} reviews processed.")
    return df_clean


def split_data(df: pd.DataFrame, 
               test_size: float = 0.2, 
               val_size: float = 0.1) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: DataFrame with 'review_clean' and 'label' columns
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"\nSplitting data...")
    print(f"Test size: {test_size*100}%, Validation size: {val_size*100}%")
    
    # Get features and labels
    X = df['review_clean'].values
    y = df['label'].values
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    
    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\nClass distribution:")
    print(f"Train - Negative: {(y_train==0).sum()}, Positive: {(y_train==1).sum()}")
    print(f"Val   - Negative: {(y_val==0).sum()}, Positive: {(y_val==1).sum()}")
    print(f"Test  - Negative: {(y_test==0).sum()}, Positive: {(y_test==1).sum()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_vocabulary_stats(texts: List[str]) -> dict:
    """
    Calculate vocabulary statistics from text corpus.
    
    Args:
        texts: List of text documents
        
    Returns:
        Dictionary with vocabulary statistics
    """
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    vocab = set(all_words)
    
    stats = {
        'total_words': len(all_words),
        'unique_words': len(vocab),
        'avg_words_per_doc': len(all_words) / len(texts),
        'vocabulary_richness': len(vocab) / len(all_words)
    }
    
    return stats


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...\n")
    
    # Test single text preprocessing
    sample_text = "<p>This is a GREAT movie!!! I loved it so much. http://example.com</p>"
    print(f"Original: {sample_text}")
    print(f"Cleaned: {preprocess_text(sample_text)}\n")
    
    # Test with dataset (if available)
    try:
        df = load_imdb_data('data/raw/IMDB Dataset.csv')
        df_clean = preprocess_dataset(df)
        print(f"\nSample cleaned review:")
        print(df_clean['review_clean'].iloc[0][:200])
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)
        
        # Get vocabulary stats
        vocab_stats = get_vocabulary_stats(X_train)
        print(f"\nVocabulary Statistics:")
        for key, value in vocab_stats.items():
            print(f"  {key}: {value:.2f}")
            
    except FileNotFoundError:
        print("IMDB Dataset.csv not found. Please place it in data/raw/ directory.")
