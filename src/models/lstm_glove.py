"""
LSTM with GloVe Embeddings
Group 9 - Text Classification (LSTM)
Uses pre-trained GloVe word vectors in an Embedding layer with LSTM.

Author: Member 2 (Gustav)
"""

import numpy as np
import time
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import set_random_seed

from preprocessing import load_imdb_data, preprocess_dataset, split_data
from evaluation import (
    evaluate_model,
    print_evaluation_report,
    plot_confusion_matrix,
    plot_roc_curve,
    save_results,
)

# Constants
RANDOM_SEED = 42
MODEL_NAME = "LSTM + GloVe"
RESULTS_DIR = PROJECT_ROOT / "results" / "member3_lstm"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"
FIGURES_LSTM = PROJECT_ROOT / "figures" / "lstm"
GLOVE_PATH = PROJECT_ROOT / "data" / "embeddings" / "glove.6B.100d.txt"
MAXLEN = 300
EMBED_DIM = 100
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 15

np.random.seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)


def load_glove(path):
    """Load GloVe embeddings into a dictionary."""
    print(f"Loading GloVe embeddings from {path}...")
    embeddings_index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            # Skip header line in Gensim-saved format (vocab_size vector_size)
            if len(values) < 10:
                continue
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
            except (ValueError, IndexError):
                continue
    return embeddings_index


def build_embedding_matrix(tokenizer, embeddings_index, embed_dim):
    """Build embedding matrix for vocabulary from GloVe."""
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    
    oov_count = 0
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
        else:
            oov_count += 1
    
    oov_rate = oov_count / len(word_index) * 100
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Out-of-vocabulary words: {oov_count:,} ({oov_rate:.2f}%)")
    
    return embedding_matrix, vocab_size


def build_lstm_glove(vocab_size, embed_dim, embedding_matrix, maxlen):
    """Build LSTM model with GloVe embeddings."""
    model = Sequential([
        Embedding(
            vocab_size, 
            embed_dim, 
            weights=[embedding_matrix], 
            input_length=maxlen, 
            trainable=False,
            mask_zero=False
        ),
        LSTM(LSTM_UNITS),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    print(f"\n{'#'*60}\n  {MODEL_NAME}\n{'#'*60}\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_LSTM.mkdir(parents=True, exist_ok=True)

    # Load GloVe embeddings
    embeddings_index = load_glove(str(GLOVE_PATH))
    embed_dim = len(next(iter(embeddings_index.values())))
    print(f"  Loaded {len(embeddings_index):,} GloVe vectors, dimension={embed_dim}")

    # Load and preprocess (keep stopwords for sequence models)
    data_path = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
    df = load_imdb_data(str(data_path))
    df_clean = preprocess_dataset(df, remove_stopwords=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    print(f"\nTrain size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # Tokenize and pad sequences
    print("\nTokenizing and padding sequences...")
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train), 
        maxlen=MAXLEN, 
        padding="post", 
        truncating="post"
    )
    X_val_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_val), 
        maxlen=MAXLEN, 
        padding="post", 
        truncating="post"
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_test), 
        maxlen=MAXLEN, 
        padding="post", 
        truncating="post"
    )
    print(f"Sequence shape: {X_train_seq.shape}")

    # Build embedding matrix
    print("\nBuilding embedding matrix...")
    embedding_matrix, vocab_size = build_embedding_matrix(tokenizer, embeddings_index, embed_dim)

    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_glove(vocab_size, embed_dim, embedding_matrix, MAXLEN)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    print("\nTraining LSTM...")
    start = time.time()
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - start
    print(f"\nTraining completed in {train_time:.2f} seconds")

    # Evaluate
    print("\nEvaluating on test set...")
    y_prob = model.predict(X_test_seq).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["training_time"] = train_time
    print_evaluation_report(metrics, MODEL_NAME)

    # Plots
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=str(FIGURES_DIR / "confusion_matrix_lstm_glove.png"),
        title=f"{MODEL_NAME} - Confusion Matrix",
    )
    plot_roc_curve(
        y_test, y_prob,
        save_path=str(FIGURES_DIR / "roc_curve_lstm_glove.png"),
        title=f"{MODEL_NAME} - ROC Curve",
    )
    
    for f in ["confusion_matrix_lstm_glove.png", "roc_curve_lstm_glove.png"]:
        shutil.copy(FIGURES_DIR / f, FIGURES_LSTM / f)

    # Training history plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f"{MODEL_NAME} - Training History", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_LSTM / "lstm_glove_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Training history saved")

    # Save results
    save_results(
        metrics,
        str(RESULTS_DIR / "results_lstm_glove.json"),
        additional_info={
            "model": MODEL_NAME, 
            "embedding": "GloVe", 
            "embed_dim": embed_dim, 
            "maxlen": MAXLEN,
            "lstm_units": LSTM_UNITS,
            "epochs_trained": len(history.history["loss"]),
        },
    )
    model.save(str(RESULTS_DIR / "model_lstm_glove.keras"))
    
    import joblib
    joblib.dump(tokenizer, str(RESULTS_DIR / "tokenizer_glove.pkl"))
    
    print(f"✓ Model and results saved under {RESULTS_DIR}")
    return model, tokenizer, metrics


if __name__ == "__main__":
    main()
