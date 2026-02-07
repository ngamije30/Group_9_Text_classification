"""
BiLSTM with Word2Vec (Skip-gram) Embeddings
Group 9 - Text Classification (BiLSTM)
Trains Word2Vec on corpus and uses it in an Embedding layer with BiLSTM.
"""

import numpy as np
import time
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Embedding, Dropout, Masking
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
MODEL_NAME = "BiLSTM + Word2Vec"
RESULTS_DIR = PROJECT_ROOT / "results" / "member2_bilstm"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"
FIGURES_BILSTM = PROJECT_ROOT / "figures" / "bilstm"
MAXLEN = 300
EMBED_DIM = 100
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 15
W2V_EPOCHS = 10

np.random.seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)


def train_word2vec(texts, vector_size=100, window=5, min_count=2):
    """Train Word2Vec (Skip-gram) on text list."""
    sentences = [t.split() for t in texts]
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,
        seed=RANDOM_SEED,
        epochs=W2V_EPOCHS,
    )
    return w2v


def build_embedding_matrix_w2v(tokenizer, w2v_model, embed_dim):
    """Build embedding matrix from Word2Vec for tokenizer vocabulary."""
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix, vocab_size


def main():
    print(f"\n{'#'*60}\n  {MODEL_NAME}\n{'#'*60}\n")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_BILSTM.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    data_path = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
    df = load_imdb_data(str(data_path))
    df_clean = preprocess_dataset(df, remove_stopwords=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    # Train Word2Vec
    print("Training Word2Vec (Skip-gram)...")
    w2v = train_word2vec(X_train, vector_size=EMBED_DIM, window=5, min_count=2)

    # Tokenize and pad
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAXLEN, padding="post", truncating="post")
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=MAXLEN, padding="post", truncating="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAXLEN, padding="post", truncating="post")

    embedding_matrix, vocab_size = build_embedding_matrix_w2v(tokenizer, w2v, EMBED_DIM)

    # Model
    model = Sequential([
        Embedding(vocab_size, EMBED_DIM, weights=[embedding_matrix], input_length=MAXLEN, trainable=False),
        Masking(mask_value=0.0),
        Bidirectional(LSTM(LSTM_UNITS)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

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

    # Evaluate
    y_prob = model.predict(X_test_seq).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics["training_time"] = train_time
    print_evaluation_report(metrics, MODEL_NAME)

    # Plots
    plot_confusion_matrix(
        y_test, y_pred,
        save_path=str(FIGURES_DIR / "confusion_matrix_bilstm_word2vec.png"),
        title=f"{MODEL_NAME} - Confusion Matrix",
    )
    plot_roc_curve(
        y_test, y_prob,
        save_path=str(FIGURES_DIR / "roc_curve_bilstm_word2vec.png"),
        title=f"{MODEL_NAME} - ROC Curve",
    )
    for f in ["confusion_matrix_bilstm_word2vec.png", "roc_curve_bilstm_word2vec.png"]:
        shutil.copy(FIGURES_DIR / f, FIGURES_BILSTM / f)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle(f"{MODEL_NAME} - Training History", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "bilstm_word2vec_history.png", dpi=300, bbox_inches="tight")
    plt.show()
    shutil.copy(FIGURES_DIR / "bilstm_word2vec_history.png", FIGURES_BILSTM / "bilstm_word2vec_history.png")

    # Save
    save_results(
        metrics,
        str(RESULTS_DIR / "results_bilstm_word2vec.json"),
        additional_info={"model": MODEL_NAME, "embedding": "Word2Vec (Skip-gram)", "embed_dim": EMBED_DIM, "maxlen": MAXLEN},
    )
    model.save(str(RESULTS_DIR / "model_bilstm_word2vec.keras"))
    w2v.save(str(RESULTS_DIR / "word2vec_model.bin"))
    import joblib
    joblib.dump(tokenizer, str(RESULTS_DIR / "tokenizer_word2vec.pkl"))
    print(f"✓ Model and results saved under {RESULTS_DIR}")
    return model, tokenizer, w2v, metrics


if __name__ == "__main__":
    main()
