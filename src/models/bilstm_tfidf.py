"""
BiLSTM with TF-IDF Embeddings
Group 9 - Text Classification (BiLSTM)
Uses TF-IDF document vectors as a single timestep input to BiLSTM.
"""

import numpy as np
import time
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Dropout
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
MODEL_NAME = "BiLSTM + TF-IDF"
RESULTS_DIR = PROJECT_ROOT / "results" / "member2_bilstm"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"
FIGURES_BILSTM = PROJECT_ROOT / "figures" / "bilstm"
MAX_FEATURES = 5000
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 15

np.random.seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)


def create_tfidf_sequences(X_train, X_val, X_test, max_features=5000):
    """Create TF-IDF features and reshape to (samples, 1, features) for BiLSTM."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=5, max_df=0.8, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_val_tfidf = vectorizer.transform(X_val).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    # Reshape to (samples, timesteps=1, features)
    X_train_seq = X_train_tfidf.reshape(X_train_tfidf.shape[0], 1, X_train_tfidf.shape[1])
    X_val_seq = X_val_tfidf.reshape(X_val_tfidf.shape[0], 1, X_val_tfidf.shape[1])
    X_test_seq = X_test_tfidf.reshape(X_test_tfidf.shape[0], 1, X_test_tfidf.shape[1])
    return X_train_seq, X_val_seq, X_test_seq, vectorizer


def build_bilstm_tfidf(input_shape):
    """Build BiLSTM for TF-IDF (1 timestep, many features)."""
    model = Sequential([
        Bidirectional(LSTM(LSTM_UNITS), input_shape=input_shape),
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
    FIGURES_BILSTM.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    data_path = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
    df = load_imdb_data(str(data_path))
    df_clean = preprocess_dataset(df, remove_stopwords=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    # TF-IDF features
    X_train_seq, X_val_seq, X_test_seq, vectorizer = create_tfidf_sequences(
        X_train, X_val, X_test, max_features=MAX_FEATURES
    )

    # Model
    model = build_bilstm_tfidf((X_train_seq.shape[1], X_train_seq.shape[2]))
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
        save_path=str(FIGURES_DIR / "confusion_matrix_bilstm_tfidf.png"),
        title=f"{MODEL_NAME} - Confusion Matrix",
    )
    plot_roc_curve(
        y_test, y_prob,
        save_path=str(FIGURES_DIR / "roc_curve_bilstm_tfidf.png"),
        title=f"{MODEL_NAME} - ROC Curve",
    )
    for f in ["confusion_matrix_bilstm_tfidf.png", "roc_curve_bilstm_tfidf.png"]:
        shutil.copy(FIGURES_DIR / f, FIGURES_BILSTM / f)

    # Training history plot
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
    plt.savefig(FIGURES_DIR / "bilstm_tfidf_history.png", dpi=300, bbox_inches="tight")
    plt.show()
    shutil.copy(FIGURES_DIR / "bilstm_tfidf_history.png", FIGURES_BILSTM / "bilstm_tfidf_history.png")

    # Save
    save_results(
        metrics,
        str(RESULTS_DIR / "results_bilstm_tfidf.json"),
        additional_info={"model": MODEL_NAME, "embedding": "TF-IDF", "max_features": MAX_FEATURES},
    )
    model.save(str(RESULTS_DIR / "model_bilstm_tfidf.keras"))
    import joblib
    joblib.dump(vectorizer, str(RESULTS_DIR / "vectorizer_tfidf.pkl"))
    print(f"✓ Model and results saved under {RESULTS_DIR}")
    return model, vectorizer, metrics


if __name__ == "__main__":
    main()
