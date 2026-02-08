import json
from pathlib import Path

# Notebook metadata
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

# --- Content Generation ---

add_markdown("# LSTM Text Classification - IMDB Sentiment Analysis\n\n**Group 9 - Text Classification Project (Member 3)**\n\n- **Dataset:** IMDB Movie Reviews\n- **Model:** Long Short-Term Memory (LSTM)\n- **Embeddings:** TF-IDF, GloVe, Word2Vec\n\nThis notebook demonstrates the implementation and training of LSTM models using three different embedding techniques.")

add_markdown("## 1. Setup and Imports")

add_code("""import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Add src to path
PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from preprocessing import load_imdb_data, preprocess_dataset, split_data
from evaluation import evaluate_model, print_evaluation_report

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Project Root: {PROJECT_ROOT}")""")

add_markdown("## 2. Data Loading & Preprocessing\n\nWe use the shared preprocessing pipeline to load and clean the IMDB dataset.")

add_code("""# Load data
data_path = PROJECT_ROOT / "data" / "raw" / "IMDB Dataset.csv"
df = load_imdb_data(str(data_path))

# Preprocess (stopwords removal is model-dependent, handled below)
print(f"Loaded {len(df)} reviews")
df.head()""")

add_markdown("## 3. Model 1: LSTM with TF-IDF\n\nTF-IDF vectors are used as input features. Since LSTM expects a sequence, we reshape the TF-IDF vector to `(samples, 1, features)`.")

add_code("""from models.lstm_tfidf import create_tfidf_sequences, build_lstm_tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF requires stopwords removal for better performance
df_tfidf = preprocess_dataset(df.copy(), remove_stopwords=True)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_tfidf)

# Create features
print("Creating TF-IDF features...")
X_train_seq, X_val_seq, X_test_seq, vectorizer = create_tfidf_sequences(
    X_train, X_val, X_test, max_features=5000
)

print(f"Input shape: {X_train_seq.shape}")

# Build and Train
model_tfidf = build_lstm_tfidf((X_train_seq.shape[1], X_train_seq.shape[2]))
model_tfidf.summary()

history_tfidf = model_tfidf.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=5,  # Reduced for demonstration
    batch_size=64,
    verbose=1
)""")

add_markdown("### TF-IDF Results")
add_code("""# Evaluate
y_prob = model_tfidf.predict(X_test_seq).flatten()
y_pred = (y_prob > 0.5).astype(int)
metrics_tfidf = evaluate_model(y_test, y_pred, y_prob)
print_evaluation_report(metrics_tfidf, "LSTM + TF-IDF")""")


add_markdown("## 4. Model 2: LSTM with GloVe Embeddings\n\nUsing pre-trained GloVe vectors (100d) to initialize the Embedding layer.")

add_code("""from models.lstm_glove import load_glove, build_embedding_matrix, build_lstm_glove
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load GloVe
glove_path = PROJECT_ROOT / "data" / "embeddings" / "glove.6B.100d.txt"
embeddings_index = load_glove(str(glove_path))
embed_dim = 100

# Preprocess (keep stopwords for sequence models)
df_glove = preprocess_dataset(df.copy(), remove_stopwords=False)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_glove)

# Tokenize
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

maxlen = 300
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=maxlen)
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=maxlen)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=maxlen)

# Embedding Matrix
embedding_matrix, vocab_size = build_embedding_matrix(tokenizer, embeddings_index, embed_dim)

# Build and Train
model_glove = build_lstm_glove(vocab_size, embed_dim, embedding_matrix, maxlen)
model_glove.summary()

history_glove = model_glove.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=5,
    batch_size=64,
    verbose=1
)""")

add_markdown("### GloVe Results")
add_code("""# Evaluate
y_prob = model_glove.predict(X_test_seq).flatten()
y_pred = (y_prob > 0.5).astype(int)
metrics_glove = evaluate_model(y_test, y_pred, y_prob)
print_evaluation_report(metrics_glove, "LSTM + GloVe")""")


add_markdown("## 5. Model 3: LSTM with Word2Vec\n\nTraining a custom Word2Vec model on the IMDB corpus.")

add_code("""from models.lstm_word2vec import train_word2vec, build_lstm_word2vec
# Note: Reuse tokenizer logic or import specific functions if needed
# For demo, we'll assume a similar flow but with custom embeddings

# Preprocess (keep stopwords)
df_w2v = preprocess_dataset(df.copy(), remove_stopwords=False)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_w2v)

# Train Word2Vec
w2v_model = train_word2vec(X_train, vector_size=100, window=5, min_count=2)

# Prepare Tokenizer matching Word2Vec vocab
tokenizer_w2v = Tokenizer()
w2v_vocab = w2v_model.wv.key_to_index
tokenizer_w2v.word_index = {word: i+1 for word, i in w2v_vocab.items()}
tokenizer_w2v.num_words = len(w2v_vocab) + 1

X_train_seq = pad_sequences(tokenizer_w2v.texts_to_sequences(X_train), maxlen=300)
X_val_seq = pad_sequences(tokenizer_w2v.texts_to_sequences(X_val), maxlen=300)
X_test_seq = pad_sequences(tokenizer_w2v.texts_to_sequences(X_test), maxlen=300)

# Build Embedding Matrix
vocab_size = len(tokenizer_w2v.word_index) + 1
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer_w2v.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# Build and Train
model_w2v = build_lstm_word2vec(vocab_size, 100, embedding_matrix, 300)
model_w2v.summary()

history_w2v = model_w2v.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=5,
    batch_size=64,
    verbose=1
)""")

add_markdown("### Word2Vec Results")
add_code("""# Evaluate
y_prob = model_w2v.predict(X_test_seq).flatten()
y_pred = (y_prob > 0.5).astype(int)
metrics_w2v = evaluate_model(y_test, y_pred, y_prob)
print_evaluation_report(metrics_w2v, "LSTM + Word2Vec")""")

add_markdown("## 6. Comparison\n\nComparison of model performance.")

add_code("""results = {
    "TF-IDF": metrics_tfidf,
    "GloVe": metrics_glove,
    "Word2Vec": metrics_w2v
}

df_results = pd.DataFrame(results).T
print(df_results)

# Plot accuracy
df_results["accuracy"].plot(kind="bar", figsize=(8, 5), title="Model Accuracy")
plt.ylabel("Accuracy")
plt.show()""")


# Save file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
output_path = PROJECT_ROOT / "notebooks" / "03_LSTM_IMDB.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook created at: {output_path}")
