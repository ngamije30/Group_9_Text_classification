#!/usr/bin/env bash
# Run all BiLSTM training scripts and comparison from project root.
# Usage: ./run_bilstm.sh   OR   bash run_bilstm.sh
# Ensure you have activated your venv/conda env with: numpy, pandas, nltk, scikit-learn, tensorflow, gensim, matplotlib, seaborn, joblib

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "  BiLSTM + TF-IDF"
echo "=============================================="
python3 -m src.models.bilstm_tfidf

echo ""
echo "=============================================="
echo "  BiLSTM + GloVe"
echo "=============================================="
python3 -m src.models.bilstm_glove

echo ""
echo "=============================================="
echo "  BiLSTM + Word2Vec"
echo "=============================================="
python3 -m src.models.bilstm_word2vec

echo ""
echo "=============================================="
echo "  BiLSTM comparison"
echo "=============================================="
python3 -m src.models.compare_results_bilstm

echo ""
echo "Done."
