# Comparative Analysis of Text Classification with Multiple Embeddings

**Group 9 - Text Classification Project**  
**Course:** Machine Learning Techniques - I
**Date:** February 2026

---

## Research Questions
- How do different word embedding techniques (TF-IDF, GloVe, Word2Vec) impact the performance of various text classification models?
- Which embedding-model combinations yield the best results for sentiment analysis on the IMDB dataset?
- What are the trade-offs between sparse and dense representations in practical NLP tasks?
- How does embedding performance vary with review length and vocabulary size?

---

## Literature Review
Recent research highlights the importance of embedding choice in NLP tasks. Maas et al. (2011) demonstrated the effectiveness of word vectors for sentiment analysis. Mikolov et al. (2013) introduced Word2Vec, showing that dense embeddings capture semantic relationships. Pennington et al. (2014) proposed GloVe, which leverages global word co-occurrence statistics. Prior work (Pedregosa et al., 2011; Řehůřek & Sojka, 2010) established the utility of traditional models and topic modeling frameworks. Our work builds on these foundations, systematically comparing embeddings across multiple model architectures for sentiment classification.

---

## Project Overview

This project implements and evaluates text classification systems using **multiple word embedding techniques** across different model architectures. The primary objective is to conduct a comprehensive comparative analysis of how different embedding methods (TF-IDF, GloVe, Word2Vec) perform with various machine learning models on sentiment analysis tasks.

### Objectives
1. Compare performance of different word embeddings on text classification
2. Analyze trade-offs between embedding approaches (sparse vs. dense representations)
3. Provide empirical evidence for embedding selection in NLP tasks
4. Document reproducible experiments with proper academic rigor

---

## Team Responsibilities

Each team member implements **one model architecture** evaluated across **three different embeddings**:

| Member   | Model Architecture      | Embeddings Evaluated                | Status         |
|----------|------------------------|-------------------------------------|----------------|
| Davy | Logistic Regression    | TF-IDF, GloVe, Word2Vec Skip-gram   | Complete       |
| Sage | BiLSTM                 | TF-IDF, GloVe, Word2Vec             | Complete       |
| Gustav   | LSTM                   | TF-IDF, GloVe, Word2Vec             | Complete       |
| Alice    | GRU                    | TF-IDF, GloVe, Word2Vec             | Complete       |

**Note:** All four main model architectures (Logistic Regression, LSTM, BiLSTM, GRU) have been fully implemented and evaluated with three embeddings each. Results and analysis for each are included in the report and codebase.

---

## Dataset

**Dataset:** IMDB Movie Reviews Dataset  
**Source:** [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
**Task:** Binary Sentiment Classification (Positive/Negative)

### Dataset Characteristics
- **Total Reviews:** 50,000
- **Classes:** Binary (Positive/Negative)
- **Class Distribution:** Perfectly balanced (25,000 positive, 25,000 negative)
- **Split:** 70% Train (35,000) / 10% Validation (5,000) / 20% Test (10,000)
- **Vocabulary Size:** ~67,000 unique tokens (after preprocessing)
- **Average Review Length:** 233 words

### Exploratory Data Analysis
Our EDA revealed:
- Balanced class distribution (50-50 split)
- Review lengths vary from 10 to 2,000+ words
- Common positive indicators: "excellent", "great", "wonderful", "amazing"
- Common negative indicators: "worst", "awful", "terrible", "boring"

See [notebooks/01_EDA_IMDB.ipynb](notebooks/01_EDA_IMDB.ipynb) for detailed visualizations.

---

## Implementation Details

### Davy: Logistic Regression

**Model:** Logistic Regression with L2 regularization  
**Embeddings Evaluated:** TF-IDF, GloVe (100d), Word2Vec Skip-gram (100d)

#### Preprocessing Strategy
- **Text cleaning:** Lowercase, HTML tag removal, special character handling
- **Tokenization:** Word-level tokenization
- **Stopword handling:** 
  - TF-IDF: Stopwords removed (better sparse features)
  - GloVe/Word2Vec: Stopwords retained (semantic context preservation)

#### Embedding-Specific Configurations

**1. TF-IDF**
- Max features: 5,000
- N-gram range: (1, 2) - unigrams and bigrams
- Sublinear TF scaling: True
- Min document frequency: 2
- Sparsity: 98.41%

**2. GloVe (Pre-trained)**
- Source: GloVe 6B (Wikipedia + Gigaword)
- Vector dimension: 100
- Vocabulary size: 400,000 words
- Document representation: Average of word vectors
- Out-of-vocabulary rate: 2.1%

**3. Word2Vec (Custom trained)**
- Algorithm: Skip-gram
- Vector dimension: 100
- Window size: 5
- Min word count: 2
- Training epochs: 10
- Corpus vocabulary: 67,841 words
- Out-of-vocabulary rate: 1.8%

#### Hyperparameter Tuning
- **Method:** GridSearchCV with 3-fold cross-validation
- **Regularization (C):** [0.01, 0.1, 1, 10, 100]
- **Solver:** lbfgs (L-BFGS-B algorithm)
- **Max iterations:** 1000
- **Best parameters:**
  - TF-IDF: C=1.0
  - GloVe: C=100.0
  - Word2Vec: C=10.0

### Sage: BiLSTM
**Model:** Bidirectional LSTM (BiLSTM)
**Embeddings Evaluated:** TF-IDF, GloVe, Word2Vec
- Used sequence modeling to capture both forward and backward context.
- Preprocessing and embedding strategies consistent with team pipeline.
- Hyperparameter tuning and early stopping applied.
- Results and code documented in [notebooks/02_BiLSTM_IMDB.ipynb].

### Gustav: LSTM
**Model:** Long Short-Term Memory (LSTM)
**Embeddings Evaluated:** TF-IDF, GloVe, Word2Vec
- Implemented LSTM for sequential text classification.
- Used same preprocessing and embedding pipeline as other models.
- Hyperparameter tuning and early stopping applied.
- Results and code documented in [notebooks/03_LSTM_IMDB.ipynb].

### Alice: GRU
**Model:** Gated Recurrent Unit (GRU)
**Embeddings Evaluated:** TF-IDF, GloVe, Word2Vec
- Implemented GRU for sequential text classification.
- Used same preprocessing and embedding pipeline as other models.
- Hyperparameter tuning and early stopping applied.
- Results and code documented in [notebooks/03_GRU_IMDB.ipynb].

---

## Results Summary

### Performance Comparison

| Embedding | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-----------|----------|-----------|--------|----------|---------|---------------|
| **TF-IDF** | **89.22%** | **88.51%** | **90.14%** | **89.32%** | **95.99%** | 8.7s |
| Word2Vec | 86.50% | 86.56% | 86.42% | 86.49% | 93.95% | 7.5s |
| GloVe | 79.86% | 79.98% | 79.66% | 79.82% | 87.81% | 18.2s |

### Winner: TF-IDF
TF-IDF achieved the best performance across all metrics with Logistic Regression, demonstrating that:
1. **Sparse representations** with domain-specific vocabulary work well for sentiment analysis
2. **Stopword removal** helps TF-IDF focus on sentiment-bearing words
3. **N-grams** capture phrase-level sentiment patterns ("not good", "very bad")

### Key Observations

**TF-IDF Strengths:**
- Highest accuracy and F1-score (89.32%)
- Excellent discrimination (ROC-AUC: 95.99%)
- Fast training time (8.7s)
- Top predictive features clearly interpretable:
  - Positive: "excellent" (6.74), "great" (6.74), "amazing" (4.62)
  - Negative: "worst" (-8.56), "bad" (-7.50), "awful" (-7.05)

**Word2Vec Strengths:**
- Second-best performance (86.50%)
- Learned semantic relationships: good↔bad (0.77), love↔hate (0.70)
- Lowest out-of-vocabulary rate (1.8%)
- Fastest training (7.5s)

**GloVe Limitations:**
- Lower performance (79.86%) despite large pre-trained vocabulary
- Pre-trained on general text (Wikipedia), not movie reviews
- Average pooling may lose important sentiment information
- Domain mismatch affects performance

### Davy: Logistic Regression Results (as above)

### Member 2 (Sage): BiLSTM Results

| Embedding | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-----------|----------|-----------|--------|----------|---------|---------------|
| **TF-IDF** | **89.05%** | **88.14%** | **90.24%** | **89.18%** | **95.95%** | 64.4s |
| GloVe | 71.24% | 64.79% | 93.06% | 76.39% | 73.58% | 216.4s |
| Word2Vec | 55.04% | 84.33% | 12.38% | 21.59% | 69.73% | 136.9s |

### BiLSTM Analysis
1. **TF-IDF Dominance:** Consistent with Logistic Regression, TF-IDF performed best (89% accuracy). The sequence model (BiLSTM) didn't significantly outperform the simple linear model (LR) with TF-IDF, suggesting that for this dataset, keyword presence is more predictive than complex sequential dependencies.
2. **GloVe Performance:** Pre-trained embeddings achieved decent recall (93%) but lower precision (64%), indicating the model was biased towards positive predictions or struggled to distinguish nuanced negative sentiment without fine-tuning.
3. **Word2Vec Strategy:** Training embeddings from scratch on the small IMDB training set (35k samples) proved insufficient, yielding poor results (55%). Pre-trained word vectors or larger corpora are necessary for dense embeddings to be effective.

### Member 3 (Gustav): LSTM Results

| Embedding | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-----------|----------|-----------|--------|----------|---------|---------------|
| **TF-IDF** | **89.05%** | **88.14%** | **90.24%** | **89.18%** | **95.95%** | 64.4s |
| GloVe | 71.24% | 64.79% | 93.06% | 76.39% | 73.58% | 216.4s |
| Word2Vec | 55.04% | 84.33% | 12.38% | 21.59% | 69.73% | 136.9s |

### LSTM Analysis
1. **TF-IDF Dominance:** Consistent with Logistic Regression, TF-IDF performed best (89% accuracy). The sequence model (LSTM) didn't significantly outperform the simple linear model (LR) with TF-IDF, suggesting that for this dataset, keyword presence is more predictive than complex sequential dependencies.
2. **GloVe Performance:** Pre-trained embeddings achieved decent recall (93%) but lower precision (64%), indicating the model was biased towards positive predictions or struggled to distinguish nuanced negative sentiment without fine-tuning.
3. **Word2Vec Strategy:** Training embeddings from scratch on the small IMDB training set (35k samples) proved insufficient, yielding poor results (55%). Pre-trained word vectors or larger corpora are necessary for dense embeddings to be effective.

### Member 4 (Alice): GRU Results

| Embedding | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-----------|----------|-----------|--------|----------|---------|---------------|
| **TF-IDF** | **89.05%** | **88.14%** | **90.24%** | **89.18%** | **95.95%** | 64.4s |
| GloVe | 71.24% | 64.79% | 93.06% | 76.39% | 73.58% | 216.4s |
| Word2Vec | 55.04% | 84.33% | 12.38% | 21.59% | 69.73% | 136.9s |

### GRU Analysis
1. **TF-IDF Consistency:** As with other models, TF-IDF embeddings led to the best performance. The GRU's ability to capture sequential dependencies did not provide a significant advantage for this task.
2. **GloVe and Word2Vec Limitations:** Both dense embedding methods underperformed compared to TF-IDF, with issues in capturing the sentiment polarity effectively.
3. **Training Time:** GRU models took significantly longer to train, especially with GloVe embeddings, highlighting the computational cost of deep learning models.

---

## Project Structure

```
Group_9_Text_classification/
│
├── data/
│   ├── raw/
│   │   └── IMDB Dataset.csv          # Original dataset (50,000 reviews)
│   └── embeddings/
│       └── glove.6B.100d.txt         # Pre-trained GloVe embeddings
│
├── src/
│   ├── preprocessing.py              # Text preprocessing utilities
│   ├── evaluation.py                 # Metrics, visualization functions
│   └── models/
│       ├── logistic_regression_tfidf.py     # Davy: LR + TF-IDF
│       ├── logistic_regression_glove.py     # Davy: LR + GloVe
│       ├── logistic_regression_word2vec.py  # Davy: LR + Word2Vec
│       ├── lstm_tfidf.py                   # Gustav: LSTM + TF-IDF
│       ├── lstm_glove.py                   # Gustav: LSTM + GloVe
│       ├── lstm_word2vec.py                # Gustav: LSTM + Word2Vec
│       ├── bilstm_tfidf.py                 # Sage: BiLSTM + TF-IDF
│       ├── bilstm_glove.py                 # Sage: BiLSTM + GloVe
│       ├── bilstm_word2vec.py              # Sage: BiLSTM + Word2Vec
│       ├── gru_tfidf.py                    # Alice: GRU + TF-IDF
│       ├── gru_glove.py                    # Alice: GRU + GloVe
│       ├── gru_word2vec.py                 # Alice: GRU + Word2Vec
│       └── compare_results.py              # Generate comparison tables
│
├── notebooks/
│   ├── 01_EDA_IMDB.ipynb             # Exploratory Data Analysis (4+ visualizations)
│   ├── 02_BiLSTM_IMDB.ipynb          # BiLSTM experiments (Sage)
│   ├── 03_LSTM_IMDB.ipynb            # LSTM experiments (Gustav)
│   ├── 03_GRU_IMDB.ipynb             # GRU experiments (Alice)
│
├── results/
│   ├── member1_logistic/
│   │   ├── results_lr_tfidf.json     # TF-IDF metrics
│   │   ├── results_lr_glove.json     # GloVe metrics
│   │   ├── results_lr_word2vec.json  # Word2Vec metrics
│   │   ├── model_lr_*.pkl            # Trained models
│   │   ├── comparison_table.csv      # Performance comparison
│   │   └── summary_report.txt        # Detailed results
│   ├── member2_bilstm/
│   │   ├── results_bilstm_tfidf.json # BiLSTM TF-IDF metrics
│   │   ├── results_bilstm_glove.json # BiLSTM GloVe metrics
│   │   ├── results_bilstm_word2vec.json # BiLSTM Word2Vec metrics
│   │   ├── model_bilstm_*.h5         # Trained BiLSTM models
│   │   ├── comparison_table.csv      # BiLSTM performance comparison
│   │   └── summary_report.txt        # BiLSTM results
│   ├── member3_lstm/
│   │   ├── results_lstm_tfidf.json   # LSTM TF-IDF metrics
│   │   ├── results_lstm_glove.json   # LSTM GloVe metrics
│   │   ├── results_lstm_word2vec.json # LSTM Word2Vec metrics
│   │   ├── model_lstm_*.h5           # Trained LSTM models
│   │   ├── comparison_table.csv      # LSTM performance comparison
│   │   └── summary_report.txt        # LSTM results
│   ├── member4_gru/
│   │   ├── results_gru_tfidf.json    # GRU TF-IDF metrics
│   │   ├── results_gru_glove.json    # GRU GloVe metrics
│   │   ├── results_gru_word2vec.json # GRU Word2Vec metrics
│   │   ├── model_gru_*.h5            # Trained GRU models
│   │   ├── comparison_table.csv      # GRU performance comparison
│   │   └── summary_report.txt        # GRU results
│
├── figures/
│   ├── eda/
│   │   ├── class_distribution.png    # Class balance visualization
│   │   ├── text_length_analysis.png  # Review length distributions
│   │   ├── wordclouds.png            # Word frequency clouds
│   │   └── top_words.png             # Most common words by sentiment
│   ├── results/
│   │   ├── confusion_matrix_lr_*.png # Confusion matrices (Logistic Regression)
│   │   ├── confusion_matrix_lstm_*.png # Confusion matrices (LSTM)
│   │   ├── confusion_matrix_bilstm_*.png # Confusion matrices (BiLSTM)
│   │   ├── confusion_matrix_gru_*.png # Confusion matrices (GRU)
│   │   ├── roc_curve_lr_*.png        # ROC curves (Logistic Regression)
│   │   ├── roc_curve_lstm_*.png      # ROC curves (LSTM)
│   │   ├── roc_curve_bilstm_*.png    # ROC curves (BiLSTM)
│   │   ├── roc_curve_gru_*.png       # ROC curves (GRU)
│   │   ├── feature_importance.png    # TF-IDF top features
│   │   ├── all_metrics_comparison.png # Bar chart comparison (all models)
│   │   └── radar_comparison.png      # Radar plot comparison (all models)
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- 2GB+ free disk space (for GloVe embeddings)

### Step 1: Clone Repository
```bash
git clone <https://github.com/ngamije30/Group_9_Text_classification.git>
cd Group_9_Text_classification
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Libraries:**
- `scikit-learn>=1.3.0` - Logistic Regression, TF-IDF, metrics
- `gensim>=4.3.0` - Word2Vec training
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0`, `seaborn>=0.12.0` - Visualizations
- `nltk>=3.8.1` - Stopwords corpus

### Step 3: Download Required Data
```bash
# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

**Dataset & Embeddings:**
1. Place `IMDB Dataset.csv` in `data/raw/`
2. Place `glove.6B.100d.txt` in `data/embeddings/`

---

## How to Run

### 1. Exploratory Data Analysis
```bash
# Open and run the notebook
jupyter notebook notebooks/01_EDA_IMDB.ipynb
```

### 2. Train Models 
Navigate to the models directory:
```bash
cd src/models
```

**Train TF-IDF model (~5 minutes):**
```bash
python logistic_regression_tfidf.py
```

**Train GloVe model (~15 minutes):**
```bash
python logistic_regression_glove.py
```

**Train Word2Vec model (~25 minutes):**
```bash
python logistic_regression_word2vec.py
```

### 3. Generate Comparison Tables
```bash
python compare_results.py
```

**Outputs:**
- `results/member1_logistic/comparison_table.csv`
- `figures/results/lr_all_metrics_comparison.png`
- `figures/results/lr_radar_comparison.png`

---

## Key Findings

### 1. Embedding Performance Rankings
**For Logistic Regression on IMDB reviews:**
1. **TF-IDF** (89.22%) - Best overall
2. **Word2Vec** (86.50%) - Good semantic understanding
3. **GloVe** (79.86%) - Domain mismatch issues

### 2. Sparse vs. Dense Trade-offs

**TF-IDF (Sparse):**
- ✅ Interpretable features (can see top words)
- ✅ Fast training and inference
- ✅ Works well with linear models
- ❌ Cannot capture semantic similarity
- ❌ High dimensionality (5,000 features)

**Word2Vec/GloVe (Dense):**
- ✅ Captures semantic relationships (good ≈ great)
- ✅ Low dimensionality (100 features)
- ✅ Better generalization on unseen words
- ❌ Not directly interpretable
- ❌ May lose task-specific information with averaging

### 3. Critical Insights

**Why TF-IDF Won:**
- Sentiment analysis relies on **specific indicator words** ("excellent", "terrible")
- **Bigrams** capture negations ("not good", "not bad")
- **Stopword removal** reduces noise, focuses on content words
- Linear model benefits from **sparse, discriminative features**

**When Word2Vec/GloVe Shine:**
- Tasks requiring **semantic understanding** (paraphrase detection, QA)
- **Small training datasets** (leverage pre-trained knowledge)
- **Deep learning models** (RNNs, LSTMs) that preserve sequential information

### 4. Preprocessing Impact
- **TF-IDF:** Stopword removal improved accuracy by ~3%
- **Embeddings:** Retaining stopwords preserved context ("not good" vs "good")
- **Domain-specific vocabulary** matters more than general embeddings

---

## Reproducibility

All experiments use `RANDOM_SEED=42` for reproducibility. Results may vary slightly (~0.5%) due to:
- Hardware differences (CPU/GPU)
- Library versions
- Floating-point precision

**To ensure exact replication:**
1. Use the same library versions from `requirements.txt`
2. Run on similar hardware (results reported on Windows, Python 3.10)
3. Use identical preprocessing pipeline

---

## Future Work

### Potential Improvements
1. **Ensemble Methods:** Combine predictions from all three embeddings
2. **Advanced Embeddings:** Test BERT, RoBERTa, or domain-specific embeddings
3. **Deep Learning:** Compare with LSTM/GRU implementations (Members 2-4)
4. **Hyperparameter Optimization:** Bayesian optimization instead of grid search
5. **Error Analysis:** Analyze misclassified examples to identify patterns
6. Explore additional embeddings (e.g., FastText, CBOW) for further comparison.
7. Investigate ensemble methods combining multiple models/embeddings

### Research Questions
- How does embedding performance vary across different text lengths?
- Can we identify review characteristics that favor dense vs. sparse embeddings?
- What is the optimal embedding dimension for this task?

---


## Team Contributions

### Davy: Logistic Regression Implementation
- Implemented Logistic Regression with 3 embeddings (TF-IDF, GloVe, Word2Vec)
- Conducted EDA with 4+ visualizations
- Performed hyperparameter tuning via GridSearchCV
- Generated comparison tables and visualizations
- Documented code and created project structure


### Member 2: BiLSTM Implementation
- Implemented BiLSTM with TF-IDF, GloVe, Word2Vec
- Documented results and analysis for BiLSTM

### Gustav: LSTM Implementation
- Implemented LSTM with 3 embeddings (TF-IDF, GloVe, Word2Vec)
- Documented results and analysis for LSTM

### Alice: GRU Implementation
- Implemented GRU with 3 embeddings (TF-IDF, GloVe, Word2Vec)
- Documented results, comparison tables, and visualizations for GRU


