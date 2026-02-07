# Comparative Analysis of Text Classification with Multiple Embeddings

**Group 9 - Text Classification Project**  
**Course:** Machine Learning / Natural Language Processing  
**Date:** February 2026


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

| Member | Model Architecture | Embeddings Evaluated |
|--------|-------------------|---------------------|
| **Member 1** | Logistic Regression | TF-IDF, GloVe, Word2Vec Skip-gram |
| Member 2 | LSTM | TF-IDF, GloVe, Word2Vec |
| Member 3 | GRU | TF-IDF, GloVe, Word2Vec |
| Member 4 | RNN | TF-IDF, GloVe, Word2Vec |

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

### Member 1: Logistic Regression

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
│       ├── logistic_regression_tfidf.py     # Member 1: LR + TF-IDF
│       ├── logistic_regression_glove.py     # Member 1: LR + GloVe
│       ├── logistic_regression_word2vec.py  # Member 1: LR + Word2Vec
│       └── compare_results.py               # Generate comparison tables
│
├── notebooks/
│   └── 01_EDA_IMDB.ipynb             # Exploratory Data Analysis (4+ visualizations)
│
├── results/
│   └── member1_logistic/
│       ├── results_lr_tfidf.json     # TF-IDF metrics
│       ├── results_lr_glove.json     # GloVe metrics
│       ├── results_lr_word2vec.json  # Word2Vec metrics
│       ├── model_lr_*.pkl            # Trained models
│       ├── comparison_table.csv      # Performance comparison
│       └── summary_report.txt        # Detailed results
│
├── figures/
│   ├── eda/
│   │   ├── class_distribution.png    # Class balance visualization
│   │   ├── text_length_analysis.png  # Review length distributions
│   │   ├── wordclouds.png            # Word frequency clouds
│   │   └── top_words.png             # Most common words by sentiment
│   └── results/
│       ├── confusion_matrix_lr_*.png # Confusion matrices (3 models)
│       ├── roc_curve_lr_*.png        # ROC curves (3 models)
│       ├── feature_importance.png    # TF-IDF top features
│       ├── lr_all_metrics_comparison.png  # Bar chart comparison
│       └── lr_radar_comparison.png   # Radar plot comparison
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
git clone <your-repo-url>
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

### 2. Train Models (Member 1)
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

### Research Questions
- How does embedding performance vary across different text lengths?
- Can we identify review characteristics that favor dense vs. sparse embeddings?
- What is the optimal embedding dimension for this task?

---

## References

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics*, 142-150.

2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.

3. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of EMNLP*, 1532-1543.

4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

5. Řehůřek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. *Proceedings of LREC Workshop on New Challenges for NLP Frameworks*.

---

## Team Contributions

### Member 1: Logistic Regression Implementation
- Implemented Logistic Regression with 3 embeddings (TF-IDF, GloVe, Word2Vec)
- Conducted EDA with 4+ visualizations
- Performed hyperparameter tuning via GridSearchCV
- Generated comparison tables and visualizations
- Documented code and created project structure

**Code Locations:**
- `src/models/logistic_regression_*.py`
- `notebooks/01_EDA_IMDB.ipynb`
- `src/preprocessing.py`, `src/evaluation.py`

### Members 2-4: [To be completed by team members]
- Member 2: LSTM implementation
- Member 3: GRU implementation  
- Member 4: RNN implementation

---

## License

This project is created for academic purposes as part of a university course assignment.

---

## Contact

**Group 9**  
For questions or collaboration:
- [Add team members' contact information]

---

## Acknowledgments

- Dataset: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Pre-trained embeddings: [Stanford GloVe Project](https://nlp.stanford.edu/projects/glove/)
- Course instructors and teaching assistants

---

**Last Updated:** February 7, 2026
