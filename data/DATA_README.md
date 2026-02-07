# Data Directory

This folder contains the datasets and embeddings used in the project.

## Required Files

### 1. IMDB Dataset (data/raw/)
- **File:** `IMDB Dataset.csv`
- **Size:** ~25 MB
- **Source:** [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Description:** 50,000 movie reviews labeled as positive/negative

**Download Instructions:**
1. Visit the Kaggle link above
2. Download `IMDB Dataset.csv`
3. Place in `data/raw/` folder

### 2. GloVe Embeddings (data/embeddings/)
- **File:** `glove.6B.100d.txt`
- **Size:** ~330 MB
- **Source:** [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- **Description:** Pre-trained word vectors

**Download Instructions:**
1. Visit https://nlp.stanford.edu/projects/glove/
2. Download `glove.6B.zip` (862 MB)
3. Extract `glove.6B.100d.txt`
4. Place in `data/embeddings/` folder

**Note:** These files are too large for git and are excluded via `.gitignore`.
