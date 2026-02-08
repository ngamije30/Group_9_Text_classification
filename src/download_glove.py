"""
Script to download GloVe embeddings using Gensim.
This is more reliable than direct downloads from Stanford servers.
"""

import gensim.downloader as api
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
GLOVE_PATH = EMBEDDINGS_DIR / "glove.6B.100d.txt"

def download_glove():
    print(f"Downloading GloVe embeddings to {GLOVE_PATH}...")
    
    # Create directory if needed
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download (this might take a few minutes)
    # glove-wiki-gigaword-100 is equivalent to glove.6B.100d (400k vocab)
    try:
        print("Loading model from Gensim API (glove-wiki-gigaword-100)...")
        model = api.load("glove-wiki-gigaword-100")
        
        print(f"Saving to {GLOVE_PATH}...")
        # Save in Word2Vec format (text)
        model.save_word2vec_format(str(GLOVE_PATH), binary=False)
        
        print("✓ GloVe embeddings downloaded and saved successfully.")
        
    except Exception as e:
        print(f"❌ Error downloading GloVe: {e}")

if __name__ == "__main__":
    download_glove()
