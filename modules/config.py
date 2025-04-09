"""
Configuration settings for the HOMO-LAT project.
"""
import os
from pathlib import Path

# Paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = ROOT_DIR
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data files
TRAIN_FILE = DATA_DIR / "train_homolat25.csv"
DEV_FILE = DATA_DIR / "dev.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Text preprocessing parameters
MAX_FEATURES = 10000  # For vectorizers like CountVectorizer and TfidfVectorizer
MAX_LENGTH = 300  # For sequences in deep learning models
EMBEDDING_DIM = 100  # For word embeddings