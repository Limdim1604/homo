"""
Data processing module for the HOMO-LAT project.
Contains functions for loading and preprocessing text data.
"""
import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import spacy
import emoji
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from modules.config import TRAIN_FILE, DEV_FILE, RANDOM_STATE, TEST_SIZE

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Initialize Spanish language tools
spanish_stopwords = set(stopwords.words('spanish'))
stemmer = SnowballStemmer('spanish')
lemmatizer = WordNetLemmatizer()

# Try to load Spanish spaCy model
try:
    nlp = spacy.load("es_core_news_sm")
except:
    print("Spanish spaCy model not found. Installing...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

def load_data(train_file=TRAIN_FILE, dev_file=DEV_FILE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and dev datasets.
    
    Returns:
        Tuple of train and dev dataframes
    """
    train_df = pd.read_csv(train_file)
    dev_df = pd.read_csv(dev_file)
    
    print(f"Loaded training set: {train_df.shape}")
    print(f"Loaded development set: {dev_df.shape}")
    
    return train_df, dev_df

def split_data(df: pd.DataFrame, test_size=TEST_SIZE, random_state=RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train and validation sets.
    
    Args:
        df: Input dataframe
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of train and validation dataframes
    """
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    
    print(f"Training set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    
    return train_df, val_df

def encode_labels(train_labels: pd.Series, val_labels: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode categorical labels to numerical values.
    
    Args:
        train_labels: Training set labels
        val_labels: Validation set labels (optional)
        
    Returns:
        Tuple of encoded train labels, encoded validation labels (if provided), and the encoder
    """
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    
    if val_labels is not None:
        y_val = encoder.transform(val_labels)
        return y_train, y_val, encoder
    
    return y_train, None, encoder

# Text Preprocessing Functions

def basic_preprocessing(text: str) -> str:
    """
    Basic text preprocessing: lowercase, remove special characters and extra spaces.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters and punctuation (keep letters, numbers, spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_accents(text: str) -> str:
    """
    Normalize accented characters.
    
    Args:
        text: Input text
        
    Returns:
        Text with accents normalized
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def remove_stopwords(text: str) -> str:
    """
    Remove Spanish stopwords from text.
    
    Args:
        text: Input text
        
    Returns:
        Text without stopwords
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    tokens = word_tokenize(text, language='spanish')
    filtered_tokens = [word for word in tokens if word.lower() not in spanish_stopwords]
    return ' '.join(filtered_tokens)

def stemming(text: str) -> str:
    """
    Apply stemming to text.
    
    Args:
        text: Input text
        
    Returns:
        Stemmed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    tokens = word_tokenize(text, language='spanish')
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def lemmatization_spacy(text: str) -> str:
    """
    Apply lemmatization to text using spaCy.
    
    Args:
        text: Input text
        
    Returns:
        Lemmatized text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)

def extract_emoji_features(text: str) -> Dict[str, int]:
    """
    Extract emoji features from text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with emoji counts
    """
    if pd.isna(text) or not isinstance(text, str):
        return {}
    
    emoji_dict = {}
    for char in text:
        if char in emoji.EMOJI_DATA:
            if char in emoji_dict:
                emoji_dict[char] += 1
            else:
                emoji_dict[char] = 1
                
    return emoji_dict

def preprocess_pipeline(text: str, 
                       lowercase: bool = True,
                       remove_special_chars: bool = True, 
                       normalize_accent: bool = False,
                       remove_stop: bool = False,
                       stem: bool = False,
                       lemmatize: bool = False) -> str:
    """
    Complete text preprocessing pipeline with configurable steps.
    
    Args:
        text: Input text
        lowercase: Whether to convert text to lowercase
        remove_special_chars: Whether to remove special characters
        normalize_accent: Whether to normalize accented characters
        remove_stop: Whether to remove stopwords
        stem: Whether to apply stemming
        lemmatize: Whether to apply lemmatization
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    processed_text = text
    
    # Lowercase
    if lowercase:
        processed_text = processed_text.lower()
    
    # Remove URLs
    processed_text = re.sub(r'https?://\S+|www\.\S+', '', processed_text)
    
    # Remove special characters
    if remove_special_chars:
        processed_text = re.sub(r'[^\w\s]', '', processed_text)
    
    # Remove extra whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    # Normalize accents
    if normalize_accent:
        processed_text = normalize_accents(processed_text)
    
    # Tokenize
    tokens = word_tokenize(processed_text, language='spanish')
    
    # Remove stopwords
    if remove_stop:
        tokens = [word for word in tokens if word.lower() not in spanish_stopwords]
    
    # Apply stemming or lemmatization
    if stem:
        tokens = [stemmer.stem(word) for word in tokens]
    elif lemmatize:
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]
    
    return " ".join(tokens)