"""
Feature engineering module for the HOMO-LAT project.
Contains functions for text vectorization and feature extraction.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Callable
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import gensim
from gensim.models import Word2Vec, FastText
import nltk
from nltk.tokenize import word_tokenize
from modules.config import MAX_FEATURES, EMBEDDING_DIM

# Ensure proper tokenization for Spanish
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a column from a DataFrame
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert sparse matrix to dense numpy array
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()

def create_bow_vectorizer(max_features: int = MAX_FEATURES, ngram_range: Tuple[int, int] = (1, 1)) -> CountVectorizer:
    """
    Create a Bag-of-Words vectorizer.
    
    Args:
        max_features: Maximum number of features
        ngram_range: Range of n-grams to consider
        
    Returns:
        CountVectorizer instance
    """
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents='unicode',
        token_pattern=r'\b\w+\b'  # Match words with at least 1 character
    )

def create_tfidf_vectorizer(max_features: int = MAX_FEATURES, ngram_range: Tuple[int, int] = (1, 1)) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer.
    
    Args:
        max_features: Maximum number of features
        ngram_range: Range of n-grams to consider
        
    Returns:
        TfidfVectorizer instance
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        strip_accents='unicode',
        token_pattern=r'\b\w+\b',  # Match words with at least 1 character
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )

def create_word2vec_model(sentences: List[List[str]], vector_size: int = EMBEDDING_DIM, window: int = 5, 
                          min_count: int = 1, workers: int = 4, sg: int = 1) -> Word2Vec:
    """
    Train a Word2Vec model on the given sentences.
    
    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Ignores words with frequency less than this
        workers: Number of threads to train the model
        sg: Training algorithm: 1 for skip-gram; 0 for CBOW
        
    Returns:
        Trained Word2Vec model
    """
    return Word2Vec(sentences=sentences, vector_size=vector_size, window=window, 
                    min_count=min_count, workers=workers, sg=sg)

def create_fasttext_model(sentences: List[List[str]], vector_size: int = EMBEDDING_DIM, window: int = 5, 
                          min_count: int = 1, workers: int = 4, sg: int = 1) -> FastText:
    """
    Train a FastText model on the given sentences.
    
    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Ignores words with frequency less than this
        workers: Number of threads to train the model
        sg: Training algorithm: 1 for skip-gram; 0 for CBOW
        
    Returns:
        Trained FastText model
    """
    return FastText(sentences=sentences, vector_size=vector_size, window=window, 
                    min_count=min_count, workers=workers, sg=sg)

def load_pretrained_word_vectors(path: str) -> Dict[str, np.ndarray]:
    """
    Load pretrained word vectors from file (e.g., GloVe or Word2Vec pre-trained embeddings).
    
    Args:
        path: Path to the embeddings file
        
    Returns:
        Dictionary mapping words to their embedding vectors
    """
    embeddings_index = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeffs
            
    print(f"Loaded {len(embeddings_index)} word vectors from {path}")
    return embeddings_index

class MeanEmbeddingVectorizer:
    """
    Transforms sentences to document vectors by averaging word vectors.
    """
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors
        # Nếu word_vectors có thuộc tính 'vectors' (như trong Gensim KeyedVectors)
        if hasattr(word_vectors, 'vectors'):
            self.vector_size = word_vectors.vectors.shape[1]
        else:
            # Nếu word_vectors là một dictionary thông thường
            self.vector_size = next(iter(word_vectors.values())).shape[0] if word_vectors else 0

    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Average word vectors in each document.
        
        Args:
            X: List of texts
            
        Returns:
            Document vectors
        """
        doc_vectors = np.zeros((len(X), self.vector_size))
        
        for i, text in enumerate(X):
            if not isinstance(text, str):
                continue
                
            tokens = word_tokenize(text.lower(), language='spanish')
            vectors = []
            
            for token in tokens:
                if token in self.word_vectors:
                    vectors.append(self.word_vectors[token])
            
            if vectors:
                doc_vectors[i] = np.mean(vectors, axis=0)
                
        return doc_vectors

class TfidfEmbeddingVectorizer:
    """
    Transforms sentences to document vectors using TF-IDF weighted averaging of word vectors.
    """
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors
        self.vector_size = next(iter(word_vectors.values())).shape[0] if word_vectors else 0
        self.word_tfidf_map = {}
        self.tfidf = TfidfVectorizer(strip_accents='unicode')
    
    def fit(self, X, y=None):
        self.tfidf.fit(X)
        feature_names = self.tfidf.get_feature_names_out()
        idf_values = self.tfidf._tfidf.idf_
        
        # Map words to their IDF values
        for word, idf in zip(feature_names, idf_values):
            self.word_tfidf_map[word] = idf
            
        return self
    
    def transform(self, X):
        """
        Create document vectors by averaging word vectors weighted by TF-IDF.
        
        Args:
            X: List of texts
            
        Returns:
            Document vectors
        """
        doc_vectors = np.zeros((len(X), self.vector_size))
        
        for i, text in enumerate(X):
            if not isinstance(text, str):
                continue
                
            tokens = word_tokenize(text.lower(), language='spanish')
            word_counts = {}
            
            for token in tokens:
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1
            
            vectors = []
            weights = []
            
            for token, count in word_counts.items():
                if token in self.word_vectors and token in self.word_tfidf_map:
                    tf = count / len(tokens)
                    tfidf = tf * self.word_tfidf_map[token]
                    vectors.append(self.word_vectors[token] * tfidf)
                    weights.append(tfidf)
            
            if vectors:
                doc_vectors[i] = np.sum(vectors, axis=0) / (np.sum(weights) or 1)
                
        return doc_vectors

def extract_ngrams(texts: List[str], ngram_range: Tuple[int, int] = (2, 3)) -> List[Dict[str, int]]:
    """
    Extract n-gram features from texts.
    
    Args:
        texts: List of texts
        ngram_range: Range of n-grams to consider
        
    Returns:
        List of dictionaries mapping n-grams to their counts
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    result = []
    for i in range(X.shape[0]):
        ngram_counts = {}
        for j, count in zip(X[i].indices, X[i].data):
            ngram_counts[feature_names[j]] = count
        result.append(ngram_counts)
    
    return result

def extract_pos_tags(texts: List[str], nlp) -> List[Dict[str, int]]:
    """
    Extract part-of-speech tags as features.
    
    Args:
        texts: List of texts
        nlp: SpaCy language model
        
    Returns:
        List of dictionaries mapping POS tags to their counts
    """
    pos_features = []
    
    for text in texts:
        if not isinstance(text, str) or pd.isna(text):
            pos_features.append({})
            continue
            
        doc = nlp(text)
        pos_counts = {}
        
        for token in doc:
            pos = token.pos_
            if pos in pos_counts:
                pos_counts[pos] += 1
            else:
                pos_counts[pos] = 1
        
        pos_features.append(pos_counts)
    
    return pos_features

def extract_lexical_features(texts: List[str]) -> List[Dict[str, float]]:
    """
    Extract lexical features from texts.
    
    Args:
        texts: List of texts
        
    Returns:
        List of dictionaries with lexical features
    """
    lexical_features = []
    
    for text in texts:
        if not isinstance(text, str) or pd.isna(text):
            lexical_features.append({})
            continue
            
        # Tokenize
        tokens = word_tokenize(text.lower(), language='spanish')
        
        if not tokens:
            lexical_features.append({})
            continue
        
        # Calculate features
        features = {
            'text_length': len(text),
            'word_count': len(tokens),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'unique_word_ratio': len(set(tokens)) / len(tokens) if tokens else 0,
        }
        
        lexical_features.append(features)
    
    return lexical_features

def create_feature_union(preprocessor: Optional[Callable] = None) -> Pipeline:
    """
    Create a feature union pipeline with multiple feature extractors.
    
    Args:
        preprocessor: Text preprocessing function
        
    Returns:
        Pipeline with feature union
    """
    bow_vectorizer = create_bow_vectorizer()
    tfidf_vectorizer = create_tfidf_vectorizer()
    
    transformers = [
        ('bow', Pipeline([
            ('selector', TextSelector('post content')),
            ('vectorizer', bow_vectorizer)
        ])),
        ('tfidf', Pipeline([
            ('selector', TextSelector('post content')),
            ('vectorizer', tfidf_vectorizer)
        ]))
    ]
    
    if preprocessor:
        for name, pipeline in transformers:
            pipeline.steps.insert(1, ('preprocessor', preprocessor))
    
    return Pipeline([
        ('features', FeatureUnion(transformer_list=transformers)),
        ('to_dense', DenseTransformer())  # Convert to dense if using with non-sparse-compatible models
    ])