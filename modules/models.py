"""
Main script for the HOMO-LAT sentiment analysis project.
This script provides a complete pipeline for training and evaluating various models.
"""
import os
import time
import numpy as np
import pandas as pd
import json
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any, Optional

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.base import BaseEstimator

# Import project modules
from modules.config import TRAIN_FILE, DEV_FILE, MODELS_DIR, RESULTS_DIR, RANDOM_STATE
from modules.data_processing import (
    load_data, split_data, encode_labels,
    basic_preprocessing, normalize_accents, remove_stopwords,
    stemming, lemmatization_spacy, preprocess_pipeline
)
from modules.feature_engineering import (
    create_bow_vectorizer, create_tfidf_vectorizer,
    create_word2vec_model, create_fasttext_model,
    MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer,
    extract_ngrams, extract_pos_tags, extract_lexical_features
)

# Try to import deep learning modules
try:
    from modules.deep_learning_models import (
        tokenize_and_pad, create_embedding_matrix,
        create_lstm_model, create_bilstm_model, create_cnn_model, create_cnn_lstm_model,
        prepare_data_for_deep_learning, train_deep_learning_model, evaluate_deep_learning_model
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Deep learning models will not be used.")
    TENSORFLOW_AVAILABLE = False

def preprocess_text(texts: List[str], preprocessing_pipeline: str = 'basic') -> List[str]:
    """
    Apply different text preprocessing techniques based on the selected pipeline.
    
    Args:
        texts: List of texts to preprocess
        preprocessing_pipeline: Name of the preprocessing pipeline to use
        
    Returns:
        List of preprocessed texts
    """
    if preprocessing_pipeline == 'none':
        return texts
    
    if preprocessing_pipeline == 'basic':
        return [basic_preprocessing(text) for text in texts]
    
    if preprocessing_pipeline == 'normalize_accents':
        return [normalize_accents(basic_preprocessing(text)) for text in texts]
    
    if preprocessing_pipeline == 'remove_stopwords':
        return [remove_stopwords(basic_preprocessing(text)) for text in texts]
    
    if preprocessing_pipeline == 'stemming':
        return [stemming(basic_preprocessing(text)) for text in texts]
    
    if preprocessing_pipeline == 'lemmatization':
        return [lemmatization_spacy(basic_preprocessing(text)) for text in texts]
    
    if preprocessing_pipeline == 'full':
        return [preprocess_pipeline(
            text, lowercase=True, remove_special_chars=True,
            normalize_accent=True, remove_stop=True, lemmatize=True
        ) for text in texts]
    
    print(f"Unknown preprocessing pipeline: {preprocessing_pipeline}. Using basic preprocessing.")
    return [basic_preprocessing(text) for text in texts]

def vectorize_text(texts: List[str], 
                  vectorization_method: str = 'tfidf',
                  ngram_range: Tuple[int, int] = (1, 1),
                  max_features: int = 10000,
                  embedding_model=None) -> Tuple[np.ndarray, Any]:
    """
    Convert texts to numerical features using different vectorization methods.
    
    Args:
        texts: List of texts to vectorize
        vectorization_method: Method to use ('bow', 'tfidf', 'word2vec', 'fasttext')
        ngram_range: Range of n-grams for BoW and TF-IDF
        max_features: Maximum number of features for BoW and TF-IDF
        embedding_model: Pre-trained embedding model (for word2vec and fasttext)
        
    Returns:
        Tuple of (feature matrix, vectorizer/model)
    """
    if vectorization_method == 'bow':
        vectorizer = create_bow_vectorizer(max_features=max_features, ngram_range=ngram_range)
        X = vectorizer.fit_transform(texts)
        return X, vectorizer
    
    if vectorization_method == 'tfidf':
        vectorizer = create_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range)
        X = vectorizer.fit_transform(texts)
        return X, vectorizer
    
    if vectorization_method == 'word2vec' and embedding_model is not None:
        vectorizer = MeanEmbeddingVectorizer(embedding_model.wv)
        X = vectorizer.transform(texts)
        return X, vectorizer
    
    if vectorization_method == 'fasttext' and embedding_model is not None:
        vectorizer = MeanEmbeddingVectorizer(embedding_model.wv)
        X = vectorizer.transform(texts)
        return X, vectorizer
    
    # Default to TF-IDF if method not recognized or embedding model not provided
    print(f"Using TF-IDF vectorization")
    vectorizer = create_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_traditional_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                         class_weight: str = None, perform_hyperparameter_tuning: bool = False) -> Any:
    """ 
    Train a traditional machine learning model.
    """
    model = None
    
    # Print model parameters before training
    print(f"\n=== Model Parameters for {model_name.upper()} ===")
    
    if model_name == 'naive_bayes':
        model = create_naive_bayes_model(alpha=1.0)
        print(f"Initial parameters:\n{model.get_params()}")
        
        if perform_hyperparameter_tuning:
            param_grid = {
                'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
                'fit_prior': [True, False]
            }
            print(f"\nTuning parameters:\n{param_grid}")
            grid_search = perform_grid_search(model, param_grid, X_train, y_train)
            model = grid_search.best_estimator_
            print(f"\nBest parameters:\n{model.get_params()}")
    
    elif model_name == 'logistic_regression':
        model = create_logistic_regression_model(C=1.0, max_iter=1000, class_weight=class_weight)
        print(f"Initial parameters:\n{model.get_params()}")
        
        if perform_hyperparameter_tuning:
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l1', 'l2'],
                'max_iter': [1000]
            }
            print(f"\nTuning parameters:\n{param_grid}")
            grid_search = perform_grid_search(model, param_grid, X_train, y_train)
            model = grid_search.best_estimator_
            print(f"\nBest parameters:\n{model.get_params()}")
    
    elif model_name == 'svm':
        model = create_svm_model(kernel='rbf', C=1.0, class_weight=class_weight)
        print(f"Initial parameters:\n{model.get_params()}")
        
        if perform_hyperparameter_tuning:
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            print(f"\nTuning parameters:\n{param_grid}")
            grid_search = perform_grid_search(model, param_grid, X_train, y_train)
            model = grid_search.best_estimator_
            print(f"\nBest parameters:\n{model.get_params()}")
    
    elif model_name == 'random_forest':
        model = create_random_forest_model(n_estimators=100, class_weight=class_weight)
        print(f"Initial parameters:\n{model.get_params()}")
        
        if perform_hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            print(f"\nTuning parameters:\n{param_grid}")
            grid_search = perform_grid_search(model, param_grid, X_train, y_train)
            model = grid_search.best_estimator_
            print(f"\nBest parameters:\n{model.get_params()}")
    
    elif model_name == 'gradient_boosting':
        model = create_gradient_boosting_model()
        print(f"Initial parameters:\n{model.get_params()}")
        
        if perform_hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            }
            print(f"\nTuning parameters:\n{param_grid}")
            grid_search = perform_grid_search(model, param_grid, X_train, y_train)
            model = grid_search.best_estimator_
            print(f"\nBest parameters:\n{model.get_params()}")
    
    elif model_name == 'knn':
        model = create_knn_model()
        print(f"Initial parameters:\n{model.get_params()}")
        
        if perform_hyperparameter_tuning:
            param_grid = {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1: Manhattan, 2: Euclidean
            }
            print(f"\nTuning parameters:\n{param_grid}")
            grid_search = perform_grid_search(model, param_grid, X_train, y_train)
            model = grid_search.best_estimator_
            print(f"\nBest parameters:\n{model.get_params()}")
    
    elif model_name == 'decision_tree':
        model = create_decision_tree_model(class_weight=class_weight)
        print(f"Initial parameters:\n{model.get_params()}")
        
        if perform_hyperparameter_tuning:
            param_grid = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            print(f"\nTuning parameters:\n{param_grid}")
            grid_search = perform_grid_search(model, param_grid, X_train, y_train)
            model = grid_search.best_estimator_
            print(f"\nBest parameters:\n{model.get_params()}")
    
    elif model_name == 'mlp':
        model = create_mlp_model()
        print(f"Initial parameters:\n{model.get_params()}")
        
        if perform_hyperparameter_tuning:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [200, 500]
            }
            print(f"\nTuning parameters:\n{param_grid}")
            grid_search = perform_grid_search(model, param_grid, X_train, y_train)
            model = grid_search.best_estimator_
            print(f"\nBest parameters:\n{model.get_params()}")
    
    # Print class balance information
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"\nClass distribution in training data:\n{class_distribution}")
    
    if class_weight:
        print(f"\nUsing class weights: {class_weight}")
        if hasattr(model, 'class_weight_'):
            print(f"Computed class weights:\n{model.class_weight_}")
    
    # Train the model if hyperparameter tuning wasn't performed
    if model is not None and not perform_hyperparameter_tuning:
        model.fit(X_train, y_train)
        # Print final model parameters if they changed during training
        print(f"\nFinal model parameters:\n{model.get_params()}")
    
    return model

def experiment_models(X_train, y_train, X_val, y_val, class_names):
    """
    Thực hiện thử nghiệm với nhiều mô hình và tìm kiếm tham số tối ưu cho từng mô hình.
    
    Args:
        X_train: Đặc trưng huấn luyện
        y_train: Nhãn huấn luyện
        X_val: Đặc trưng kiểm thử
        y_val: Nhãn kiểm thử
        class_names: Tên các lớp
        
    Returns:
        Dictionary với kết quả phân tích và mô hình tốt nhất
    """
    models_to_test = {
        'naive_bayes': {
            'model': create_naive_bayes_model(),
            'param_grid': {
                'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'fit_prior': [True, False]
            }
        },
        'logistic_regression': {
            'model': create_logistic_regression_model(),
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'penalty': ['l1', 'l2', None],
                'max_iter': [1000, 2000, 5000],
                'class_weight': [None, 'balanced']
            }
        },
        'svm': {
            'model': create_svm_model(probability=True),
            'param_grid': {
                'C': [0.1, 1.0, 10.0, 100.0, 1000.0],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'degree': [2, 3, 4],  # Cho kernel poly
                'class_weight': [None, 'balanced']
            }
        },
        'linear_svc': {
            'model': LinearSVC(random_state=RANDOM_STATE, max_iter=10000),
            'param_grid': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'loss': ['squared_hinge'],  # Removido 'hinge' para evitar conflitos
                'penalty': ['l1', 'l2'],
                'dual': [True, False],
                'class_weight': [None, 'balanced']
            }
        },
        'random_forest': {
            'model': create_random_forest_model(),
            'param_grid': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }
        },
        'gradient_boosting': {
            'model': create_gradient_boosting_model(),
            'param_grid': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5],
                'subsample': [0.7, 0.8, 0.9, 1.0]
            }
        },
        'knn': {
            'model': create_knn_model(),
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],  # 1: Manhattan, 2: Euclidean
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [20, 30, 40]
            }
        },
        'decision_tree': {
            'model': create_decision_tree_model(),
            'param_grid': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            }
        }
    }
    
    best_models = {}
    
    for model_name, model_config in models_to_test.items():
        print(f"\n\n{'='*80}\nTuning model: {model_name}\n{'='*80}\n")
        model = model_config['model']
        param_grid = model_config['param_grid']
        
        # Chạy GridSearchCV với tất cả tham số
        try:
            grid_search = perform_grid_search(
                model=model, 
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                cv=5,  # Sử dụng 5-fold cross-validation
                scoring='f1_macro',  # Tối ưu hóa F1-macro
                verbose=1
            )
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_results = grid_search.cv_results_
            
            # Đánh giá mô hình với tập kiểm thử
            val_results = evaluate_model(best_model, X_val, y_val, class_names)
            
            # Lưu kết quả
            best_models[model_name] = {
                'model': best_model,
                'params': best_params,
                'cv_results': cv_results,
                'val_results': val_results
            }
            
            print(f"\n\n{'='*80}\nBest model: {model_name}\n{'='*80}\n")
            print(f"Best parameters: {best_params}")
            print(f"Best F1-macro: {val_results['f1_macro']:.4f}")
            print(f"Validation accuracy: {val_results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error tuning model: {model_name}")
            print(f"Exception: {e}")
    
    return best_models

def perform_grid_search(model: BaseEstimator, param_grid: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                      cv: int = 5, scoring: str = 'f1_macro') -> GridSearchCV:
    """
    Perform grid search for hyperparameter optimization.
    
    Args:
        model: Base model
        param_grid: Dictionary with parameters names as keys and lists of parameter values
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Fitted GridSearchCV object
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best {scoring}: {grid_search.best_score_:.4f}")
    
    return grid_search

def perform_random_search(model: BaseEstimator, param_distributions: Dict[str, Any],
                        X_train: np.ndarray, y_train: np.ndarray,
                        n_iter: int = 20, cv: int = 5, scoring: str = 'f1_macro') -> RandomizedSearchCV:
    """
    Perform randomized search for hyperparameter optimization.
    
    Args:
        model: Base model
        param_distributions: Dictionary with parameters names as keys and distributions as values
        X_train: Training features
        y_train: Training labels
        n_iter: Number of parameter settings sampled
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Fitted RandomizedSearchCV object
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best {scoring}: {random_search.best_score_:.4f}")
    
    return random_search

def fine_tune_model(model_name, X_train, y_train, class_names=None, cv=5, 
                   save_results=True, output_dir='results/tuning'):
    """     
    Fine-tune a specified model with comprehensive parameter grids
    
    Args:
        model_name: Name of the model to fine-tune
        X_train: Training features
        y_train: Training labels
        class_names: List of class names
        cv: Number of cross-validation folds
        save_results: Whether to save results to disk
        output_dir: Directory to save results
        
    Returns:
        best_model: The best model found
        best_params: The best parameters found
        results_df: DataFrame with all results
    """
    print(f"Starting fine-tuning for {model_name}...")
            
    # Create output directory if it doesn't exist
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get base model and parameter grid based on model name
    base_model, param_grid = get_model_and_params(model_name)
    
    # Run grid search
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    
    grid_search = perform_grid_search(
        model=base_model,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring='f1_macro',
        verbose=1
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Extract results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Print results
    print(f"Fine-tuning completed in {duration:.1f} seconds")
    print(f"Best score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Sort results by rank
    results_df = results_df.sort_values('rank_test_score')
    
    # Save results if requested
    if save_results:
        model_dir = f"{output_dir}/{model_name}_{timestamp}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save model
        joblib.dump(best_model, f"{model_dir}/best_model.pkl")
        
        # Save best params
        with open(f"{model_dir}/best_params.txt", 'w') as f:
            f.write(f"Best score (f1_macro): {best_score:.4f}\n\n")
            f.write("Best parameters:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"\nTime taken: {duration:.1f} seconds")
        
        # Save full results
        results_df.to_csv(f"{model_dir}/all_results.csv", index=False)
        
        # Save top 10 results in separate file for easy viewing
        top_results = results_df.head(10)
        top_results.to_csv(f"{model_dir}/top_results.csv", index=False)
    
    return best_model, best_params, results_df

def get_model_and_params(model_name):
    """
    Get base model and comprehensive parameter grid based on model name
    """
    # Comprehensive parameter grids for different models
    param_grids = {
        'naive_bayes': {
            'alpha': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0],
            'fit_prior': [True, False]
        },
        'logistic_regression': {
            'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [1000, 2000, 5000],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'class_weight': [None, 'balanced'],
            'tol': [1e-5, 1e-4, 1e-3]
        },
        'svm': {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],  # For polynomial kernel
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'class_weight': [None, 'balanced'],
            'probability': [True]  # Keep True for probability estimates
        },
        'linear_svc': {
            'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'loss': ['hinge', 'squared_hinge'],
            'penalty': ['l1', 'l2'],
            'max_iter': [1000, 2000, 5000, 10000],
            'tol': [1e-5, 1e-4, 1e-3],
            'class_weight': [None, 'balanced'],
            'dual': [True, False]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [None, 5, 10, 15, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None, 0.7, 0.8],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'criterion': ['gini', 'entropy'],
            'min_impurity_decrease': [0.0, 0.01, 0.05]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [2, 3, 4, 5, 6, 8, 10],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None, 0.7, 0.8],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'validation_fraction': [0.1, 0.15, 0.2],
            'n_iter_no_change': [5, 10, 15]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 21],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40, 50],
            'p': [1, 2, 3],  # 1: Manhattan, 2: Euclidean, 3: Minkowski
            'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
        },
        'decision_tree': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15, 20, 30, 40],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None, 0.7, 0.8],
            'class_weight': [None, 'balanced'],
            'min_impurity_decrease': [0.0, 0.01, 0.05]
        }
    }
    
    # Create base model based on model name
    if model_name == 'naive_bayes':
        model = create_naive_bayes_model()
        param_grid = param_grids['naive_bayes']
    elif model_name == 'logistic_regression':
        model = create_logistic_regression_model()
        param_grid = param_grids['logistic_regression']
    elif model_name == 'svm':
        model = create_svm_model(probability=True)
        param_grid = param_grids['svm']
    elif model_name == 'linear_svc':
        model = create_svm_model(kernel='linear')
        param_grid = param_grids['linear_svc']
    elif model_name == 'random_forest':
        model = create_random_forest_model()
        param_grid = param_grids['random_forest']
    elif model_name == 'gradient_boosting':
        model = create_gradient_boosting_model()
        param_grid = param_grids['gradient_boosting']
    elif model_name == 'knn':
        model = create_knn_model()
        param_grid = param_grids['knn']
    elif model_name == 'decision_tree':
        model = create_decision_tree_model()
        param_grid = param_grids['decision_tree']
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model, param_grid

def adaptive_parameter_search(model: BaseEstimator, 
                            param_grid: Dict[str, Any], 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            dataset_size: str = 'auto', 
                            cv: int = 5, 
                            scoring: str = 'f1_macro',
                            output_dir: str = 'results/tuning',
                            model_name: str = 'model'):
    """
    Perform parameter search with adaptive strategy based on dataset size
    
    Args:
        model: Base model to tune
        param_grid: Full parameter grid
        X_train: Training features
        y_train: Training labels
        dataset_size: 'small', 'medium', 'large', or 'auto' to determine automatically
        cv: Number of cross-validation folds
        scoring: Scoring metric
        output_dir: Directory to save results
        model_name: Name of the model for saving files
        
    Returns:
        best_model: Best tuned model
        best_params: Best parameters
        results_df: DataFrame with all results
    """
    # Determine dataset size if set to auto
    if dataset_size == 'auto':
        n_samples = X_train.shape[0]
        if n_samples < 1000:
            dataset_size = 'small'
        elif n_samples < 10000:
            dataset_size = 'medium'
        else:
            dataset_size = 'large'
    
    print(f"Dataset size determined as: {dataset_size}")
    
    # Create output directory if needed
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = f"{output_dir}/{model_name}_{timestamp}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Adapt search strategy based on dataset size
    if (dataset_size == 'small'):
        # For small datasets, we can do a full grid search
        grid_search = perform_grid_search(
            model=model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train, 
            cv=cv, 
            scoring=scoring,
            verbose=1
        )
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        results_df = pd.DataFrame(grid_search.cv_results_)
        
    elif dataset_size == 'medium':
        # For medium datasets, use a two-stage approach
        # First, do a randomized search to narrow down the parameter space
        print("First stage: Randomized search to narrow parameter space")
        n_iter = min(50, np.prod([len(v) for v in param_grid.values()]) // 10)
        random_search = perform_random_search(
            model=model,
            param_distributions=param_grid, 
            X_train=X_train,
            y_train=y_train, 
            n_iter=n_iter, 
            cv=cv, 
            scoring=scoring
        )
        
        results_df_random = pd.DataFrame(random_search.cv_results_)
        top_params = results_df_random.sort_values('rank_test_score').head(5)
        
        # Create a focused parameter grid around the best parameters
        focused_param_grid = {}
        for param in param_grid.keys():
            param_col = f'param_{param}'
            if param_col in top_params.columns:
                unique_values = top_params[param_col].unique()
                if len(unique_values) > 0:
                    focused_param_grid[param] = list(unique_values)
                else:
                    focused_param_grid[param] = [random_search.best_params_[param]]
            else:
                focused_param_grid[param] = [random_search.best_params_[param]]
        
        # Second stage: Grid search around the best parameters
        print("Second stage: Grid search around best parameters from first stage")
        grid_search = perform_grid_search(
            model=model,
            param_grid=focused_param_grid, 
            X_train=X_train, 
            y_train=y_train, 
            cv=cv, 
            scoring=scoring,
            verbose=1
        )
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Save first stage results
        pd.DataFrame(random_search.cv_results_).to_csv(
            f"{result_dir}/first_stage_results.csv", index=False)
        
    else:  # Large dataset
        # For large datasets, we use randomized search with more iterations
        print("Large dataset: Using advanced randomized search")
        n_iter = 100  # More iterations for large datasets
        # Use randomized search with parameter importance evaluation
        random_search = perform_random_search(
            model=model,
            param_distributions=param_grid, 
            X_train=X_train,
            y_train=y_train, 
            n_iter=n_iter, 
            cv=3,  # Reduce CV folds for large datasets 
            scoring=scoring
        )
        
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        results_df = pd.DataFrame(random_search.cv_results_)
    
    # Save results
    joblib.dump(best_model, f"{result_dir}/best_model.pkl")
    with open(f"{result_dir}/best_params.txt", 'w') as f:
        f.write(f"Best score ({scoring}): {best_score:.4f}\n\n")
        f.write("Best parameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    results_df.to_csv(f"{result_dir}/all_results.csv", index=False)
    results_df.sort_values('rank_test_score').head(10).to_csv(
        f"{result_dir}/top_results.csv", index=False)
    
    # Try to extract parameter importance
    try:
        param_importance = calculate_parameter_importance(results_df)
        param_importance.to_csv(f"{result_dir}/parameter_importance.csv")
    except Exception as e:
        print(f"Could not calculate parameter importance: {e}")
    
    return best_model, best_params, results_df

def calculate_parameter_importance(results_df):
    """
    Calculate importance of each parameter based on CV results
    
    Args:
        results_df: DataFrame with CV results
        
    Returns:
        DataFrame with parameter importance scores
    """
    # Get all parameter columns
    param_cols = [c for c in results_df.columns if c.startswith('param_')]
    
    # Dictionary to store importance for each parameter
    importance_dict = {}
    
    for param in param_cols:
        param_name = param.replace('param_', '')
        
        # Group by this parameter and get mean and std of scores
        try:
            param_data = results_df.groupby(param)['mean_test_score'].agg(['mean', 'std', 'count']).reset_index()
            
            # Calculate the range of means (max - min)
            score_range = param_data['mean'].max() - param_data['mean'].min()
            
            # Calculate weighted std (weighted by count)
            weighted_std = np.average(param_data['std'], weights=param_data['count'])
            
            # Importance = range / weighted_std (to normalize by variation)
            # If weighted_std is 0, set importance to score_range
            importance = score_range / weighted_std if weighted_std > 0 else score_range
            
            importance_dict[param_name] = {
                'importance': importance,
                'score_range': score_range,
                'n_unique_values': len(param_data),
                'best_value': results_df.loc[results_df['rank_test_score'] == 1, param].values[0]
            }
        except Exception:
            # Skip parameters that cause errors (e.g., non-numeric values)
            pass
    
    # Convert to DataFrame and sort by importance
    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index')
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df

def create_svm_model(C: float = 1.0, kernel: str = 'linear', gamma: str = 'scale',
                   class_weight: Optional[str] = None, probability: bool = False) -> SVC:
    """
    Create a Support Vector Machine model.
    """
    if kernel == 'linear':
        return LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            max_iter=10000,
            dual=True,  # Definir explicitamente como True
            loss='squared_hinge'  # Usar apenas squared_hinge para evitar conflitos
        )
    else:
        return SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=RANDOM_STATE
        )

# Definição das funções de criação de modelos
def create_naive_bayes_model(alpha: float = 1.0, fit_prior: bool = True, is_complement: bool = False) -> BaseEstimator:
    """
    Create a Naive Bayes model.
    """
    if is_complement:
        return ComplementNB(alpha=alpha, fit_prior=fit_prior)
    else:
        return MultinomialNB(alpha=alpha, fit_prior=fit_prior)

def create_logistic_regression_model(C: float = 1.0, max_iter: int = 1000, 
                                   solver: str = 'liblinear', class_weight: Optional[str] = None) -> LogisticRegression:
    """
    Create a Logistic Regression model.
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )

def create_svm_model(C: float = 1.0, kernel: str = 'linear', gamma: str = 'scale',
                   class_weight: Optional[str] = None, probability: bool = False) -> BaseEstimator:
    """
    Create a Support Vector Machine model.
    """
    if kernel == 'linear':
        return LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            max_iter=10000,
            dual=True,
            loss='squared_hinge'
        )
    else:
        return SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=RANDOM_STATE
        )

def create_random_forest_model(n_estimators: int = 100, max_depth: Optional[int] = None,
                             min_samples_split: int = 2, class_weight: Optional[str] = None) -> RandomForestClassifier:
    """
    Create a Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=RANDOM_STATE
    )

def create_gradient_boosting_model(n_estimators: int = 100, learning_rate: float = 0.1,
                                 max_depth: int = 3, subsample: float = 1.0) -> GradientBoostingClassifier:
    """
    Create a Gradient Boosting classifier.
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=RANDOM_STATE
    )

def create_knn_model(n_neighbors: int = 5, weights: str = 'uniform',
                   algorithm: str = 'auto', leaf_size: int = 30) -> KNeighborsClassifier:
    """
    Create a K-Nearest Neighbors classifier.
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size
    )

def create_decision_tree_model(max_depth: Optional[int] = None, min_samples_split: int = 2,
                             class_weight: Optional[str] = None) -> DecisionTreeClassifier:
    """
    Create a Decision Tree classifier.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=RANDOM_STATE
    )

def create_mlp_model(hidden_layer_sizes=(100,), activation='relu', alpha=0.0001,
                   learning_rate='constant', max_iter=200) -> MLPClassifier:
    """
    Create a Multi-layer Perceptron classifier.
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=RANDOM_STATE
    )

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, label_names=None) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model with predict method
        X_test: Test features
        y_test: Test labels
        label_names: Names of the classes
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Verifica se os dados são multi-rótulo (cada amostra pode ter múltiplas classes)
    is_multilabel = hasattr(y_test, 'shape') and len(y_test.shape) > 1 and y_test.shape[1] > 1
    
    f1_samples = None
    if is_multilabel:
        # Só calcula f1_samples se for realmente um problema multilabel
        f1_samples = f1_score(y_test, y_pred, average='samples')
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Cria o relatório de classificação com os nomes das classes
    if label_names is not None:
        report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    else:
        report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_samples': f1_samples,
        'confusion_matrix': cm,
        'classification_report': report
    }

def print_evaluation_results(results: Dict[str, Any], model_name: str) -> None:
    """
    Print evaluation results in a readable format.
    """
    print(f"\n=== Evaluation Results for {model_name.upper()} ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Macro: {results['f1_macro']:.4f}")
    print(f"F1 Weighted: {results['f1_weighted']:.4f}")
    
    if results['f1_samples'] is not None:
        print(f"F1 Samples: {results['f1_samples']:.4f}")
    
    print("\nClassification Report:")
    for class_name, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):  # Ignorar entradas que não são dicionários
            print(f"Class: {class_name}")
            print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"  F1-score: {metrics.get('f1-score', 'N/A'):.4f}")
            print(f"  Support: {metrics.get('support', 'N/A')}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str = 'Confusion Matrix') -> None:
    """
    Plot a confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()