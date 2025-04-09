"""
Main script for the HOMO-LAT sentiment analysis project.
This script provides a complete pipeline for training and evaluating various models.
"""
import os
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
import pickle
from typing import Dict, List, Tuple, Union, Any, Optional

# Import project modules
from modules.config import TRAIN_FILE, DEV_FILE, MODELS_DIR, RESULTS_DIR
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
from modules.models import (
    create_naive_bayes_model, create_logistic_regression_model,
    create_svm_model, create_random_forest_model, create_gradient_boosting_model,
    create_knn_model, create_decision_tree_model, create_mlp_model,
    evaluate_model, print_evaluation_results, plot_confusion_matrix,
    perform_grid_search
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
            'model': create_svm_model(kernel='linear'),
            'param_grid': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'loss': ['hinge', 'squared_hinge'],
                'dual': [True, False],
                'max_iter': [1000, 2000, 5000],
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
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }
        },
        'gradient_boosting': {
            'model': create_gradient_boosting_model(),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 0.9, 1.0]
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
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced']
            }
        }
    }
    
    # Tạo thư mục kết quả nếu chưa có
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(RESULTS_DIR, f"complete_experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Lưu kết quả vào: {experiment_dir}")
    
    # Theo dõi kết quả của tất cả mô hình
    all_results = {}
    best_models = {}
    comparison_results = []
    
    # Lặp qua từng mô hình để thử nghiệm
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
                cv=5,  # 5-fold cross-validation
                scoring='f1_macro',  # Tối ưu hóa F1-macro
                verbose=1
            )
            
            # Lấy mô hình tốt nhất 
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            cv_results = grid_search.cv_results_
            
            # Đánh giá mô hình trên tập validation
            val_results = evaluate_model(best_model, X_val, y_val, class_names)
            
            # Lưu kết quả
            model_dir = os.path.join(experiment_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Lưu mô hình tốt nhất
            model_path = os.path.join(model_dir, f"{model_name}_best_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            # Lưu thông số tốt nhất
            params_path = os.path.join(model_dir, f"{model_name}_best_params.json")
            with open(params_path, 'w') as f:
                # Chuyển đổi các giá trị không serializable
                serializable_params = {}
                for k, v in best_params.items():
                    if v is None:
                        serializable_params[k] = None
                    elif isinstance(v, (list, tuple)) and not v:
                        serializable_params[k] = []
                    else:
                        serializable_params[k] = str(v)
                json.dump(serializable_params, f, indent=2)
                
            # Lưu kết quả validation
            val_results_path = os.path.join(model_dir, f"{model_name}_validation_results.json")
            with open(val_results_path, 'w') as f:
                val_results_json = {
                    'accuracy': float(val_results['accuracy']),
                    'f1_macro': float(val_results['f1_macro']),
                    'f1_weighted': float(val_results['f1_weighted']),
                    'confusion_matrix': val_results['confusion_matrix'].tolist()
                }
                json.dump(val_results_json, f, indent=2)
            
            # Lưu kết quả cross-validation
            cv_results_path = os.path.join(model_dir, f"{model_name}_cv_results.csv")
            cv_results_df = pd.DataFrame(cv_results)
            cv_results_df.to_csv(cv_results_path, index=False)
                
            # Theo dõi kết quả cho so sánh
            comparison_results.append({
                'model': model_name,
                'best_params': best_params,
                'cv_f1_macro': best_score,
                'val_f1_macro': val_results['f1_macro'],
                'val_accuracy': val_results['accuracy']
            })
            
            # Lưu mô hình và kết quả tốt nhất
            best_models[model_name] = {
                'model': best_model,
                'params': best_params,
                'cv_score': best_score,
                'val_score': val_results
            }
            
            print(f"\nModel: {model_name}")
            print(f"Best parameters: {best_params}")
            print(f"Cross-validation F1-macro: {best_score:.4f}")
            print(f"Validation F1-macro: {val_results['f1_macro']:.4f}")
            print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error tuning {model_name}: {str(e)}")
    
    # Tạo bảng so sánh các mô hình
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values('val_f1_macro', ascending=False)
    comparison_path = os.path.join(experiment_dir, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    
    print("\n\n=== Comparison of all models ===\n")
    print(comparison_df[['model', 'cv_f1_macro', 'val_f1_macro', 'val_accuracy']])
    
    # Tìm mô hình tốt nhất
    best_model_name = comparison_df.iloc[0]['model']
    best_model_info = best_models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best parameters: {best_model_info['params']}")
    print(f"Validation F1-macro: {best_model_info['val_score']['f1_macro']:.4f}")
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_model_info['model'],
        'best_params': best_model_info['params'],
        'all_models': best_models,
        'comparison_df': comparison_df,
        'experiment_dir': experiment_dir
    }

def run_experiment(preprocessing_pipeline: str, vectorization_method: str,
                 model_name: str, use_dev_set: bool = False,
                 use_hyperparameter_tuning: bool = False,
                 handle_class_imbalance: bool = False,
                 ngram_range: Tuple[int, int] = (1, 1),
                 embedding_dim: int = 100,
                 save_results: bool = True) -> Dict[str, Any]:
    """
    Run a complete experiment with the specified configuration.
    
    Args:
        preprocessing_pipeline: Name of the preprocessing pipeline to use
        vectorization_method: Vectorization method to use
        model_name: Name of the model to train
        use_dev_set: Whether to use the development set for evaluation
        use_hyperparameter_tuning: Whether to perform hyperparameter tuning
        handle_class_imbalance: Whether to handle class imbalance
        ngram_range: Range of n-grams for BoW and TF-IDF
        embedding_dim: Dimension of word embeddings
        save_results: Whether to save the results
        
    Returns:
        Dictionary with experiment results
    """
    # Start timing
    start_time = time.time()
    
    # Load data
    train_df, dev_df = load_data()
    
    # Generate a timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a unique experiment name
    experiment_name = f"{preprocessing_pipeline}_{vectorization_method}_{model_name}_{timestamp}"
    
    print(f"\n=== Running experiment: {experiment_name} ===\n")
    
    # Prepare train and validation sets
    if use_dev_set:
        print("Using development set for evaluation.")
        X_train, y_train = train_df['post content'], train_df['label']
        X_val, y_val = dev_df['post content'], dev_df['label']
    else:
        print("Splitting training data for validation.")
        train_df, val_df = split_data(train_df)
        X_train, y_train = train_df['post content'], train_df['label']
        X_val, y_val = val_df['post content'], val_df['label']
    
    # Encode labels
    y_train_encoded, y_val_encoded, label_encoder = encode_labels(y_train, y_val)
    
    # Get class names
    class_names = label_encoder.classes_
    
    print(f"Classes: {class_names}")
    print(f"Distribution in training set: {np.bincount(y_train_encoded)}")
    print(f"Distribution in validation set: {np.bincount(y_val_encoded)}")
    
    # Preprocess text
    print(f"\nApplying preprocessing pipeline: {preprocessing_pipeline}")
    X_train_preprocessed = preprocess_text(X_train, preprocessing_pipeline)
    X_val_preprocessed = preprocess_text(X_val, preprocessing_pipeline)
    
    # Create word embeddings if needed
    embedding_model = None
    if vectorization_method in ['word2vec', 'fasttext']:
        print(f"\nTraining {vectorization_method} embeddings...")
        tokenized_sentences = [text.split() for text in X_train_preprocessed if isinstance(text, str)]
        
        if vectorization_method == 'word2vec':
            embedding_model = create_word2vec_model(
                sentences=tokenized_sentences, 
                vector_size=embedding_dim
            )
        elif vectorization_method == 'fasttext':
            embedding_model = create_fasttext_model(
                sentences=tokenized_sentences, 
                vector_size=embedding_dim
            )
    
    # Vectorize text
    print(f"\nVectorizing texts using method: {vectorization_method}")
    X_train_vectorized, vectorizer = vectorize_text(
        X_train_preprocessed, vectorization_method,
        ngram_range=ngram_range, embedding_model=embedding_model
    )
    
    # Transform validation data using the same vectorizer
    if vectorization_method in ['word2vec', 'fasttext']:
        X_val_vectorized = vectorizer.transform(X_val_preprocessed)
    else:
        X_val_vectorized = vectorizer.transform(X_val_preprocessed)
    
    print(f"Training features shape: {X_train_vectorized.shape}")
    print(f"Validation features shape: {X_val_vectorized.shape}")
    
    # Set class_weight if handling class imbalance
    class_weight = None
    if handle_class_imbalance:
        print("\nHandling class imbalance with 'balanced' weights")
        class_weight = 'balanced'
    
    # Train the model
    print(f"\nTraining model: {model_name}")
    if model_name in ['lstm', 'bilstm', 'cnn', 'cnn_lstm'] and TENSORFLOW_AVAILABLE:
        # Deep learning models
        # First convert data to the format expected by deep learning models
        X_train_dl, X_val_dl, y_train_dl, y_val_dl, tokenizer = prepare_data_for_deep_learning(
            X_train_preprocessed, X_val_preprocessed,
            y_train_encoded, y_val_encoded,
            num_classes=len(class_names)
        )
        
        # Create embedding matrix if using pre-trained embeddings
        embedding_matrix = None
        if embedding_model is not None:
            embedding_matrix = create_embedding_matrix(
                tokenizer, embedding_dim,
                {w: embedding_model.wv[w] for w in embedding_model.wv.key_to_index}
            )
        
        # Create the deep learning model
        vocab_size = len(tokenizer.word_index) + 1
        if model_name == 'lstm':
            model = create_lstm_model(
                vocab_size, embedding_dim, embedding_matrix=embedding_matrix,
                num_classes=len(class_names)
            )
        elif model_name == 'bilstm':
            model = create_bilstm_model(
                vocab_size, embedding_dim, embedding_matrix=embedding_matrix,
                num_classes=len(class_names)
            )
        elif model_name == 'cnn':
            model = create_cnn_model(
                vocab_size, embedding_dim, embedding_matrix=embedding_matrix,
                num_classes=len(class_names)
            )
        elif model_name == 'cnn_lstm':
            model = create_cnn_lstm_model(
                vocab_size, embedding_dim, embedding_matrix=embedding_matrix,
                num_classes=len(class_names)
            )
        
        # Train the model
        model = train_deep_learning_model(
            model, X_train_dl, y_train_dl, X_val_dl, y_val_dl,
            experiment_name, epochs=10
        )
        
        # Evaluate the model
        results = evaluate_deep_learning_model(
            model, X_val_dl, y_val_dl,
            num_classes=len(class_names),
            label_names=class_names
        )
    else:
        # Traditional machine learning models
        model = train_traditional_model(
            model_name, X_train_vectorized, y_train_encoded,
            class_weight=class_weight,
            perform_hyperparameter_tuning=use_hyperparameter_tuning
        )
        
        # Evaluate the model
        results = evaluate_model(model, X_val_vectorized, y_val_encoded, class_names)
    
    # Print evaluation results
    print("\n=== Evaluation Results ===\n")
    print_evaluation_results(results, model_name)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    
    # Save results if requested
    if save_results:
        # Create directories if they don't exist
        results_dir = os.path.join(RESULTS_DIR, experiment_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(results_dir, f"{experiment_name}_model.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to: {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
        
        # Save the vectorizer
        vectorizer_path = os.path.join(results_dir, f"{experiment_name}_vectorizer.pkl")
        try:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"Vectorizer saved to: {vectorizer_path}")
        except Exception as e:
            print(f"Failed to save vectorizer: {e}")
        
        # Save evaluation results
        results_path = os.path.join(results_dir, f"{experiment_name}_results.json")
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'accuracy': float(results['accuracy']),
            'f1_macro': float(results['f1_macro']),
            'f1_weighted': float(results['f1_weighted']),
            'class_report': {
                k: (v if isinstance(v, dict) else float(v)) 
                for k, v in results['classification_report'].items()
            },
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'experiment_config': {
                'preprocessing_pipeline': preprocessing_pipeline,
                'vectorization_method': vectorization_method,
                'model_name': model_name,
                'use_dev_set': use_dev_set,
                'use_hyperparameter_tuning': use_hyperparameter_tuning,
                'handle_class_imbalance': handle_class_imbalance,
                'ngram_range': ngram_range,
                'execution_time': execution_time
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to: {results_path}")
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'results': results,
        'experiment_name': experiment_name
    }

def main():
    """Main function for the HOMO-LAT project."""
    parser = argparse.ArgumentParser(description='Run HOMO-LAT sentiment analysis experiments.')
    
    parser.add_argument('--preprocessing', type=str, default='basic',
                      choices=['none', 'basic', 'normalize_accents', 'remove_stopwords', 'stemming', 'lemmatization', 'full'],
                      help='Preprocessing pipeline to use')
    
    parser.add_argument('--vectorization', type=str, default='bow',
                      choices=['bow', 'tfidf', 'word2vec', 'fasttext'],
                      help='Vectorization method to use')
    
    parser.add_argument('--model', type=str, default='logistic_regression',
                      choices=['naive_bayes', 'logistic_regression', 'svm', 'random_forest', 
                             'gradient_boosting', 'knn', 'decision_tree', 'mlp',
                             'lstm', 'bilstm', 'cnn', 'cnn_lstm'],
                      help='Model to train')
    
    parser.add_argument('--dev', action='store_true', help='Use development set for evaluation')
    
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    parser.add_argument('--balance', action='store_true', help='Handle class imbalance')
    
    parser.add_argument('--ngram_min', type=int, default=1, help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=1, help='Maximum n-gram size')
    
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension')
    
    parser.add_argument('--no_save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    # Run the experiment
    run_experiment(
        preprocessing_pipeline=args.preprocessing,
        vectorization_method=args.vectorization,
        model_name=args.model, 
        use_dev_set=args.dev,
        use_hyperparameter_tuning=args.tune,
        handle_class_imbalance=args.balance,
        ngram_range=(args.ngram_min, args.ngram_max),
        embedding_dim=args.embedding_dim,
        save_results=not args.no_save
    )

if __name__ == '__main__':
    main()