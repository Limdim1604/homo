"""
Models module for the HOMO-LAT project.
Contains functions for creating and evaluating different machine learning models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterSampler
import matplotlib.pyplot as plt
import seaborn as sns
from modules.config import RANDOM_STATE
# Xóa circular import
import joblib
import os
import time

# Traditional ML Models

def create_naive_bayes_model(alpha: float = 1.0, fit_prior: bool = True, is_complement: bool = False) -> BaseEstimator:
    """
    Create a Naive Bayes model.
    
    Args:
        alpha: Smoothing parameter
        fit_prior: Whether to learn class prior probabilities
        is_complement: Whether to use Complement Naive Bayes
        
    Returns:
        Naive Bayes model
    """
    if is_complement:
        return ComplementNB(alpha=alpha, fit_prior=fit_prior)
    else:
        return MultinomialNB(alpha=alpha, fit_prior=fit_prior)

def create_logistic_regression_model(C: float = 1.0, max_iter: int = 1000, 
                                   solver: str = 'liblinear', class_weight: Optional[str] = None) -> LogisticRegression:
    """
    Create a Logistic Regression model.
    
    Args:
        C: Inverse of regularization strength
        max_iter: Maximum number of iterations
        solver: Algorithm to use
        class_weight: Weights associated with classes
        
    Returns:
        Logistic Regression model
    """
    return LogisticRegression(
        C=C, 
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        multi_class='auto'
    )

def create_svm_model(C: float = 1.0, kernel: str = 'linear', gamma: str = 'scale',
                   class_weight: Optional[str] = None, probability: bool = False) -> SVC:
    """
    Create a Support Vector Machine model.
    
    Args:
        C: Regularization parameter
        kernel: Kernel type to be used
        gamma: Kernel coefficient
        class_weight: Weights associated with classes
        probability: Whether to enable probability estimates
        
    Returns:
        SVM model
    """
    if kernel == 'linear':
        return LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            max_iter=10000
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
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        class_weight: Weights associated with classes
        
    Returns:
        Random Forest model
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
    
    Args:
        n_estimators: Number of boosting stages
        learning_rate: Shrinks the contribution of each tree
        max_depth: Maximum depth of the individual regression estimators
        subsample: Fraction of samples to be used for fitting the individual base learners
        
    Returns:
        Gradient Boosting model
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
    Create a K-Neighbors classifier.
    
    Args:
        n_neighbors: Number of neighbors
        weights: Weight function used in prediction
        algorithm: Algorithm used to compute the nearest neighbors
        leaf_size: Leaf size passed to BallTree or KDTree
        
    Returns:
        KNN model
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
    
    Args:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        class_weight: Weights associated with classes
        
    Returns:
        Decision Tree model
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=RANDOM_STATE
    )

def create_mlp_model(hidden_layer_sizes: Tuple = (100,), activation: str = 'relu',
                   solver: str = 'adam', alpha: float = 0.0001,
                   learning_rate: str = 'adaptive', max_iter: int = 200) -> MLPClassifier:
    """
    Create a Multi-layer Perceptron classifier.
    
    Args:
        hidden_layer_sizes: Number of neurons in each hidden layer
        activation: Activation function
        solver: Solver for weight optimization
        alpha: L2 penalty parameter
        learning_rate: Learning rate schedule for weight updates
        max_iter: Maximum number of iterations
        
    Returns:
        MLP model
    """
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=RANDOM_STATE
    )

def create_ensemble_model(estimators: List[Tuple[str, BaseEstimator]], voting: str = 'hard') -> VotingClassifier:
    """
    Create an ensemble of models using voting.
    
    Args:
        estimators: List of (name, estimator) tuples
        voting: Voting type ('hard' or 'soft')
        
    Returns:
        Voting ensemble model
    """
    return VotingClassifier(
        estimators=estimators,
        voting=voting
    )

# Model Evaluation Functions

def evaluate_model(model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray, 
                 label_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        label_names: Class label names
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_samples = None
    
    # If multi-label classification
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        f1_samples = f1_score(y_test, y_pred, average='samples')
    
    # Get detailed classification report
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_samples': f1_samples,
        'classification_report': report,
        'confusion_matrix': cm
    }

def print_evaluation_results(results: Dict[str, Any], model_name: str) -> None:
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary with evaluation metrics
        model_name: Name of the model
    """
    print(f"===== {model_name} Evaluation Results =====")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {results['f1_weighted']:.4f}")
    
    if results['f1_samples'] is not None:
        print(f"F1 Score (Samples): {results['f1_samples']:.4f}")
    
    print("\nClassification Report:")
    report = results['classification_report']
    
    # Print per-class metrics
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg', 'samples avg']:
            print(f"  Class {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
            print(f"    Support: {metrics['support']}")
    
    # Print average metrics
    print("\n  Average Metrics:")
    if 'macro avg' in report:
        print(f"    Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    if 'weighted avg' in report:
        print(f"    Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
    if 'samples avg' in report:
        print(f"    Samples Avg F1-Score: {report['samples avg']['f1-score']:.4f}")

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], model_name: str, 
                        normalize: bool = False, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        model_name: Name of the model
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = f'Normalized Confusion Matrix - {model_name}'
        fmt = '.2f'
    else:
        title = f'Confusion Matrix - {model_name}'
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

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
        scoring='f1_macro'
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Extract results
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
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
            scoring=scoring
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
        
        # Extract top parameter combinations
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
            scoring=scoring
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