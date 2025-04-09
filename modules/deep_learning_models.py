"""
Deep Learning models module for the HOMO-LAT project.
Contains functions for creating and evaluating different deep learning models.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Try importing TensorFlow and Keras, with graceful fallback if not installed
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D
    from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D, Bidirectional
    from tensorflow.keras.layers import Input, Concatenate, Flatten, SpatialDropout1D
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Deep learning models will not work.")
    TENSORFLOW_AVAILABLE = False

from modules.config import MAX_LENGTH, EMBEDDING_DIM, RANDOM_STATE, MODELS_DIR

# Text Preprocessing for Deep Learning

def tokenize_and_pad(texts: List[str], max_length: int = MAX_LENGTH, 
                    max_words: int = 10000) -> Tuple[np.ndarray, Tokenizer]:
    """
    Tokenize texts and pad sequences.
    
    Args:
        texts: List of texts
        max_length: Maximum length of sequences
        max_words: Maximum number of words in vocabulary
        
    Returns:
        Tuple of padded sequences and tokenizer
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot tokenize and pad sequences.")
        return None, None
        
    # Create tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"Padded sequences shape: {padded_sequences.shape}")
    
    return padded_sequences, tokenizer

def create_embedding_matrix(tokenizer: Tokenizer, embedding_dim: int = EMBEDDING_DIM, 
                          pretrained_embeddings: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """
    Create embedding matrix.
    
    Args:
        tokenizer: Fitted tokenizer
        embedding_dim: Dimensionality of the embeddings
        pretrained_embeddings: Dictionary of pretrained word embeddings (optional)
        
    Returns:
        Embedding matrix
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot create embedding matrix.")
        return None
        
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    if pretrained_embeddings is not None:
        for word, i in tokenizer.word_index.items():
            if i >= vocab_size:
                continue
                
            embedding_vector = pretrained_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
        print(f"Created embedding matrix with shape: {embedding_matrix.shape}")
        
    return embedding_matrix

# Deep Learning Models

def create_lstm_model(vocab_size: int, embedding_dim: int = EMBEDDING_DIM, 
                    max_length: int = MAX_LENGTH, embedding_matrix: Optional[np.ndarray] = None,
                    num_classes: int = 3, dropout_rate: float = 0.2) -> Model:
    """
    Create an LSTM model.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimensionality of the embeddings
        max_length: Maximum length of sequences
        embedding_matrix: Pretrained embedding matrix (optional)
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot create LSTM model.")
        return None
        
    model = Sequential()
    
    # Embedding layer
    if embedding_matrix is not None:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False
        ))
    else:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ))
    
    # Add SpatialDropout
    model.add(SpatialDropout1D(dropout_rate))
    
    # LSTM layer
    model.add(LSTM(100, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    
    # Dense output layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    if num_classes == 2:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_bilstm_model(vocab_size: int, embedding_dim: int = EMBEDDING_DIM,
                       max_length: int = MAX_LENGTH, embedding_matrix: Optional[np.ndarray] = None,
                       num_classes: int = 3, dropout_rate: float = 0.2) -> Model:
    """
    Create a Bidirectional LSTM model.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimensionality of the embeddings
        max_length: Maximum length of sequences
        embedding_matrix: Pretrained embedding matrix (optional)
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        Bidirectional LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot create Bidirectional LSTM model.")
        return None
        
    model = Sequential()
    
    # Embedding layer
    if embedding_matrix is not None:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False
        ))
    else:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ))
    
    # Add SpatialDropout
    model.add(SpatialDropout1D(dropout_rate))
    
    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(100, dropout=dropout_rate, recurrent_dropout=dropout_rate)))
    
    # Dense output layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    if num_classes == 2:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_cnn_model(vocab_size: int, embedding_dim: int = EMBEDDING_DIM,
                   max_length: int = MAX_LENGTH, embedding_matrix: Optional[np.ndarray] = None,
                   num_classes: int = 3, dropout_rate: float = 0.2) -> Model:
    """
    Create a CNN model for text classification.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimensionality of the embeddings
        max_length: Maximum length of sequences
        embedding_matrix: Pretrained embedding matrix (optional)
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        CNN model
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot create CNN model.")
        return None
        
    model = Sequential()
    
    # Embedding layer
    if embedding_matrix is not None:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False
        ))
    else:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ))
    
    # Add SpatialDropout
    model.add(SpatialDropout1D(dropout_rate))
    
    # Add CNN layers
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    
    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    if num_classes == 2:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

def create_cnn_lstm_model(vocab_size: int, embedding_dim: int = EMBEDDING_DIM,
                        max_length: int = MAX_LENGTH, embedding_matrix: Optional[np.ndarray] = None,
                        num_classes: int = 3, dropout_rate: float = 0.2) -> Model:
    """
    Create a hybrid CNN-LSTM model for text classification.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimensionality of the embeddings
        max_length: Maximum length of sequences
        embedding_matrix: Pretrained embedding matrix (optional)
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        CNN-LSTM model
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot create CNN-LSTM model.")
        return None
        
    model = Sequential()
    
    # Embedding layer
    if embedding_matrix is not None:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False
        ))
    else:
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ))
    
    # Add SpatialDropout
    model.add(SpatialDropout1D(dropout_rate))
    
    # Add CNN layer for feature extraction
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    
    # Add LSTM layer
    model.add(LSTM(100, dropout=dropout_rate, recurrent_dropout=dropout_rate))
    
    # Dense layers
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    if num_classes == 2:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model

# Model Training and Evaluation

def prepare_data_for_deep_learning(X_train: List[str], X_val: List[str], y_train: np.ndarray, y_val: np.ndarray,
                                max_length: int = MAX_LENGTH, max_words: int = 10000, 
                                num_classes: int = 3) -> Tuple:
    """
    Prepare data for deep learning models.
    
    Args:
        X_train: Training texts
        X_val: Validation texts
        y_train: Training labels
        y_val: Validation labels
        max_length: Maximum length of sequences
        max_words: Maximum number of words in vocabulary
        num_classes: Number of output classes
        
    Returns:
        Tuple of (X_train_padded, X_val_padded, y_train_cat, y_val_cat, tokenizer)
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot prepare data for deep learning.")
        return None
        
    # Tokenize and pad sequences
    X_train_padded, tokenizer = tokenize_and_pad(X_train, max_length, max_words)
    X_val_padded = pad_sequences(
        tokenizer.texts_to_sequences(X_val),
        maxlen=max_length,
        padding='post'
    )
    
    # Convert labels to categorical format if needed
    if num_classes == 2:
        y_train_cat = y_train
        y_val_cat = y_val
    else:
        y_train_cat = to_categorical(y_train, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)
    
    return X_train_padded, X_val_padded, y_train_cat, y_val_cat, tokenizer

def train_deep_learning_model(model: Model, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            model_name: str, batch_size: int = 32,
                            epochs: int = 20, patience: int = 3) -> Model:
    """
    Train a deep learning model.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name of the model (for saving checkpoints)
        batch_size: Batch size
        epochs: Number of epochs
        patience: Patience for early stopping
        
    Returns:
        Trained model
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot train deep learning model.")
        return None
        
    # Create model directory if it doesn't exist
    model_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define callbacks
    checkpoint_path = os.path.join(model_dir, f"{model_name}_best.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"Model training completed. Best model saved to {checkpoint_path}")
    
    return model

def evaluate_deep_learning_model(model: Model, X_test: np.ndarray, y_test: np.ndarray,
                               num_classes: int = 3, label_names: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate a deep learning model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        num_classes: Number of output classes
        label_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Cannot evaluate deep learning model.")
        return None
        
    # Get predictions
    if num_classes == 2:
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert one-hot encoded y_test to class indices if necessary
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Get detailed classification report
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm
    }