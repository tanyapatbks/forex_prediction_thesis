"""
Modified CNN-LSTM model for continuous target prediction.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import matplotlib.pyplot as plt

from src.utils.config import TRAINED_MODELS_DIR
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger("cnn_lstm")


def create_sequences(
    data: pd.DataFrame,
    target_col: str = 'target',
    sequence_length: int = 60,
    prediction_steps: int = 1,
    shuffle: bool = True,
    scaling_method: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and target values for CNN-LSTM model.
    
    Args:
        data (pd.DataFrame): DataFrame with features and target
        target_col (str, optional): Target column name. Defaults to 'target'.
        sequence_length (int, optional): Length of input sequences. Defaults to 60.
        prediction_steps (int, optional): Number of steps ahead to predict. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the sequences. Defaults to True.
        scaling_method (str, optional): Method for scaling features. 
                                      Options: 'standard', 'minmax', 'robust', None.
                                      Defaults to 'standard'.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (input sequences) and y (target values)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    # Separate features and target
    features = data.drop(columns=[target_col]).values
    target = data[target_col].values
    
    # Scale features if specified
    if scaling_method == 'standard':
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
    elif scaling_method == 'robust':
        scaler = RobustScaler()
        features = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_steps + 1):
        X.append(features[i:(i + sequence_length)])
        y.append(target[i + sequence_length + prediction_steps - 1])
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle sequences if specified
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    logger.info(f"Created {len(X)} sequences of length {sequence_length}")
    
    return X, y


def build_cnn_lstm_model(
    input_shape: Tuple[int, int],
    output_size: int = 1,
    regression: bool = True,  # Added parameter to indicate regression task
    cnn_layers: List[Dict] = None,
    lstm_layers: List[Dict] = None,
    dense_layers: List[Dict] = None,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0
) -> Model:
    """
    Build a CNN-LSTM hybrid model for Forex prediction.
    
    Args:
        input_shape (Tuple[int, int]): Shape of input sequences (sequence_length, n_features)
        output_size (int, optional): Size of output (1 for regression or binary classification). Defaults to 1.
        regression (bool, optional): Whether the task is regression. Defaults to True.
        cnn_layers (List[Dict], optional): List of CNN layer configurations. Defaults to None.
        lstm_layers (List[Dict], optional): List of LSTM layer configurations. Defaults to None.
        dense_layers (List[Dict], optional): List of dense layer configurations. Defaults to None.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.3.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.001.
        l1_reg (float, optional): L1 regularization factor. Defaults to 0.0.
        l2_reg (float, optional): L2 regularization factor. Defaults to 0.0.
        
    Returns:
        Model: Compiled CNN-LSTM model
    """
    # Use default configurations if not provided
    if cnn_layers is None:
        cnn_layers = [
            {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
            {'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
        ]
    
    if lstm_layers is None:
        lstm_layers = [
            {'units': 100, 'return_sequences': True},
            {'units': 50, 'return_sequences': False},
        ]
    
    if dense_layers is None:
        dense_layers = [
            {'units': 50, 'activation': 'relu'},
            {'units': 25, 'activation': 'relu'},
        ]
    
    # Create regularizer
    regularizer = l1_l2(l1=l1_reg, l2=l2_reg)
    
    # Build model
    inputs = Input(shape=input_shape)
    
    # CNN layers
    x = inputs
    for i, layer_config in enumerate(cnn_layers):
        x = Conv1D(
            filters=layer_config['filters'],
            kernel_size=layer_config['kernel_size'],
            activation=layer_config['activation'],
            padding='same',
            kernel_regularizer=regularizer
        )(x)
        x = BatchNormalization()(x)
        
        # Add pooling after pairs of CNN layers
        if (i + 1) % 2 == 0:
            x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers
    for i, layer_config in enumerate(lstm_layers):
        return_sequences = layer_config.get('return_sequences', i < len(lstm_layers) - 1)
        x = LSTM(
            units=layer_config['units'],
            return_sequences=return_sequences,
            kernel_regularizer=regularizer
        )(x)
        x = BatchNormalization()(x)
        if return_sequences:
            x = Dropout(dropout_rate)(x)
    
    # Additional dropout after LSTM
    x = Dropout(dropout_rate)(x)
    
    # Dense layers
    for layer_config in dense_layers:
        x = Dense(
            units=layer_config['units'],
            activation=layer_config['activation'],
            kernel_regularizer=regularizer
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
    # Output layer
    if regression:
        # For regression, use linear activation
        outputs = Dense(output_size, activation='linear')(x)
    else:
        # For classification (binary or multiclass)
        if output_size == 1:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(output_size, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    
    if regression:
        # For regression tasks
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean squared error loss
            metrics=['mae', 'mse']  # Track mean absolute error and mean squared error
        )
    else:
        # For classification tasks
        if output_size == 1:
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    logger.info(f"Built CNN-LSTM model with input shape {input_shape}, {'regression' if regression else 'classification'} task")
    
    return model


def train_cnn_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    regression: bool = True,  # Added parameter for regression
    model_config: Dict = None,
    training_config: Dict = None,
    model_name: str = 'cnn_lstm',
    save_model: bool = True
) -> Tuple[Model, Dict]:
    """
    Train a CNN-LSTM model for Forex prediction.
    
    Args:
        X_train (np.ndarray): Training input sequences
        y_train (np.ndarray): Training target values
        X_val (np.ndarray): Validation input sequences
        y_val (np.ndarray): Validation target values
        regression (bool, optional): Whether the task is regression. Defaults to True.
        model_config (Dict, optional): Model configuration. Defaults to None.
        training_config (Dict, optional): Training configuration. Defaults to None.
        model_name (str, optional): Name for the saved model. Defaults to 'cnn_lstm'.
        save_model (bool, optional): Whether to save the trained model. Defaults to True.
        
    Returns:
        Tuple[Model, Dict]: Trained model and training history
    """
    # Use default configurations if not provided
    if model_config is None:
        model_config = {
            'cnn_layers': [
                {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
            ],
            'lstm_layers': [
                {'units': 100, 'return_sequences': True},
                {'units': 50, 'return_sequences': False},
            ],
            'dense_layers': [
                {'units': 50, 'activation': 'relu'},
                {'units': 25, 'activation': 'relu'},
            ],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'l1_reg': 0.0,
            'l2_reg': 0.0
        }
    
    if training_config is None:
        training_config = {
            'batch_size': 32,
            'epochs': 100,
            'patience': 15,
            'min_delta': 0.001,
            'validation_split': 0.0,  # We're providing validation data explicitly
            'shuffle': True
        }
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = 1  # Usually 1 for regression or binary classification
    
    model = build_cnn_lstm_model(
        input_shape=input_shape,
        output_size=output_size,
        regression=regression,  # Pass the regression flag
        **model_config
    )
    
    # Define callbacks
    callbacks = []
    
    # Early stopping
    monitor_metric = 'val_loss'  # Use loss for both regression and classification
    
    early_stop = EarlyStopping(
        monitor=monitor_metric,
        patience=training_config['patience'],
        min_delta=training_config['min_delta'],
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor=monitor_metric,
        factor=0.5,
        patience=5,
        min_delta=0.001,
        min_lr=0.00001,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model checkpoint
    if save_model:
        os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
        checkpoint_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_checkpoint.h5")
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor=monitor_metric,
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        shuffle=training_config['shuffle'],
        verbose=1
    )
    
    # Save trained model
    if save_model:
        model_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}.h5")
        model.save(model_path)
        
        # Save model configuration
        config_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': model_config,
                'training_config': training_config,
                'input_shape': input_shape,
                'output_size': output_size,
                'regression': regression  # Save regression flag
            }, f, indent=4)
        
        logger.info(f"Saved trained model to {model_path}")
    
    logger.info("CNN-LSTM model training completed")
    
    return model, history.history


def evaluate_cnn_lstm_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    regression: bool = True  # Added parameter for regression
) -> Dict[str, float]:
    """
    Evaluate a trained CNN-LSTM model.
    
    Args:
        model (Model): Trained CNN-LSTM model
        X_test (np.ndarray): Test input sequences
        y_test (np.ndarray): Test target values
        regression (bool, optional): Whether the task is regression. Defaults to True.
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    # Evaluate model
    evaluation = model.evaluate(X_test, y_test, verbose=1)
    
    # Get metrics
    metrics = {}
    
    if regression:
        # Regression metrics
        metrics['loss'] = evaluation[0]  # MSE
        metrics['mae'] = evaluation[1]   # Mean Absolute Error
        metrics['mse'] = evaluation[2]   # Mean Squared Error
        
        # Additional regression metrics
        y_pred = model.predict(X_test)
        
        # Root Mean Squared Error (RMSE)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # R-squared (coefficient of determination)
        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
        ss_residual = np.sum((y_test - y_pred.flatten()) ** 2)
        metrics['r2'] = 1 - (ss_residual / ss_total)
        
        # Mean Absolute Percentage Error (MAPE)
        # For direction strength prediction, MAPE might not be meaningful
        
        # Directional Accuracy (how often the sign of prediction matches the sign of actual)
        correct_direction = np.sum(np.sign(y_pred.flatten()) == np.sign(y_test))
        metrics['directional_accuracy'] = correct_direction / len(y_test)
        
    else:
        # Classification metrics
        if len(model.metrics_names) >= 1:
            metrics['loss'] = evaluation[0]
        
        if len(model.metrics_names) >= 2:
            metrics['accuracy'] = evaluation[1]
        
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int) if y_pred_proba.shape[1:] == () or y_pred_proba.shape[1] == 1 else np.argmax(y_pred_proba, axis=1)
        
        # Additional metrics for binary classification
        if y_test.ndim == 1 or y_test.shape[1] == 1:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            # Flatten predictions if needed
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba.flatten()
            
            # Calculate metrics
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.5  # Default value if AUC cannot be calculated
    
    logger.info(f"CNN-LSTM model evaluation: {metrics}")
    
    return metrics


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = 'cnn_lstm',
    regression: bool = True,  # Added parameter for regression
    save_plot: bool = True,
    show_plot: bool = True
) -> None:
    """
    Plot training history of a CNN-LSTM model.
    
    Args:
        history (Dict[str, List[float]]): Training history
        model_name (str, optional): Name of the model. Defaults to 'cnn_lstm'.
        regression (bool, optional): Whether the task is regression. Defaults to True.
        save_plot (bool, optional): Whether to save the plot. Defaults to True.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot training & validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot second metric (accuracy for classification, MAE for regression)
    plt.subplot(2, 1, 2)
    if regression:
        # For regression, plot Mean Absolute Error
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title(f'{model_name} - Mean Absolute Error')
        plt.ylabel('MAE')
    else:
        # For classification, plot accuracy
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} - Accuracy')
        plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plot_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {plot_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def load_cnn_lstm_model(model_name: str) -> Tuple[Model, Dict]:
    """
    Load a trained CNN-LSTM model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Tuple[Model, Dict]: Loaded model and configuration
    """
    model_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}.h5")
    config_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_config.json")
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        error_msg = f"Model or config file not found: {model_path}, {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded CNN-LSTM model from {model_path}")
    
    return model, config


def predict_with_cnn_lstm(
    model: Model,
    X: np.ndarray,
    regression: bool = True,  # Added parameter for regression
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using a trained CNN-LSTM model.
    
    Args:
        model (Model): Trained CNN-LSTM model
        X (np.ndarray): Input sequences
        regression (bool, optional): Whether the task is regression. Defaults to True.
        threshold (float, optional): Decision threshold for binary classification. Defaults to 0.5.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted labels and raw predictions (regression values or probabilities)
    """
    # Get predictions
    y_pred_raw = model.predict(X)
    
    if regression:
        # For regression, return the raw predictions and sign of predictions as labels
        y_pred_labels = np.sign(y_pred_raw)  # Sign of the prediction (-1, 0, 1)
        return y_pred_labels, y_pred_raw
    else:
        # For classification, convert probabilities to labels
        if y_pred_raw.shape[1:] == () or y_pred_raw.shape[1] == 1:
            # Binary classification
            y_pred_raw = y_pred_raw.flatten()
            y_pred_labels = (y_pred_raw > threshold).astype(int)
        else:
            # Multi-class classification
            y_pred_labels = np.argmax(y_pred_raw, axis=1)
    
        return y_pred_labels, y_pred_raw