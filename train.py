"""
Script for training Forex prediction models.
"""

import os
import argparse
import time
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.utils.logger import setup_logger
from src.utils.config import (
    CURRENCY_PAIRS, TRAINED_MODELS_DIR, CONFIG_DIR, 
    MODEL_TYPES, TRAIN_START_DATE, TRAIN_END_DATE
)

# Set up argument parser
parser = argparse.ArgumentParser(description='Train Forex Prediction Models')
parser.add_argument('--model', type=str, choices=MODEL_TYPES + ['all'], default='all',
                    help=f'Model to train. Options: {", ".join(MODEL_TYPES)} or all. Default: all')
parser.add_argument('--currency', type=str, choices=CURRENCY_PAIRS + ['all', 'bagging'], default='all',
                    help=f'Currency pair to use. Options: {", ".join(CURRENCY_PAIRS)}, all, or bagging. Default: all')
parser.add_argument('--config', type=str, default='default_config.json',
                    help='Configuration file to use. Default: default_config.json')
parser.add_argument('--log', type=str, default='info',
                    help='Logging level. Options: debug, info, warning, error. Default: info')
parser.add_argument('--tuning', action='store_true',
                    help='Perform hyperparameter tuning. Default: False')
parser.add_argument('--visualize', action='store_true',
                    help='Generate visualizations during training. Default: False')

args = parser.parse_args()

# Set up logger
log_level = getattr(logging, args.log.upper())
logger = setup_logger("train", level=log_level)

# Load configuration
config_path = os.path.join(CONFIG_DIR, args.config)
if not os.path.exists(config_path):
    logger.warning(f"Configuration file {config_path} not found. Using default parameters.")
    config = {}
else:
    with open(config_path, 'r') as f:
        config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")

# Determine currency pairs to process
if args.currency == 'all':
    pairs = CURRENCY_PAIRS
elif args.currency == 'bagging':
    pairs = ['BAGGING']
else:
    pairs = [args.currency]

# Determine models to train
if args.model == 'all':
    models = MODEL_TYPES
else:
    models = [args.model]

def train_models():
    """Train Forex prediction models."""
    logger.info("Starting model training")
    
    # Record start time for performance tracking
    start_time = time.time()
    
    from src.features.feature_enhancement import load_feature_data
    
    # Get model training parameters from config
    model_params = config.get('model_training', {})
    
    # Determine currency pairs to train models for
    train_pairs = [p for p in pairs if p != 'BAGGING']
    
    for model_type in models:
        logger.info(f"Training {model_type} models")
        
        for pair in train_pairs:
            logger.info(f"Training {model_type} model for {pair}")
            
            try:
                # Load feature data
                train_data = load_feature_data(pair, 'train')
                val_data = load_feature_data(pair, 'val')
                test_data = load_feature_data(pair, 'test')
            except FileNotFoundError as e:
                logger.error(f"Feature data for {pair} not found. Run feature enhancement first. Error: {str(e)}")
                continue
            
            # Perform hyperparameter tuning if requested
            if args.tuning:
                logger.info(f"Performing hyperparameter tuning for {model_type} on {pair}")
                
                if model_type == 'CNN-LSTM':
                    from src.models.cnn_lstm import create_sequences
                    from hyperparameter_tuning import tune_cnn_lstm
                    
                    # Get model-specific parameters
                    cnn_lstm_params = model_params.get('CNN-LSTM', {})
                    sequence_length = cnn_lstm_params.get('sequence_length', 60)
                    
                    # Prepare sequences
                    X_train, y_train = create_sequences(
                        train_data,
                        target_col='target',
                        sequence_length=sequence_length,
                        prediction_steps=1,
                        shuffle=True
                    )
                    
                    X_val, y_val = create_sequences(
                        val_data,
                        target_col='target',
                        sequence_length=sequence_length,
                        prediction_steps=1,
                        shuffle=False
                    )
                    
                    # Tune hyperparameters
                    best_params = tune_cnn_lstm(
                        X_train, y_train,
                        X_val, y_val,
                        n_trials=50,
                        study_name=f"cnn_lstm_{pair}_tuning"
                    )
                    
                    # Update model config with best parameters
                    if 'model_config' not in cnn_lstm_params:
                        cnn_lstm_params['model_config'] = {}
                    
                    cnn_lstm_params['model_config'].update(best_params)
                    model_params['CNN-LSTM'] = cnn_lstm_params
                
                elif model_type == 'TFT':
                    from src.models.tft import prepare_data_for_tft
                    from hyperparameter_tuning import tune_tft
                    
                    # Get model-specific parameters
                    tft_params = model_params.get('TFT', {})
                    max_encoder_length = tft_params.get('max_encoder_length', 24)
                    
                    # Prepare data for TFT
                    training, validation, testing, data_params = prepare_data_for_tft(
                        train_data,
                        val_data,
                        test_data,
                        target_variable='target',
                        max_encoder_length=max_encoder_length,
                        max_prediction_length=1
                    )
                    
                    # Tune hyperparameters
                    best_params = tune_tft(
                        training,
                        validation,
                        n_trials=50,
                        study_name=f"tft_{pair}_tuning"
                    )
                    
                    # Update model config with best parameters
                    if 'model_config' not in tft_params:
                        tft_params['model_config'] = {}
                    
                    tft_params['model_config'].update(best_params)
                    model_params['TFT'] = tft_params
                
                elif model_type == 'XGBoost':
                    from src.models.xgboost_model import prepare_data_for_xgboost
                    from hyperparameter_tuning import tune_xgboost
                    
                    # Get model-specific parameters
                    xgboost_params = model_params.get('XGBoost', {})
                    sequence_length = xgboost_params.get('sequence_length', 1)
                    
                    # Prepare data for XGBoost
                    X_train, y_train, feature_names = prepare_data_for_xgboost(
                        train_data,
                        target_col='target',
                        sequence_length=sequence_length,
                        prediction_steps=1,
                        shuffle=True
                    )
                    
                    X_val, y_val, _ = prepare_data_for_xgboost(
                        val_data,
                        target_col='target',
                        sequence_length=sequence_length,
                        prediction_steps=1,
                        shuffle=False
                    )
                    
                    # Tune hyperparameters
                    best_params = tune_xgboost(
                        X_train, y_train,
                        X_val, y_val,
                        n_trials=100,
                        study_name=f"xgboost_{pair}_tuning"
                    )
                    
                    # Update model config with best parameters
                    if 'model_config' not in xgboost_params:
                        xgboost_params['model_config'] = {}
                    
                    xgboost_params['model_config'].update(best_params)
                    model_params['XGBoost'] = xgboost_params
                
                logger.info(f"Hyperparameter tuning completed for {model_type} on {pair}")
            
            # Train model based on type
            if model_type == 'CNN-LSTM':
                from src.models.cnn_lstm import (
                    create_sequences, 
                    train_cnn_lstm_model, 
                    plot_training_history
                )
                
                # Get model-specific parameters
                cnn_lstm_params = model_params.get('CNN-LSTM', {})
                sequence_length = cnn_lstm_params.get('sequence_length', 60)
                
                # Prepare sequences
                X_train, y_train = create_sequences(
                    train_data,
                    target_col='target',
                    sequence_length=sequence_length,
                    prediction_steps=1,
                    shuffle=True
                )
                
                X_val, y_val = create_sequences(
                    val_data,
                    target_col='target',
                    sequence_length=sequence_length,
                    prediction_steps=1,
                    shuffle=False
                )
                
                # Train model
                model, history = train_cnn_lstm_model(
                    X_train, y_train,
                    X_val, y_val,
                    model_config=cnn_lstm_params.get('model_config', None),
                    training_config=cnn_lstm_params.get('training_config', None),
                    model_name=f"cnn_lstm_{pair}",
                    save_model=True
                )
                
                # Plot training history
                if args.visualize:
                    plot_training_history(
                        history,
                        model_name=f"cnn_lstm_{pair}",
                        save_plot=True,
                        show_plot=args.visualize
                    )
            
            elif model_type == 'TFT':
                from src.models.tft import (
                    prepare_data_for_tft,
                    train_tft_model
                )
                
                # Get model-specific parameters
                tft_params = model_params.get('TFT', {})
                max_encoder_length = tft_params.get('max_encoder_length', 24)
                
                # Prepare data for TFT
                training, validation, testing, data_params = prepare_data_for_tft(
                    train_data,
                    val_data,
                    test_data,
                    target_variable='target',
                    max_encoder_length=max_encoder_length,
                    max_prediction_length=1
                )
                
                # Train model
                model, history = train_tft_model(
                    training,
                    validation,
                    model_config=tft_params.get('model_config', None),
                    training_config=tft_params.get('training_config', None),
                    model_name=f"tft_{pair}",
                    save_model=True
                )
            
            elif model_type == 'XGBoost':
                from src.models.xgboost_model import (
                    prepare_data_for_xgboost,
                    train_xgboost_model,
                    plot_xgboost_training_history,
                    plot_xgboost_feature_importance
                )
                
                # Get model-specific parameters
                xgboost_params = model_params.get('XGBoost', {})
                sequence_length = xgboost_params.get('sequence_length', 1)
                
                # Prepare data for XGBoost
                X_train, y_train, feature_names = prepare_data_for_xgboost(
                    train_data,
                    target_col='target',
                    sequence_length=sequence_length,
                    prediction_steps=1,
                    shuffle=True
                )
                
                X_val, y_val, _ = prepare_data_for_xgboost(
                    val_data,
                    target_col='target',
                    sequence_length=sequence_length,
                    prediction_steps=1,
                    shuffle=False
                )
                
                # Train model
                model, history = train_xgboost_model(
                    X_train, y_train,
                    X_val, y_val,
                    feature_names=feature_names,
                    model_config=xgboost_params.get('model_config', None),
                    model_name=f"xgboost_{pair}",
                    save_model=True
                )
                
                # Plot training history and feature importance
                if args.visualize:
                    plot_xgboost_training_history(
                        history,
                        model_name=f"xgboost_{pair}",
                        save_plot=True,
                        show_plot=args.visualize
                    )
                    
                    plot_xgboost_feature_importance(
                        model,
                        feature_names=feature_names,
                        model_name=f"xgboost_{pair}",
                        importance_type='weight',
                        max_features=20,
                        save_plot=True,
                        show_plot=args.visualize
                    )
            
            logger.info(f"Completed training {model_type} model for {pair}")
    
    # Train Bagging model if requested
    if 'BAGGING' in pairs:
        logger.info("Training Bagging models")
        from src.models.bagging_model import create_bagging_model
        
        for model_type in models:
            logger.info(f"Creating Bagging model for {model_type}")
            bagging_model = create_bagging_model(
                model_type,
                CURRENCY_PAIRS,
                model_name=f"bagging_{model_type.lower()}"
            )
            logger.info(f"Completed creating Bagging model for {model_type}")
    
    # Record end time and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Model training completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    train_models()