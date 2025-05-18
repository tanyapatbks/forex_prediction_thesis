"""
Hyperparameter tuning utilities for Forex prediction models.
"""

import os
import argparse
import time
import json
import pandas as pd
import numpy as np
import logging
import optuna
from optuna.trial import Trial
from datetime import datetime

from src.utils.logger import setup_logger
from src.utils.config import (
    TRAINED_MODELS_DIR, CONFIG_DIR, HYPERPARAMS_DIR,
    MODEL_TYPES, CURRENCY_PAIRS
)

# Set up argument parser
parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Forex Prediction Models')
parser.add_argument('--model', type=str, choices=MODEL_TYPES, required=True,
                    help=f'Model to tune. Options: {", ".join(MODEL_TYPES)}.')
parser.add_argument('--currency', type=str, choices=CURRENCY_PAIRS, required=True,
                    help=f'Currency pair to use for tuning. Options: {", ".join(CURRENCY_PAIRS)}.')
parser.add_argument('--trials', type=int, default=50,
                    help='Number of hyperparameter trials. Default: 50')
parser.add_argument('--timeout', type=int, default=3600,
                    help='Timeout in seconds. Default: 3600 (1 hour)')
parser.add_argument('--log', type=str, default='info',
                    help='Logging level. Options: debug, info, warning, error. Default: info')
parser.add_argument('--visualize', action='store_true',
                    help='Visualize tuning results. Default: False')

args = parser.parse_args()

# Set up logger
log_level = getattr(logging, args.log.upper())
logger = setup_logger("hyperparameter_tuning", level=log_level)

# Create hyperparameters directory if it doesn't exist
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)


def tune_cnn_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout: int = 3600,
    study_name: str = 'cnn_lstm_optimization',
    direction: str = 'maximize',
    metric: str = 'val_accuracy'
) -> dict:
    """
    Hyperparameter tuning for CNN-LSTM model.
    
    Args:
        X_train (np.ndarray): Training input sequences
        y_train (np.ndarray): Training target values
        X_val (np.ndarray): Validation input sequences
        y_val (np.ndarray): Validation target values
        n_trials (int, optional): Number of trials. Defaults to 50.
        timeout (int, optional): Timeout in seconds. Defaults to 3600.
        study_name (str, optional): Name of the study. Defaults to 'cnn_lstm_optimization'.
        direction (str, optional): Direction of optimization. Defaults to 'maximize'.
        metric (str, optional): Metric to optimize. Defaults to 'val_accuracy'.
        
    Returns:
        dict: Best hyperparameters
    """
    import tensorflow as tf
    from src.models.cnn_lstm import build_cnn_lstm_model
    
    def objective(trial: Trial) -> float:
        # Define hyperparameters to tune
        cnn_layers = [
            {
                'filters': trial.suggest_int('cnn_filters_1', 32, 128, step=32),
                'kernel_size': trial.suggest_int('cnn_kernel_1', 3, 7, step=2),
                'activation': 'relu'
            },
            {
                'filters': trial.suggest_int('cnn_filters_2', 64, 256, step=64),
                'kernel_size': trial.suggest_int('cnn_kernel_2', 3, 7, step=2),
                'activation': 'relu'
            }
        ]
        
        lstm_layers = [
            {
                'units': trial.suggest_int('lstm_units_1', 50, 200, step=50),
                'return_sequences': True
            },
            {
                'units': trial.suggest_int('lstm_units_2', 25, 100, step=25),
                'return_sequences': False
            }
        ]
        
        dense_layers = [
            {
                'units': trial.suggest_int('dense_units_1', 25, 100, step=25),
                'activation': 'relu'
            },
            {
                'units': trial.suggest_int('dense_units_2', 10, 50, step=10),
                'activation': 'relu'
            }
        ]
        
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-3, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Build model
        model = build_cnn_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            output_size=1,  # Binary classification
            cnn_layers=cnn_layers,
            lstm_layers=lstm_layers,
            dense_layers=dense_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            l1_reg=l1_reg,
            l2_reg=l2_reg
        )
        
        # Define callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Get best validation metric
        if metric == 'val_accuracy':
            best_value = max(history.history['val_accuracy'])
        elif metric == 'val_loss':
            best_value = min(history.history['val_loss'])
            # Convert to maximization problem
            best_value = -best_value
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_value
    
    # Create study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_params
    
    # Convert to the format expected by our model building function
    model_params = {
        'cnn_layers': [
            {
                'filters': best_params['cnn_filters_1'],
                'kernel_size': best_params['cnn_kernel_1'],
                'activation': 'relu'
            },
            {
                'filters': best_params['cnn_filters_2'],
                'kernel_size': best_params['cnn_kernel_2'],
                'activation': 'relu'
            }
        ],
        'lstm_layers': [
            {
                'units': best_params['lstm_units_1'],
                'return_sequences': True
            },
            {
                'units': best_params['lstm_units_2'],
                'return_sequences': False
            }
        ],
        'dense_layers': [
            {
                'units': best_params['dense_units_1'],
                'activation': 'relu'
            },
            {
                'units': best_params['dense_units_2'],
                'activation': 'relu'
            }
        ],
        'dropout_rate': best_params['dropout_rate'],
        'learning_rate': best_params['learning_rate'],
        'l1_reg': best_params['l1_reg'],
        'l2_reg': best_params['l2_reg'],
        'batch_size': best_params['batch_size']
    }
    
    # Save results
    results_path = os.path.join(HYPERPARAMS_DIR, f"{study_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': n_trials,
            'model_params': model_params
        }, f, indent=4)
    
    # Visualize if requested
    if args.visualize:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(HYPERPARAMS_DIR, f"{study_name}_history.png"), dpi=300)
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(os.path.join(HYPERPARAMS_DIR, f"{study_name}_importances.png"), dpi=300)
        except:
            logger.warning("Could not plot parameter importances")
        
        # Plot contour plots for pairs of important parameters
        plt.figure(figsize=(10, 6))
        try:
            param_names = list(best_params.keys())
            for i in range(min(3, len(param_names))):
                for j in range(i+1, min(4, len(param_names))):
                    plt.figure(figsize=(10, 6))
                    optuna.visualization.matplotlib.plot_contour(
                        study, 
                        params=[param_names[i], param_names[j]]
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(HYPERPARAMS_DIR, f"{study_name}_contour_{param_names[i]}_{param_names[j]}.png"), dpi=300)
        except:
            logger.warning("Could not plot contour plots")
    
    logger.info(f"Best CNN-LSTM parameters: {best_params}")
    logger.info(f"Best value: {study.best_value}")
    
    return model_params


def tune_tft(
    training_data,
    validation_data,
    n_trials: int = 50,
    timeout: int = 3600,
    study_name: str = 'tft_optimization',
    direction: str = 'minimize',
    metric: str = 'val_loss'
) -> dict:
    """
    Hyperparameter tuning for TFT model.
    
    Args:
        training_data: Training dataset for TFT
        validation_data: Validation dataset for TFT
        n_trials (int, optional): Number of trials. Defaults to 50.
        timeout (int, optional): Timeout in seconds. Defaults to 3600.
        study_name (str, optional): Name of the study. Defaults to 'tft_optimization'.
        direction (str, optional): Direction of optimization. Defaults to 'minimize'.
        metric (str, optional): Metric to optimize. Defaults to 'val_loss'.
        
    Returns:
        dict: Best hyperparameters
    """
    import pytorch_lightning as pl
    from pytorch_forecasting.metrics import QuantileLoss
    import torch
    
    def objective(trial: Trial) -> float:
        # Define hyperparameters to tune
        hidden_size = trial.suggest_int('hidden_size', 16, 64, step=16)
        attention_head_size = trial.suggest_int('attention_head_size', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        hidden_continuous_size = trial.suggest_int('hidden_continuous_size', 8, 32, step=8)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        
        # Create data loaders
        train_dataloader = training_data.to_dataloader(
            batch_size=batch_size,
            train=True,
            num_workers=os.cpu_count() // 2,
            pin_memory=True,
        )
        
        val_dataloader = validation_data.to_dataloader(
            batch_size=batch_size,
            train=False,
            num_workers=os.cpu_count() // 2,
            pin_memory=True,
        )
        
        # Create model
        from pytorch_forecasting import TemporalFusionTransformer
        
        model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=5,
        )
        
        # Create trainer
        early_stop = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=False,
            mode="min"
        )
        
        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="auto",
            enable_progress_bar=False,
            callbacks=[early_stop],
            gradient_clip_val=0.1,
            logger=False
        )
        
        # Train model
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Get best validation metric
        if metric == 'val_loss':
            best_value = trainer.callback_metrics.get('val_loss', torch.tensor(float('inf'))).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_value
    
    # Create study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_params
    
    # Save results
    results_path = os.path.join(HYPERPARAMS_DIR, f"{study_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': n_trials
        }, f, indent=4)
    
    # Visualize if requested
    if args.visualize:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(HYPERPARAMS_DIR, f"{study_name}_history.png"), dpi=300)
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(os.path.join(HYPERPARAMS_DIR, f"{study_name}_importances.png"), dpi=300)
        except:
            logger.warning("Could not plot parameter importances")
    
    logger.info(f"Best TFT parameters: {best_params}")
    logger.info(f"Best value: {study.best_value}")
    
    return best_params


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 100,
    timeout: int = 3600,
    study_name: str = 'xgboost_optimization',
    direction: str = 'maximize',
    metric: str = 'auc'
) -> dict:
    """
    Hyperparameter tuning for XGBoost model.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation targets
        n_trials (int, optional): Number of trials. Defaults to 100.
        timeout (int, optional): Timeout in seconds. Defaults to 3600.
        study_name (str, optional): Name of the study. Defaults to 'xgboost_optimization'.
        direction (str, optional): Direction of optimization. Defaults to 'maximize'.
        metric (str, optional): Metric to optimize. Defaults to 'auc'.
        
    Returns:
        dict: Best hyperparameters
    """
    import xgboost as xgb
    
    def objective(trial: Trial) -> float:
        # Define hyperparameters to tune
        params = {
            'objective': 'binary:logistic',
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.1),
            'gamma': trial.suggest_float('gamma', 0, 5, step=0.1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5, step=0.1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 5, step=0.1),
            'random_state': 42,
            'use_label_encoder': False,
            'verbosity': 0
        }
        
        # Create model
        model = xgb.XGBClassifier(**params)
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['logloss', 'auc', 'error'],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Get best validation metric
        if metric == 'auc':
            best_value = model.evals_result()['validation_1']['auc'][-1]
        elif metric == 'logloss':
            best_value = -model.evals_result()['validation_1']['logloss'][-1]  # Convert to maximization
        elif metric == 'error':
            best_value = -model.evals_result()['validation_1']['error'][-1]  # Convert to maximization
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_value
    
    # Create study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        pruner=pruner
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['random_state'] = 42
    best_params['use_label_encoder'] = False
    
    # Save results
    results_path = os.path.join(HYPERPARAMS_DIR, f"{study_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_value': study.best_value,
            'n_trials': n_trials
        }, f, indent=4)
    
    # Visualize if requested
    if args.visualize:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(HYPERPARAMS_DIR, f"{study_name}_history.png"), dpi=300)
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        try:
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(os.path.join(HYPERPARAMS_DIR, f"{study_name}_importances.png"), dpi=300)
        except:
            logger.warning("Could not plot parameter importances")
        
        # Plot contour plots for pairs of important parameters
        plt.figure(figsize=(10, 6))
        try:
            param_names = list(best_params.keys())
            for i in range(min(3, len(param_names))):
                for j in range(i+1, min(4, len(param_names))):
                    plt.figure(figsize=(10, 6))
                    optuna.visualization.matplotlib.plot_contour(
                        study, 
                        params=[param_names[i], param_names[j]]
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(HYPERPARAMS_DIR, f"{study_name}_contour_{param_names[i]}_{param_names[j]}.png"), dpi=300)
        except:
            logger.warning("Could not plot contour plots")
    
    logger.info(f"Best XGBoost parameters: {best_params}")
    logger.info(f"Best value: {study.best_value}")
    
    return best_params


def run_hyperparameter_tuning():
    """Run hyperparameter tuning for the specified model and currency pair."""
    logger.info(f"Starting hyperparameter tuning for {args.model} on {args.currency}")
    
    # Record start time for performance tracking
    start_time = time.time()
    
    from src.features.feature_enhancement import load_feature_data
    
    try:
        # Load feature data
        train_data = load_feature_data(args.currency, 'train')
        val_data = load_feature_data(args.currency, 'val')
        test_data = load_feature_data(args.currency, 'test')  # Needed for some models
    except FileNotFoundError as e:
        logger.error(f"Feature data for {args.currency} not found. Run feature enhancement first. Error: {str(e)}")
        return
    
    # Run tuning based on model type
    if args.model == 'CNN-LSTM':
        from src.models.cnn_lstm import create_sequences
        
        # Prepare sequences
        X_train, y_train = create_sequences(
            train_data,
            target_col='target',
            sequence_length=60,  # Default value, can be tuned as well
            prediction_steps=1,
            shuffle=True
        )
        
        X_val, y_val = create_sequences(
            val_data,
            target_col='target',
            sequence_length=60,
            prediction_steps=1,
            shuffle=False
        )
        
        # Run tuning
        best_params = tune_cnn_lstm(
            X_train, y_train,
            X_val, y_val,
            n_trials=args.trials,
            timeout=args.timeout,
            study_name=f"cnn_lstm_{args.currency}_tuning"
        )
    
    elif args.model == 'TFT':
        from src.models.tft import prepare_data_for_tft
        
        # Prepare data for TFT
        training, validation, testing, data_params = prepare_data_for_tft(
            train_data,
            val_data,
            test_data,
            target_variable='target',
            max_encoder_length=24,  # Default value, can be tuned as well
            max_prediction_length=1
        )
        
        # Run tuning
        best_params = tune_tft(
            training,
            validation,
            n_trials=args.trials,
            timeout=args.timeout,
            study_name=f"tft_{args.currency}_tuning"
        )
    
    elif args.model == 'XGBoost':
        from src.models.xgboost_model import prepare_data_for_xgboost
        
        # Prepare data for XGBoost
        X_train, y_train, feature_names = prepare_data_for_xgboost(
            train_data,
            target_col='target',
            sequence_length=1,  # Default value for XGBoost
            prediction_steps=1,
            shuffle=True
        )
        
        X_val, y_val, _ = prepare_data_for_xgboost(
            val_data,
            target_col='target',
            sequence_length=1,
            prediction_steps=1,
            shuffle=False
        )
        
        # Run tuning
        best_params = tune_xgboost(
            X_train, y_train,
            X_val, y_val,
            n_trials=args.trials,
            timeout=args.timeout,
            study_name=f"xgboost_{args.currency}_tuning"
        )
    
    # Record end time and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best parameters saved to {os.path.join(HYPERPARAMS_DIR, f'{args.model.lower()}_{args.currency}_tuning_results.json')}")


if __name__ == "__main__":
    run_hyperparameter_tuning()