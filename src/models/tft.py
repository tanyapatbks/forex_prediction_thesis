"""
Temporal Fusion Transformer (TFT) model for Forex prediction.
"""

import os
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE, RMSE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import matplotlib.pyplot as plt
import pickle

from src.utils.config import TRAINED_MODELS_DIR
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger("tft_model")


class TFTModule(pl.LightningModule):
    """
    PyTorch Lightning module for TFT model training.
    This wrapper is mainly for hyperparameter tuning compatibility.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self.training_set = None
        self.validation_set = None
        self.target_variable = kwargs.get('target_variable', 'target')
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        
    def setup_model(self, training_set, validation_set):
        """Set up the TFT model with datasets"""
        self.training_set = training_set
        self.validation_set = validation_set
        
        # Define the model
        self.model = TemporalFusionTransformer.from_dataset(
            training_set,
            learning_rate=self.learning_rate,
            hidden_size=self.hparams.get('hidden_size', 32),
            attention_head_size=self.hparams.get('attention_head_size', 4),
            dropout=self.hparams.get('dropout', 0.1),
            hidden_continuous_size=self.hparams.get('hidden_continuous_size', 16),
            loss=QuantileLoss(),
            log_interval=self.hparams.get('log_interval', 10),
            reduce_on_plateau_patience=self.hparams.get('reduce_on_plateau_patience', 5),
        )
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)


def prepare_data_for_tft(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_variable: str = 'target',
    time_idx_column: str = None,
    static_categoricals: List[str] = None,
    static_reals: List[str] = None,
    time_varying_known_categoricals: List[str] = None,
    time_varying_known_reals: List[str] = None,
    time_varying_unknown_categoricals: List[str] = None,
    time_varying_unknown_reals: List[str] = None,
    max_encoder_length: int = 24,
    max_prediction_length: int = 1,
    group_ids: List[str] = None,
    min_encoder_length: int = None,
    target_normalizer: GroupNormalizer = None
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, Dict]:
    """
    Prepare Forex data for the TFT model.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        target_variable (str, optional): Target variable name. Defaults to 'target'.
        time_idx_column (str, optional): Column name for time index. If None, one will be created.
        static_categoricals (List[str], optional): Static categorical variables. Defaults to None.
        static_reals (List[str], optional): Static real variables. Defaults to None.
        time_varying_known_categoricals (List[str], optional): Known time-varying categorical variables. Defaults to None.
        time_varying_known_reals (List[str], optional): Known time-varying real variables. Defaults to None.
        time_varying_unknown_categoricals (List[str], optional): Unknown time-varying categorical variables. Defaults to None.
        time_varying_unknown_reals (List[str], optional): Unknown time-varying real variables. Defaults to None.
        max_encoder_length (int, optional): Maximum encoder length. Defaults to 24.
        max_prediction_length (int, optional): Maximum prediction length. Defaults to 1.
        group_ids (List[str], optional): Group IDs. Defaults to None.
        min_encoder_length (int, optional): Minimum encoder length. Defaults to None.
        target_normalizer (GroupNormalizer, optional): Target normalizer. Defaults to None.
        
    Returns:
        Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, Dict]: Training, validation, and test datasets, and data parameters
    """
    # Create a time index if not provided
    if time_idx_column is None:
        logger.info("Creating time index column")
        # Reset index to get the datetime as a column
        train_df = train_df.reset_index()
        val_df = val_df.reset_index()
        test_df = test_df.reset_index()
        
        # Create a time index column
        time_idx_column = 'time_idx'
        
        # Create a numeric time index
        for df in [train_df, val_df, test_df]:
            df[time_idx_column] = range(len(df))
    
    # Default to a single group if not provided
    if group_ids is None:
        group_id_column = 'group_id'
        for df in [train_df, val_df, test_df]:
            df[group_id_column] = 0  # single group
        group_ids = [group_id_column]
    
    # Set default variable groups if not provided
    if static_categoricals is None:
        static_categoricals = []
    
    if static_reals is None:
        static_reals = []
    
    if time_varying_known_categoricals is None:
        time_varying_known_categoricals = []
    
    if time_varying_known_reals is None:
        time_varying_known_reals = []
    
    # Cyclical time features often make good known variables
    time_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
                    'month_sin', 'month_cos']
    
    for feature in time_features:
        if feature in train_df.columns:
            if feature not in time_varying_known_reals:
                time_varying_known_reals.append(feature)
    
    if time_varying_unknown_categoricals is None:
        time_varying_unknown_categoricals = []
    
    if time_varying_unknown_reals is None:
        time_varying_unknown_reals = []
        # Add all numeric columns except target and time columns
        for col in train_df.select_dtypes(include=['float', 'int']).columns:
            if (col != target_variable and 
                col != time_idx_column and 
                col not in static_reals and
                col not in static_categoricals and
                col not in time_varying_known_reals and
                col not in time_varying_known_categoricals and
                col not in time_varying_unknown_categoricals and
                col not in group_ids):
                time_varying_unknown_reals.append(col)
    
    # Set minimum encoder length if not provided
    if min_encoder_length is None:
        min_encoder_length = max_encoder_length // 2
    
    # Set target normalizer if not provided
    if target_normalizer is None:
        target_normalizer = GroupNormalizer(
            groups=group_ids,
            transformation="softplus"
        )
    
    # Create training dataset
    training = TimeSeriesDataSet(
        data=train_df,
        time_idx=time_idx_column,
        target=target_variable,
        group_ids=group_ids,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_encoder_length=min_encoder_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset based on training dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        val_df, 
        stop_randomization=True
    )
    
    # Create test dataset based on training dataset
    testing = TimeSeriesDataSet.from_dataset(
        training, 
        test_df, 
        stop_randomization=True
    )
    
    # Store data parameters for future reference
    data_params = {
        'time_idx_column': time_idx_column,
        'group_ids': group_ids,
        'static_categoricals': static_categoricals,
        'static_reals': static_reals,
        'time_varying_known_categoricals': time_varying_known_categoricals,
        'time_varying_known_reals': time_varying_known_reals,
        'time_varying_unknown_categoricals': time_varying_unknown_categoricals,
        'time_varying_unknown_reals': time_varying_unknown_reals,
        'max_encoder_length': max_encoder_length,
        'max_prediction_length': max_prediction_length,
        'min_encoder_length': min_encoder_length,
        'target_variable': target_variable
    }
    
    logger.info(f"Created TimeSeriesDataSets with {len(training)} training, {len(validation)} validation, and {len(testing)} testing samples")
    
    return training, validation, testing, data_params


def train_tft_model(
    training_data: TimeSeriesDataSet,
    validation_data: TimeSeriesDataSet,
    model_config: Dict = None,
    training_config: Dict = None,
    model_name: str = 'tft',
    save_model: bool = True
) -> Tuple[TFTModule, Dict]:
    """
    Train a TFT model for Forex prediction.
    
    Args:
        training_data (TimeSeriesDataSet): Training dataset
        validation_data (TimeSeriesDataSet): Validation dataset
        model_config (Dict, optional): Model configuration. Defaults to None.
        training_config (Dict, optional): Training configuration. Defaults to None.
        model_name (str, optional): Name for the saved model. Defaults to 'tft'.
        save_model (bool, optional): Whether to save the trained model. Defaults to True.
        
    Returns:
        Tuple[TFTModule, Dict]: Trained model and training history
    """
    # Use default configurations if not provided
    if model_config is None:
        model_config = {
            'hidden_size': 32,
            'attention_head_size': 4,
            'dropout': 0.1,
            'hidden_continuous_size': 16,
            'learning_rate': 0.001,
            'log_interval': 10,
            'reduce_on_plateau_patience': 5,
        }
    
    if training_config is None:
        training_config = {
            'batch_size': 128,
            'max_epochs': 100,
            'patience': 10,
            'min_delta': 0.001,
            'gradient_clip_val': 0.1,
        }
    
    # Create data loaders
    train_dataloader = training_data.to_dataloader(
        batch_size=training_config['batch_size'],
        train=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )
    
    val_dataloader = validation_data.to_dataloader(
        batch_size=training_config['batch_size'],
        train=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )
    
    # Create lightning module
    pl_module = TFTModule(**model_config)
    pl_module.setup_model(training_data, validation_data)
    
    # Create callbacks
    callbacks = []
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=training_config['min_delta'],
        patience=training_config['patience'],
        verbose=True,
        mode="min"
    )
    callbacks.append(early_stop)
    
    # Model checkpoint
    if save_model:
        os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
        checkpoint_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_checkpoint")
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True
        )
        callbacks.append(checkpoint)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config['max_epochs'],
        accelerator="auto",  # Auto-detect GPU/CPU
        gradient_clip_val=training_config['gradient_clip_val'],
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=True,
    )
    
    # Train model
    trainer.fit(
        pl_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Load best model
    if save_model and checkpoint.best_model_path:
        pl_module = TFTModule.load_from_checkpoint(checkpoint.best_model_path)
        pl_module.setup_model(training_data, validation_data)
    
    # Save model and configuration
    if save_model:
        # Save TFT model
        model_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}.pth")
        torch.save(pl_module.model.state_dict(), model_path)
        
        # Save the complete module with pickle
        module_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_module.pkl")
        with open(module_path, 'wb') as f:
            pickle.dump(pl_module, f)
        
        # Save training and validation datasets
        datasets_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_datasets.pkl")
        with open(datasets_path, 'wb') as f:
            pickle.dump((training_data, validation_data), f)
        
        # Save configurations
        config_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': model_config,
                'training_config': training_config,
            }, f, indent=4)
        
        logger.info(f"Saved trained TFT model to {model_path}")
    
    # Extract training history
    history = {
        'train_loss': trainer.callback_metrics.get('train_loss', 0.0),
        'val_loss': trainer.callback_metrics.get('val_loss', 0.0),
    }
    
    logger.info("TFT model training completed")
    
    return pl_module, history


def evaluate_tft_model(
    model: TFTModule,
    test_data: TimeSeriesDataSet,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    Evaluate a trained TFT model.
    
    Args:
        model (TFTModule): Trained TFT model
        test_data (TimeSeriesDataSet): Test dataset
        return_predictions (bool, optional): Whether to return predictions. Defaults to False.
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    # Create test dataloader
    test_dataloader = test_data.to_dataloader(
        batch_size=128,
        train=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )
    
    # Get predictions
    trainer = pl.Trainer(
        accelerator="auto",
        enable_progress_bar=True,
    )
    
    # Get predictions
    predictions = trainer.predict(model, dataloaders=test_dataloader)
    
    # Concatenate predictions
    y_hat = torch.cat([p.prediction for p in predictions])
    
    # Get actuals (this is a bit hacky but necessary to get the actuals)
    actuals = torch.cat([p.x["encoder_target"][Ellipsis, -1] for p in predictions])
    
    # Convert to binary predictions if needed
    y_pred = (y_hat > 0.5).float()
    
    # Calculate metrics
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = ((y_pred == actuals).float().mean()).item()
    
    # Create an array for storing intermediate 0-1 predictions for calculating other metrics
    y_pred_np = y_pred.cpu().numpy()
    actuals_np = actuals.cpu().numpy()
    
    # Calculate additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics['sklearn_accuracy'] = accuracy_score(actuals_np, y_pred_np)
    metrics['precision'] = precision_score(actuals_np, y_pred_np, zero_division=0)
    metrics['recall'] = recall_score(actuals_np, y_pred_np, zero_division=0)
    metrics['f1_score'] = f1_score(actuals_np, y_pred_np, zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(actuals_np, y_hat.cpu().numpy())
    except:
        metrics['roc_auc'] = 0.5  # Default value if AUC cannot be calculated
    
    logger.info(f"TFT model evaluation: {metrics}")
    
    if return_predictions:
        return metrics, (y_hat.cpu().numpy(), actuals_np)
    else:
        return metrics


def load_tft_model(model_name: str) -> Tuple[TFTModule, Tuple[TimeSeriesDataSet, TimeSeriesDataSet]]:
    """
    Load a trained TFT model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Tuple[TFTModule, Tuple[TimeSeriesDataSet, TimeSeriesDataSet]]: Loaded model and datasets
    """
    module_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_module.pkl")
    datasets_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_datasets.pkl")
    
    if not os.path.exists(module_path) or not os.path.exists(datasets_path):
        error_msg = f"Model or datasets file not found: {module_path}, {datasets_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load module
    with open(module_path, 'rb') as f:
        pl_module = pickle.load(f)
    
    # Load datasets
    with open(datasets_path, 'rb') as f:
        training_data, validation_data = pickle.load(f)
    
    logger.info(f"Loaded TFT model from {module_path}")
    
    return pl_module, (training_data, validation_data)


def predict_with_tft(
    model: TFTModule,
    data: TimeSeriesDataSet,
    return_x: bool = False
) -> np.ndarray:
    """
    Make predictions using a trained TFT model.
    
    Args:
        model (TFTModule): Trained TFT model
        data (TimeSeriesDataSet): Data to predict
        return_x (bool, optional): Whether to return the input data. Defaults to False.
        
    Returns:
        np.ndarray: Predicted values
    """
    # Create dataloader
    dataloader = data.to_dataloader(
        batch_size=128,
        train=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )
    
    # Get predictions
    trainer = pl.Trainer(
        accelerator="auto",
        enable_progress_bar=True,
    )
    
    predictions = trainer.predict(model, dataloaders=dataloader)
    
    # Convert predictions to numpy array
    y_hat = torch.cat([p.prediction for p in predictions]).cpu().numpy()
    
    if return_x:
        # Get input data
        x = torch.cat([p.x["encoder_target"][Ellipsis, -1] for p in predictions]).cpu().numpy()
        return y_hat, x
    else:
        return y_hat