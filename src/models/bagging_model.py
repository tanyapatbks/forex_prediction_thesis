"""
Bagging models for Forex prediction.
Uses data from multiple currency pairs to improve model robustness.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from src.utils.config import TRAINED_MODELS_DIR, CURRENCY_PAIRS
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger("bagging_model")


def combine_datasets(
    datasets: Dict[str, pd.DataFrame],
    add_pair_identifier: bool = True
) -> pd.DataFrame:
    """
    Combine datasets from multiple currency pairs.
    
    Args:
        datasets (Dict[str, pd.DataFrame]): Dictionary mapping currency pair names to DataFrames
        add_pair_identifier (bool, optional): Whether to add a pair identifier column. Defaults to True.
        
    Returns:
        pd.DataFrame: Combined dataset
    """
    combined_data = []
    
    for pair_name, df in datasets.items():
        # Create a copy of the dataframe
        temp_df = df.copy()
        
        # Add pair identifier if requested
        if add_pair_identifier:
            temp_df['pair'] = pair_name
        
        # Add to list
        combined_data.append(temp_df)
    
    # Concatenate all dataframes
    result = pd.concat(combined_data, axis=0)
    
    logger.info(f"Combined {len(datasets)} datasets with total shape: {result.shape}")
    
    return result


def split_by_pair(
    data: pd.DataFrame,
    pair_column: str = 'pair'
) -> Dict[str, pd.DataFrame]:
    """
    Split a combined dataset back into separate dataframes by currency pair.
    
    Args:
        data (pd.DataFrame): Combined DataFrame
        pair_column (str, optional): Column name with pair identifiers. Defaults to 'pair'.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping currency pair names to DataFrames
    """
    if pair_column not in data.columns:
        error_msg = f"Pair column '{pair_column}' not found in data"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get unique pairs
    pairs = data[pair_column].unique()
    
    # Split data
    result = {}
    for pair in pairs:
        pair_data = data[data[pair_column] == pair].copy()
        # Drop pair column
        if pair_column in pair_data.columns:
            pair_data = pair_data.drop(columns=[pair_column])
        result[pair] = pair_data
    
    logger.info(f"Split dataset into {len(result)} pair-specific datasets")
    
    return result


def prepare_bagging_data(
    currency_pairs_data: Dict[str, Dict[str, pd.DataFrame]],
    dataset_type: str = 'train',
    add_pair_identifier: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for bagging by combining data from multiple currency pairs.
    
    Args:
        currency_pairs_data (Dict[str, Dict[str, pd.DataFrame]]): Dictionary with data for each currency pair:
            - Outer key: Currency pair code
            - Inner key: Dataset type ('train', 'val', 'test')
            - Value: DataFrame with features and target
        dataset_type (str, optional): Type of dataset to prepare. Defaults to 'train'.
        add_pair_identifier (bool, optional): Whether to add a pair identifier column. Defaults to True.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with combined datasets:
            - Key: Dataset type ('train', 'val', 'test')
            - Value: Combined DataFrame
    """
    # Extract datasets of the specified type
    datasets = {}
    for pair_name, pair_data in currency_pairs_data.items():
        if dataset_type in pair_data:
            datasets[pair_name] = pair_data[dataset_type]
    
    # Combine datasets
    combined_data = combine_datasets(datasets, add_pair_identifier)
    
    result = {dataset_type: combined_data}
    
    return result


def prepare_all_bagging_data(
    currency_pairs_data: Dict[str, Dict[str, pd.DataFrame]],
    add_pair_identifier: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Prepare all data types for bagging.
    
    Args:
        currency_pairs_data (Dict[str, Dict[str, pd.DataFrame]]): Dictionary with data for each currency pair
        add_pair_identifier (bool, optional): Whether to add a pair identifier column. Defaults to True.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with combined datasets for all types
    """
    result = {}
    
    # Process each dataset type
    for dataset_type in ['train', 'val', 'test']:
        # Extract datasets of the specified type
        datasets = {}
        for pair_name, pair_data in currency_pairs_data.items():
            if dataset_type in pair_data:
                datasets[pair_name] = pair_data[dataset_type]
        
        # Combine datasets if any exist
        if datasets:
            result[dataset_type] = combine_datasets(datasets, add_pair_identifier)
        else:
            logger.warning(f"No datasets found for type '{dataset_type}'")
    
    logger.info(f"Prepared bagging data for {len(result)} dataset types")
    
    return result


def bootstrap_sample(
    data: pd.DataFrame,
    n_samples: int = None,
    random_state: int = None
) -> pd.DataFrame:
    """
    Create a bootstrap sample from the data.
    
    Args:
        data (pd.DataFrame): Input data
        n_samples (int, optional): Number of samples to draw. If None, same as data size. 
                                 Defaults to None.
        random_state (int, optional): Random seed. Defaults to None.
        
    Returns:
        pd.DataFrame: Bootstrap sample
    """
    if n_samples is None:
        n_samples = len(data)
    
    # Create bootstrap sample
    sample = resample(
        data,
        replace=True,
        n_samples=n_samples,
        random_state=random_state
    )
    
    return sample


def create_bootstrap_samples(
    data: pd.DataFrame,
    n_bootstraps: int = 10,
    sample_size: float = 1.0,
    random_state: int = None
) -> List[pd.DataFrame]:
    """
    Create multiple bootstrap samples from the data.
    
    Args:
        data (pd.DataFrame): Input data
        n_bootstraps (int, optional): Number of bootstrap samples to create. Defaults to 10.
        sample_size (float, optional): Size of each sample relative to data size. Defaults to 1.0.
        random_state (int, optional): Random seed. Defaults to None.
        
    Returns:
        List[pd.DataFrame]: List of bootstrap samples
    """
    # Calculate sample size
    n_samples = int(len(data) * sample_size)
    
    # Create bootstrap samples
    samples = []
    for i in range(n_bootstraps):
        # Set seed for reproducibility, but different for each bootstrap
        seed = None if random_state is None else random_state + i
        
        # Create sample
        sample = bootstrap_sample(data, n_samples, seed)
        samples.append(sample)
    
    logger.info(f"Created {n_bootstraps} bootstrap samples of size {n_samples}")
    
    return samples


class BaggingWrapper:
    """
    Wrapper class for bagging models.
    Combines predictions from multiple models.
    """
    
    def __init__(self, model_type: str = 'CNN-LSTM', currency_pairs: List[str] = None):
        """
        Initialize the BaggingWrapper.
        
        Args:
            model_type (str, optional): Type of models to bag. Defaults to 'CNN-LSTM'.
            currency_pairs (List[str], optional): List of currency pairs to include. Defaults to None.
        """
        self.model_type = model_type
        self.currency_pairs = currency_pairs or CURRENCY_PAIRS
        self.models = {}
        self.feature_names = None
    
    def add_model(self, pair_name: str, model: Any, feature_names: List[str] = None):
        """
        Add a model to the bagging ensemble.
        
        Args:
            pair_name (str): Currency pair name
            model (Any): Trained model
            feature_names (List[str], optional): Feature names for the model. Defaults to None.
        """
        self.models[pair_name] = model
        if feature_names is not None and self.feature_names is None:
            self.feature_names = feature_names
    
    def predict(self, X: np.ndarray, mode: str = 'majority', threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions by combining predictions from all models.
        
        Args:
            X (np.ndarray): Input features
            mode (str, optional): Aggregation mode. Options: 'majority', 'average'. Defaults to 'majority'.
            threshold (float, optional): Threshold for converting probabilities to binary predictions. 
                                      Defaults to 0.5.
        
        Returns:
            np.ndarray: Aggregated predictions
        """
        if not self.models:
            error_msg = "No models available for prediction"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from each model
        for pair_name, model in self.models.items():
            if self.model_type == 'CNN-LSTM':
                from src.models.cnn_lstm import predict_with_cnn_lstm
                pred, prob = predict_with_cnn_lstm(model, X, threshold)
                all_predictions.append(pred)
                all_probabilities.append(prob)
            
            elif self.model_type == 'TFT':
                from src.models.tft import predict_with_tft
                pred = predict_with_tft(model, X)
                binary_pred = (pred > threshold).astype(int)
                all_predictions.append(binary_pred)
                all_probabilities.append(pred)
            
            elif self.model_type == 'XGBoost':
                from src.models.xgboost_model import predict_with_xgboost
                prob = predict_with_xgboost(model, X, return_probabilities=True)
                pred = (prob > threshold).astype(int)
                all_predictions.append(pred)
                all_probabilities.append(prob)
            
            else:
                error_msg = f"Unsupported model type: {self.model_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Stack predictions and probabilities
        all_predictions = np.stack(all_predictions, axis=0)
        all_probabilities = np.stack(all_probabilities, axis=0)
        
        # Aggregate predictions
        if mode == 'majority':
            # Use majority voting
            final_predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
        else:
            # Use probability averaging
            final_probabilities = np.mean(all_probabilities, axis=0)
            final_predictions = (final_probabilities > threshold).astype(int)
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions by averaging probabilities from all models.
        
        Args:
            X (np.ndarray): Input features
        
        Returns:
            np.ndarray: Averaged probability predictions
        """
        if not self.models:
            error_msg = "No models available for prediction"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        all_probabilities = []
        
        # Get probability predictions from each model
        for pair_name, model in self.models.items():
            if self.model_type == 'CNN-LSTM':
                from src.models.cnn_lstm import predict_with_cnn_lstm
                _, prob = predict_with_cnn_lstm(model, X, 0.5)
                all_probabilities.append(prob)
            
            elif self.model_type == 'TFT':
                from src.models.tft import predict_with_tft
                pred = predict_with_tft(model, X)
                all_probabilities.append(pred)
            
            elif self.model_type == 'XGBoost':
                from src.models.xgboost_model import predict_with_xgboost
                prob = predict_with_xgboost(model, X, return_probabilities=True)
                all_probabilities.append(prob)
            
            else:
                error_msg = f"Unsupported model type: {self.model_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Stack probabilities
        all_probabilities = np.stack(all_probabilities, axis=0)
        
        # Average probabilities
        avg_probabilities = np.mean(all_probabilities, axis=0)
        
        return avg_probabilities
    
    def save(self, model_name: str = None):
        """
        Save the bagging ensemble.
        
        Args:
            model_name (str, optional): Name for the saved ensemble. Defaults to None.
        """
        if model_name is None:
            model_name = f"bagging_{self.model_type.lower()}"
        
        os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
        
        # Save wrapper metadata
        metadata = {
            'model_type': self.model_type,
            'currency_pairs': self.currency_pairs,
        }
        
        metadata_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save feature names if available
        if self.feature_names is not None:
            feature_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_features.json")
            with open(feature_path, 'w') as f:
                json.dump(self.feature_names, f, indent=4)
        
        logger.info(f"Saved bagging ensemble metadata to {metadata_path}")
    
    @classmethod
    def load(cls, model_name: str, load_models: bool = True) -> 'BaggingWrapper':
        """
        Load a saved bagging ensemble.
        
        Args:
            model_name (str): Name of the saved ensemble
            load_models (bool, optional): Whether to load the individual models. Defaults to True.
        
        Returns:
            BaggingWrapper: Loaded bagging ensemble
        """
        metadata_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_metadata.json")
        
        if not os.path.exists(metadata_path):
            error_msg = f"Metadata file not found: {metadata_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create wrapper
        wrapper = cls(
            model_type=metadata['model_type'],
            currency_pairs=metadata['currency_pairs']
        )
        
        # Load feature names if available
        feature_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_features.json")
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                wrapper.feature_names = json.load(f)
        
        # Load individual models if requested
        if load_models:
            for pair_name in metadata['currency_pairs']:
                pair_model_name = f"{metadata['model_type'].lower()}_{pair_name}"
                
                if metadata['model_type'] == 'CNN-LSTM':
                    from src.models.cnn_lstm import load_cnn_lstm_model
                    try:
                        model, _ = load_cnn_lstm_model(pair_model_name)
                        wrapper.add_model(pair_name, model)
                    except Exception as e:
                        logger.warning(f"Failed to load model for {pair_name}: {str(e)}")
                
                elif metadata['model_type'] == 'TFT':
                    from src.models.tft import load_tft_model
                    try:
                        model, _ = load_tft_model(pair_model_name)
                        wrapper.add_model(pair_name, model)
                    except Exception as e:
                        logger.warning(f"Failed to load model for {pair_name}: {str(e)}")
                
                elif metadata['model_type'] == 'XGBoost':
                    from src.models.xgboost_model import load_xgboost_model
                    try:
                        model, _, feature_names = load_xgboost_model(pair_model_name)
                        wrapper.add_model(pair_name, model, feature_names)
                    except Exception as e:
                        logger.warning(f"Failed to load model for {pair_name}: {str(e)}")
                
                else:
                    error_msg = f"Unsupported model type: {metadata['model_type']}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
        
        logger.info(f"Loaded bagging ensemble from {metadata_path}")
        
        return wrapper


def create_bagging_model(
    model_type: str,
    currency_pairs: List[str],
    model_name: str = None
) -> BaggingWrapper:
    """
    Create a bagging model by loading individual models for each currency pair.
    
    Args:
        model_type (str): Type of models to bag ('CNN-LSTM', 'TFT', or 'XGBoost')
        currency_pairs (List[str]): List of currency pairs to include
        model_name (str, optional): Name for the saved ensemble. Defaults to None.
        
    Returns:
        BaggingWrapper: Bagging ensemble
    """
    # Create wrapper
    wrapper = BaggingWrapper(model_type, currency_pairs)
    
    # Load models for each currency pair
    for pair_name in currency_pairs:
        pair_model_name = f"{model_type.lower()}_{pair_name}"
        
        if model_type == 'CNN-LSTM':
            from src.models.cnn_lstm import load_cnn_lstm_model
            try:
                model, _ = load_cnn_lstm_model(pair_model_name)
                wrapper.add_model(pair_name, model)
            except Exception as e:
                logger.warning(f"Failed to load model for {pair_name}: {str(e)}")
        
        elif model_type == 'TFT':
            from src.models.tft import load_tft_model
            try:
                model, _ = load_tft_model(pair_model_name)
                wrapper.add_model(pair_name, model)
            except Exception as e:
                logger.warning(f"Failed to load model for {pair_name}: {str(e)}")
        
        elif model_type == 'XGBoost':
            from src.models.xgboost_model import load_xgboost_model
            try:
                model, _, feature_names = load_xgboost_model(pair_model_name)
                wrapper.add_model(pair_name, model, feature_names)
            except Exception as e:
                logger.warning(f"Failed to load model for {pair_name}: {str(e)}")
        
        else:
            error_msg = f"Unsupported model type: {model_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Save the ensemble
    if model_name is None:
        model_name = f"bagging_{model_type.lower()}"
    
    wrapper.save(model_name)
    
    logger.info(f"Created bagging ensemble with {len(wrapper.models)} models")
    
    return wrapper


def evaluate_bagging_model(
    model: BaggingWrapper,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mode: str = 'majority',
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a bagging model.
    
    Args:
        model (BaggingWrapper): Bagging model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        mode (str, optional): Aggregation mode. Defaults to 'majority'.
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test, mode, threshold)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    except:
        metrics['roc_auc'] = 0.5  # Default value if AUC cannot be calculated
    
    logger.info(f"Bagging model evaluation: {metrics}")
    
    return metrics


def compare_with_individual_models(
    bagging_model: BaggingWrapper,
    individual_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Compare bagging model with individual models.
    
    Args:
        bagging_model (BaggingWrapper): Bagging model
        individual_models (Dict[str, Any]): Dictionary mapping pair names to individual models
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        model_type (str): Type of models ('CNN-LSTM', 'TFT', or 'XGBoost')
        threshold (float, optional): Decision threshold. Defaults to 0.5.
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping model names to their metrics
    """
    # Evaluate bagging model
    bagging_metrics = evaluate_bagging_model(bagging_model, X_test, y_test, threshold=threshold)
    
    # Evaluate individual models
    individual_metrics = {}
    
    for pair_name, model in individual_models.items():
        if model_type == 'CNN-LSTM':
            from src.models.cnn_lstm import evaluate_cnn_lstm_model
            metrics = evaluate_cnn_lstm_model(model, X_test, y_test)
        
        elif model_type == 'TFT':
            from src.models.tft import evaluate_tft_model
            metrics = evaluate_tft_model(model, X_test, y_test)
        
        elif model_type == 'XGBoost':
            from src.models.xgboost_model import evaluate_xgboost_model
            metrics = evaluate_xgboost_model(model, X_test, y_test)
        
        else:
            error_msg = f"Unsupported model type: {model_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        individual_metrics[pair_name] = metrics
    
    # Combine results
    all_metrics = {
        'bagging': bagging_metrics,
        **individual_metrics
    }
    
    return all_metrics


def plot_model_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'accuracy',
    title: str = None,
    save_path: str = None,
    show_plot: bool = True
):
    """
    Plot comparison of models based on a specific metric.
    
    Args:
        metrics (Dict[str, Dict[str, float]]): Dictionary mapping model names to their metrics
        metric_name (str, optional): Metric to compare. Defaults to 'accuracy'.
        title (str, optional): Plot title. Defaults to None.
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    # Extract metric values
    models = []
    values = []
    
    for model_name, model_metrics in metrics.items():
        if metric_name in model_metrics:
            models.append(model_name)
            values.append(model_metrics[metric_name])
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Model': models,
        metric_name: values
    })
    
    # Sort by metric value
    df = df.sort_values(metric_name, ascending=False)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df['Model'], df[metric_name])
    
    # Color the 'bagging' bar differently
    for i, model in enumerate(df['Model']):
        if model == 'bagging':
            bars[i].set_color('green')
        else:
            bars[i].set_color('blue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height * 1.01,
            f'{height:.4f}',
            ha='center',
            va='bottom'
        )
    
    # Set title and labels
    if title is None:
        title = f'Comparison of Models ({metric_name.capitalize()})'
    
    plt.title(title)
    plt.ylabel(metric_name.capitalize())
    plt.ylim(0, min(1.0, max(values) * 1.2))  # Set y-limit
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def train_bagging_model(
    combined_data: Dict[str, pd.DataFrame],
    model_type: str,
    target_col: str = 'target',
    sequence_length: int = 60,
    prediction_steps: int = 1,
    n_bootstraps: int = 5,
    sample_size: float = 0.8,
    model_config: Dict = None,
    training_config: Dict = None,
    model_name: str = None,
    random_state: int = 42
) -> BaggingWrapper:
    """
    Train a bagging model using bootstrap samples from combined data.
    
    Args:
        combined_data (Dict[str, pd.DataFrame]): Dictionary with combined datasets
        model_type (str): Type of models to use ('CNN-LSTM', 'TFT', or 'XGBoost')
        target_col (str, optional): Target column name. Defaults to 'target'.
        sequence_length (int, optional): Length of input sequences. Defaults to 60.
        prediction_steps (int, optional): Number of steps ahead to predict. Defaults to 1.
        n_bootstraps (int, optional): Number of bootstrap samples to create. Defaults to 5.
        sample_size (float, optional): Size of each sample relative to data size. Defaults to 0.8.
        model_config (Dict, optional): Model configuration. Defaults to None.
        training_config (Dict, optional): Training configuration. Defaults to None.
        model_name (str, optional): Name for the saved ensemble. Defaults to None.
        random_state (int, optional): Random seed. Defaults to 42.
        
    Returns:
        BaggingWrapper: Trained bagging ensemble
    """
    # Create bootstrap samples
    train_samples = create_bootstrap_samples(
        combined_data['train'],
        n_bootstraps=n_bootstraps,
        sample_size=sample_size,
        random_state=random_state
    )
    
    # Create wrapper
    wrapper = BaggingWrapper(model_type)
    
    # Train a model for each bootstrap sample
    for i, sample in enumerate(train_samples):
        logger.info(f"Training model for bootstrap sample {i+1}/{n_bootstraps}")
        
        # Extract validation data
        validation_data = combined_data['val']
        
        # Prepare X and y for training
        if model_type == 'CNN-LSTM':
            from src.models.cnn_lstm import create_sequences, build_cnn_lstm_model, train_cnn_lstm_model
            
            # Create sequences for training and validation
            X_train, y_train = create_sequences(
                sample,
                target_col=target_col,
                sequence_length=sequence_length,
                prediction_steps=prediction_steps,
                shuffle=True
            )
            
            X_val, y_val = create_sequences(
                validation_data,
                target_col=target_col,
                sequence_length=sequence_length,
                prediction_steps=prediction_steps,
                shuffle=False
            )
            
            # Train model
            model, _ = train_cnn_lstm_model(
                X_train, y_train,
                X_val, y_val,
                model_config=model_config,
                training_config=training_config,
                model_name=f"bootstrap_{i}_{model_type.lower()}",
                save_model=True
            )
            
            # Add to wrapper
            wrapper.add_model(f"bootstrap_{i}", model)
        
        elif model_type == 'TFT':
            from src.models.tft import prepare_data_for_tft, train_tft_model
            
            # Prepare data for TFT
            training, validation, _, _ = prepare_data_for_tft(
                sample,
                validation_data,
                validation_data,  # Use validation data as test data as well
                target_variable=target_col,
                max_encoder_length=sequence_length,
                max_prediction_length=prediction_steps
            )
            
            # Train model
            model, _ = train_tft_model(
                training,
                validation,
                model_config=model_config,
                training_config=training_config,
                model_name=f"bootstrap_{i}_{model_type.lower()}",
                save_model=True
            )
            
            # Add to wrapper
            wrapper.add_model(f"bootstrap_{i}", model)
        
        elif model_type == 'XGBoost':
            from src.models.xgboost_model import prepare_data_for_xgboost, train_xgboost_model
            
            # Prepare data for XGBoost
            X_train, y_train, feature_names = prepare_data_for_xgboost(
                sample,
                target_col=target_col,
                sequence_length=sequence_length,
                prediction_steps=prediction_steps,
                shuffle=True
            )
            
            X_val, y_val, _ = prepare_data_for_xgboost(
                validation_data,
                target_col=target_col,
                sequence_length=sequence_length,
                prediction_steps=prediction_steps,
                shuffle=False
            )
            
            # Train model
            model, _ = train_xgboost_model(
                X_train, y_train,
                X_val, y_val,
                feature_names=feature_names,
                model_config=model_config,
                model_name=f"bootstrap_{i}_{model_type.lower()}",
                save_model=True
            )
            
            # Add to wrapper
            wrapper.add_model(f"bootstrap_{i}", model, feature_names)
        
        else:
            error_msg = f"Unsupported model type: {model_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Save wrapper
    if model_name is None:
        model_name = f"bagging_bootstrap_{model_type.lower()}"
    
    wrapper.save(model_name)
    
    logger.info(f"Trained bagging model with {n_bootstraps} bootstrap models")
    
    return wrapper