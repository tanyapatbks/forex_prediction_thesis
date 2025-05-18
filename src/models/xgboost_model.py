"""
XGBoost model for Forex prediction.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import matplotlib.pyplot as plt
import joblib
import optuna

from src.utils.config import TRAINED_MODELS_DIR
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger("xgboost_model")


def prepare_data_for_xgboost(
    data: pd.DataFrame,
    target_col: str = 'target',
    sequence_length: int = 1,
    prediction_steps: int = 1,
    shuffle: bool = True,
    scaling_method: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for XGBoost model.
    For XGBoost, we typically don't need to create sequences as it can handle tabular data directly.
    However, we might want to include lagged features to capture temporal patterns.
    
    Args:
        data (pd.DataFrame): DataFrame with features and target
        target_col (str, optional): Target column name. Defaults to 'target'.
        sequence_length (int, optional): Number of lagged features to include. Defaults to 1.
        prediction_steps (int, optional): Number of steps ahead to predict. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        scaling_method (str, optional): Method for scaling features. 
                                      Options: 'standard', 'minmax', 'robust', None.
                                      Defaults to None.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (feature matrix) and y (target values)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    # Create a copy of the data
    df = data.copy()
    
    # Add lagged features if sequence_length > 1
    if sequence_length > 1:
        # Get feature columns (exclude target)
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Add lagged features
        for lag in range(1, sequence_length):
            for col in feature_cols:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        # Drop rows with NaN values from lagging
        df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Scale features if specified
    if scaling_method == 'standard':
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    elif scaling_method == 'robust':
        scaler = RobustScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Convert to numpy arrays
    X_array = X.values
    y_array = y.values
    
    # Shuffle if specified
    if shuffle:
        indices = np.arange(X_array.shape[0])
        np.random.shuffle(indices)
        X_array = X_array[indices]
        y_array = y_array[indices]
    
    logger.info(f"Prepared data for XGBoost: X shape {X_array.shape}, y shape {y_array.shape}")
    
    return X_array, y_array, X.columns.tolist()


def build_xgboost_model(
    objective: str = 'binary:logistic',
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0,
    reg_alpha: float = 0,
    reg_lambda: float = 1,
    scale_pos_weight: float = 1,
    random_state: int = 42,
    **kwargs
) -> xgb.XGBClassifier:
    """
    Build an XGBoost model for Forex prediction.
    
    Args:
        objective (str, optional): Learning objective. Defaults to 'binary:logistic'.
        n_estimators (int, optional): Number of boosting rounds. Defaults to 100.
        max_depth (int, optional): Maximum depth of trees. Defaults to 6.
        learning_rate (float, optional): Step size shrinkage. Defaults to 0.1.
        min_child_weight (int, optional): Minimum sum of instance weight in a child. Defaults to 1.
        subsample (float, optional): Subsample ratio of the training instances. Defaults to 0.8.
        colsample_bytree (float, optional): Subsample ratio of columns for each tree. Defaults to 0.8.
        gamma (float, optional): Minimum loss reduction required to make a further partition. Defaults to 0.
        reg_alpha (float, optional): L1 regularization term on weights. Defaults to 0.
        reg_lambda (float, optional): L2 regularization term on weights. Defaults to 1.
        scale_pos_weight (float, optional): Balance of positive and negative weights. Defaults to 1.
        random_state (int, optional): Random number seed. Defaults to 42.
        **kwargs: Additional parameters to pass to the XGBoost model.
        
    Returns:
        xgb.XGBClassifier: Configured XGBoost model
    """
    # Create model
    model = xgb.XGBClassifier(
        objective=objective,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,  # To avoid warning
        verbosity=0,  # Be quiet
        **kwargs
    )
    
    logger.info(f"Built XGBoost model with {n_estimators} estimators")
    
    return model


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str] = None,
    model_config: Dict = None,
    model_name: str = 'xgboost',
    save_model: bool = True
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train an XGBoost model for Forex prediction.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation targets
        feature_names (List[str], optional): Names of features. Defaults to None.
        model_config (Dict, optional): Model configuration. Defaults to None.
        model_name (str, optional): Name for the saved model. Defaults to 'xgboost'.
        save_model (bool, optional): Whether to save the trained model. Defaults to True.
        
    Returns:
        Tuple[xgb.XGBClassifier, Dict]: Trained model and training history
    """
    # Use default configurations if not provided
    if model_config is None:
        model_config = {
            'objective': 'binary:logistic',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'random_state': 42
        }
    
    # Build model
    model = build_xgboost_model(**model_config)
    
    # Set feature names if provided
    if feature_names is not None:
        feature_param = {'feature_names': feature_names}
    else:
        feature_param = {}
    
    # Train model with early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric=['logloss', 'error'],
        early_stopping_rounds=20,
        verbose=True,
        **feature_param
    )
    
    # Get training history
    history = {
        'training_loss': model.evals_result()['validation_0']['logloss'],
        'validation_loss': model.evals_result()['validation_1']['logloss'],
        'training_error': model.evals_result()['validation_0']['error'],
        'validation_error': model.evals_result()['validation_1']['error'],
    }
    
    # Save model
    if save_model:
        os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
        model_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}.json")
        model.save_model(model_path)
        
        # Save feature names if provided
        if feature_names is not None:
            feature_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_features.json")
            with open(feature_path, 'w') as f:
                json.dump(feature_names, f)
        
        # Save model configuration
        config_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        
        logger.info(f"Saved trained XGBoost model to {model_path}")
    
    logger.info("XGBoost model training completed")
    
    return model, history


def evaluate_xgboost_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a trained XGBoost model.
    
    Args:
        model (xgb.XGBClassifier): Trained XGBoost model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
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
    
    logger.info(f"XGBoost model evaluation: {metrics}")
    
    return metrics


def plot_xgboost_training_history(
    history: Dict[str, List[float]],
    model_name: str = 'xgboost',
    save_plot: bool = True,
    show_plot: bool = True
) -> None:
    """
    Plot training history of an XGBoost model.
    
    Args:
        history (Dict[str, List[float]]): Training history
        model_name (str, optional): Name of the model. Defaults to 'xgboost'.
        save_plot (bool, optional): Whether to save the plot. Defaults to True.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot training & validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history['training_loss'], label='Training Loss')
    plt.plot(history['validation_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.ylabel('Log Loss')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation error
    plt.subplot(2, 1, 2)
    plt.plot(history['training_error'], label='Training Error')
    plt.plot(history['validation_error'], label='Validation Error')
    plt.title(f'{model_name} - Error')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
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


def plot_xgboost_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: List[str] = None,
    model_name: str = 'xgboost',
    importance_type: str = 'weight',
    max_features: int = 20,
    save_plot: bool = True,
    show_plot: bool = True
) -> None:
    """
    Plot feature importance of an XGBoost model.
    
    Args:
        model (xgb.XGBClassifier): Trained XGBoost model
        feature_names (List[str], optional): Names of features. Defaults to None.
        model_name (str, optional): Name of the model. Defaults to 'xgboost'.
        importance_type (str, optional): Type of importance. Defaults to 'weight'.
        max_features (int, optional): Maximum number of features to show. Defaults to 20.
        save_plot (bool, optional): Whether to save the plot. Defaults to True.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    # Get feature importance
    importance = model.get_booster().get_score(importance_type=importance_type)
    
    # Convert to DataFrame
    if feature_names is not None:
        # Make sure all feature names are in the importance dictionary
        # XGBoost might exclude features with zero importance
        tuples = [(feature_names[int(k[1:])] if k.startswith('f') else k, v) 
                 for k, v in importance.items()]
    else:
        tuples = [(k, v) for k, v in importance.items()]
    
    importance_df = pd.DataFrame(tuples, columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Limit to max_features
    if max_features is not None and max_features < len(importance_df):
        importance_df = importance_df.head(max_features)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.title(f'{model_name} - Feature Importance ({importance_type})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()  # To show most important features at the top
    plt.grid(True, axis='x')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        plot_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_feature_importance.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {plot_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def load_xgboost_model(model_name: str) -> Tuple[xgb.XGBClassifier, Dict, Optional[List[str]]]:
    """
    Load a trained XGBoost model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Tuple[xgb.XGBClassifier, Dict, Optional[List[str]]]: Loaded model, configuration, and feature names
    """
    model_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}.json")
    config_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_config.json")
    feature_path = os.path.join(TRAINED_MODELS_DIR, f"{model_name}_features.json")
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        error_msg = f"Model or config file not found: {model_path}, {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Build model with the same configuration
    model = build_xgboost_model(**config)
    
    # Load model parameters
    model.load_model(model_path)
    
    # Load feature names if available
    feature_names = None
    if os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            feature_names = json.load(f)
    
    logger.info(f"Loaded XGBoost model from {model_path}")
    
    return model, config, feature_names


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_space: Dict = None,
    n_trials: int = 100,
    timeout: int = 3600,
    study_name: str = 'xgboost_optimization',
    direction: str = 'maximize',
    metric: str = 'roc_auc'
) -> Dict:
    """
    Perform hyperparameter tuning for XGBoost model using Optuna.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation targets
        param_space (Dict, optional): Parameter space for tuning. Defaults to None.
        n_trials (int, optional): Number of trials. Defaults to 100.
        timeout (int, optional): Timeout in seconds. Defaults to 3600.
        study_name (str, optional): Name of the study. Defaults to 'xgboost_optimization'.
        direction (str, optional): Direction of optimization. Defaults to 'maximize'.
        metric (str, optional): Metric to optimize. Defaults to 'roc_auc'.
        
    Returns:
        Dict: Best parameters
    """
    # Use default parameter space if not provided
    if param_space is None:
        param_space = {
            'n_estimators': (50, 500),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'gamma': (0, 5),
            'reg_alpha': (0, 5),
            'reg_lambda': (0, 5),
        }
    
    def objective(trial):
        # Sample parameters
        params = {
            'objective': 'binary:logistic',
            'n_estimators': trial.suggest_int('n_estimators', param_space['n_estimators'][0], param_space['n_estimators'][1]),
            'max_depth': trial.suggest_int('max_depth', param_space['max_depth'][0], param_space['max_depth'][1]),
            'learning_rate': trial.suggest_float('learning_rate', param_space['learning_rate'][0], param_space['learning_rate'][1], log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', param_space['min_child_weight'][0], param_space['min_child_weight'][1]),
            'subsample': trial.suggest_float('subsample', param_space['subsample'][0], param_space['subsample'][1]),
            'colsample_bytree': trial.suggest_float('colsample_bytree', param_space['colsample_bytree'][0], param_space['colsample_bytree'][1]),
            'gamma': trial.suggest_float('gamma', param_space['gamma'][0], param_space['gamma'][1]),
            'reg_alpha': trial.suggest_float('reg_alpha', param_space['reg_alpha'][0], param_space['reg_alpha'][1]),
            'reg_lambda': trial.suggest_float('reg_lambda', param_space['reg_lambda'][0], param_space['reg_lambda'][1]),
            'random_state': 42,
            'use_label_encoder': False,
            'verbosity': 0
        }
        
        # Build and train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='logloss',
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate model based on the specified metric
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        if metric == 'accuracy':
            score = accuracy_score(y_val, y_pred)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        elif metric == 'f1_score':
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == 'roc_auc':
            score = roc_auc_score(y_val, y_pred_proba)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return score
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_params
    logger.info(f"Hyperparameter tuning completed. Best parameters: {best_params}")
    
    return best_params


def predict_with_xgboost(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    return_probabilities: bool = False
) -> np.ndarray:
    """
    Make predictions using a trained XGBoost model.
    
    Args:
        model (xgb.XGBClassifier): Trained XGBoost model
        X (np.ndarray): Input features
        return_probabilities (bool, optional): Whether to return probabilities. Defaults to False.
        
    Returns:
        np.ndarray: Predicted values
    """
    if return_probabilities:
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X)