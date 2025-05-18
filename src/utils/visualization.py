"""
Visualization utilities for the Forex prediction project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union

from src.utils.config import RESULTS_DIR


def set_plot_style():
    """Set the default style for plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    sns.set_palette("deep")


def plot_time_series(
    data: pd.DataFrame, 
    column: str,
    title: str = '',
    save_path: Optional[str] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a time series from a DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with datetime index
        column (str): Column to plot
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the plot. Defaults to True.
        ax (plt.Axes, optional): Existing axes to plot on
        **kwargs: Additional arguments to pass to the plot function
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    
    data[column].plot(ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(column)
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return ax


def plot_multiple_time_series(
    data: pd.DataFrame,
    columns: List[str],
    title: str = '',
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot multiple time series from a DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with datetime index
        columns (List[str]): Columns to plot
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        show (bool, optional): Whether to display the plot. Defaults to True.
        **kwargs: Additional arguments to pass to the plot function
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    _, ax = plt.subplots(figsize=(12, 6))
    
    for col in columns:
        data[col].plot(ax=ax, label=col, **kwargs)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return ax


def plot_currency_pairs_comparison(
    data_dict: Dict[str, pd.DataFrame],
    column: str = 'Close',
    normalize: bool = True,
    title: str = 'Currency Pairs Comparison',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot a comparison of the same column across different currency pairs.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary mapping currency pair names to DataFrames
        column (str, optional): Column to plot. Defaults to 'Close'.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
        title (str, optional): Plot title. Defaults to 'Currency Pairs Comparison'.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    _, ax = plt.subplots(figsize=(12, 6))
    
    for pair_name, df in data_dict.items():
        values = df[column].copy()
        
        if normalize:
            values = (values - values.iloc[0]) / values.iloc[0] * 100
        
        values.plot(ax=ax, label=pair_name)
    
    if normalize:
        ax.set_ylabel(f'{column} (% change)')
    else:
        ax.set_ylabel(column)
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return ax


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = 'Feature Importance',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot feature importance.
    
    Args:
        feature_names (List[str]): List of feature names
        importances (np.ndarray): Array of feature importance values
        title (str, optional): Plot title. Defaults to 'Feature Importance'.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    # Sort features by importance
    indices = np.argsort(importances)
    feature_names = [feature_names[i] for i in indices]
    importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
    
    ax.barh(range(len(feature_names)), importances, align='center')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return ax


def plot_model_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    dates: np.ndarray,
    title: str = 'Model Predictions vs Actual',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot model predictions against actual values.
    
    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        dates (np.ndarray): Dates corresponding to values
        title (str, optional): Plot title. Defaults to 'Model Predictions vs Actual'.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates, actual, label='Actual', marker='o', linestyle='-', markersize=3)
    plt.plot(dates, predicted, label='Predicted', marker='x', linestyle='-', markersize=3)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    if len(dates) > 20:
        plt.xticks(rotation=45)
        # Show only a subset of dates to avoid overcrowding
        step = max(1, len(dates) // 20)
        plt.xticks(range(0, len(dates), step), [dates[i] for i in range(0, len(dates), step)], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return plt.gca()


def plot_model_comparisons(
    models_performance: Dict[str, Dict[str, float]],
    metric: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot performance comparison of different models.
    
    Args:
        models_performance (Dict[str, Dict[str, float]]): Dictionary with model names as keys and 
                                                         performance metrics as nested dictionaries
        metric (str): Metric to plot
        title (str, optional): Plot title. If None, a default title will be created.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    # Extract model names and performance values
    model_names = list(models_performance.keys())
    metric_values = [models_performance[model][metric] for model in model_names]
    
    if title is None:
        title = f'{metric} Comparison'
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': model_names,
        metric: metric_values
    })
    
    # Sort by the metric value
    df = df.sort_values(by=metric, ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric, data=df)
    
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return plt.gca()


def plot_returns_comparison(
    returns_data: Dict[str, pd.Series],
    title: str = 'Cumulative Returns Comparison',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot cumulative returns comparison between different strategies.
    
    Args:
        returns_data (Dict[str, pd.Series]): Dictionary mapping strategy names to their returns
        title (str, optional): Plot title. Defaults to 'Cumulative Returns Comparison'.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    plt.figure(figsize=(12, 6))
    
    for strategy_name, returns in returns_data.items():
        cumulative_returns = (1 + returns).cumprod() - 1
        cumulative_returns.plot(label=strategy_name)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return plt.gca()


def visualize_performance_metrics(
    model_name: str,
    currency_pair: str,
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize key performance metrics for a model.
    
    Args:
        model_name (str): Name of the model
        currency_pair (str): Currency pair code
        metrics (Dict[str, float]): Dictionary of performance metrics
        save_path (str, optional): Path to save the figure. If None, a default will be used.
        show (bool, optional): Whether to display the plot. Defaults to True.
    """
    # If no save path is provided, create one
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"{model_name}_{currency_pair}_metrics.png")
    
    metrics_to_plot = {
        'Annual Return (%)': metrics.get('annual_return', 0) * 100,
        'Win Rate (%)': metrics.get('win_rate', 0) * 100,
        'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
        'Max Drawdown (%)': abs(metrics.get('max_drawdown', 0)) * 100,
    }
    
    # Define colors based on values
    colors = []
    for metric, value in metrics_to_plot.items():
        if 'Return' in metric or 'Win Rate' in metric:
            colors.append('green' if value > 0 else 'red')
        elif 'Sharpe' in metric:
            colors.append('green' if value > 1 else 'orange' if value > 0 else 'red')
        elif 'Drawdown' in metric:
            colors.append('red' if value > 20 else 'orange' if value > 10 else 'green')
        else:
            colors.append('blue')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        metrics_to_plot.keys(),
        metrics_to_plot.values(),
        color=colors
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.title(f'Performance Metrics: {model_name} on {currency_pair}')
    plt.ylabel('Value')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()


def plot_market_conditions_performance(
    performance_data: Dict[str, Dict[str, float]],
    metric: str = 'return',
    title: str = 'Performance in Different Market Conditions',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot model performance under different market conditions.
    
    Args:
        performance_data (Dict[str, Dict[str, float]]): Dictionary mapping market conditions 
                                                       to performance metrics
        metric (str, optional): Metric to plot. Defaults to 'return'.
        title (str, optional): Plot title. Defaults to 'Performance in Different Market Conditions'.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    conditions = list(performance_data.keys())
    values = [performance_data[condition][metric] for condition in conditions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(conditions, values)
    
    # Color bars based on values
    for i, bar in enumerate(bars):
        bar.set_color('green' if values[i] > 0 else 'red')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height * (1.01 if height > 0 else 0.9),
            f'{height:.2%}',
            ha='center',
            va='bottom' if height > 0 else 'top'
        )
    
    plt.title(title)
    plt.ylabel(f'{metric.capitalize()} (%)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return plt.gca()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Axes:
    """
    Plot confusion matrix for binary classification (up/down prediction).
    
    Args:
        y_true (np.ndarray): True direction labels (0 for down, 1 for up)
        y_pred (np.ndarray): Predicted direction labels
        title (str, optional): Plot title. Defaults to 'Confusion Matrix'.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to True.
        
    Returns:
        plt.Axes: The matplotlib axes object
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Down', 'Up'],
        yticklabels=['Down', 'Up']
    )
    
    plt.title(title)
    plt.ylabel('True Direction')
    plt.xlabel('Predicted Direction')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return plt.gca()