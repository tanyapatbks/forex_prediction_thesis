"""
Performance metrics and evaluation utilities for Forex prediction models.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from src.utils.config import RESULTS_DIR
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger("performance_metrics")


def calculate_returns(
    price_data: pd.DataFrame,
    positions: np.ndarray,
    start_idx: int = 0,
    transaction_cost: float = 0.0001,
    initial_capital: float = 10000,
    position_size: float = 1.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate returns based on price data and positions.
    
    Args:
        price_data (pd.DataFrame): DataFrame with price data
        positions (np.ndarray): Array of positions (1 for long, 0 for neutral, -1 for short)
        start_idx (int, optional): Starting index. Defaults to 0.
        transaction_cost (float, optional): Transaction cost in percentage. Defaults to 0.0001 (1 pip).
        initial_capital (float, optional): Initial capital. Defaults to 10000.
        position_size (float, optional): Position size as a fraction of capital. Defaults to 1.0.
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: Returns, equity curve, and drawdowns
    """
    # Convert positions to pandas Series with the same index as price_data
    positions_series = pd.Series(positions, index=price_data.index[start_idx:start_idx+len(positions)])
    
    # Calculate price changes
    price_changes = price_data['Close'].pct_change()
    
    # Calculate returns based on positions
    strategy_returns = positions_series.shift(1) * price_changes.loc[positions_series.index]
    
    # Calculate transaction costs
    position_changes = positions_series.diff().abs()
    transaction_costs = position_changes * transaction_cost
    
    # Subtract transaction costs from returns
    net_returns = strategy_returns - transaction_costs
    
    # Calculate equity curve
    equity_curve = (1 + net_returns).cumprod() * initial_capital * position_size
    
    # Calculate drawdowns
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    
    return net_returns, equity_curve, drawdown


def calculate_trading_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    drawdown: pd.Series,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate trading performance metrics.
    
    Args:
        returns (pd.Series): Series of returns
        equity_curve (pd.Series): Series of equity values
        drawdown (pd.Series): Series of drawdowns
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
        
    Returns:
        Dict[str, float]: Dictionary of performance metrics
    """
    # Calculate basic metrics
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    
    # Calculate annual return
    days = (returns.index[-1] - returns.index[0]).days
    annual_return = ((1 + total_return) ** (365 / days)) - 1 if days > 0 else 0
    
    # Calculate Sharpe ratio
    if len(returns) > 1:
        excess_returns = returns - risk_free_rate / 252  # Assuming 252 trading days per year
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Calculate Sortino ratio
    negative_returns = returns[returns < 0]
    sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
    
    # Calculate maximum drawdown
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    # Calculate profit factor
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    profit_factor = winning_trades.sum() / abs(losing_trades.sum()) if len(losing_trades) > 0 and losing_trades.sum() < 0 else 0
    
    # Calculate average win/loss ratio
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else 0
    
    # Create metrics dictionary
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'win_loss_ratio': win_loss_ratio,
        'num_trades': len(returns),
    }
    
    return metrics


def convert_predictions_to_positions(
    predictions: np.ndarray,
    multi_class: bool = False
) -> np.ndarray:
    """
    Convert model predictions to trading positions.
    
    Args:
        predictions (np.ndarray): Model predictions
        multi_class (bool, optional): Whether predictions are multi-class. Defaults to False.
        
    Returns:
        np.ndarray: Trading positions (1 for long, 0 for neutral, -1 for short)
    """
    if multi_class:
        # Assuming predictions are class indices: 0 (short), 1 (neutral), 2 (long)
        return predictions - 1
    else:
        # Assuming binary predictions: 0 (short) or 1 (long)
        return 2 * predictions - 1


def evaluate_trading_performance(
    model: Any,
    X_test: np.ndarray,
    price_data: pd.DataFrame,
    test_start_idx: int,
    model_type: str,
    multi_class: bool = False,
    transaction_cost: float = 0.0001,
    risk_free_rate: float = 0.0
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate trading performance of a model.
    
    Args:
        model (Any): Trained model
        X_test (np.ndarray): Test features
        price_data (pd.DataFrame): DataFrame with price data
        test_start_idx (int): Starting index of test data in price_data
        model_type (str): Type of model ('CNN-LSTM', 'TFT', 'XGBoost', 'Bagging')
        multi_class (bool, optional): Whether model makes multi-class predictions. Defaults to False.
        transaction_cost (float, optional): Transaction cost in percentage. Defaults to 0.0001 (1 pip).
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
        
    Returns:
        Tuple[Dict[str, float], pd.DataFrame]: Trading metrics and trading summary DataFrame
    """
    # Get predictions
    if model_type == 'CNN-LSTM':
        from src.models.cnn_lstm import predict_with_cnn_lstm
        predictions, _ = predict_with_cnn_lstm(model, X_test)
    
    elif model_type == 'TFT':
        from src.models.tft import predict_with_tft
        predictions = predict_with_tft(model, X_test)
        predictions = (predictions > 0.5).astype(int)
    
    elif model_type == 'XGBoost':
        from src.models.xgboost_model import predict_with_xgboost
        predictions = predict_with_xgboost(model, X_test)
    
    elif model_type == 'Bagging':
        predictions = model.predict(X_test)
    
    else:
        error_msg = f"Unsupported model type: {model_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Convert predictions to positions
    positions = convert_predictions_to_positions(predictions, multi_class)
    
    # Calculate returns and equity curve
    returns, equity_curve, drawdown = calculate_returns(
        price_data,
        positions,
        start_idx=test_start_idx,
        transaction_cost=transaction_cost
    )
    
    # Calculate performance metrics
    metrics = calculate_trading_metrics(returns, equity_curve, drawdown, risk_free_rate)
    
    # Create trading summary DataFrame
    summary = pd.DataFrame({
        'Close': price_data['Close'].iloc[test_start_idx:test_start_idx+len(positions)],
        'Position': positions,
        'Return': returns,
        'Equity': equity_curve,
        'Drawdown': drawdown
    })
    
    logger.info(f"Trading performance evaluation for {model_type} model: {metrics}")
    
    return metrics, summary


def calculate_buy_hold_performance(
    price_data: pd.DataFrame,
    test_start_idx: int,
    test_length: int,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate performance of a buy and hold strategy.
    
    Args:
        price_data (pd.DataFrame): DataFrame with price data
        test_start_idx (int): Starting index of test data
        test_length (int): Length of test data
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
        
    Returns:
        Dict[str, float]: Dictionary of performance metrics
    """
    # Create positions array (all 1s for always long)
    positions = np.ones(test_length)
    
    # Calculate returns and equity curve
    returns, equity_curve, drawdown = calculate_returns(
        price_data,
        positions,
        start_idx=test_start_idx,
        transaction_cost=0.0001  # Only applies to the initial trade
    )
    
    # Calculate performance metrics
    metrics = calculate_trading_metrics(returns, equity_curve, drawdown, risk_free_rate)
    
    logger.info(f"Buy & Hold performance: {metrics}")
    
    return metrics


def identify_market_conditions(
    price_data: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.05
) -> pd.Series:
    """
    Identify market conditions based on price trend and volatility.
    
    Args:
        price_data (pd.DataFrame): DataFrame with price data
        window (int, optional): Window for trend calculation. Defaults to 20.
        threshold (float, optional): Threshold for trend identification. Defaults to 0.05.
        
    Returns:
        pd.Series: Series with market condition labels
    """
    # Calculate short-term trend
    price_data['SMA'] = price_data['Close'].rolling(window=window).mean()
    price_data['Trend'] = (price_data['Close'] - price_data['SMA']) / price_data['SMA']
    
    # Calculate volatility
    price_data['Returns'] = price_data['Close'].pct_change()
    price_data['Volatility'] = price_data['Returns'].rolling(window=window).std()
    
    # Classify market conditions
    conditions = []
    
    for i, row in price_data.iterrows():
        if pd.isna(row['Trend']) or pd.isna(row['Volatility']):
            conditions.append('Unknown')
        elif row['Trend'] > threshold:
            if row['Volatility'] > row['Volatility'].mean():
                conditions.append('Bullish Volatile')
            else:
                conditions.append('Bullish Stable')
        elif row['Trend'] < -threshold:
            if row['Volatility'] > row['Volatility'].mean():
                conditions.append('Bearish Volatile')
            else:
                conditions.append('Bearish Stable')
        else:
            if row['Volatility'] > row['Volatility'].mean():
                conditions.append('Sideways Volatile')
            else:
                conditions.append('Sideways Stable')
    
    market_conditions = pd.Series(conditions, index=price_data.index)
    
    return market_conditions


def evaluate_performance_by_market_condition(
    returns: pd.Series,
    market_conditions: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate performance of a strategy by market condition.
    
    Args:
        returns (pd.Series): Series of returns
        market_conditions (pd.Series): Series of market condition labels
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary mapping market conditions to performance metrics
    """
    # Align market conditions with returns
    aligned_conditions = market_conditions.loc[returns.index]
    
    # Get unique market conditions
    unique_conditions = aligned_conditions.unique()
    
    # Calculate performance metrics for each condition
    condition_performance = {}
    
    for condition in unique_conditions:
        condition_returns = returns[aligned_conditions == condition]
        
        if len(condition_returns) > 0:
            # Calculate basic metrics
            total_return = (1 + condition_returns).prod() - 1
            avg_return = condition_returns.mean()
            win_rate = (condition_returns > 0).sum() / len(condition_returns)
            
            # Calculate Sharpe ratio (annualized)
            sharpe = condition_returns.mean() / condition_returns.std() * np.sqrt(252) if condition_returns.std() > 0 else 0
            
            condition_performance[condition] = {
                'count': len(condition_returns),
                'total_return': total_return,
                'avg_return': avg_return,
                'win_rate': win_rate,
                'sharpe': sharpe
            }
        else:
            condition_performance[condition] = {
                'count': 0,
                'total_return': 0.0,
                'avg_return': 0.0,
                'win_rate': 0.0,
                'sharpe': 0.0
            }
    
    logger.info(f"Performance by market condition calculated for {len(unique_conditions)} conditions")
    
    return condition_performance


def compare_models_performance(
    models_metrics: Dict[str, Dict[str, float]],
    buy_hold_metrics: Dict[str, float],
    key_metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compare performance metrics of multiple models.
    
    Args:
        models_metrics (Dict[str, Dict[str, float]]): Dictionary mapping model names to their metrics
        buy_hold_metrics (Dict[str, float]): Dictionary of buy & hold metrics
        key_metrics (List[str], optional): List of key metrics to compare. Defaults to None.
        
    Returns:
        pd.DataFrame: DataFrame with model performance comparison
    """
    # Use default key metrics if not provided
    if key_metrics is None:
        key_metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    # Add buy & hold metrics to models_metrics
    all_metrics = {
        **models_metrics,
        'Buy & Hold': buy_hold_metrics
    }
    
    # Create comparison DataFrame
    comparison = {}
    
    for metric in key_metrics:
        comparison[metric] = {model_name: metrics.get(metric, np.nan) for model_name, metrics in all_metrics.items()}
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison)
    
    # Calculate relative performance vs buy & hold
    for metric in key_metrics:
        if metric in buy_hold_metrics:
            # For metrics where higher is better (except drawdown)
            if metric != 'max_drawdown':
                comparison_df[f'{metric}_vs_BH'] = comparison_df[metric] / buy_hold_metrics[metric] - 1
            else:
                # For drawdown, lower is better
                comparison_df[f'{metric}_vs_BH'] = buy_hold_metrics[metric] / comparison_df[metric] - 1
    
    return comparison_df


def plot_equity_curves(
    equity_curves: Dict[str, pd.Series],
    title: str = 'Equity Curves Comparison',
    y_label: str = 'Equity ($)',
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None,
    show_plot: bool = True
) -> None:
    """
    Plot equity curves for multiple strategies.
    
    Args:
        equity_curves (Dict[str, pd.Series]): Dictionary mapping strategy names to their equity curves
        title (str, optional): Plot title. Defaults to 'Equity Curves Comparison'.
        y_label (str, optional): Y-axis label. Defaults to 'Equity ($)'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    plt.figure(figsize=figsize)
    
    for name, curve in equity_curves.items():
        plt.plot(curve.index, curve, label=name)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved equity curves plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_drawdowns(
    drawdowns: Dict[str, pd.Series],
    title: str = 'Drawdowns Comparison',
    y_label: str = 'Drawdown (%)',
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None,
    show_plot: bool = True
) -> None:
    """
    Plot drawdowns for multiple strategies.
    
    Args:
        drawdowns (Dict[str, pd.Series]): Dictionary mapping strategy names to their drawdowns
        title (str, optional): Plot title. Defaults to 'Drawdowns Comparison'.
        y_label (str, optional): Y-axis label. Defaults to 'Drawdown (%)'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    plt.figure(figsize=figsize)
    
    for name, curve in drawdowns.items():
        plt.plot(curve.index, curve * 100, label=name)  # Convert to percentage
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Invert y-axis since drawdowns are negative
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved drawdowns plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_returns_heatmap(
    returns: pd.Series,
    title: str = 'Returns Heatmap',
    figsize: Tuple[int, int] = (12, 8),
    save_path: str = None,
    show_plot: bool = True
) -> None:
    """
    Plot a heatmap of returns by day of week and hour.
    
    Args:
        returns (pd.Series): Series of returns with DatetimeIndex
        title (str, optional): Plot title. Defaults to 'Returns Heatmap'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    import seaborn as sns
    
    # Add day of week and hour information
    returns_df = pd.DataFrame(returns)
    returns_df['Day'] = returns_df.index.day_name()
    returns_df['Hour'] = returns_df.index.hour
    
    # Sort days of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    returns_df['Day'] = pd.Categorical(returns_df['Day'], categories=days_order, ordered=True)
    
    # Create pivot table
    pivot = returns_df.pivot_table(
        values=returns_df.columns[0],
        index='Day',
        columns='Hour',
        aggfunc='mean'
    )
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot,
        cmap='RdYlGn',
        center=0,
        linewidths=0.5,
        annot=True,
        fmt='.4f'
    )
    
    plt.title(title)
    plt.tight_layout()
    
    # Save plot if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved returns heatmap to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_market_conditions_performance(
    condition_performance: Dict[str, Dict[str, float]],
    metric: str = 'total_return',
    title: str = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None,
    show_plot: bool = True
) -> None:
    """
    Plot performance by market condition.
    
    Args:
        condition_performance (Dict[str, Dict[str, float]]): Dictionary mapping conditions to metrics
        metric (str, optional): Metric to plot. Defaults to 'total_return'.
        title (str, optional): Plot title. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    # Extract conditions and values
    conditions = []
    values = []
    counts = []
    
    for condition, metrics in condition_performance.items():
        conditions.append(condition)
        values.append(metrics.get(metric, 0))
        counts.append(metrics.get('count', 0))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Condition': conditions,
        metric: values,
        'Count': counts
    })
    
    # Sort by metric value
    df = df.sort_values(metric, ascending=False)
    
    # Create plot
    plt.figure(figsize=figsize)
    bars = plt.bar(df['Condition'], df[metric])
    
    # Color bars based on value
    for i, bar in enumerate(bars):
        bar.set_color('green' if df[metric].iloc[i] > 0 else 'red')
    
    # Add count labels
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            0.01,
            f'n={df["Count"].iloc[i]}',
            ha='center',
            va='bottom',
            color='black',
            fontweight='bold',
            rotation=90
        )
    
    # Set title and labels
    if title is None:
        title = f'Performance by Market Condition ({metric})'
    
    plt.title(title)
    plt.ylabel(metric)
    plt.grid(True, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved market conditions performance plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_performance_results(
    results: Dict,
    model_name: str,
    currency_pair: str
) -> str:
    """
    Save performance results to a file.
    
    Args:
        results (Dict): Dictionary of performance results
        model_name (str): Name of the model
        currency_pair (str): Currency pair code
        
    Returns:
        str: Path where the results were saved
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create filename
    filename = f"{model_name}_{currency_pair}_performance.json"
    output_path = os.path.join(RESULTS_DIR, filename)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert results
    converted_results = convert_numpy_types(results)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=4)
    
    logger.info(f"Saved performance results for {model_name} on {currency_pair} to {output_path}")
    
    return output_path


def load_performance_results(
    model_name: str,
    currency_pair: str
) -> Dict:
    """
    Load performance results from a file.
    
    Args:
        model_name (str): Name of the model
        currency_pair (str): Currency pair code
        
    Returns:
        Dict: Dictionary of performance results
    """
    # Create filename
    filename = f"{model_name}_{currency_pair}_performance.json"
    file_path = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(file_path):
        error_msg = f"Performance results file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load results
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded performance results for {model_name} on {currency_pair} from {file_path}")
    
    return results


def create_performance_report(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    buy_hold_metrics: Dict[str, Dict[str, float]],
    market_condition_performance: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    key_metrics: List[str] = None
) -> pd.DataFrame:
    """
    Create a comprehensive performance report.
    
    Args:
        all_metrics (Dict[str, Dict[str, Dict[str, float]]]): Dictionary with metrics:
            - Outer key: Currency pair
            - Middle key: Model name
            - Inner key: Metric name
            - Value: Metric value
        buy_hold_metrics (Dict[str, Dict[str, float]]): Dictionary with buy & hold metrics:
            - Outer key: Currency pair
            - Inner key: Metric name
            - Value: Metric value
        market_condition_performance (Dict[str, Dict[str, Dict[str, Dict[str, float]]]]): 
            Dictionary with market condition performance:
            - Outer key: Currency pair
            - Second key: Model name
            - Third key: Market condition
            - Inner key: Metric name
            - Value: Metric value
        key_metrics (List[str], optional): List of key metrics to include. Defaults to None.
        
    Returns:
        pd.DataFrame: DataFrame with performance report
    """
    # Use default key metrics if not provided
    if key_metrics is None:
        key_metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    # Create report data
    report_data = []
    
    for pair, pair_metrics in all_metrics.items():
        for model, model_metrics in pair_metrics.items():
            # Add basic metrics
            row = {
                'Currency Pair': pair,
                'Model': model
            }
            
            # Add key metrics
            for metric in key_metrics:
                if metric in model_metrics:
                    row[metric] = model_metrics[metric]
            
            # Add comparison to buy & hold
            if pair in buy_hold_metrics:
                for metric in key_metrics:
                    if metric in model_metrics and metric in buy_hold_metrics[pair]:
                        if metric != 'max_drawdown':
                            # For metrics where higher is better
                            row[f'{metric}_vs_BH'] = model_metrics[metric] / buy_hold_metrics[pair][metric] - 1
                        else:
                            # For drawdown, lower is better
                            row[f'{metric}_vs_BH'] = buy_hold_metrics[pair][metric] / model_metrics[metric] - 1
            
            # Add market condition performance
            if pair in market_condition_performance and model in market_condition_performance[pair]:
                for condition, condition_metrics in market_condition_performance[pair][model].items():
                    if 'total_return' in condition_metrics:
                        row[f'{condition}_return'] = condition_metrics['total_return']
                    if 'win_rate' in condition_metrics:
                        row[f'{condition}_win_rate'] = condition_metrics['win_rate']
            
            report_data.append(row)
    
    # Create DataFrame
    report_df = pd.DataFrame(report_data)
    
    return report_df


def save_trading_summary(
    summary: pd.DataFrame,
    model_name: str,
    currency_pair: str
) -> str:
    """
    Save trading summary to a CSV file.
    
    Args:
        summary (pd.DataFrame): Trading summary DataFrame
        model_name (str): Name of the model
        currency_pair (str): Currency pair code
        
    Returns:
        str: Path where the summary was saved
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create filename
    filename = f"{model_name}_{currency_pair}_summary.csv"
    output_path = os.path.join(RESULTS_DIR, filename)
    
    # Save summary
    summary.to_csv(output_path)
    
    logger.info(f"Saved trading summary for {model_name} on {currency_pair} to {output_path}")
    
    return output_path


def load_trading_summary(
    model_name: str,
    currency_pair: str
) -> pd.DataFrame:
    """
    Load trading summary from a CSV file.
    
    Args:
        model_name (str): Name of the model
        currency_pair (str): Currency pair code
        
    Returns:
        pd.DataFrame: Trading summary DataFrame
    """
    # Create filename
    filename = f"{model_name}_{currency_pair}_summary.csv"
    file_path = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(file_path):
        error_msg = f"Trading summary file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load summary
    summary = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    logger.info(f"Loaded trading summary for {model_name} on {currency_pair} from {file_path}")
    
    return summary