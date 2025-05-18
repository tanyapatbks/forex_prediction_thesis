"""
Modified performance metrics for continuous target prediction.
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


def convert_continuous_to_positions(
    predictions: np.ndarray,
    threshold: float = 0.1  # Threshold for determining significant movement
) -> np.ndarray:
    """
    Convert continuous model predictions to trading positions.
    
    Args:
        predictions (np.ndarray): Continuous model predictions (-1 to 1 scale)
        threshold (float, optional): Threshold for determining significant movement.
                                  Values between -threshold and threshold are considered neutral.
                                  Defaults to 0.1.
        
    Returns:
        np.ndarray: Trading positions (1 for long, 0 for neutral, -1 for short)
    """
    positions = np.zeros_like(predictions)
    
    # Assign positions based on prediction strength
    positions[predictions > threshold] = 1       # Strong up trend - go long
    positions[predictions < -threshold] = -1     # Strong down trend - go short
    # Values between -threshold and threshold remain 0 (neutral)
    
    return positions


def calculate_returns(
    price_data: pd.DataFrame,
    positions: np.ndarray,
    start_idx: int = 0,
    position_sizing: bool = True,  # Whether to use position sizing based on signal strength
    raw_predictions: Optional[np.ndarray] = None,  # Raw prediction values for position sizing
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
        position_sizing (bool, optional): Whether to use position sizing based on signal strength.
                                        Defaults to True.
        raw_predictions (np.ndarray, optional): Raw prediction values for position sizing (required if position_sizing=True).
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
    
    # Apply position sizing if enabled and raw predictions provided
    if position_sizing and raw_predictions is not None:
        # Create a series of position sizes based on signal strength
        # Scale from 0 to 1 based on absolute value of prediction
        # This ensures that stronger signals get larger position sizes
        signal_strength = pd.Series(
            np.abs(raw_predictions) / np.max([1.0, np.max(np.abs(raw_predictions))]),
            index=positions_series.index
        )
        
        # Calculate returns with dynamic position sizing
        strategy_returns = positions_series * price_changes.loc[positions_series.index] * signal_strength
    else:
        # Standard returns calculation with fixed position sizes
        strategy_returns = positions_series * price_changes.loc[positions_series.index]
    
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
    multi_class: bool = False,
    is_continuous: bool = True,  # Added parameter for continuous predictions
    threshold: float = 0.1      # Added threshold for continuous predictions
) -> np.ndarray:
    """
    Convert model predictions to trading positions.
    
    Args:
        predictions (np.ndarray): Model predictions
        multi_class (bool, optional): Whether predictions are multi-class. Defaults to False.
        is_continuous (bool, optional): Whether predictions are continuous values. Defaults to True.
        threshold (float, optional): Threshold for continuous predictions. Defaults to 0.1.
        
    Returns:
        np.ndarray: Trading positions (1 for long, 0 for neutral, -1 for short)
    """
    if is_continuous:
        # For continuous predictions (-1 to 1 range)
        return convert_continuous_to_positions(predictions, threshold)
    elif multi_class:
        # For multi-class predictions (0=short, 1=neutral, 2=long)
        return predictions - 1
    else:
        # For binary predictions (0=short, 1=long)
        return 2 * predictions - 1


def evaluate_trading_performance(
    model: Any,
    X_test: np.ndarray,
    price_data: pd.DataFrame,
    test_start_idx: int,
    model_type: str,
    is_continuous: bool = True,  # Added parameter for continuous predictions
    multi_class: bool = False,
    threshold: float = 0.1,      # Added threshold for continuous predictions
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
        is_continuous (bool, optional): Whether model makes continuous predictions. Defaults to True.
        multi_class (bool, optional): Whether model makes multi-class predictions. Defaults to False.
        threshold (float, optional): Threshold for continuous predictions. Defaults to 0.1.
        transaction_cost (float, optional): Transaction cost in percentage. Defaults to 0.0001 (1 pip).
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
        
    Returns:
        Tuple[Dict[str, float], pd.DataFrame]: Trading metrics and trading summary DataFrame
    """
    # Get predictions
    raw_predictions = None
    
    if model_type == 'CNN-LSTM':
        from src.models.cnn_lstm import predict_with_cnn_lstm
        predictions, raw_predictions = predict_with_cnn_lstm(
            model, X_test, regression=is_continuous, threshold=threshold
        )
    
    elif model_type == 'TFT':
        from src.models.tft import predict_with_tft
        raw_predictions = predict_with_tft(model, X_test)
        # For continuous predictions, use raw values
        # For binary classification, threshold the values
        if not is_continuous:
            predictions = (raw_predictions > threshold).astype(int)
        else:
            predictions = raw_predictions
    
    elif model_type == 'XGBoost':
        from src.models.xgboost_model import predict_with_xgboost
        if is_continuous:
            # For regression XGBoost, get raw predictions
            raw_predictions = predict_with_xgboost(model, X_test, return_probabilities=False)
            predictions = raw_predictions
        else:
            # For classification XGBoost
            predictions = predict_with_xgboost(model, X_test)
            raw_predictions = predict_with_xgboost(model, X_test, return_probabilities=True)
    
    elif model_type == 'Bagging':
        if is_continuous:
            # For continuous predictions
            raw_predictions = model.predict_proba(X_test)  # Get continuous values
            predictions = raw_predictions
        else:
            # For classification
            predictions = model.predict(X_test)
            raw_predictions = model.predict_proba(X_test)
    
    else:
        error_msg = f"Unsupported model type: {model_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Convert predictions to positions
    positions = convert_predictions_to_positions(
        predictions, 
        multi_class=multi_class, 
        is_continuous=is_continuous,
        threshold=threshold
    )
    
    # Calculate returns and equity curve
    returns, equity_curve, drawdown = calculate_returns(
        price_data,
        positions,
        start_idx=test_start_idx,
        position_sizing=is_continuous,  # Use position sizing for continuous predictions
        raw_predictions=raw_predictions if is_continuous else None,
        transaction_cost=transaction_cost
    )
    
    # Calculate performance metrics
    metrics = calculate_trading_metrics(returns, equity_curve, drawdown, risk_free_rate)
    
    # Create trading summary DataFrame
    summary = pd.DataFrame({
        'Close': price_data['Close'].iloc[test_start_idx:test_start_idx+len(positions)],
        'Position': positions,
        'Signal': raw_predictions if raw_predictions is not None else positions,  # Store raw predictions if available
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
        position_sizing=False,  # No position sizing for buy & hold
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


def plot_signal_strength_vs_returns(
    summary: pd.DataFrame,
    title: str = 'Signal Strength vs Returns',
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot signal strength vs returns.
    
    Args:
        summary (pd.DataFrame): Trading summary DataFrame with Signal and Return columns
        title (str, optional): Plot title. Defaults to 'Signal Strength vs Returns'.
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    # Check if Signal column contains continuous values
    if not pd.api.types.is_numeric_dtype(summary['Signal']):
        logger.warning("Signal column is not numeric. Cannot plot signal strength vs returns.")
        return
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot signal strength vs returns
    plt.scatter(
        summary['Signal'],
        summary['Return'] * 100,  # Convert to percentage
        alpha=0.5,
        s=30,
        c=summary['Return'] > 0,  # Color by profit/loss
        cmap='RdYlGn'
    )
    
    # Add trend line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(summary['Signal'], summary['Return'] * 100)
    x = np.array([summary['Signal'].min(), summary['Signal'].max()])
    y = slope * x + intercept
    plt.plot(x, y, 'r--', label=f'Trend (r={r_value:.3f})')
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    plt.title(title)
    plt.xlabel('Signal Strength')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved signal strength vs returns plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_signal_distribution(
    summary: pd.DataFrame,
    title: str = 'Signal Strength Distribution',
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot distribution of signal strengths.
    
    Args:
        summary (pd.DataFrame): Trading summary DataFrame with Signal column
        title (str, optional): Plot title. Defaults to 'Signal Strength Distribution'.
        save_path (str, optional): Path to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
    """
    # Check if Signal column contains continuous values
    if not pd.api.types.is_numeric_dtype(summary['Signal']):
        logger.warning("Signal column is not numeric. Cannot plot signal distribution.")
        return
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Calculate profitable signals
    profitable = summary[summary['Return'] > 0]['Signal']
    losing = summary[summary['Return'] <= 0]['Signal']
    
    # Plot distributions
    plt.hist(profitable, bins=20, alpha=0.5, label='Profitable', color='green')
    plt.hist(losing, bins=20, alpha=0.5, label='Losing', color='red')
    
    plt.title(title)
    plt.xlabel('Signal Strength')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved signal distribution plot to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_returns_heatmap(
    returns: pd.Series,
    signal_strength: Optional[pd.Series] = None,
    title: str = 'Returns Heatmap',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot a heatmap of returns by day of week and hour, optionally colored by signal strength.
    
    Args:
        returns (pd.Series): Series of returns with DatetimeIndex
        signal_strength (pd.Series, optional): Series of signal strengths. Defaults to None.
        title (str, optional): Plot title. Defaults to 'Returns Heatmap'.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).
        save_path (str, optional): Path to save the figure. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
    """
    import seaborn as sns
    
    # Add day of week and hour information
    returns_df = pd.DataFrame(returns)
    returns_df['Day'] = returns_df.index.day_name()
    returns_df['Hour'] = returns_df.index.hour
    
    # Add signal strength if provided
    if signal_strength is not None:
        returns_df['Signal'] = signal_strength
    
    # Sort days of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    returns_df['Day'] = pd.Categorical(returns_df['Day'], categories=days_order, ordered=True)
    
    # Create pivot table for returns
    pivot_returns = returns_df.pivot_table(
        values=returns_df.columns[0],
        index='Day',
        columns='Hour',
        aggfunc='mean'
    )
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    if signal_strength is not None:
        # Create pivot table for signal strength
        pivot_signal = returns_df.pivot_table(
            values='Signal',
            index='Day',
            columns='Hour',
            aggfunc='mean'
        )
        
        # Plot returns with signal strength as size
        ax = sns.heatmap(
            pivot_returns * 100,  # Convert to percentage
            cmap='RdYlGn',
            center=0,
            linewidths=0.5,
            annot=True,
            fmt='.2f'
        )
        
        # Overlay signal strength as markers
        for i in range(len(pivot_returns.index)):
            for j in range(len(pivot_returns.columns)):
                if not pd.isna(pivot_signal.iloc[i, j]):
                    size = abs(pivot_signal.iloc[i, j]) * 100  # Scale marker size
                    plt.scatter(
                        j + 0.5,
                        i + 0.5,
                        s=size,
                        color='black' if pivot_signal.iloc[i, j] >= 0 else 'white',
                        alpha=0.5,
                        edgecolors='gray'
                    )
    else:
        # Plot just returns
        sns.heatmap(
            pivot_returns * 100,  # Convert to percentage
            cmap='RdYlGn',
            center=0,
            linewidths=0.5,
            annot=True,
            fmt='.2f'
        )
    
    plt.title(title)
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved returns heatmap to {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()