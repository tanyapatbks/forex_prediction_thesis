"""
Feature enhancement utilities for the Forex prediction project.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from src.utils.config import FEATURES_DATA_DIR, CURRENCY_PAIRS
from src.utils.config import TREND_INDICATORS, MOMENTUM_INDICATORS, VOLATILITY_INDICATORS
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger("feature_enhancement")


def add_trend_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend indicators to the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with trend indicators added
    """
    df = data.copy()
    
    # Simple Moving Average (SMA)
    for window in [5, 10, 20, 50, 100]:
        indicator = SMAIndicator(close=df['Close'], window=window)
        df[f'SMA_{window}'] = indicator.sma_indicator()
    
    # Exponential Moving Average (EMA)
    for window in [5, 10, 20, 50, 100]:
        indicator = EMAIndicator(close=df['Close'], window=window)
        df[f'EMA_{window}'] = indicator.ema_indicator()
    
    # Moving Average Convergence Divergence (MACD)
    macd = MACD(
        close=df['Close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['MACD_line'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Price relative to moving averages
    for window in [10, 20, 50]:
        df[f'Close_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}'] - 1
        df[f'Close_to_EMA_{window}'] = df['Close'] / df[f'EMA_{window}'] - 1
    
    # Calculate crossovers
    df['SMA_5_10_cross'] = ((df['SMA_5'] > df['SMA_10']) & 
                            (df['SMA_5'].shift(1) <= df['SMA_10'].shift(1))).astype(int)
    df['SMA_10_20_cross'] = ((df['SMA_10'] > df['SMA_20']) & 
                             (df['SMA_10'].shift(1) <= df['SMA_20'].shift(1))).astype(int)
    df['EMA_5_10_cross'] = ((df['EMA_5'] > df['EMA_10']) & 
                            (df['EMA_5'].shift(1) <= df['EMA_10'].shift(1))).astype(int)
    df['EMA_10_20_cross'] = ((df['EMA_10'] > df['EMA_20']) & 
                             (df['EMA_10'].shift(1) <= df['EMA_20'].shift(1))).astype(int)
    
    logger.info("Added trend indicators")
    
    return df


def add_momentum_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum indicators to the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with momentum indicators added
    """
    df = data.copy()
    
    # Relative Strength Index (RSI)
    for window in [7, 14, 21]:
        indicator = RSIIndicator(close=df['Close'], window=window)
        df[f'RSI_{window}'] = indicator.rsi()
    
    # Stochastic Oscillator
    for window in [7, 14, 21]:
        indicator = StochasticOscillator(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=window,
            smooth_window=3
        )
        df[f'Stoch_{window}'] = indicator.stoch()
        df[f'Stoch_signal_{window}'] = indicator.stoch_signal()
    
    # Rate of Change (ROC)
    for window in [5, 10, 20]:
        indicator = ROCIndicator(close=df['Close'], window=window)
        df[f'ROC_{window}'] = indicator.roc()
    
    # RSI Divergence (simplified)
    for window in [14]:
        # Find local price highs and lows
        price_diff = df['Close'].diff()
        price_high = ((price_diff > 0) & (price_diff.shift(-1) < 0)).astype(int)
        price_low = ((price_diff < 0) & (price_diff.shift(-1) > 0)).astype(int)
        
        # Find local RSI highs and lows
        rsi = df[f'RSI_{window}']
        rsi_diff = rsi.diff()
        rsi_high = ((rsi_diff > 0) & (rsi_diff.shift(-1) < 0)).astype(int)
        rsi_low = ((rsi_diff < 0) & (rsi_diff.shift(-1) > 0)).astype(int)
        
        # Bullish divergence: price makes lower low but RSI makes higher low
        bullish_div = ((price_low == 1) & (price_low.shift(1) == 1) & 
                       (df['Close'] < df['Close'].shift(1)) & 
                       (rsi_low == 1) & (rsi_low.shift(1) == 1) & 
                       (rsi > rsi.shift(1))).astype(int)
        
        # Bearish divergence: price makes higher high but RSI makes lower high
        bearish_div = ((price_high == 1) & (price_high.shift(1) == 1) & 
                       (df['Close'] > df['Close'].shift(1)) & 
                       (rsi_high == 1) & (rsi_high.shift(1) == 1) & 
                       (rsi < rsi.shift(1))).astype(int)
        
        df[f'RSI_{window}_bullish_div'] = bullish_div
        df[f'RSI_{window}_bearish_div'] = bearish_div
    
    logger.info("Added momentum indicators")
    
    return df


def add_volatility_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility indicators to the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with volatility indicators added
    """
    df = data.copy()
    
    # Bollinger Bands
    for window in [20]:
        for std in [2.0]:
            indicator = BollingerBands(
                close=df['Close'],
                window=window,
                window_dev=std
            )
            df[f'BB_{window}_middle'] = indicator.bollinger_mavg()
            df[f'BB_{window}_upper'] = indicator.bollinger_hband()
            df[f'BB_{window}_lower'] = indicator.bollinger_lband()
            df[f'BB_{window}_width'] = (df[f'BB_{window}_upper'] - df[f'BB_{window}_lower']) / df[f'BB_{window}_middle']
            df[f'BB_{window}_pct_b'] = indicator.bollinger_pband()
    
    # Average True Range (ATR)
    for window in [7, 14, 21]:
        indicator = AverageTrueRange(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            window=window
        )
        df[f'ATR_{window}'] = indicator.average_true_range()
        # Normalized ATR (ATR relative to price)
        df[f'ATR_{window}_pct'] = df[f'ATR_{window}'] / df['Close'] * 100
    
    # Historical Volatility
    for window in [10, 20, 30]:
        # Calculate log returns
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        # Standard deviation of log returns (annualized)
        df[f'Volatility_{window}'] = log_returns.rolling(window=window).std() * np.sqrt(252)
    
    logger.info("Added volatility indicators")
    
    return df


def add_price_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add price pattern indicators to the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with price pattern indicators added
    """
    df = data.copy()
    
    # Price momentum and reversion
    for window in [1, 3, 5, 10, 20]:
        # Price changes
        df[f'Price_change_{window}'] = df['Close'].pct_change(window) * 100
        
        # Normalized price
        if window > 1:
            df[f'Normalized_price_{window}'] = df['Close'] / df['Close'].rolling(window=window).mean()
    
    # Candlestick patterns (simplified)
    # Body size
    df['Candle_body'] = abs(df['Close'] - df['Open'])
    df['Candle_body_pct'] = df['Candle_body'] / df['Open'] * 100
    
    # Upper shadow
    df['Upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Upper_shadow_pct'] = df['Upper_shadow'] / df['Open'] * 100
    
    # Lower shadow
    df['Lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Lower_shadow_pct'] = df['Lower_shadow'] / df['Open'] * 100
    
    # Candle direction
    df['Candle_direction'] = np.sign(df['Close'] - df['Open'])
    
    # Major price levels
    for window in [20, 50]:
        # Highest high and lowest low
        df[f'Highest_high_{window}'] = df['High'].rolling(window=window).max()
        df[f'Lowest_low_{window}'] = df['Low'].rolling(window=window).min()
        
        # Price distance from high/low
        df[f'Pct_from_high_{window}'] = (df['Close'] - df[f'Highest_high_{window}']) / df[f'Highest_high_{window}'] * 100
        df[f'Pct_from_low_{window}'] = (df['Close'] - df[f'Lowest_low_{window}']) / df[f'Lowest_low_{window}'] * 100
    
    logger.info("Added price pattern indicators")
    
    return df


def add_volume_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based indicators to the DataFrame if volume data is available.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with volume indicators added
    """
    df = data.copy()
    
    # Check if Volume column exists
    if 'Volume' not in df.columns:
        logger.warning("Volume data not available - skipping volume indicators")
        return df
    
    # Basic volume indicators
    for window in [5, 10, 20]:
        # Volume moving average
        df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window).mean()
        # Volume relative to its moving average
        df[f'Volume_ratio_{window}'] = df['Volume'] / df[f'Volume_SMA_{window}']
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Chaikin Money Flow (CMF)
    for window in [20]:
        mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        mf_volume = mf_multiplier * df['Volume']
        df[f'CMF_{window}'] = (mf_volume.rolling(window=window).sum() / 
                              df['Volume'].rolling(window=window).sum())
    
    # Volume weighted price
    df['VWAP_daily'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    logger.info("Added volume indicators")
    
    return df


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with datetime index
        
    Returns:
        pd.DataFrame: DataFrame with time features added
    """
    df = data.copy()
    
    # Basic time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Session indicators
    # Asia session (approximately)
    df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
    # Europe session (approximately)
    df['europe_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    # US session (approximately)
    df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
    
    # Day type indicators
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = (df['day_of_month'] == 1).astype(int)
    df['is_month_end'] = (df['day_of_month'] == df.index.days_in_month).astype(int)
    df['is_quarter_start'] = ((df['month'] - 1) % 3 == 0).astype(int) & (df['day_of_month'] == 1).astype(int)
    df['is_quarter_end'] = (df['month'] % 3 == 0).astype(int) & (df['is_month_end'] == 1).astype(int)
    
    # Cyclical encoding of time features
    # Hour of day (0-23) -> cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week (0-6) -> cyclical encoding
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Month (1-12) -> cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    logger.info("Added time features")
    
    return df


def add_all_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators and features to the DataFrame.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        
    Returns:
        pd.DataFrame: DataFrame with all features added
    """
    # Apply transformations sequentially
    df = data.copy()
    
    df = add_trend_indicators(df)
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_price_patterns(df)
    df = add_volume_indicators(df)
    df = add_time_features(df)
    
    # Drop rows with NaN values
    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)
    
    if initial_len > final_len:
        logger.info(f"Dropped {initial_len - final_len} rows with NaN values")
    
    return df


def select_features_correlation(
    data: pd.DataFrame,
    target_col: str = 'target',
    threshold: float = 0.05,
    top_n: int = 50
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features based on their correlation with the target variable.
    
    Args:
        data (pd.DataFrame): DataFrame with features
        target_col (str, optional): Target column name. Defaults to 'target'.
        threshold (float, optional): Correlation threshold. Defaults to 0.05.
        top_n (int, optional): Maximum number of features to select. Defaults to 50.
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with selected features and list of selected feature names
    """
    # Calculate correlation with target
    correlation = data.corr()[target_col].abs().sort_values(ascending=False)
    
    # Remove target itself from correlation list
    correlation = correlation.drop(target_col)
    
    # Select features above threshold
    selected_features = correlation[correlation > threshold].index.tolist()
    
    # Limit to top_n features
    selected_features = selected_features[:top_n]
    
    # Create DataFrame with selected features and target
    data_selected = data[selected_features + [target_col]].copy()
    
    logger.info(f"Selected {len(selected_features)} features using correlation method")
    
    return data_selected, selected_features


def select_features_mutual_info(
    data: pd.DataFrame,
    target_col: str = 'target',
    top_n: int = 50
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features based on mutual information with the target variable.
    
    Args:
        data (pd.DataFrame): DataFrame with features
        target_col (str, optional): Target column name. Defaults to 'target'.
        top_n (int, optional): Maximum number of features to select. Defaults to 50.
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with selected features and list of selected feature names
    """
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Calculate mutual information
    if y.nunique() <= 2:  # Binary classification
        mi = mutual_info_classif(X, y)
    else:  # Regression or multi-class
        mi = mutual_info_classif(X, y)
    
    # Create DataFrame with feature names and their importance
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': mi
    }).sort_values('MI_Score', ascending=False)
    
    # Select top N features
    selected_features = mi_df.head(top_n)['Feature'].tolist()
    
    # Create DataFrame with selected features and target
    data_selected = data[selected_features + [target_col]].copy()
    
    logger.info(f"Selected {len(selected_features)} features using mutual information method")
    
    return data_selected, selected_features


def select_features_random_forest(
    data: pd.DataFrame,
    target_col: str = 'target',
    top_n: int = 50
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features based on Random Forest importance.
    
    Args:
        data (pd.DataFrame): DataFrame with features
        target_col (str, optional): Target column name. Defaults to 'target'.
        top_n (int, optional): Maximum number of features to select. Defaults to 50.
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame with selected features and list of selected feature names
    """
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Initialize Random Forest model
    if y.nunique() <= 2:  # Binary classification
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # Regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit model
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame with feature names and their importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Select top N features
    selected_features = importance_df.head(top_n)['Feature'].tolist()
    
    # Create DataFrame with selected features and target
    data_selected = data[selected_features + [target_col]].copy()
    
    logger.info(f"Selected {len(selected_features)} features using random forest importance method")
    
    return data_selected, selected_features


def save_selected_features(
    features_list: List[str],
    currency_pair: str,
    method: str,
    dataset_type: str = 'train',
    timeframe: str = "1H"
) -> str:
    """
    Save list of selected features to a file.
    
    Args:
        features_list (List[str]): List of selected feature names
        currency_pair (str): Currency pair code
        method (str): Feature selection method name
        dataset_type (str, optional): Type of dataset ('train'). Defaults to 'train'.
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        
    Returns:
        str: Path where the features list was saved
    """
    # Create filename
    filename = f"{currency_pair}_{timeframe}_{dataset_type}_{method}_features.txt"
    output_path = os.path.join(FEATURES_DATA_DIR, filename)
    
    # Save features list
    with open(output_path, 'w') as f:
        for feature in features_list:
            f.write(f"{feature}\n")
    
    logger.info(f"Saved selected features for {currency_pair} to {output_path}")
    
    return output_path


def load_selected_features(
    currency_pair: str,
    method: str,
    dataset_type: str = 'train',
    timeframe: str = "1H"
) -> List[str]:
    """
    Load list of selected features from a file.
    
    Args:
        currency_pair (str): Currency pair code
        method (str): Feature selection method name
        dataset_type (str, optional): Type of dataset ('train'). Defaults to 'train'.
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        
    Returns:
        List[str]: List of selected feature names
    """
    # Create filename
    filename = f"{currency_pair}_{timeframe}_{dataset_type}_{method}_features.txt"
    file_path = os.path.join(FEATURES_DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        error_msg = f"Selected features file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load features list
    with open(file_path, 'r') as f:
        features_list = [line.strip() for line in f.readlines()]
    
    logger.info(f"Loaded selected features for {currency_pair} from {file_path}")
    
    return features_list


def save_feature_data(
    data: pd.DataFrame,
    currency_pair: str,
    dataset_type: str,
    timeframe: str = "1H"
) -> str:
    """
    Save feature data to a CSV file.
    
    Args:
        data (pd.DataFrame): DataFrame with features
        currency_pair (str): Currency pair code
        dataset_type (str): Type of dataset ('train', 'val', 'test')
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        
    Returns:
        str: Path where the data was saved
    """
    # Create filename
    filename = f"{currency_pair}_{timeframe}_{dataset_type}_features.csv"
    output_path = os.path.join(FEATURES_DATA_DIR, filename)
    
    # Save data
    data.to_csv(output_path)
    logger.info(f"Saved feature data for {currency_pair} ({dataset_type}) to {output_path}")
    
    return output_path


def load_feature_data(
    currency_pair: str,
    dataset_type: str,
    timeframe: str = "1H"
) -> pd.DataFrame:
    """
    Load feature data from a CSV file.
    
    Args:
        currency_pair (str): Currency pair code
        dataset_type (str): Type of dataset ('train', 'val', 'test')
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        
    Returns:
        pd.DataFrame: DataFrame with features
    """
    # Create filename
    filename = f"{currency_pair}_{timeframe}_{dataset_type}_features.csv"
    file_path = os.path.join(FEATURES_DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        error_msg = f"Feature data file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load data
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded feature data for {currency_pair} ({dataset_type}) from {file_path}")
    
    return data


def enhance_and_select_features(
    data_dict: Dict[str, Dict[str, pd.DataFrame]],
    selection_method: str = 'random_forest',
    top_n: int = 50,
    timeframe: str = "1H",
    save_data: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Complete pipeline for feature enhancement and selection.
    
    Args:
        data_dict (Dict[str, Dict[str, pd.DataFrame]]): Dictionary with processed data:
            - Outer key: Currency pair code
            - Inner key: 'train', 'val', or 'test'
            - Value: Processed DataFrame
        selection_method (str, optional): Method for feature selection.
                                        Options: 'correlation', 'mutual_info', 'random_forest'.
                                        Defaults to 'random_forest'.
        top_n (int, optional): Maximum number of features to select. Defaults to 50.
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        save_data (bool, optional): Whether to save enhanced feature data to files. Defaults to True.
        
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Dictionary with enhanced data:
            - Outer key: Currency pair code
            - Inner key: 'train', 'val', or 'test'
            - Value: DataFrame with enhanced and selected features
    """
    logger.info(f"Starting feature enhancement and selection pipeline using {selection_method} method")
    
    enhanced_data = {}
    
    for pair_name, datasets in data_dict.items():
        logger.info(f"Enhancing features for {pair_name}")
        
        # Add features to each dataset
        train_enhanced = add_all_features(datasets['train'])
        val_enhanced = add_all_features(datasets['val'])
        test_enhanced = add_all_features(datasets['test'])
        
        # Select features based on the training set
        if selection_method == 'correlation':
            train_selected, selected_features = select_features_correlation(
                train_enhanced,
                top_n=top_n
            )
        elif selection_method == 'mutual_info':
            train_selected, selected_features = select_features_mutual_info(
                train_enhanced,
                top_n=top_n
            )
        elif selection_method == 'random_forest':
            train_selected, selected_features = select_features_random_forest(
                train_enhanced,
                top_n=top_n
            )
        else:
            logger.error(f"Unknown feature selection method: {selection_method}")
            raise ValueError(f"Unknown feature selection method: {selection_method}")
        
        # Apply the same feature selection to validation and test sets
        # Make sure 'target' is included in the selected features list
        if 'target' not in selected_features:
            selected_features.append('target')
            
        val_selected = val_enhanced[selected_features].copy()
        test_selected = test_enhanced[selected_features].copy()
        
        # Save to dictionary
        enhanced_data[pair_name] = {
            'train': train_selected,
            'val': val_selected,
            'test': test_selected
        }
        
        # Save data and feature list to files if requested
        if save_data:
            save_selected_features(
                selected_features,
                pair_name,
                selection_method,
                'train',
                timeframe
            )
            
            save_feature_data(train_selected, pair_name, 'train', timeframe)
            save_feature_data(val_selected, pair_name, 'val', timeframe)
            save_feature_data(test_selected, pair_name, 'test', timeframe)
    
    logger.info("Feature enhancement and selection pipeline completed successfully")
    
    return enhanced_data


def visualize_feature_importance(
    data: pd.DataFrame,
    feature_selection_method: str,
    currency_pair: str,
    target_col: str = 'target',
    top_n: int = 20,
    show_plot: bool = True,
    save_plot: bool = True
) -> None:
    """
    Visualize feature importance based on different methods.
    
    Args:
        data (pd.DataFrame): DataFrame with features
        feature_selection_method (str): Method for feature selection
        currency_pair (str): Currency pair code
        target_col (str, optional): Target column name. Defaults to 'target'.
        top_n (int, optional): Number of top features to display. Defaults to 20.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
        save_plot (bool, optional): Whether to save the plot. Defaults to True.
    """
    from src.utils.visualization import plot_feature_importance
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Calculate feature importance based on the selected method
    if feature_selection_method == 'correlation':
        # Calculate correlation with target
        importances = data.corr()[target_col].abs()
        importances = importances.drop(target_col)
        feature_names = importances.index.tolist()
        importances = importances.values
        
    elif feature_selection_method == 'mutual_info':
        # Calculate mutual information
        importances = mutual_info_classif(X, y)
        feature_names = X.columns.tolist()
        
    elif feature_selection_method == 'random_forest':
        # Train a Random Forest model
        if y.nunique() <= 2:  # Binary classification
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        importances = model.feature_importances_
        feature_names = X.columns.tolist()
        
    else:
        logger.error(f"Unknown feature selection method: {feature_selection_method}")
        raise ValueError(f"Unknown feature selection method: {feature_selection_method}")
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    
    # Prepare data for plotting
    top_feature_names = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    # Create plot
    title = f"Top {top_n} Features Importance for {currency_pair} using {feature_selection_method.capitalize()}"
    save_path = os.path.join(FEATURES_DATA_DIR, f"{currency_pair}_{feature_selection_method}_importance.png") if save_plot else None
    
    plot_feature_importance(
        top_feature_names,
        top_importances,
        title=title,
        save_path=save_path,
        show=show_plot
    )
    
    logger.info(f"Feature importance visualization created for {currency_pair}")