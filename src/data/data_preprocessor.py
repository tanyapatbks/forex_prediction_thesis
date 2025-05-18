"""
Enhanced data preprocessor with continuous target variable.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

from src.utils.config import PROCESSED_DATA_DIR, CURRENCY_PAIRS
from src.utils.logger import setup_logger
from src.data.data_loader import load_all_currency_pairs, save_processed_data, split_data

# Set up logger
logger = setup_logger("data_preprocessor")


def check_missing_values(
    data_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Check for missing values in the data and report statistics.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary mapping currency pair names to DataFrames
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with missing value statistics for each currency pair
    """
    missing_stats = {}
    
    for pair_name, df in data_dict.items():
        # Get missing values count
        missing_count = df.isna().sum()
        
        # Calculate percentage of missing values
        missing_percentage = (missing_count / len(df)) * 100
        
        # Create statistics DataFrame
        stats_df = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing Percentage': missing_percentage
        })
        
        missing_stats[pair_name] = stats_df
        
        # Log the results
        if missing_count.sum() > 0:
            logger.warning(f"Missing values detected in {pair_name}:\n{stats_df[stats_df['Missing Count'] > 0]}")
        else:
            logger.info(f"No missing values found in {pair_name}")
    
    return missing_stats


def handle_missing_values(
    data_dict: Dict[str, pd.DataFrame],
    method: str = 'interpolate'
) -> Dict[str, pd.DataFrame]:
    """
    Handle missing values in the data.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary mapping currency pair names to DataFrames
        method (str, optional): Method to handle missing values. 
                              Options: 'interpolate', 'forward_fill', 'mean'.
                              Defaults to 'interpolate'.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with cleaned DataFrames
    """
    cleaned_data = {}
    
    for pair_name, df in data_dict.items():
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Count missing values before cleaning
        missing_before = cleaned_df.isna().sum().sum()
        
        # Apply the specified method
        if method == 'interpolate':
            cleaned_df = cleaned_df.interpolate(method='time')
            # Fill any remaining NaNs at the beginning or end
            cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
            
        elif method == 'forward_fill':
            cleaned_df = cleaned_df.fillna(method='ffill')
            # Fill any remaining NaNs at the beginning
            cleaned_df = cleaned_df.fillna(method='bfill')
            
        elif method == 'mean':
            cleaned_df = cleaned_df.fillna(cleaned_df.mean())
            # Fill any remaining NaNs (e.g., if a whole column is NaN)
            cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
            
        else:
            logger.error(f"Unknown method for handling missing values: {method}")
            raise ValueError(f"Unknown method for handling missing values: {method}")
        
        # Count missing values after cleaning
        missing_after = cleaned_df.isna().sum().sum()
        
        # Log the results
        if missing_before > 0:
            logger.info(f"Handled {missing_before} missing values in {pair_name} using {method}")
            if missing_after > 0:
                logger.warning(f"Still {missing_after} missing values in {pair_name} after cleaning")
                
        cleaned_data[pair_name] = cleaned_df
    
    return cleaned_data


def check_timeframe_consistency(
    data_dict: Dict[str, pd.DataFrame]
) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    """
    Check if all currency pairs have data for the same timeframe.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary mapping currency pair names to DataFrames
        
    Returns:
        Tuple[bool, Dict[str, Dict[str, int]]]: Tuple containing:
            - Boolean indicating if all pairs have consistent timeframes
            - Dictionary with timeframe statistics for each pair
    """
    timeframe_stats = {}
    
    # Get min and max date for each DataFrame
    for pair_name, df in data_dict.items():
        timeframe_stats[pair_name] = {
            'min_date': df.index.min(),
            'max_date': df.index.max(),
            'total_rows': len(df)
        }
    
    # Check if all pairs have the same min and max date
    first_pair = list(timeframe_stats.keys())[0]
    first_min = timeframe_stats[first_pair]['min_date']
    first_max = timeframe_stats[first_pair]['max_date']
    first_total = timeframe_stats[first_pair]['total_rows']
    
    all_consistent = True
    
    for pair_name, stats in timeframe_stats.items():
        if pair_name == first_pair:
            continue
            
        if stats['min_date'] != first_min or stats['max_date'] != first_max or stats['total_rows'] != first_total:
            all_consistent = False
            logger.warning(f"Timeframe inconsistency detected for {pair_name}")
            logger.warning(f"{pair_name}: {stats}")
            logger.warning(f"{first_pair}: {timeframe_stats[first_pair]}")
    
    if all_consistent:
        logger.info("All currency pairs have consistent timeframes")
    else:
        logger.warning("Timeframe inconsistency detected between currency pairs")
    
    return all_consistent, timeframe_stats


def align_timeframes(
    data_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Align the timeframes of all currency pairs to ensure consistency.
    
    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary mapping currency pair names to DataFrames
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with aligned DataFrames
    """
    # Get the common date range
    all_dates = set()
    
    # First, collect all unique dates across all pairs
    for df in data_dict.values():
        all_dates.update(df.index)
    
    all_dates = sorted(all_dates)
    
    # Create a new dictionary with aligned DataFrames
    aligned_data = {}
    
    for pair_name, df in data_dict.items():
        # Reindex to include all dates
        aligned_df = df.reindex(all_dates)
        
        # Count missing values after reindexing
        missing_count = aligned_df.isna().sum().sum()
        
        if missing_count > 0:
            logger.warning(f"After alignment, {pair_name} has {missing_count} missing values")
            
            # Interpolate missing values
            aligned_df = aligned_df.interpolate(method='time')
            aligned_df = aligned_df.fillna(method='ffill').fillna(method='bfill')
            
            # Check if any missing values remain
            remaining_missing = aligned_df.isna().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"After interpolation, {pair_name} still has {remaining_missing} missing values")
        
        aligned_data[pair_name] = aligned_df
    
    logger.info(f"All currency pairs aligned to a common timeframe with {len(all_dates)} data points")
    
    return aligned_data


def create_target_variable(
    data: pd.DataFrame,
    target_type: str = 'continuous',
    look_ahead: int = 1,
    threshold: float = 0.0,
    normalization_window: int = 20,
    min_max_scale: bool = True
) -> pd.DataFrame:
    """
    Create target variable for prediction.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        target_type (str, optional): Type of target variable to create. 
                                    Options: 'binary' (up/down), 'ternary' (-1/0/1), 'continuous' (normalized price change).
                                    Defaults to 'continuous'.
        look_ahead (int, optional): Number of periods to look ahead for target. Defaults to 1.
        threshold (float, optional): Threshold for classification (in percentage). Defaults to 0.0.
        normalization_window (int, optional): Window size for normalizing continuous target. Defaults to 20.
        min_max_scale (bool, optional): Whether to scale continuous target to [-1, 1]. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame with target variable added
    """
    # Make a copy of the data
    data_with_target = data.copy()
    
    # Calculate future price
    future_price = data_with_target['Close'].shift(-look_ahead)
    
    # Calculate price change and percentage
    price_change = future_price - data_with_target['Close']
    price_change_pct = price_change / data_with_target['Close'] * 100
    
    if target_type == 'binary':
        # Create binary target based on threshold (0 for down, 1 for up)
        data_with_target['target'] = (price_change_pct > threshold).astype(int)
        logger.info(f"Created binary target variable with look_ahead={look_ahead}, threshold={threshold}%")
        
    elif target_type == 'ternary':
        # Create ternary target (-1 for down, 0 for neutral, 1 for up)
        data_with_target['target'] = np.sign(price_change_pct - threshold)
        logger.info(f"Created ternary target variable with look_ahead={look_ahead}, threshold={threshold}%")
        
    elif target_type == 'continuous':
        # For continuous target, we want to normalize the price change to account for volatility
        # Calculate the volatility (standard deviation of returns) over a rolling window
        returns = data_with_target['Close'].pct_change()
        volatility = returns.rolling(window=normalization_window).std()
        
        # Normalize the price change by the volatility to get a measure of "strength"
        # This gives higher values when the price change is large relative to recent volatility
        normalized_change = price_change_pct / (volatility * 100 + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Optionally scale to range [-1, 1] using min-max scaling or tanh
        if min_max_scale:
            # Use hyperbolic tangent (tanh) to scale to [-1, 1]
            data_with_target['target'] = np.tanh(normalized_change)
        else:
            data_with_target['target'] = normalized_change
            
        logger.info(f"Created continuous target variable (normalized price change) with look_ahead={look_ahead}")
        
    else:
        # Use the actual price change percentage as target
        data_with_target['target'] = price_change_pct
        logger.info(f"Created price change percentage target variable with look_ahead={look_ahead}")
    
    # Drop the last 'look_ahead' rows since we don't have targets for them
    data_with_target = data_with_target.iloc[:-look_ahead]
    
    return data_with_target


def preprocess_data(
    currency_pairs: List[str] = None,
    timeframe: str = "1H",
    handle_missing_method: str = 'interpolate',
    target_type: str = 'continuous',  # Changed default to continuous
    look_ahead: int = 1,
    threshold: float = 0.0,
    normalization_window: int = 20,  # Added parameter for volatility window
    min_max_scale: bool = True,      # Added parameter for scaling
    train_start: str = None,
    train_end: str = None,
    test_start: str = None,
    test_end: str = None,
    validation_size: float = 0.2,
    save_data: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Complete preprocessing pipeline for Forex data.
    
    Args:
        currency_pairs (List[str], optional): List of currency pairs to process.
                                           If None, all configured pairs will be used.
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        handle_missing_method (str, optional): Method to handle missing values.
                                            Defaults to 'interpolate'.
        target_type (str, optional): Type of target variable to create. 
                                    Options: 'binary', 'ternary', 'continuous'.
                                    Defaults to 'continuous'.
        look_ahead (int, optional): Number of periods to look ahead for target. Defaults to 1.
        threshold (float, optional): Threshold for classification. Defaults to 0.0.
        normalization_window (int, optional): Window size for normalizing continuous target. Defaults to 20.
        min_max_scale (bool, optional): Whether to scale continuous target to [-1, 1]. Defaults to True.
        train_start (str, optional): Start date for training data. If None, uses default config.
        train_end (str, optional): End date for training data. If None, uses default config.
        test_start (str, optional): Start date for test data. If None, uses default config.
        test_end (str, optional): End date for test data. If None, uses default config.
        validation_size (float, optional): Fraction of training data for validation. 
                                        Defaults to 0.2.
        save_data (bool, optional): Whether to save processed data to files. Defaults to True.
        
    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Nested dictionary with processed data:
            - Outer key: Currency pair code
            - Inner key: 'train', 'val', or 'test'
            - Value: Processed DataFrame
    """
    from src.utils.config import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
    
    # Use default values from config if not provided
    train_start = train_start or TRAIN_START_DATE
    train_end = train_end or TRAIN_END_DATE
    test_start = test_start or TEST_START_DATE
    test_end = test_end or TEST_END_DATE
    
    # Use all configured currency pairs if none provided
    if currency_pairs is None:
        currency_pairs = CURRENCY_PAIRS
    
    logger.info(f"Starting preprocessing pipeline for currency pairs: {currency_pairs}")
    logger.info(f"Using target type: {target_type}")
    
    # Step 1: Load raw data
    raw_data_dict = load_all_currency_pairs(currency_pairs, timeframe)
    
    # Step 2: Check and handle missing values
    check_missing_values(raw_data_dict)
    cleaned_data_dict = handle_missing_values(raw_data_dict, method=handle_missing_method)
    
    # Step 3: Check and align timeframes
    is_consistent, _ = check_timeframe_consistency(cleaned_data_dict)
    if not is_consistent:
        logger.info("Aligning timeframes across all currency pairs")
        aligned_data_dict = align_timeframes(cleaned_data_dict)
    else:
        aligned_data_dict = cleaned_data_dict
    
    # Step 4: Create target variables for each currency pair
    data_with_targets = {}
    for pair_name, df in aligned_data_dict.items():
        data_with_targets[pair_name] = create_target_variable(
            df,
            target_type=target_type,
            look_ahead=look_ahead,
            threshold=threshold,
            normalization_window=normalization_window,
            min_max_scale=min_max_scale
        )
    
    # Step 5: Split data into train, validation, and test sets
    processed_data = {}
    
    for pair_name, df in data_with_targets.items():
        train_data, val_data, test_data = split_data(
            df,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            validation_size=validation_size
        )
        
        processed_data[pair_name] = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        
        # Save processed data if requested
        if save_data:
            save_processed_data(train_data, pair_name, 'train', timeframe)
            save_processed_data(val_data, pair_name, 'val', timeframe)
            save_processed_data(test_data, pair_name, 'test', timeframe)
    
    logger.info("Preprocessing pipeline completed successfully")
    
    return processed_data