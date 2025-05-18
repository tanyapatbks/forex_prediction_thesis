"""
Data loading utilities for the Forex prediction project.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.config import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger("data_loader")


def load_raw_data(
    currency_pair: str, 
    timeframe: str = "1H"
) -> pd.DataFrame:
    """
    Load raw data for a specific currency pair and timeframe.
    
    Args:
        currency_pair (str): Currency pair code (e.g., "EURUSD", "GBPUSD")
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        
    Returns:
        pd.DataFrame: DataFrame containing the raw data
    """
    file_path = os.path.join(RAW_DATA_DIR, f"{currency_pair}_{timeframe}.csv")
    
    if not os.path.exists(file_path):
        error_msg = f"Data file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading raw data for {currency_pair} ({timeframe}) from {file_path}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Convert 'Time' column to datetime and set as index
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
        
        # Sort by time index
        df.sort_index(inplace=True)
        
        logger.info(f"Successfully loaded data for {currency_pair}. Shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data for {currency_pair}: {str(e)}")
        raise


def load_all_currency_pairs(
    currency_pairs: List[str] = None,
    timeframe: str = "1H"
) -> Dict[str, pd.DataFrame]:
    """
    Load raw data for multiple currency pairs.
    
    Args:
        currency_pairs (List[str], optional): List of currency pairs to load. 
                                            If None, all available pairs will be loaded.
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping currency pair codes to their data
    """
    from src.utils.config import CURRENCY_PAIRS
    
    if currency_pairs is None:
        currency_pairs = CURRENCY_PAIRS
    
    data_dict = {}
    
    for pair in currency_pairs:
        try:
            data_dict[pair] = load_raw_data(pair, timeframe)
        except Exception as e:
            logger.error(f"Failed to load {pair}: {str(e)}")
    
    logger.info(f"Loaded data for {len(data_dict)} currency pairs")
    return data_dict


def split_data(
    data: pd.DataFrame,
    train_start: str = TRAIN_START_DATE,
    train_end: str = TRAIN_END_DATE,
    test_start: str = TEST_START_DATE,
    test_end: str = TEST_END_DATE,
    validation_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data (pd.DataFrame): DataFrame with datetime index
        train_start (str, optional): Start date for training data. Defaults to config.TRAIN_START_DATE.
        train_end (str, optional): End date for training data. Defaults to config.TRAIN_END_DATE.
        test_start (str, optional): Start date for test data. Defaults to config.TEST_START_DATE.
        test_end (str, optional): End date for test data. Defaults to config.TEST_END_DATE.
        validation_size (float, optional): Fraction of training data to use for validation. 
                                        Defaults to 0.2.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and test data
    """
    # Filter data for the training and test periods
    train_data = data.loc[train_start:train_end].copy()
    test_data = data.loc[test_start:test_end].copy()
    
    # Calculate the split point for validation
    num_train_samples = len(train_data)
    num_val_samples = int(num_train_samples * validation_size)
    split_idx = num_train_samples - num_val_samples
    
    # Split training data into train and validation
    train_data_final = train_data.iloc[:split_idx].copy()
    val_data = train_data.iloc[split_idx:].copy()
    
    logger.info(f"Data split - Train: {train_data_final.shape}, Validation: {val_data.shape}, Test: {test_data.shape}")
    
    return train_data_final, val_data, test_data


def save_processed_data(
    data: pd.DataFrame,
    currency_pair: str,
    dataset_type: str,
    timeframe: str = "1H"
) -> str:
    """
    Save processed data to the processed data directory.
    
    Args:
        data (pd.DataFrame): Processed data
        currency_pair (str): Currency pair code
        dataset_type (str): Type of dataset ('train', 'val', 'test')
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        
    Returns:
        str: Path where the data was saved
    """
    # Create filename
    filename = f"{currency_pair}_{timeframe}_{dataset_type}.csv"
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    
    # Save data
    data.to_csv(output_path)
    logger.info(f"Saved {dataset_type} data for {currency_pair} to {output_path}")
    
    return output_path


def load_processed_data(
    currency_pair: str,
    dataset_type: str,
    timeframe: str = "1H"
) -> pd.DataFrame:
    """
    Load processed data from the processed data directory.
    
    Args:
        currency_pair (str): Currency pair code
        dataset_type (str): Type of dataset ('train', 'val', 'test')
        timeframe (str, optional): Timeframe of the data. Defaults to "1H".
        
    Returns:
        pd.DataFrame: Processed data
    """
    # Create filename
    filename = f"{currency_pair}_{timeframe}_{dataset_type}.csv"
    file_path = os.path.join(PROCESSED_DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        error_msg = f"Processed data file not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load data
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {dataset_type} data for {currency_pair} from {file_path}")
    
    return data