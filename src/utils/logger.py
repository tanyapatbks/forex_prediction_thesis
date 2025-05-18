"""
Logging utility for the Forex prediction project.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

from src.utils.config import LOGS_DIR


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name (str): Name of the logger
        log_file (str, optional): Path to the log file. If None, a default name will be generated.
        level (int, optional): Logging level. Defaults to logging.INFO.
        console_output (bool, optional): Whether to output logs to console. Defaults to True.
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOGS_DIR, f"{name}_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_experiment_logger(
    experiment_name: str,
    model_name: str,
    currency_pair: str,
) -> logging.Logger:
    """
    Get a logger specifically configured for a model training/evaluation experiment.
    
    Args:
        experiment_name (str): Name of the experiment (e.g., "training", "evaluation")
        model_name (str): Name of the model (e.g., "CNN-LSTM", "TFT")
        currency_pair (str): Currency pair code (e.g., "EURUSD", "GBPUSD")
        
    Returns:
        logging.Logger: Configured logger
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_name = f"{experiment_name}_{model_name}_{currency_pair}"
    log_file = os.path.join(LOGS_DIR, f"{logger_name}_{timestamp}.log")
    
    return setup_logger(logger_name, log_file)