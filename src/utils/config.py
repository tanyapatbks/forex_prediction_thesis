"""
Configuration module for the Forex prediction project.
Contains constants and settings used throughout the project.
"""
import os
import json
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

# Model directories
MODELS_DIR = ROOT_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
RESULTS_DIR = MODELS_DIR / "results"

# Logs directory
LOGS_DIR = ROOT_DIR / "logs"

# Configuration directory
CONFIG_DIR = ROOT_DIR / "config"
HYPERPARAMS_DIR = CONFIG_DIR / "hyperparameters"

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)

# Currency pairs
CURRENCY_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
CURRENCY_CODES = {"EURUSD": "E+", "GBPUSD": "G+", "USDJPY": "J+", "BAGGING": "B"}

# Time periods
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2021-12-31"
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2022-04-30"

# Model types
MODEL_TYPES = ["CNN-LSTM", "TFT", "XGBoost"]

# Technical indicators categories
TREND_INDICATORS = ["SMA", "EMA", "MACD"]
MOMENTUM_INDICATORS = ["RSI", "Stochastic", "ROC"]
VOLATILITY_INDICATORS = ["Bollinger_Bands", "ATR"]

# All indicators
ALL_INDICATORS = TREND_INDICATORS + MOMENTUM_INDICATORS + VOLATILITY_INDICATORS

def load_config(config_file):
    """
    Load a configuration from a JSON file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = CONFIG_DIR / config_file
    
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config_data, config_file):
    """
    Save a configuration to a JSON file.
    
    Args:
        config_data (dict): Configuration dictionary
        config_file (str): Path to the configuration file
    """
    config_path = CONFIG_DIR / config_file
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)