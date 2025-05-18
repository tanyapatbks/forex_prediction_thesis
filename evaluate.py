"""
Script for evaluating Forex prediction models.
"""

import os
import argparse
import time
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.utils.logger import setup_logger
from src.utils.config import (
    CURRENCY_PAIRS, RESULTS_DIR, CONFIG_DIR, 
    MODEL_TYPES, TEST_START_DATE, TEST_END_DATE
)

# Set up argument parser
parser = argparse.ArgumentParser(description='Evaluate Forex Prediction Models')
parser.add_argument('--model', type=str, choices=MODEL_TYPES + ['all', 'bagging'], default='all',
                    help=f'Model to evaluate. Options: {", ".join(MODEL_TYPES)}, bagging, or all. Default: all')
parser.add_argument('--currency', type=str, choices=CURRENCY_PAIRS + ['all'], default='all',
                    help=f'Currency pair to evaluate on. Options: {", ".join(CURRENCY_PAIRS)}, or all. Default: all')
parser.add_argument('--config', type=str, default='default_config.json',
                    help='Configuration file to use. Default: default_config.json')
parser.add_argument('--log', type=str, default='info',
                    help='Logging level. Options: debug, info, warning, error. Default: info')
parser.add_argument('--visualize', action='store_true',
                    help='Generate visualizations. Default: False')
parser.add_argument('--compare', action='store_true',
                    help='Compare different models. Default: False')
parser.add_argument('--market_conditions', action='store_true',
                    help='Analyze performance by market conditions. Default: False')

args = parser.parse_args()

# Set up logger
log_level = getattr(logging, args.log.upper())
logger = setup_logger("evaluate", level=log_level)

# Load configuration
config_path = os.path.join(CONFIG_DIR, args.config)
if not os.path.exists(config_path):
    logger.warning(f"Configuration file {config_path} not found. Using default parameters.")
    config = {}
else:
    with open(config_path, 'r') as f:
        config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")

# Determine models to evaluate
if args.model == 'all':
    models = MODEL_TYPES
elif args.model == 'bagging':
    models = [f"Bagging_{m}" for m in MODEL_TYPES]
else:
    models = [args.model]

# Determine currency pairs to evaluate on
if args.currency == 'all':
    pairs = CURRENCY_PAIRS
else:
    pairs = [args.currency]

def evaluate_models():
    """Evaluate Forex prediction models."""
    logger.info("Starting model evaluation")
    
    # Record start time for performance tracking
    start_time = time.time()
    
    from src.features.feature_enhancement import load_feature_data
    from src.data.data_loader import load_raw_data
    from src.evaluation.performance_metrics import (
        evaluate_trading_performance, 
        calculate_buy_hold_performance,
        identify_market_conditions,
        evaluate_performance_by_market_condition,
        compare_models_performance,
        save_performance_results,
        save_trading_summary,
        plot_equity_curves,
        plot_drawdowns,
        plot_market_conditions_performance,
        create_performance_report
    )
    
    # Get evaluation parameters from config
    eval_params = config.get('evaluation', {})
    
    # Dictionary to store all metrics
    all_metrics = {}
    buy_hold_metrics = {}
    market_condition_performance = {}
    equity_curves = {}
    drawdowns = {}
    
    # Evaluate models for each currency pair
    for pair in pairs:
        all_metrics[pair] = {}
        market_condition_performance[pair] = {}
        
        try:
            # Load test data
            test_data = load_feature_data(pair, 'test')
            # Load raw price data for trading simulation
            price_data = load_raw_data(pair)
        except FileNotFoundError as e:
            logger.error(f"Data for {pair} not found. Error: {str(e)}")
            continue
        
        # Find test data start index in price data
        test_start_date = test_data.index[0]
        try:
            test_start_idx = price_data.index.get_loc(test_start_date)
        except KeyError:
            # Find the closest date
            test_start_idx = price_data.index.searchsorted(test_start_date)
            logger.warning(f"Exact test start date not found in price data. Using closest date at index {test_start_idx}.")
        
        # Calculate buy & hold performance
        buy_hold_metrics[pair] = calculate_buy_hold_performance(
            price_data,
            test_start_idx,
            len(test_data),
            risk_free_rate=eval_params.get('risk_free_rate', 0.0)
        )
        
        # Identify market conditions if requested
        if args.market_conditions:
            market_conditions = identify_market_conditions(
                price_data.iloc[test_start_idx:test_start_idx+len(test_data)].copy(),
                window=eval_params.get('market_condition_window', 20),
                threshold=eval_params.get('market_condition_threshold', 0.05)
            )
        else:
            market_conditions = None
        
        # Evaluate each model
        for model_type in models:
            logger.info(f"Evaluating {model_type} model for {pair}")
            
            try:
                # Handle different model types
                if model_type.startswith('Bagging_'):
                    # For Bagging models
                    base_model_type = model_type.split('_')[1]
                    
                    from src.models.bagging_model import BaggingWrapper
                    model = BaggingWrapper.load(f"bagging_{base_model_type.lower()}")
                    
                    # Prepare test data based on base model type
                    if base_model_type == 'CNN-LSTM':
                        from src.models.cnn_lstm import create_sequences
                        
                        # Get model-specific parameters
                        cnn_lstm_params = eval_params.get('CNN-LSTM', {})
                        sequence_length = cnn_lstm_params.get('sequence_length', 60)
                        
                        # Prepare sequences
                        X_test, y_test = create_sequences(
                            test_data,
                            target_col='target',
                            sequence_length=sequence_length,
                            prediction_steps=1,
                            shuffle=False
                        )
                    
                    elif base_model_type == 'TFT':
                        from src.models.tft import prepare_data_for_tft
                        
                        # Get one of the component models to extract parameters
                        component_model = next(iter(model.models.values()))
                        training_data = component_model.training_set
                        
                        # Create test dataset with the same parameters as training
                        _, _, X_test, _ = prepare_data_for_tft(
                            test_data,  # Use test data for all splits as we only need the test part
                            test_data,
                            test_data,
                            target_variable='target',
                            max_encoder_length=training_data.max_encoder_length,
                            max_prediction_length=training_data.max_prediction_length,
                            time_varying_known_categoricals=training_data.time_varying_known_categoricals,
                            time_varying_known_reals=training_data.time_varying_known_reals,
                            time_varying_unknown_categoricals=training_data.time_varying_unknown_categoricals,
                            time_varying_unknown_reals=training_data.time_varying_unknown_reals,
                            static_categoricals=training_data.static_categoricals,
                            static_reals=training_data.static_reals,
                            target_normalizer=training_data.target_normalizer
                        )
                        
                        y_test = None  # Not used directly in TFT evaluation
                    
                    elif base_model_type == 'XGBoost':
                        from src.models.xgboost_model import prepare_data_for_xgboost
                        
                        # Get model-specific parameters
                        xgboost_params = eval_params.get('XGBoost', {})
                        sequence_length = xgboost_params.get('sequence_length', 1)
                        
                        # Prepare data for XGBoost
                        X_test, y_test, _ = prepare_data_for_xgboost(
                            test_data,
                            target_col='target',
                            sequence_length=sequence_length,
                            prediction_steps=1,
                            shuffle=False
                        )
                    
                    # Model type for evaluation function
                    eval_model_type = 'Bagging'
                
                else:
                    # For regular models
                    if model_type == 'CNN-LSTM':
                        from src.models.cnn_lstm import load_cnn_lstm_model, create_sequences
                        
                        # Get model-specific parameters
                        cnn_lstm_params = eval_params.get('CNN-LSTM', {})
                        sequence_length = cnn_lstm_params.get('sequence_length', 60)
                        
                        # Prepare sequences
                        X_test, y_test = create_sequences(
                            test_data,
                            target_col='target',
                            sequence_length=sequence_length,
                            prediction_steps=1,
                            shuffle=False
                        )
                        
                        # Load model
                        model, _ = load_cnn_lstm_model(f"cnn_lstm_{pair}")
                    
                    elif model_type == 'TFT':
                        from src.models.tft import load_tft_model, prepare_data_for_tft
                        
                        # Load model and datasets
                        model, (training_data, validation_data) = load_tft_model(f"tft_{pair}")
                        
                        # Create test dataset with the same parameters as training
                        _, _, X_test, _ = prepare_data_for_tft(
                            test_data,  # Use test data for all splits as we only need the test part
                            test_data,
                            test_data,
                            target_variable='target',
                            max_encoder_length=training_data.max_encoder_length,
                            max_prediction_length=training_data.max_prediction_length,
                            time_varying_known_categoricals=training_data.time_varying_known_categoricals,
                            time_varying_known_reals=training_data.time_varying_known_reals,
                            time_varying_unknown_categoricals=training_data.time_varying_unknown_categoricals,
                            time_varying_unknown_reals=training_data.time_varying_unknown_reals,
                            static_categoricals=training_data.static_categoricals,
                            static_reals=training_data.static_reals,
                            target_normalizer=training_data.target_normalizer
                        )
                        
                        y_test = None  # Not used directly in TFT evaluation
                    
                    elif model_type == 'XGBoost':
                        from src.models.xgboost_model import load_xgboost_model, prepare_data_for_xgboost
                        
                        # Get model-specific parameters
                        xgboost_params = eval_params.get('XGBoost', {})
                        sequence_length = xgboost_params.get('sequence_length', 1)
                        
                        # Prepare data for XGBoost
                        X_test, y_test, _ = prepare_data_for_xgboost(
                            test_data,
                            target_col='target',
                            sequence_length=sequence_length,
                            prediction_steps=1,
                            shuffle=False
                        )
                        
                        # Load model
                        model, _, _ = load_xgboost_model(f"xgboost_{pair}")
                    
                    # Model type for evaluation function
                    eval_model_type = model_type
                
                # Evaluate trading performance
                metrics, summary = evaluate_trading_performance(
                    model,
                    X_test,
                    price_data,
                    test_start_idx,
                    eval_model_type,
                    multi_class=False,
                    transaction_cost=eval_params.get('transaction_cost', 0.0001),
                    risk_free_rate=eval_params.get('risk_free_rate', 0.0)
                )
                
                # Store metrics
                all_metrics[pair][model_type] = metrics
                
                # Store equity curve and drawdown for plotting
                equity_curves[f"{model_type}_{pair}"] = summary['Equity']
                drawdowns[f"{model_type}_{pair}"] = summary['Drawdown']
                
                # Evaluate performance by market condition if requested
                if args.market_conditions and market_conditions is not None:
                    condition_perf = evaluate_performance_by_market_condition(
                        summary['Return'],
                        market_conditions
                    )
                    
                    market_condition_performance[pair][model_type] = condition_perf
                
                # Save results
                save_performance_results(
                    {
                        'metrics': metrics,
                        'market_condition_performance': condition_perf if args.market_conditions and market_conditions is not None else None
                    },
                    model_type,
                    pair
                )
                
                save_trading_summary(summary, model_type, pair)
                
                # Plot results if requested
                if args.visualize:
                    # Plot equity curve for this model
                    plot_equity_curves(
                        {model_type: summary['Equity'], 'Buy & Hold': summary['Equity'].iloc[0] * (1 + price_data['Close'].pct_change().iloc[test_start_idx:test_start_idx+len(summary)].cumsum())},
                        title=f'Equity Curve: {model_type} vs Buy & Hold ({pair})',
                        save_path=os.path.join(RESULTS_DIR, f"{model_type}_{pair}_equity.png"),
                        show_plot=args.visualize
                    )
                    
                    # Plot drawdown
                    plot_drawdowns(
                        {model_type: summary['Drawdown']},
                        title=f'Drawdown: {model_type} ({pair})',
                        save_path=os.path.join(RESULTS_DIR, f"{model_type}_{pair}_drawdown.png"),
                        show_plot=args.visualize
                    )
                    
                    # Plot performance by market condition if requested
                    if args.market_conditions and market_conditions is not None:
                        plot_market_conditions_performance(
                            condition_perf,
                            metric='total_return',
                            title=f'Performance by Market Condition: {model_type} ({pair})',
                            save_path=os.path.join(RESULTS_DIR, f"{model_type}_{pair}_market_conditions.png"),
                            show_plot=args.visualize
                        )
                
                logger.info(f"Completed evaluation of {model_type} model for {pair}")
            
            except Exception as e:
                logger.error(f"Error evaluating {model_type} model for {pair}: {str(e)}")
                continue
    
    # Compare models if requested
    if args.compare and all_metrics:
        logger.info("Comparing models performance")
        
        # Create and save performance report
        report = create_performance_report(
            all_metrics,
            buy_hold_metrics,
            market_condition_performance if args.market_conditions else {}
        )
        
        report_path = os.path.join(RESULTS_DIR, "performance_report.csv")
        report.to_csv(report_path)
        logger.info(f"Saved performance report to {report_path}")
        
        # Plot comparison across all models if requested
        if args.visualize and equity_curves:
            # Group equity curves by currency pair
            for pair in pairs:
                pair_equity_curves = {k: v for k, v in equity_curves.items() if k.endswith(f"_{pair}")}
                
                if pair_equity_curves:
                    # Rename keys to remove pair name
                    renamed_curves = {k.split('_')[0]: v for k, v in pair_equity_curves.items()}
                    
                    # Add buy and hold for comparison
                    buy_hold_equity = equity_curves[list(pair_equity_curves.keys())[0]].iloc[0] * (1 + price_data['Close'].pct_change().iloc[test_start_idx:test_start_idx+len(equity_curves[list(pair_equity_curves.keys())[0]])].cumsum())
                    renamed_curves['Buy & Hold'] = buy_hold_equity
                    
                    # Plot all equity curves together
                    plot_equity_curves(
                        renamed_curves,
                        title=f'Equity Curves Comparison - {pair}',
                        save_path=os.path.join(RESULTS_DIR, f"all_models_{pair}_equity.png"),
                        show_plot=args.visualize
                    )
            
            # Plot all equity curves together (all pairs and models)
            plot_equity_curves(
                equity_curves,
                title='Equity Curves Comparison (All Models and Pairs)',
                save_path=os.path.join(RESULTS_DIR, "all_models_equity.png"),
                show_plot=args.visualize
            )
            
            # Plot all drawdowns together
            plot_drawdowns(
                drawdowns,
                title='Drawdowns Comparison (All Models)',
                save_path=os.path.join(RESULTS_DIR, "all_models_drawdown.png"),
                show_plot=args.visualize
            )
    
    # Record end time and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Model evaluation completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    evaluate_models()