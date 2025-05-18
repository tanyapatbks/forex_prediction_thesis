"""
Main script for the Forex prediction project.
Run the entire pipeline from data preprocessing to model evaluation.
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
    CURRENCY_PAIRS, TRAINED_MODELS_DIR, RESULTS_DIR, CONFIG_DIR, 
    MODEL_TYPES, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE
)

# Set up argument parser
parser = argparse.ArgumentParser(description='Forex Prediction Pipeline')
parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], default=0,
                    help='Stage to run: 1-Data Preprocessing, 2-Feature Enhancement, 3-Model Training, 4-Model Evaluation. Default: run all stages.')
parser.add_argument('--model', type=str, choices=MODEL_TYPES + ['all'], default='all',
                    help=f'Model to use. Options: {", ".join(MODEL_TYPES)} or all. Default: all')
parser.add_argument('--currency', type=str, choices=CURRENCY_PAIRS + ['all', 'bagging'], default='all',
                    help=f'Currency pair to process. Options: {", ".join(CURRENCY_PAIRS)}, all, or bagging. Default: all')
parser.add_argument('--config', type=str, default='default_config.json',
                    help='Configuration file to use. Default: default_config.json')
parser.add_argument('--log', type=str, default='info',
                    help='Logging level. Options: debug, info, warning, error. Default: info')
parser.add_argument('--no_tuning', action='store_true',
                    help='Skip hyperparameter tuning. Default: False')
parser.add_argument('--force', action='store_true',
                    help='Force reprocessing of all data and retraining of all models. Default: False')
parser.add_argument('--visualize', action='store_true',
                    help='Generate visualizations. Default: False')

args = parser.parse_args()

# Set up logger
log_level = getattr(logging, args.log.upper())
logger = setup_logger("main", level=log_level)

# Load configuration
config_path = os.path.join(CONFIG_DIR, args.config)
if not os.path.exists(config_path):
    logger.warning(f"Configuration file {config_path} not found. Using default parameters.")
    config = {}
else:
    with open(config_path, 'r') as f:
        config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")

# Determine currency pairs to process
if args.currency == 'all':
    pairs = CURRENCY_PAIRS
elif args.currency == 'bagging':
    pairs = ['BAGGING']
else:
    pairs = [args.currency]

# Determine models to use
if args.model == 'all':
    models = MODEL_TYPES
else:
    models = [args.model]

# Main pipeline function
def run_pipeline():
    logger.info("Starting Forex Prediction Pipeline")
    
    # Record start time for performance tracking
    start_time = time.time()
    
    # Stage 1: Data Preprocessing
    if args.stage == 0 or args.stage == 1:
        logger.info("Stage 1: Data Preprocessing")
        from src.data.data_preprocessor import preprocess_data
        
        # Get preprocessing parameters from config
        preprocess_params = config.get('preprocessing', {})
        
        # Set dates from config if provided
        train_start = preprocess_params.get('train_start_date', TRAIN_START_DATE)
        train_end = preprocess_params.get('train_end_date', TRAIN_END_DATE)
        test_start = preprocess_params.get('test_start_date', TEST_START_DATE)
        test_end = preprocess_params.get('test_end_date', TEST_END_DATE)
        
        # Process all currency pairs except 'BAGGING' which is handled differently
        process_pairs = [p for p in pairs if p != 'BAGGING']
        if process_pairs:
            processed_data = preprocess_data(
                currency_pairs=process_pairs,
                timeframe=preprocess_params.get('timeframe', "1H"),
                handle_missing_method=preprocess_params.get('handle_missing_method', 'interpolate'),
                target_type=preprocess_params.get('target_type', 'binary'),
                look_ahead=preprocess_params.get('look_ahead', 1),
                threshold=preprocess_params.get('threshold', 0.0),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                validation_size=preprocess_params.get('validation_size', 0.2),
                save_data=True
            )
            logger.info(f"Processed data for {len(process_pairs)} currency pairs")
        else:
            logger.info("No currency pairs selected for preprocessing")
    
    # Stage 2: Feature Enhancement
    if args.stage == 0 or args.stage == 2:
        logger.info("Stage 2: Feature Enhancement")
        from src.features.feature_enhancement import enhance_and_select_features
        from src.data.data_loader import load_processed_data
        
        # Get feature enhancement parameters from config
        feature_params = config.get('feature_enhancement', {})
        
        # Load processed data
        process_pairs = [p for p in pairs if p != 'BAGGING']
        
        if process_pairs:
            processed_data = {}
            for pair in process_pairs:
                try:
                    # Load processed data for each pair
                    train_data = load_processed_data(pair, 'train')
                    val_data = load_processed_data(pair, 'val')
                    test_data = load_processed_data(pair, 'test')
                    
                    processed_data[pair] = {
                        'train': train_data,
                        'val': val_data,
                        'test': test_data
                    }
                except FileNotFoundError as e:
                    logger.error(f"Processed data for {pair} not found. Run Stage 1 first or check paths. Error: {str(e)}")
                    return
            
            # Enhance and select features
            enhanced_data = enhance_and_select_features(
                processed_data,
                selection_method=feature_params.get('selection_method', 'random_forest'),
                top_n=feature_params.get('top_n', 50),
                timeframe=feature_params.get('timeframe', "1H"),
                save_data=True
            )
            
            # Visualize feature importance if requested
            if args.visualize:
                from src.features.feature_enhancement import visualize_feature_importance
                for pair in process_pairs:
                    visualize_feature_importance(
                        enhanced_data[pair]['train'],
                        feature_params.get('selection_method', 'random_forest'),
                        pair,
                        top_n=20,
                        show_plot=args.visualize
                    )
                    
            logger.info(f"Enhanced features for {len(process_pairs)} currency pairs")
        else:
            logger.info("No currency pairs selected for feature enhancement")
    
    # Stage 3: Model Training
    if args.stage == 0 or args.stage == 3:
        logger.info("Stage 3: Model Training")
        from src.features.feature_enhancement import load_feature_data
        
        # Get model training parameters from config
        model_params = config.get('model_training', {})
        
        # Determine currency pairs to train models for
        train_pairs = [p for p in pairs if p != 'BAGGING']
        
        for model_type in models:
            logger.info(f"Training {model_type} models")
            
            for pair in train_pairs:
                logger.info(f"Training {model_type} model for {pair}")
                
                try:
                    # Load feature data
                    train_data = load_feature_data(pair, 'train')
                    val_data = load_feature_data(pair, 'val')
                    test_data = load_feature_data(pair, 'test')
                except FileNotFoundError as e:
                    logger.error(f"Feature data for {pair} not found. Run Stage 2 first or check paths. Error: {str(e)}")
                    continue
                
                # Train model based on type
                if model_type == 'CNN-LSTM':
                    from src.models.cnn_lstm import (
                        create_sequences, 
                        train_cnn_lstm_model, 
                        evaluate_cnn_lstm_model,
                        plot_training_history
                    )
                    
                    # Get model-specific parameters
                    cnn_lstm_params = model_params.get('CNN-LSTM', {})
                    sequence_length = cnn_lstm_params.get('sequence_length', 60)
                    
                    # Prepare sequences
                    X_train, y_train = create_sequences(
                        train_data,
                        target_col='target',
                        sequence_length=sequence_length,
                        prediction_steps=1,
                        shuffle=True
                    )
                    
                    X_val, y_val = create_sequences(
                        val_data,
                        target_col='target',
                        sequence_length=sequence_length,
                        prediction_steps=1,
                        shuffle=False
                    )
                    
                    # Train model
                    model, history = train_cnn_lstm_model(
                        X_train, y_train,
                        X_val, y_val,
                        model_config=cnn_lstm_params.get('model_config', None),
                        training_config=cnn_lstm_params.get('training_config', None),
                        model_name=f"cnn_lstm_{pair}",
                        save_model=True
                    )
                    
                    # Plot training history
                    if args.visualize:
                        plot_training_history(
                            history,
                            model_name=f"cnn_lstm_{pair}",
                            save_plot=True,
                            show_plot=args.visualize
                        )
                
                elif model_type == 'TFT':
                    from src.models.tft import (
                        prepare_data_for_tft,
                        train_tft_model
                    )
                    
                    # Get model-specific parameters
                    tft_params = model_params.get('TFT', {})
                    max_encoder_length = tft_params.get('max_encoder_length', 24)
                    
                    # Prepare data for TFT
                    training, validation, testing, data_params = prepare_data_for_tft(
                        train_data,
                        val_data,
                        test_data,
                        target_variable='target',
                        max_encoder_length=max_encoder_length,
                        max_prediction_length=1
                    )
                    
                    # Train model
                    model, history = train_tft_model(
                        training,
                        validation,
                        model_config=tft_params.get('model_config', None),
                        training_config=tft_params.get('training_config', None),
                        model_name=f"tft_{pair}",
                        save_model=True
                    )
                
                elif model_type == 'XGBoost':
                    from src.models.xgboost_model import (
                        prepare_data_for_xgboost,
                        train_xgboost_model,
                        plot_xgboost_training_history,
                        plot_xgboost_feature_importance
                    )
                    
                    # Get model-specific parameters
                    xgboost_params = model_params.get('XGBoost', {})
                    sequence_length = xgboost_params.get('sequence_length', 1)
                    
                    # Prepare data for XGBoost
                    X_train, y_train, feature_names = prepare_data_for_xgboost(
                        train_data,
                        target_col='target',
                        sequence_length=sequence_length,
                        prediction_steps=1,
                        shuffle=True
                    )
                    
                    X_val, y_val, _ = prepare_data_for_xgboost(
                        val_data,
                        target_col='target',
                        sequence_length=sequence_length,
                        prediction_steps=1,
                        shuffle=False
                    )
                    
                    # Train model
                    model, history = train_xgboost_model(
                        X_train, y_train,
                        X_val, y_val,
                        feature_names=feature_names,
                        model_config=xgboost_params.get('model_config', None),
                        model_name=f"xgboost_{pair}",
                        save_model=True
                    )
                    
                    # Plot training history and feature importance
                    if args.visualize:
                        plot_xgboost_training_history(
                            history,
                            model_name=f"xgboost_{pair}",
                            save_plot=True,
                            show_plot=args.visualize
                        )
                        
                        plot_xgboost_feature_importance(
                            model,
                            feature_names=feature_names,
                            model_name=f"xgboost_{pair}",
                            importance_type='weight',
                            max_features=20,
                            save_plot=True,
                            show_plot=args.visualize
                        )
                
                logger.info(f"Completed training {model_type} model for {pair}")
        
        # Train Bagging model if requested
        if 'BAGGING' in pairs:
            logger.info("Training Bagging models")
            from src.models.bagging_model import create_bagging_model
            
            for model_type in models:
                logger.info(f"Creating Bagging model for {model_type}")
                bagging_model = create_bagging_model(
                    model_type,
                    CURRENCY_PAIRS,
                    model_name=f"bagging_{model_type.lower()}"
                )
                logger.info(f"Completed creating Bagging model for {model_type}")
    
    # Stage 4: Model Evaluation
    if args.stage == 0 or args.stage == 4:
        logger.info("Stage 4: Model Evaluation & Performance Analysis")
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
        eval_pairs = [p for p in pairs if p != 'BAGGING']
        
        for pair in eval_pairs:
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
            
            # Identify market conditions
            market_conditions = identify_market_conditions(
                price_data.iloc[test_start_idx:test_start_idx+len(test_data)].copy(),
                window=eval_params.get('market_condition_window', 20),
                threshold=eval_params.get('market_condition_threshold', 0.05)
            )
            
            # Evaluate each model
            for model_type in models:
                logger.info(f"Evaluating {model_type} model for {pair}")
                
                try:
                    # Load model
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
                        _, _, testing_data, _ = prepare_data_for_tft(
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
                        
                        # Use testing_data for evaluation
                        X_test = testing_data
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
                    
                    # Evaluate trading performance
                    metrics, summary = evaluate_trading_performance(
                        model,
                        X_test,
                        price_data,
                        test_start_idx,
                        model_type,
                        multi_class=False,
                        transaction_cost=eval_params.get('transaction_cost', 0.0001),
                        risk_free_rate=eval_params.get('risk_free_rate', 0.0)
                    )
                    
                    # Store metrics
                    all_metrics[pair][model_type] = metrics
                    
                    # Store equity curve and drawdown for plotting
                    equity_curves[f"{model_type}_{pair}"] = summary['Equity']
                    drawdowns[f"{model_type}_{pair}"] = summary['Drawdown']
                    
                    # Evaluate performance by market condition
                    condition_perf = evaluate_performance_by_market_condition(
                        summary['Return'],
                        market_conditions
                    )
                    
                    market_condition_performance[pair][model_type] = condition_perf
                    
                    # Save results
                    save_performance_results(
                        {
                            'metrics': metrics,
                            'market_condition_performance': condition_perf
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
                        
                        # Plot performance by market condition
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
        
        # Evaluate Bagging models if requested
        if 'BAGGING' in pairs:
            logger.info("Evaluating Bagging models")
            
            for model_type in models:
                logger.info(f"Evaluating Bagging model for {model_type}")
                
                try:
                    # Load Bagging model
                    from src.models.bagging_model import BaggingWrapper
                    bagging_model = BaggingWrapper.load(f"bagging_{model_type.lower()}")
                    
                    # Use each currency pair for testing
                    for pair in CURRENCY_PAIRS:
                        logger.info(f"Testing Bagging model on {pair}")
                        
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
                        
                        # Prepare test data based on model type
                        if model_type == 'CNN-LSTM':
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
                        
                        elif model_type == 'TFT':
                            from src.models.tft import prepare_data_for_tft
                            
                            # Get one of the component models to extract parameters
                            component_model = next(iter(bagging_model.models.values()))
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
                        
                        elif model_type == 'XGBoost':
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
                        
                        # Evaluate trading performance
                        metrics, summary = evaluate_trading_performance(
                            bagging_model,
                            X_test,
                            price_data,
                            test_start_idx,
                            'Bagging',
                            multi_class=False,
                            transaction_cost=eval_params.get('transaction_cost', 0.0001),
                            risk_free_rate=eval_params.get('risk_free_rate', 0.0)
                        )
                        
                        # Store metrics in all_metrics
                        if pair not in all_metrics:
                            all_metrics[pair] = {}
                        
                        bagging_name = f"Bagging_{model_type}"
                        all_metrics[pair][bagging_name] = metrics
                        
                        # Store equity curve and drawdown for plotting
                        equity_curves[f"{bagging_name}_{pair}"] = summary['Equity']
                        drawdowns[f"{bagging_name}_{pair}"] = summary['Drawdown']
                        
                        # Identify market conditions
                        if pair not in market_condition_performance:
                            market_condition_performance[pair] = {}
                        
                        market_conditions = identify_market_conditions(
                            price_data.iloc[test_start_idx:test_start_idx+len(summary)].copy(),
                            window=eval_params.get('market_condition_window', 20),
                            threshold=eval_params.get('market_condition_threshold', 0.05)
                        )
                        
                        # Evaluate performance by market condition
                        condition_perf = evaluate_performance_by_market_condition(
                            summary['Return'],
                            market_conditions
                        )
                        
                        market_condition_performance[pair][bagging_name] = condition_perf
                        
                        # Save results
                        save_performance_results(
                            {
                                'metrics': metrics,
                                'market_condition_performance': condition_perf
                            },
                            bagging_name,
                            pair
                        )
                        
                        save_trading_summary(summary, bagging_name, pair)
                        
                        # Plot results if requested
                        if args.visualize:
                            # Plot equity curve for this model
                            plot_equity_curves(
                                {bagging_name: summary['Equity'], 'Buy & Hold': summary['Equity'].iloc[0] * (1 + price_data['Close'].pct_change().iloc[test_start_idx:test_start_idx+len(summary)].cumsum())},
                                title=f'Equity Curve: {bagging_name} vs Buy & Hold ({pair})',
                                save_path=os.path.join(RESULTS_DIR, f"{bagging_name}_{pair}_equity.png"),
                                show_plot=args.visualize
                            )
                            
                            # Plot drawdown
                            plot_drawdowns(
                                {bagging_name: summary['Drawdown']},
                                title=f'Drawdown: {bagging_name} ({pair})',
                                save_path=os.path.join(RESULTS_DIR, f"{bagging_name}_{pair}_drawdown.png"),
                                show_plot=args.visualize
                            )
                            
                            # Plot performance by market condition
                            plot_market_conditions_performance(
                                condition_perf,
                                metric='total_return',
                                title=f'Performance by Market Condition: {bagging_name} ({pair})',
                                save_path=os.path.join(RESULTS_DIR, f"{bagging_name}_{pair}_market_conditions.png"),
                                show_plot=args.visualize
                            )
                        
                        logger.info(f"Completed evaluation of {bagging_name} model for {pair}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating Bagging model for {model_type}: {str(e)}")
                    continue
        
        # Create and save performance report
        if all_metrics:
            report = create_performance_report(
                all_metrics,
                buy_hold_metrics,
                market_condition_performance
            )
            
            report_path = os.path.join(RESULTS_DIR, "performance_report.csv")
            report.to_csv(report_path)
            logger.info(f"Saved performance report to {report_path}")
            
            # Plot comparison across all models if requested
            if args.visualize and equity_curves:
                # Plot all equity curves together
                plot_equity_curves(
                    equity_curves,
                    title='Equity Curves Comparison (All Models)',
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
    
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    run_pipeline()