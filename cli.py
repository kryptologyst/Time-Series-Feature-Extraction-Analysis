#!/usr/bin/env python3
"""
Command Line Interface for Time Series Analysis

This script provides a simple command-line interface for running
time series analysis with various options.
"""

import argparse
import sys
from pathlib import Path
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from time_series_analyzer import TimeSeriesAnalyzer


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Time Series Feature Extraction & Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --samples 50 --window 100
  python cli.py --config config/custom_config.yaml --output results.json
  python cli.py --quick --visualize
        """
    )
    
    # Data parameters
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=100,
        help="Number of time series samples (default: 100)"
    )
    
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=200,
        help="Window length for each time series (default: 200)"
    )
    
    parser.add_argument(
        "--noise", "-n",
        type=float,
        default=0.1,
        help="Noise level (default: 0.1)"
    )
    
    # Analysis options
    parser.add_argument(
        "--features",
        action="store_true",
        help="Extract features only"
    )
    
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Run forecasting only"
    )
    
    parser.add_argument(
        "--anomaly",
        action="store_true",
        help="Run anomaly detection only"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick analysis with minimal features"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON format)"
    )
    
    # Visualization
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation"
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize analyzer
        if args.config:
            analyzer = TimeSeriesAnalyzer(args.config)
        else:
            analyzer = TimeSeriesAnalyzer()
        
        # Update configuration based on CLI arguments
        analyzer.config["data"]["n_samples"] = args.samples
        analyzer.config["data"]["window_length"] = args.window
        analyzer.config["data"]["noise_level"] = args.noise
        analyzer.config["data"]["random_seed"] = args.seed
        
        if args.quick:
            analyzer.config["features"]["tsfresh"]["default_fc_parameters"] = "minimal"
        
        print(f"Running analysis with {args.samples} samples, window length {args.window}")
        
        # Run analysis based on options
        if args.features:
            print("Extracting features...")
            df, labels = analyzer.data_generator.generate_realistic_data()
            features = analyzer.feature_extractor.extract_tsfresh_features(df)
            statistical_features = analyzer.feature_extractor.extract_statistical_features(df)
            all_features = pd.concat([features, statistical_features.set_index('id')], axis=1)
            
            results = {
                "data_points": len(df),
                "time_series": df['id'].nunique(),
                "features_extracted": all_features.shape[1],
                "feature_names": all_features.columns.tolist()
            }
            
        elif args.forecast:
            print("Running forecasting...")
            df, labels = analyzer.data_generator.generate_realistic_data()
            first_series = df[df['id'] == 0]['value'].values
            
            arima_forecast, arima_conf = analyzer.forecasting_models.arima_forecast(first_series)
            
            prophet_df = pd.DataFrame({
                'ds': pd.date_range('2023-01-01', periods=len(first_series), freq='D'),
                'y': first_series
            })
            prophet_forecast, prophet_conf = analyzer.forecasting_models.prophet_forecast(prophet_df)
            
            results = {
                "data_points": len(df),
                "forecast_steps": len(arima_forecast),
                "arima_forecast": arima_forecast.tolist(),
                "prophet_forecast": prophet_forecast.tolist()
            }
            
        elif args.anomaly:
            print("Running anomaly detection...")
            df, labels = analyzer.data_generator.generate_realistic_data()
            features = analyzer.feature_extractor.extract_tsfresh_features(df)
            anomaly_scores = analyzer.anomaly_detector.isolation_forest_detection(features)
            
            results = {
                "data_points": len(df),
                "time_series": df['id'].nunique(),
                "anomalies_detected": int(np.sum(anomaly_scores == -1)),
                "anomaly_rate": float(np.sum(anomaly_scores == -1) / len(anomaly_scores)),
                "anomalous_series": [int(i) for i, score in enumerate(anomaly_scores) if score == -1]
            }
            
        else:
            # Run complete analysis
            print("Running complete analysis...")
            results = analyzer.run_complete_analysis()
            
            # Convert results to serializable format
            results_summary = {
                "data_points": len(results['data']),
                "time_series": results['data']['id'].nunique(),
                "features_extracted": results['features'].shape[1],
                "anomalies_detected": int(np.sum(results['anomaly_scores'] == -1)),
                "anomaly_rate": float(np.sum(results['anomaly_scores'] == -1) / len(results['anomaly_scores'])),
                "forecast_models": list(results['forecasts'].keys()),
                "visualizations_created": len(results['plots'])
            }
            
            results = results_summary
        
        # Display results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        
        for key, value in results.items():
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                print(f"{key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"{key}: {value}")
        
        print("="*50)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        
        # Generate visualizations if requested
        if args.visualize and not args.no_plots:
            print("Generating visualizations...")
            if 'plots' in results:
                # Save plots
                for plot_name, plot_fig in results['plots'].items():
                    plot_fig.write_html(f"plots/{plot_name}_plot.html")
                print("Plots saved to plots/ directory")
            else:
                print("No plots available in results")
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
