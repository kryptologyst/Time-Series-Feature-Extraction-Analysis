"""
Time Series Feature Extraction and Analysis Project

This module provides comprehensive time series analysis capabilities including:
- Feature extraction using tsfresh
- Forecasting with ARIMA, Prophet, and LSTM
- Anomaly detection
- Interactive visualization
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tsfresh
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")


class TimeSeriesDataGenerator:
    """Generate synthetic time series data for testing and demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data generator with configuration."""
        self.config = config
        self.n_samples = config["data"]["n_samples"]
        self.window_length = config["data"]["window_length"]
        self.noise_level = config["data"]["noise_level"]
        self.random_seed = config["data"]["random_seed"]
        
        np.random.seed(self.random_seed)
        logger.info(f"Initialized TimeSeriesDataGenerator with {self.n_samples} samples")
    
    def generate_synthetic_data(self) -> Tuple[pd.DataFrame, List[int]]:
        """
        Generate synthetic time series data with different patterns.
        
        Returns:
            Tuple of (DataFrame with time series data, labels)
        """
        data = []
        labels = []
        
        for i in range(self.n_samples):
            # Generate different signal types
            if i < self.n_samples // 3:
                # Low frequency sine wave
                freq = 1
                signal = np.sin(np.linspace(0, 2 * np.pi * freq, self.window_length))
                label = 0
            elif i < 2 * self.n_samples // 3:
                # High frequency sine wave
                freq = 5
                signal = np.sin(np.linspace(0, 2 * np.pi * freq, self.window_length))
                label = 1
            else:
                # Random walk with trend
                signal = np.cumsum(np.random.randn(self.window_length)) + np.linspace(0, 2, self.window_length)
                label = 2
            
            # Add noise
            signal += self.noise_level * np.random.randn(self.window_length)
            
            # Store data
            for t in range(self.window_length):
                data.append({
                    'id': i,
                    'time': t,
                    'value': signal[t]
                })
            labels.append(label)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} data points across {self.n_samples} time series")
        return df, labels
    
    def generate_realistic_data(self) -> Tuple[pd.DataFrame, List[int]]:
        """
        Generate more realistic time series data with trends and seasonality.
        
        Returns:
            Tuple of (DataFrame with time series data, labels)
        """
        data = []
        labels = []
        
        for i in range(self.n_samples):
            t = np.arange(self.window_length)
            
            if i < self.n_samples // 2:
                # Seasonal pattern with trend
                trend = 0.01 * t
                seasonal = 2 * np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 6)
                noise = self.noise_level * np.random.randn(self.window_length)
                signal = trend + seasonal + noise
                label = 0
            else:
                # Anomalous pattern (sudden change)
                trend = 0.005 * t
                seasonal = 1.5 * np.sin(2 * np.pi * t / 12)
                anomaly = np.zeros(self.window_length)
                anomaly[self.window_length//2:] = 3  # Sudden jump
                noise = self.noise_level * np.random.randn(self.window_length)
                signal = trend + seasonal + anomaly + noise
                label = 1
            
            # Store data
            for t_idx, value in enumerate(signal):
                data.append({
                    'id': i,
                    'time': t_idx,
                    'value': value
                })
            labels.append(label)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated realistic data with {len(df)} data points")
        return df, labels


class FeatureExtractor:
    """Extract features from time series data using various methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature extractor."""
        self.config = config
        logger.info("Initialized FeatureExtractor")
    
    def extract_tsfresh_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features using tsfresh library.
        
        Args:
            df: DataFrame with columns ['id', 'time', 'value']
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Extracting tsfresh features...")
        
        # Extract features
        features = extract_features(
            df,
            column_id='id',
            column_sort='time',
            default_fc_parameters=self.config["features"]["tsfresh"]["default_fc_parameters"]
        )
        
        # Handle missing values
        features = impute(features)
        
        logger.info(f"Extracted {features.shape[1]} features for {features.shape[0]} time series")
        return features
    
    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic statistical features manually.
        
        Args:
            df: DataFrame with columns ['id', 'time', 'value']
            
        Returns:
            DataFrame with statistical features
        """
        logger.info("Extracting statistical features...")
        
        features = []
        for series_id in df['id'].unique():
            series_data = df[df['id'] == series_id]['value'].values
            
            feature_dict = {
                'id': series_id,
                'mean': np.mean(series_data),
                'std': np.std(series_data),
                'min': np.min(series_data),
                'max': np.max(series_data),
                'range': np.max(series_data) - np.min(series_data),
                'skewness': pd.Series(series_data).skew(),
                'kurtosis': pd.Series(series_data).kurtosis(),
                'autocorr_lag1': pd.Series(series_data).autocorr(lag=1),
                'autocorr_lag2': pd.Series(series_data).autocorr(lag=2),
                'trend_slope': np.polyfit(range(len(series_data)), series_data, 1)[0]
            }
            features.append(feature_dict)
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted {features_df.shape[1]-1} statistical features")
        return features_df


class ForecastingModels:
    """Implement various forecasting models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize forecasting models."""
        self.config = config
        logger.info("Initialized ForecastingModels")
    
    def arima_forecast(self, series: np.ndarray, steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast using ARIMA model.
        
        Args:
            series: Time series data
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast values, confidence intervals)
        """
        logger.info("Fitting ARIMA model...")
        
        try:
            model = ARIMA(series, order=self.config["models"]["arima"]["order"])
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=steps)
            conf_int = fitted_model.get_forecast(steps=steps).conf_int()
            
            return forecast.values, conf_int.values
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}")
            return np.zeros(steps), np.zeros((steps, 2))
    
    def prophet_forecast(self, df: pd.DataFrame, steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast using Prophet model.
        
        Args:
            df: DataFrame with columns ['ds', 'y']
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast values, confidence intervals)
        """
        logger.info("Fitting Prophet model...")
        
        try:
            model = Prophet(
                yearly_seasonality=self.config["models"]["prophet"]["yearly_seasonality"],
                weekly_seasonality=self.config["models"]["prophet"]["weekly_seasonality"],
                daily_seasonality=self.config["models"]["prophet"]["daily_seasonality"]
            )
            model.fit(df)
            
            future = model.make_future_dataframe(periods=steps)
            forecast = model.predict(future)
            
            forecast_values = forecast['yhat'].tail(steps).values
            conf_int = forecast[['yhat_lower', 'yhat_upper']].tail(steps).values
            
            return forecast_values, conf_int
        except Exception as e:
            logger.warning(f"Prophet fitting failed: {e}")
            return np.zeros(steps), np.zeros((steps, 2))


class AnomalyDetector:
    """Detect anomalies in time series data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize anomaly detector."""
        self.config = config
        logger.info("Initialized AnomalyDetector")
    
    def isolation_forest_detection(self, features: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            features: Feature matrix
            
        Returns:
            Array of anomaly scores
        """
        logger.info("Running Isolation Forest anomaly detection...")
        
        model = IsolationForest(
            contamination=self.config["anomaly_detection"]["isolation_forest"]["contamination"],
            random_state=self.config["anomaly_detection"]["isolation_forest"]["random_state"]
        )
        
        anomaly_scores = model.fit_predict(features)
        return anomaly_scores


class Visualizer:
    """Create interactive visualizations for time series analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize visualizer."""
        self.config = config
        logger.info("Initialized Visualizer")
    
    def plot_time_series(self, df: pd.DataFrame, labels: List[int], title: str = "Time Series Data") -> go.Figure:
        """
        Create interactive plot of time series data.
        
        Args:
            df: DataFrame with time series data
            labels: Class labels for each series
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Sample Time Series", "Class Distribution", "Feature Distribution", "Anomaly Detection"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot sample time series
        unique_ids = df['id'].unique()[:5]  # Show first 5 series
        colors = px.colors.qualitative.Set1
        
        for i, series_id in enumerate(unique_ids):
            series_data = df[df['id'] == series_id]
            fig.add_trace(
                go.Scatter(
                    x=series_data['time'],
                    y=series_data['value'],
                    mode='lines',
                    name=f'Series {series_id}',
                    line=dict(color=colors[i % len(colors)])
                ),
                row=1, col=1
            )
        
        # Class distribution
        label_counts = pd.Series(labels).value_counts()
        fig.add_trace(
            go.Bar(
                x=label_counts.index,
                y=label_counts.values,
                name="Class Distribution"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_forecast_results(self, actual: np.ndarray, forecast: np.ndarray, 
                            conf_int: Optional[np.ndarray] = None) -> go.Figure:
        """
        Plot forecasting results.
        
        Args:
            actual: Actual values
            forecast: Forecasted values
            conf_int: Confidence intervals
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(actual))),
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Plot forecast
        forecast_x = list(range(len(actual), len(actual) + len(forecast)))
        fig.add_trace(go.Scatter(
            x=forecast_x,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Plot confidence intervals
        if conf_int is not None:
            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=np.concatenate([conf_int[:, 1], conf_int[::-1, 0]]),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title="Time Series Forecasting",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400
        )
        
        return fig


class TimeSeriesAnalyzer:
    """Main class for comprehensive time series analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the time series analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_generator = TimeSeriesDataGenerator(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.forecasting_models = ForecastingModels(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.visualizer = Visualizer(self.config)
        
        logger.info("TimeSeriesAnalyzer initialized successfully")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run complete time series analysis pipeline.
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting complete time series analysis...")
        
        # Generate data
        df, labels = self.data_generator.generate_realistic_data()
        
        # Extract features
        tsfresh_features = self.feature_extractor.extract_tsfresh_features(df)
        statistical_features = self.feature_extractor.extract_statistical_features(df)
        
        # Combine features
        all_features = pd.concat([tsfresh_features, statistical_features.set_index('id')], axis=1)
        
        # Anomaly detection
        anomaly_scores = self.anomaly_detector.isolation_forest_detection(all_features)
        
        # Forecasting (using first time series as example)
        first_series = df[df['id'] == 0]['value'].values
        arima_forecast, arima_conf = self.forecasting_models.arima_forecast(first_series)
        
        # Create Prophet DataFrame
        prophet_df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=len(first_series), freq='D'),
            'y': first_series
        })
        prophet_forecast, prophet_conf = self.forecasting_models.prophet_forecast(prophet_df)
        
        # Create visualizations
        main_plot = self.visualizer.plot_time_series(df, labels)
        arima_plot = self.visualizer.plot_forecast_results(first_series, arima_forecast, arima_conf)
        prophet_plot = self.visualizer.plot_forecast_results(first_series, prophet_forecast, prophet_conf)
        
        results = {
            'data': df,
            'labels': labels,
            'features': all_features,
            'anomaly_scores': anomaly_scores,
            'forecasts': {
                'arima': {'forecast': arima_forecast, 'conf_int': arima_conf},
                'prophet': {'forecast': prophet_forecast, 'conf_int': prophet_conf}
            },
            'plots': {
                'main': main_plot,
                'arima': arima_plot,
                'prophet': prophet_plot
            }
        }
        
        logger.info("Complete analysis finished successfully")
        return results


def main():
    """Main function to run the time series analysis."""
    analyzer = TimeSeriesAnalyzer()
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*50)
    print("TIME SERIES ANALYSIS SUMMARY")
    print("="*50)
    print(f"Data points: {len(results['data'])}")
    print(f"Time series: {results['data']['id'].nunique()}")
    print(f"Features extracted: {results['features'].shape[1]}")
    print(f"Anomalies detected: {np.sum(results['anomaly_scores'] == -1)}")
    print("="*50)
    
    return results


if __name__ == "__main__":
    main()
