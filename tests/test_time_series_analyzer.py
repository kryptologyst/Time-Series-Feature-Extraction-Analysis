"""
Unit tests for time series analysis components.

This module contains comprehensive tests for all major components
of the time series analysis system.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from time_series_analyzer import (
    TimeSeriesDataGenerator,
    FeatureExtractor,
    ForecastingModels,
    AnomalyDetector,
    Visualizer,
    TimeSeriesAnalyzer
)


class TestTimeSeriesDataGenerator(unittest.TestCase):
    """Test cases for TimeSeriesDataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "data": {
                "n_samples": 10,
                "window_length": 50,
                "noise_level": 0.1,
                "random_seed": 42
            }
        }
        self.generator = TimeSeriesDataGenerator(self.config)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.n_samples, 10)
        self.assertEqual(self.generator.window_length, 50)
        self.assertEqual(self.generator.noise_level, 0.1)
        self.assertEqual(self.generator.random_seed, 42)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        df, labels = self.generator.generate_synthetic_data()
        
        # Check data structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(labels, list)
        
        # Check DataFrame columns
        self.assertIn('id', df.columns)
        self.assertIn('time', df.columns)
        self.assertIn('value', df.columns)
        
        # Check data dimensions
        self.assertEqual(len(df), 10 * 50)  # n_samples * window_length
        self.assertEqual(len(labels), 10)
        
        # Check unique IDs
        self.assertEqual(df['id'].nunique(), 10)
        
        # Check time values
        for series_id in df['id'].unique():
            series_data = df[df['id'] == series_id]
            self.assertEqual(len(series_data), 50)
            self.assertTrue(all(series_data['time'] == range(50)))
    
    def test_generate_realistic_data(self):
        """Test realistic data generation."""
        df, labels = self.generator.generate_realistic_data()
        
        # Check data structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(labels, list)
        
        # Check data dimensions
        self.assertEqual(len(df), 10 * 50)
        self.assertEqual(len(labels), 10)
        
        # Check that we have different classes
        unique_labels = set(labels)
        self.assertGreater(len(unique_labels), 1)


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "features": {
                "tsfresh": {
                    "default_fc_parameters": "minimal",
                    "impute_function": "mean"
                }
            }
        }
        self.extractor = FeatureExtractor(self.config)
        
        # Create sample data
        self.sample_df = pd.DataFrame({
            'id': [0, 0, 0, 1, 1, 1],
            'time': [0, 1, 2, 0, 1, 2],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor.config)
    
    @patch('tsfresh.extract_features')
    def test_extract_tsfresh_features(self, mock_extract):
        """Test tsfresh feature extraction."""
        # Mock the tsfresh function
        mock_features = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'feature2': [3.0, 4.0]
        })
        mock_extract.return_value = mock_features
        
        with patch('tsfresh.utilities.dataframe_functions.impute') as mock_impute:
            mock_impute.return_value = mock_features
            
            result = self.extractor.extract_tsfresh_features(self.sample_df)
            
            # Check that tsfresh was called
            mock_extract.assert_called_once()
            mock_impute.assert_called_once()
            
            # Check result
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape, (2, 2))
    
    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        result = self.extractor.extract_statistical_features(self.sample_df)
        
        # Check result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('id', result.columns)
        
        # Check that we have statistical features
        expected_features = ['mean', 'std', 'min', 'max', 'range', 'skewness', 'kurtosis']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # Check data dimensions
        self.assertEqual(len(result), 2)  # Two unique series


class TestForecastingModels(unittest.TestCase):
    """Test cases for ForecastingModels class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "models": {
                "arima": {
                    "order": [1, 1, 1],
                    "seasonal_order": [1, 1, 1, 12]
                },
                "prophet": {
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False
                }
            }
        }
        self.models = ForecastingModels(self.config)
        
        # Create sample time series
        self.sample_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Create sample Prophet DataFrame
        self.prophet_df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10, freq='D'),
            'y': self.sample_series
        })
    
    def test_initialization(self):
        """Test models initialization."""
        self.assertIsNotNone(self.models.config)
    
    @patch('statsmodels.tsa.arima.model.ARIMA')
    def test_arima_forecast(self, mock_arima):
        """Test ARIMA forecasting."""
        # Mock ARIMA model
        mock_model = MagicMock()
        mock_fitted = MagicMock()
        mock_forecast = MagicMock()
        mock_conf_int = MagicMock()
        
        mock_forecast.values = np.array([11, 12, 13])
        mock_conf_int.values = np.array([[10, 12], [11, 13], [12, 14]])
        
        mock_fitted.forecast.return_value = mock_forecast
        mock_fitted.get_forecast.return_value.conf_int.return_value = mock_conf_int
        mock_model.fit.return_value = mock_fitted
        mock_arima.return_value = mock_model
        
        forecast, conf_int = self.models.arima_forecast(self.sample_series, steps=3)
        
        # Check results
        self.assertIsInstance(forecast, np.ndarray)
        self.assertIsInstance(conf_int, np.ndarray)
        self.assertEqual(len(forecast), 3)
        self.assertEqual(conf_int.shape, (3, 2))
    
    @patch('prophet.Prophet')
    def test_prophet_forecast(self, mock_prophet):
        """Test Prophet forecasting."""
        # Mock Prophet model
        mock_model = MagicMock()
        mock_future = pd.DataFrame({
            'ds': pd.date_range('2023-01-11', periods=3, freq='D')
        })
        mock_forecast = pd.DataFrame({
            'yhat': [11, 12, 13],
            'yhat_lower': [10, 11, 12],
            'yhat_upper': [12, 13, 14]
        })
        
        mock_model.make_future_dataframe.return_value = mock_future
        mock_model.predict.return_value = mock_forecast
        mock_prophet.return_value = mock_model
        
        forecast, conf_int = self.models.prophet_forecast(self.prophet_df, steps=3)
        
        # Check results
        self.assertIsInstance(forecast, np.ndarray)
        self.assertIsInstance(conf_int, np.ndarray)
        self.assertEqual(len(forecast), 3)
        self.assertEqual(conf_int.shape, (3, 2))


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "anomaly_detection": {
                "isolation_forest": {
                    "contamination": 0.1,
                    "random_state": 42
                }
            }
        }
        self.detector = AnomalyDetector(self.config)
        
        # Create sample features
        self.sample_features = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        })
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector.config)
    
    @patch('sklearn.ensemble.IsolationForest')
    def test_isolation_forest_detection(self, mock_isolation_forest):
        """Test Isolation Forest anomaly detection."""
        # Mock Isolation Forest
        mock_model = MagicMock()
        mock_model.fit_predict.return_value = np.array([1, 1, 1, -1, 1, 1, 1, 1, 1, 1])
        mock_isolation_forest.return_value = mock_model
        
        result = self.detector.isolation_forest_detection(self.sample_features)
        
        # Check result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 10)
        
        # Check that Isolation Forest was called
        mock_isolation_forest.assert_called_once()


class TestVisualizer(unittest.TestCase):
    """Test cases for Visualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "visualization": {
                "figure_size": [12, 8],
                "style": "seaborn-v0_8"
            }
        }
        self.visualizer = Visualizer(self.config)
        
        # Create sample data
        self.sample_df = pd.DataFrame({
            'id': [0, 0, 0, 1, 1, 1],
            'time': [0, 1, 2, 0, 1, 2],
            'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        })
        self.sample_labels = [0, 1]
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertIsNotNone(self.visualizer.config)
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        fig = self.visualizer.plot_time_series(self.sample_df, self.sample_labels)
        
        # Check that we get a Plotly figure
        self.assertIsInstance(fig, go.Figure)
    
    def test_plot_forecast_results(self):
        """Test forecast plotting."""
        actual = np.array([1, 2, 3, 4, 5])
        forecast = np.array([6, 7, 8])
        conf_int = np.array([[5, 7], [6, 8], [7, 9]])
        
        fig = self.visualizer.plot_forecast_results(actual, forecast, conf_int)
        
        # Check that we get a Plotly figure
        self.assertIsInstance(fig, go.Figure)


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """Test cases for TimeSeriesAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.config_content = """
data:
  n_samples: 5
  window_length: 20
  noise_level: 0.1
  random_seed: 42

features:
  tsfresh:
    default_fc_parameters: "minimal"
    impute_function: "mean"

models:
  arima:
    order: [1, 1, 1]
    seasonal_order: [1, 1, 1, 12]

  prophet:
    yearly_seasonality: true
    weekly_seasonality: true
    daily_seasonality: false

anomaly_detection:
  isolation_forest:
    contamination: 0.1
    random_state: 42

visualization:
  figure_size: [12, 8]
  style: "seaborn-v0_8"

logging:
  level: "INFO"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
"""
    
    @patch('builtins.open', create=True)
    @patch('yaml.safe_load')
    def test_initialization(self, mock_yaml_load, mock_open):
        """Test analyzer initialization."""
        mock_yaml_load.return_value = {
            "data": {"n_samples": 5, "window_length": 20, "noise_level": 0.1, "random_seed": 42},
            "features": {"tsfresh": {"default_fc_parameters": "minimal"}},
            "models": {"arima": {"order": [1, 1, 1]}, "prophet": {"yearly_seasonality": True}},
            "anomaly_detection": {"isolation_forest": {"contamination": 0.1}},
            "visualization": {"figure_size": [12, 8]}
        }
        
        analyzer = TimeSeriesAnalyzer()
        
        # Check that components are initialized
        self.assertIsNotNone(analyzer.data_generator)
        self.assertIsNotNone(analyzer.feature_extractor)
        self.assertIsNotNone(analyzer.forecasting_models)
        self.assertIsNotNone(analyzer.anomaly_detector)
        self.assertIsNotNone(analyzer.visualizer)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
