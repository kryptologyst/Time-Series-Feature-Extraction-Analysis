# Time Series Feature Extraction & Analysis

A comprehensive Python project for time series analysis, feature extraction, forecasting, and anomaly detection using modern machine learning libraries.

## Features

- **Feature Extraction**: Automatic feature engineering using tsfresh
- **Forecasting**: Multiple models including ARIMA, Prophet, and LSTM
- **Anomaly Detection**: Isolation Forest and other anomaly detection methods
- **Interactive Visualization**: Plotly-based interactive plots
- **Web Interface**: Streamlit dashboard for easy exploration
- **Comprehensive Testing**: Unit tests for all components
- **Modern Architecture**: Type hints, logging, configuration management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Time-Series-Feature-Extraction-Analysis.git
cd Time-Series-Feature-Extraction-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

Run the complete analysis pipeline:

```bash
python src/time_series_analyzer.py
```

### Web Interface

Launch the Streamlit dashboard:

```bash
streamlit run src/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Jupyter Notebook

For interactive exploration, use the provided notebooks in the `notebooks/` directory.

## Project Structure

```
├── src/                          # Source code
│   ├── time_series_analyzer.py  # Main analysis module
│   └── streamlit_app.py         # Web interface
├── config/                      # Configuration files
│   └── config.yaml             # Main configuration
├── data/                        # Data storage
├── models/                      # Saved models
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
│   └── test_time_series_analyzer.py
├── requirements.txt             # Dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Usage Examples

### Basic Analysis

```python
from src.time_series_analyzer import TimeSeriesAnalyzer

# Initialize analyzer
analyzer = TimeSeriesAnalyzer()

# Run complete analysis
results = analyzer.run_complete_analysis()

# Access results
print(f"Features extracted: {results['features'].shape[1]}")
print(f"Anomalies detected: {sum(results['anomaly_scores'] == -1)}")
```

### Custom Configuration

```python
import yaml

# Load custom configuration
with open('config/custom_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with custom config
analyzer = TimeSeriesAnalyzer('config/custom_config.yaml')
```

### Feature Extraction Only

```python
from src.time_series_analyzer import TimeSeriesDataGenerator, FeatureExtractor

# Generate data
generator = TimeSeriesDataGenerator(config)
df, labels = generator.generate_realistic_data()

# Extract features
extractor = FeatureExtractor(config)
features = extractor.extract_tsfresh_features(df)
statistical_features = extractor.extract_statistical_features(df)
```

### Forecasting

```python
from src.time_series_analyzer import ForecastingModels
import pandas as pd

# Prepare data
models = ForecastingModels(config)
series = df[df['id'] == 0]['value'].values

# ARIMA forecast
arima_forecast, arima_conf = models.arima_forecast(series, steps=10)

# Prophet forecast
prophet_df = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=len(series), freq='D'),
    'y': series
})
prophet_forecast, prophet_conf = models.prophet_forecast(prophet_df, steps=10)
```

### Anomaly Detection

```python
from src.time_series_analyzer import AnomalyDetector

# Detect anomalies
detector = AnomalyDetector(config)
anomaly_scores = detector.isolation_forest_detection(features)

# Get anomalous series
anomalous_series = [i for i, score in enumerate(anomaly_scores) if score == -1]
```

## Configuration

The project uses YAML configuration files. Key parameters:

```yaml
data:
  n_samples: 100              # Number of time series
  window_length: 200          # Length of each series
  noise_level: 0.1           # Noise level
  random_seed: 42            # Random seed

features:
  tsfresh:
    default_fc_parameters: "minimal"  # Feature extraction level

models:
  arima:
    order: [1, 1, 1]         # ARIMA parameters
  prophet:
    yearly_seasonality: true  # Prophet seasonality options

anomaly_detection:
  isolation_forest:
    contamination: 0.1       # Expected anomaly rate
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_time_series_analyzer.py::TestTimeSeriesDataGenerator
```

## Dependencies

### Core Libraries
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing

### Time Series Analysis
- **tsfresh**: Automatic feature extraction
- **statsmodels**: Statistical models
- **pmdarima**: Auto-ARIMA
- **prophet**: Facebook's forecasting tool
- **tslearn**: Time series machine learning
- **darts**: Time series forecasting
- **sktime**: Scikit-learn for time series

### Machine Learning
- **scikit-learn**: Machine learning utilities
- **torch**: Deep learning framework

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plotting

### Web Interface
- **streamlit**: Web application framework

### Utilities
- **pyyaml**: Configuration management
- **loguru**: Logging
- **tqdm**: Progress bars

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python -m pytest tests/`
6. Commit your changes: `git commit -am 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- tsfresh library for automatic feature extraction
- Facebook Prophet for forecasting capabilities
- Streamlit for the web interface framework
- The Python data science community for excellent libraries

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory issues**: Reduce `n_samples` or `window_length` in config
3. **Slow performance**: Use `minimal` tsfresh parameters
4. **Prophet errors**: Ensure data has proper datetime format

### Getting Help

- Check the test files for usage examples
- Review the configuration options
- Open an issue on GitHub for bugs or feature requests

## Changelog

### Version 1.0.0
- Initial release
- Feature extraction with tsfresh
- ARIMA and Prophet forecasting
- Isolation Forest anomaly detection
- Streamlit web interface
- Comprehensive test suite
- Modern Python architecture with type hints
# Time-Series-Feature-Extraction-Analysis
