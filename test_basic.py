#!/usr/bin/env python3
"""
Simple test script for the time series analysis project.

This script tests the basic functionality without requiring all dependencies.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("Testing basic functionality...")
    
    # Test data generation
    print("1. Testing data generation...")
    n_samples = 10
    window_length = 50
    
    data = []
    labels = []
    
    for i in range(n_samples):
        if i < n_samples // 2:
            # Low frequency sine wave
            freq = 1
            signal = np.sin(np.linspace(0, 2 * np.pi * freq, window_length))
            label = 0
        else:
            # High frequency sine wave
            freq = 5
            signal = np.sin(np.linspace(0, 2 * np.pi * freq, window_length))
            label = 1
        
        # Add noise
        signal += 0.1 * np.random.randn(window_length)
        
        # Store data
        for t in range(window_length):
            data.append({
                'id': i,
                'time': t,
                'value': signal[t]
            })
        labels.append(label)
    
    df = pd.DataFrame(data)
    print(f"   ✓ Generated {len(df)} data points across {df['id'].nunique()} time series")
    
    # Test statistical feature extraction
    print("2. Testing statistical feature extraction...")
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
    print(f"   ✓ Extracted {features_df.shape[1]-1} statistical features")
    
    # Test basic visualization
    print("3. Testing basic visualization...")
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot sample time series
        for i in range(min(4, df['id'].nunique())):
            series_data = df[df['id'] == i]
            axes[i//2, i%2].plot(series_data['time'], series_data['value'])
            axes[i//2, i%2].set_title(f'Series {i}')
            axes[i//2, i%2].set_xlabel('Time')
            axes[i//2, i%2].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig('plots/test_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Created test visualization: plots/test_visualization.png")
        
    except ImportError:
        print("   ⚠ Matplotlib not available, skipping visualization test")
    
    # Test data export
    print("4. Testing data export...")
    df.to_csv('data/test_data.csv', index=False)
    features_df.to_csv('data/test_features.csv', index=False)
    print("   ✓ Exported test data to data/ directory")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Data points generated: {len(df)}")
    print(f"Time series created: {df['id'].nunique()}")
    print(f"Features extracted: {features_df.shape[1]-1}")
    print(f"Class distribution: {pd.Series(labels).value_counts().to_dict()}")
    print("="*50)
    
    return True


def main():
    """Main test function."""
    print("Time Series Analysis Project - Basic Test")
    print("=" * 50)
    
    try:
        success = test_basic_functionality()
        if success:
            print("\n✓ All basic tests passed!")
            print("\nTo run the full analysis with all features:")
            print("1. Install dependencies: pip install -r requirements.txt")
            print("2. Run CLI: python cli.py --help")
            print("3. Launch Streamlit: streamlit run src/streamlit_app.py")
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
