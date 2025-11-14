"""
Streamlit Web Interface for Time Series Analysis

This module provides an interactive web interface for exploring time series
analysis results including feature extraction, forecasting, and anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from time_series_analyzer import TimeSeriesAnalyzer


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Time Series Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Time Series Feature Extraction & Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    n_samples = st.sidebar.slider("Number of Time Series", 10, 200, 100)
    window_length = st.sidebar.slider("Window Length", 50, 500, 200)
    noise_level = st.sidebar.slider("Noise Level", 0.01, 0.5, 0.1)
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    run_feature_extraction = st.sidebar.checkbox("Feature Extraction", value=True)
    run_forecasting = st.sidebar.checkbox("Forecasting", value=True)
    run_anomaly_detection = st.sidebar.checkbox("Anomaly Detection", value=True)
    
    # Model selection
    st.sidebar.subheader("Forecasting Models")
    use_arima = st.sidebar.checkbox("ARIMA", value=True)
    use_prophet = st.sidebar.checkbox("Prophet", value=True)
    
    # Main content
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running time series analysis..."):
            # Update config
            config = {
                "data": {
                    "n_samples": n_samples,
                    "window_length": window_length,
                    "noise_level": noise_level,
                    "random_seed": 42
                },
                "features": {
                    "tsfresh": {
                        "default_fc_parameters": "minimal",
                        "impute_function": "mean"
                    }
                },
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
                },
                "anomaly_detection": {
                    "isolation_forest": {
                        "contamination": 0.1,
                        "random_state": 42
                    }
                }
            }
            
            # Initialize analyzer
            analyzer = TimeSeriesAnalyzer()
            analyzer.config = config
            
            # Run analysis
            results = analyzer.run_complete_analysis()
            
            # Display results
            display_results(results, run_feature_extraction, run_forecasting, 
                          run_anomaly_detection, use_arima, use_prophet)
    
    # Show sample data without running analysis
    if st.button("📊 Show Sample Data"):
        analyzer = TimeSeriesAnalyzer()
        df, labels = analyzer.data_generator.generate_realistic_data()
        
        st.subheader("Sample Time Series Data")
        st.dataframe(df.head(20))
        
        # Plot sample time series
        fig = go.Figure()
        for i in range(min(5, df['id'].nunique())):
            series_data = df[df['id'] == i]
            fig.add_trace(go.Scatter(
                x=series_data['time'],
                y=series_data['value'],
                mode='lines',
                name=f'Series {i}',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Sample Time Series",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def display_results(results: dict, run_feature_extraction: bool, run_forecasting: bool,
                   run_anomaly_detection: bool, use_arima: bool, use_prophet: bool):
    """Display analysis results in the Streamlit interface."""
    
    # Summary statistics
    st.subheader("📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Points", len(results['data']))
    with col2:
        st.metric("Time Series", results['data']['id'].nunique())
    with col3:
        st.metric("Features", results['features'].shape[1])
    with col4:
        st.metric("Anomalies", np.sum(results['anomaly_scores'] == -1))
    
    st.markdown("---")
    
    # Data visualization
    st.subheader("📈 Time Series Visualization")
    
    # Main plot
    main_plot = results['plots']['main']
    st.plotly_chart(main_plot, use_container_width=True)
    
    # Feature extraction results
    if run_feature_extraction:
        st.subheader("🔍 Feature Extraction Results")
        
        # Show feature importance
        features_df = results['features']
        
        # Calculate feature statistics
        feature_stats = pd.DataFrame({
            'mean': features_df.mean(),
            'std': features_df.std(),
            'min': features_df.min(),
            'max': features_df.max()
        }).round(4)
        
        st.dataframe(feature_stats.head(20))
        
        # Feature distribution plot
        if len(features_df.columns) > 0:
            # Select a few features for visualization
            selected_features = features_df.columns[:min(6, len(features_df.columns))]
            
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=selected_features[:6]
            )
            
            for i, feature in enumerate(selected_features):
                row = i // 3 + 1
                col = i % 3 + 1
                
                fig.add_trace(
                    go.Histogram(x=features_df[feature], name=feature),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Forecasting results
    if run_forecasting:
        st.subheader("🔮 Forecasting Results")
        
        if use_arima and 'arima' in results['forecasts']:
            st.write("**ARIMA Forecast**")
            arima_plot = results['plots']['arima']
            st.plotly_chart(arima_plot, use_container_width=True)
        
        if use_prophet and 'prophet' in results['forecasts']:
            st.write("**Prophet Forecast**")
            prophet_plot = results['plots']['prophet']
            st.plotly_chart(prophet_plot, use_container_width=True)
    
    # Anomaly detection results
    if run_anomaly_detection:
        st.subheader("🚨 Anomaly Detection Results")
        
        anomaly_scores = results['anomaly_scores']
        anomaly_df = pd.DataFrame({
            'series_id': range(len(anomaly_scores)),
            'anomaly_score': anomaly_scores,
            'is_anomaly': anomaly_scores == -1
        })
        
        # Anomaly distribution
        anomaly_counts = anomaly_df['is_anomaly'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(x=['Normal', 'Anomaly'], 
                  y=[anomaly_counts.get(False, 0), anomaly_counts.get(True, 0)],
                  marker_color=['green', 'red'])
        ])
        
        fig.update_layout(
            title="Anomaly Distribution",
            xaxis_title="Category",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show anomalous series
        anomalous_series = anomaly_df[anomaly_df['is_anomaly']]['series_id'].tolist()
        
        if anomalous_series:
            st.write(f"**Anomalous Series IDs:** {anomalous_series[:10]}")  # Show first 10
            
            # Plot anomalous series
            fig = go.Figure()
            data_df = results['data']
            
            for series_id in anomalous_series[:5]:  # Show first 5 anomalous series
                series_data = data_df[data_df['id'] == series_id]
                fig.add_trace(go.Scatter(
                    x=series_data['time'],
                    y=series_data['value'],
                    mode='lines',
                    name=f'Anomalous Series {series_id}',
                    line=dict(color='red', width=2)
                ))
            
            fig.update_layout(
                title="Anomalous Time Series",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.subheader("💾 Download Results")
    
    # Prepare data for download
    results_summary = {
        'data_summary': {
            'total_points': len(results['data']),
            'num_series': results['data']['id'].nunique(),
            'num_features': results['features'].shape[1],
            'num_anomalies': np.sum(results['anomaly_scores'] == -1)
        }
    }
    
    # Convert to CSV for download
    summary_df = pd.DataFrame([results_summary['data_summary']])
    
    st.download_button(
        label="Download Summary",
        data=summary_df.to_csv(index=False),
        file_name="analysis_summary.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
