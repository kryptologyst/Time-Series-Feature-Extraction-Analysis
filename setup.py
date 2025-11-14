#!/usr/bin/env python3
"""
Setup script for Time Series Analysis Project

This script helps set up the project environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Time Series Analysis Project...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Create necessary directories
    directories = ['data', 'models', 'plots', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies. Please check requirements.txt")
        sys.exit(1)
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("Warning: Some tests failed, but setup can continue")
    
    # Create sample data
    print("\nCreating sample data...")
    try:
        from src.time_series_analyzer import TimeSeriesAnalyzer
        analyzer = TimeSeriesAnalyzer()
        df, labels = analyzer.data_generator.generate_realistic_data()
        df.to_csv('data/sample_data.csv', index=False)
        print("✓ Sample data created: data/sample_data.csv")
    except Exception as e:
        print(f"Warning: Could not create sample data: {e}")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the CLI: python cli.py --help")
    print("2. Launch Streamlit: streamlit run src/streamlit_app.py")
    print("3. Open Jupyter notebook: jupyter notebook notebooks/")
    print("4. Run tests: python -m pytest tests/")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
