"""
Command-Line Interface for Resource Usage Prediction Engine
Allows users to make predictions directly from the terminal
"""

import os
import sys
import argparse
import json
import numpy as np
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.predictor import ResourcePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictorCLI:
    """Command-line interface for the prediction engine."""
    
    def __init__(self, model_dir='models'):
        """Initialize the CLI with the predictor."""
        self.predictor = ResourcePredictor(model_dir=model_dir, window_size=10)
    
    def load_data_from_file(self, filepath):
        """Load recent data from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return np.array(data, dtype=float)
            elif isinstance(data, dict) and 'data' in data:
                return np.array(data['data'], dtype=float)
            else:
                raise ValueError("File must contain a list or dict with 'data' key")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def parse_inline_data(self, data_str):
        """Parse data from inline JSON string."""
        try:
            data = json.loads(data_str)
            return np.array(data, dtype=float)
        except Exception as e:
            print(f"Error parsing data: {e}")
            return None
    
    def predict_lstm(self, recent_data):
        """Predict using LSTM model."""
        print("\n" + "="*60)
        print("LSTM MODEL PREDICTION (Advanced)")
        print("="*60)
        
        try:
            prediction = self.predictor.predict_lstm(recent_data)
            print(f"CPU Usage (next timestep):    {prediction['cpu']:6.2f}%")
            print(f"Memory Usage (next timestep): {prediction['memory']:6.2f}%")
            return prediction
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def predict_baseline(self, recent_data):
        """Predict using baseline model."""
        print("\n" + "="*60)
        print("BASELINE MODEL PREDICTION (Random Forest)")
        print("="*60)
        
        try:
            prediction = self.predictor.predict_baseline(recent_data)
            print(f"CPU Usage (next timestep):    {prediction['cpu']:6.2f}%")
            print(f"Memory Usage (next timestep): {prediction['memory']:6.2f}%")
            return prediction
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def predict_ensemble(self, recent_data):
        """Predict using ensemble of both models."""
        print("\n" + "="*60)
        print("ENSEMBLE PREDICTION (Baseline + LSTM Average)")
        print("="*60)
        
        try:
            prediction = self.predictor.predict_with_confidence(recent_data)
            print(f"CPU Usage (next timestep):    {prediction['cpu']:6.2f}% ± {prediction['cpu_std']:.4f}")
            print(f"Memory Usage (next timestep): {prediction['memory']:6.2f}% ± {prediction['memory_std']:.4f}")
            print("\nNote: ± values represent standard deviation across models")
            return prediction
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def predict_multi_horizon(self, recent_data, horizons=[1, 5, 10], use_lstm=True):
        """Predict for multiple horizons."""
        model_name = "LSTM" if use_lstm else "Baseline"
        print("\n" + "="*60)
        print(f"MULTI-HORIZON PREDICTIONS ({model_name})")
        print(f"Predicting for next {horizons} timesteps")
        print("="*60)
        
        try:
            predictions = self.predictor.predict_multiple_horizons(
                recent_data, horizons=horizons, use_lstm=use_lstm
            )
            
            for horizon, pred in predictions.items():
                print(f"\n{horizon}:")
                print(f"  CPU:    {pred['cpu']:6.2f}%")
                print(f"  Memory: {pred['memory']:6.2f}%")
            
            return predictions
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def run_interactive(self):
        """Run in interactive mode."""
        print("\n" + "="*70)
        print("RESOURCE USAGE PREDICTION ENGINE - INTERACTIVE MODE")
        print("="*70)
        
        while True:
            print("\n--- Enter Recent Data ---")
            print("Format: JSON array [[cpu1, mem1], [cpu2, mem2], ...]")
            print("Example: [[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]")
            print("Or type 'quit' to exit\n")
            
            data_input = input("Data: ").strip()
            
            if data_input.lower() == 'quit':
                print("Exiting...")
                break
            
            if not data_input:
                print("Input is empty, please try again.")
                continue
            
            recent_data = self.parse_inline_data(data_input)
            if recent_data is None:
                continue
            
            print("\n--- Select Prediction Model ---")
            print("1. LSTM (Advanced)")
            print("2. Baseline (Random Forest)")
            print("3. Ensemble (Average of both)")
            print("4. Multi-Horizon (1, 5, 10 timesteps)")
            print("5. All Models\n")
            
            choice = input("Choice (1-5): ").strip()
            
            if choice == '1':
                self.predict_lstm(recent_data)
            elif choice == '2':
                self.predict_baseline(recent_data)
            elif choice == '3':
                self.predict_ensemble(recent_data)
            elif choice == '4':
                horizons_input = input("Horizons (comma-separated, default: 1,5,10): ").strip()
                if horizons_input:
                    try:
                        horizons = [int(h.strip()) for h in horizons_input.split(',')]
                    except ValueError:
                        print("Invalid horizons format")
                        continue
                else:
                    horizons = [1, 5, 10]
                self.predict_multi_horizon(recent_data, horizons=horizons)
            elif choice == '5':
                self.predict_lstm(recent_data)
                self.predict_baseline(recent_data)
                self.predict_ensemble(recent_data)
            else:
                print("Invalid choice")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Resource Usage Prediction Engine - CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interactive mode
  python cli.py --interactive
  
  # Predict using LSTM with inline data
  python cli.py --data "[[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]" --model lstm
  
  # Predict using ensemble with data from file
  python cli.py --file recent_data.json --model ensemble
  
  # Multi-horizon prediction
  python cli.py --data "[[30.5, 45.2], [31.2, 46.1]]" --model multi --horizons "1,5,10"
        '''
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Recent data as JSON string (e.g., "[[30.5, 45.2], [31.2, 46.1]]")'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Load recent data from JSON file'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['lstm', 'baseline', 'ensemble', 'multi', 'all'],
        default='ensemble',
        help='Prediction model to use (default: ensemble)'
    )
    parser.add_argument(
        '--horizons',
        type=str,
        default='1,5,10',
        help='Time horizons for multi-horizon predictions (default: 1,5,10)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    try:
        cli = PredictorCLI(model_dir=args.model_dir)
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        sys.exit(1)
    
    # Run in interactive mode
    if args.interactive:
        cli.run_interactive()
        return
    
    # Load data
    if args.data:
        recent_data = cli.parse_inline_data(args.data)
    elif args.file:
        recent_data = cli.load_data_from_file(args.file)
    else:
        parser.print_help()
        sys.exit(1)
    
    if recent_data is None:
        print("Failed to load data")
        sys.exit(1)
    
    print(f"\nInput Data Shape: {recent_data.shape}")
    print(f"Latest CPU/Memory: {recent_data[-1]}")
    
    # Make predictions
    if args.model == 'lstm':
        cli.predict_lstm(recent_data)
    elif args.model == 'baseline':
        cli.predict_baseline(recent_data)
    elif args.model == 'ensemble':
        cli.predict_ensemble(recent_data)
    elif args.model == 'multi':
        try:
            horizons = [int(h.strip()) for h in args.horizons.split(',')]
        except ValueError:
            print("Invalid horizons format")
            sys.exit(1)
        cli.predict_multi_horizon(recent_data, horizons=horizons)
    elif args.model == 'all':
        cli.predict_lstm(recent_data)
        cli.predict_baseline(recent_data)
        cli.predict_ensemble(recent_data)
        horizons = [int(h.strip()) for h in args.horizons.split(',')]
        cli.predict_multi_horizon(recent_data, horizons=horizons)


if __name__ == '__main__':
    main()
