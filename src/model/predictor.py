import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import logging
import os
import sys
from sklearn.preprocessing import MinMaxScaler

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logger = logging.getLogger(__name__)

class ResourcePredictor:
    """
    Prediction Engine for CPU and Memory usage forecasting.
    
    This class loads trained models (both baseline and advanced LSTM),
    preprocesses recent time-series data, and generates predictions for
    future resource usage (next 5-10 minutes).
    """
    
    def __init__(self, model_dir='models', window_size=10):
        """
        Initialize the prediction engine.
        
        Args:
            model_dir (str): Directory containing trained models and scaler
            window_size (int): Number of past timesteps for model input (must match training window)
        """
        self.model_dir = model_dir
        self.window_size = window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
        
        # Load baseline model
        baseline_path = os.path.join(model_dir, 'baseline_rf.pkl')
        if os.path.exists(baseline_path):
            self.baseline_model = joblib.load(baseline_path)
            logger.info(f"Loaded baseline model from {baseline_path}")
        else:
            self.baseline_model = None
            logger.warning(f"Baseline model not found at {baseline_path}")
        
        # Load advanced LSTM model
        lstm_path = os.path.join(model_dir, 'advanced_lstm.pt')
        if os.path.exists(lstm_path):
            self.lstm_model = self._load_lstm_model(lstm_path)
            logger.info(f"Loaded LSTM model from {lstm_path}")
        else:
            self.lstm_model = None
            logger.warning(f"LSTM model not found at {lstm_path}")

    def _load_lstm_model(self, model_path):
        """Load LSTM model architecture and weights."""
        from src.model.advanced_model import ResourceLSTM
        
        model = ResourceLSTM(input_size=2, hidden_size=64, num_layers=2, output_size=2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def preprocess_recent_data(self, recent_data, fit_scaler=False):
        """
        Preprocess recent time-series data.
        
        Args:
            recent_data (np.ndarray or pd.DataFrame): Recent resource usage data
                Shape: (N_timesteps, 2) where 2 = [cpu_usage, memory_usage]
            fit_scaler (bool): Whether to fit scaler (only for new data streams)
        
        Returns:
            np.ndarray: Scaled data (N_timesteps, 2)
        """
        if isinstance(recent_data, pd.DataFrame):
            # Expect columns: ['cpu_usage', 'memory_usage'] or similar
            data = recent_data[['cpu_usage', 'memory_usage']].values if 'cpu_usage' in recent_data.columns else recent_data.iloc[:, :2].values
        else:
            data = np.array(recent_data)
        
        if data.shape[1] != 2:
            raise ValueError(f"Expected 2 features (CPU, Memory), got {data.shape[1]}")
        
        if fit_scaler:
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)
        
        return scaled_data

    def predict_baseline(self, recent_data_raw):
        """
        Generate predictions using baseline Random Forest model.
        
        Args:
            recent_data_raw (np.ndarray): Recent resource data (N_timesteps, 2)
        
        Returns:
            dict: Predictions with keys 'cpu' and 'memory' (next 1 timestep)
        """
        if self.baseline_model is None:
            logger.warning("Baseline model not loaded")
            return None
        
        # Preprocess
        recent_data_scaled = self.preprocess_recent_data(recent_data_raw)
        
        # Flatten to match training format
        X_flat = recent_data_scaled.reshape(1, -1)
        
        # Predict
        pred_flat = self.baseline_model.predict(X_flat)
        pred_scaled = pred_flat.reshape(1, 1, 2)  # Shape: (1, 1, 2)
        
        # Denormalize
        pred_denorm = self.scaler.inverse_transform(pred_scaled.reshape(-1, 2))
        
        return {
            'cpu': float(pred_denorm[0, 0]),
            'memory': float(pred_denorm[0, 1])
        }

    def predict_lstm(self, recent_data_raw, horizon=1):
        """
        Generate predictions using advanced LSTM model.
        
        Args:
            recent_data_raw (np.ndarray): Recent resource data (N_timesteps, 2)
            horizon (int): Number of future timesteps to predict (1-10)
        
        Returns:
            dict: Predictions with keys 'cpu' and 'memory'
        """
        if self.lstm_model is None:
            logger.warning("LSTM model not loaded")
            return None
        
        # Preprocess
        recent_data_scaled = self.preprocess_recent_data(recent_data_raw)
        
        # Ensure we have the correct window size
        if len(recent_data_scaled) < self.window_size:
            logger.warning(f"Input has {len(recent_data_scaled)} timesteps, but window_size is {self.window_size}. Padding with zeros.")
            pad_size = self.window_size - len(recent_data_scaled)
            recent_data_scaled = np.vstack([np.zeros((pad_size, 2)), recent_data_scaled])
        elif len(recent_data_scaled) > self.window_size:
            # Take last window_size timesteps
            recent_data_scaled = recent_data_scaled[-self.window_size:]
        
        # Convert to tensor and add batch dimension
        X_tensor = torch.tensor(recent_data_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred_scaled = self.lstm_model(X_tensor).cpu().numpy()  # Shape: (1, 1, 2)
        
        # Denormalize
        pred_denorm = self.scaler.inverse_transform(pred_scaled.reshape(-1, 2))
        
        return {
            'cpu': float(pred_denorm[0, 0]),
            'memory': float(pred_denorm[0, 1])
        }

    def predict_multiple_horizons(self, recent_data_raw, horizons=None, use_lstm=True):
        """
        Generate predictions for multiple future timesteps.
        
        Args:
            recent_data_raw (np.ndarray): Recent resource data (N_timesteps, 2)
            horizons (list): List of timesteps to predict for (e.g., [1, 2, 5, 10])
            use_lstm (bool): Use LSTM model if available, else use baseline
        
        Returns:
            dict: Dictionary with predictions for each horizon
        """
        if horizons is None:
            horizons = [1, 5, 10]  # Default: predict for next 1, 5, 10 minutes
        
        predictions = {}
        model_name = 'lstm' if use_lstm else 'baseline'
        
        for h in horizons:
            if use_lstm and self.lstm_model is not None:
                pred = self.predict_lstm(recent_data_raw, horizon=h)
            else:
                pred = self.predict_baseline(recent_data_raw)
            
            if pred:
                predictions[f'horizon_{h}min'] = pred
        
        return predictions

    def predict_batch(self, recent_data_sequences, use_lstm=True):
        """
        Generate predictions for multiple data sequences (batch processing).
        
        Args:
            recent_data_sequences (list): List of recent data arrays, each (N_timesteps, 2)
            use_lstm (bool): Use LSTM model if available
        
        Returns:
            list: List of prediction dictionaries, one per input sequence
        """
        predictions = []
        for data_seq in recent_data_sequences:
            if use_lstm and self.lstm_model is not None:
                pred = self.predict_lstm(data_seq)
            else:
                pred = self.predict_baseline(data_seq)
            predictions.append(pred)
        
        return predictions

    def predict_with_confidence(self, recent_data_raw):
        """
        Generate predictions from both baseline and advanced models 
        to provide a confidence estimate via ensemble averaging.
        
        Args:
            recent_data_raw (np.ndarray): Recent resource data (N_timesteps, 2)
        
        Returns:
            dict: Ensemble predictions with 'cpu' and 'memory' keys
        """
        predictions = []
        
        if self.baseline_model is not None:
            pred_baseline = self.predict_baseline(recent_data_raw)
            if pred_baseline:
                predictions.append(pred_baseline)
        
        if self.lstm_model is not None:
            pred_lstm = self.predict_lstm(recent_data_raw)
            if pred_lstm:
                predictions.append(pred_lstm)
        
        if not predictions:
            raise RuntimeError("No models available for prediction")
        
        # Average predictions
        cpu_preds = [p['cpu'] for p in predictions]
        mem_preds = [p['memory'] for p in predictions]
        
        ensemble_pred = {
            'cpu': float(np.mean(cpu_preds)),
            'memory': float(np.mean(mem_preds)),
            'cpu_std': float(np.std(cpu_preds)) if len(cpu_preds) > 1 else 0.0,
            'memory_std': float(np.std(mem_preds)) if len(mem_preds) > 1 else 0.0
        }
        
        return ensemble_pred


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize predictor
    predictor = ResourcePredictor(model_dir='models', window_size=10)
    
    # Simulate recent data (10 timesteps of CPU and Memory readings)
    # Values should be in original scale (0-100 for percentages)
    recent_data = np.array([
        [30.5, 45.2],
        [31.2, 46.1],
        [32.1, 47.0],
        [32.8, 47.5],
        [33.5, 48.2],
        [34.2, 49.1],
        [35.0, 49.8],
        [35.7, 50.5],
        [36.4, 51.2],
        [37.1, 52.0],
    ])
    
    print("\n=== LSTM Prediction (Single Horizon) ===")
    pred_lstm = predictor.predict_lstm(recent_data)
    print(f"CPU (next min): {pred_lstm['cpu']:.2f}%")
    print(f"Memory (next min): {pred_lstm['memory']:.2f}%")
    
    print("\n=== Baseline Prediction (Single Horizon) ===")
    pred_baseline = predictor.predict_baseline(recent_data)
    print(f"CPU (next min): {pred_baseline['cpu']:.2f}%")
    print(f"Memory (next min): {pred_baseline['memory']:.2f}%")
    
    print("\n=== Ensemble Prediction with Confidence ===")
    pred_ensemble = predictor.predict_with_confidence(recent_data)
    print(f"CPU (next min): {pred_ensemble['cpu']:.2f}% ± {pred_ensemble['cpu_std']:.4f}")
    print(f"Memory (next min): {pred_ensemble['memory']:.2f}% ± {pred_ensemble['memory_std']:.4f}")
    
    print("\n=== Multi-Horizon Predictions ===")
    multi_pred = predictor.predict_multiple_horizons(recent_data, horizons=[1, 5, 10], use_lstm=True)
    for horizon, pred in multi_pred.items():
        print(f"{horizon}: CPU={pred['cpu']:.2f}%, Memory={pred['memory']:.2f}%")
