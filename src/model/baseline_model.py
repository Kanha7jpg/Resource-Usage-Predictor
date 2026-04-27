import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os

logger = logging.getLogger(__name__)

class BaselineModels:
    def __init__(self, model_type='linear_regression'):
        self.model_type = model_type
        if model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            raise ValueError("model_type must be 'linear_regression' or 'random_forest'")
        self.is_trained = False

    def _flatten_data(self, X, y=None):
        """
        Scikit-learn models expect 2D arrays (samples, features). 
        We must flatten the time-series window (samples, window_size, features)
        into (samples, window_size * features).
        """
        X_flat = X.reshape(X.shape[0], -1)
        if y is not None:
            y_flat = y.reshape(y.shape[0], -1)
            return X_flat, y_flat
        return X_flat

    def train(self, X_train, y_train):
        logger.info(f"Training {self.model_type} baseline model...")
        X_train_flat, y_train_flat = self._flatten_data(X_train, y_train)
        self.model.fit(X_train_flat, y_train_flat)
        self.is_trained = True
        logger.info("Training complete.")

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")
        X_flat = self._flatten_data(X)
        y_pred_flat = self.model.predict(X_flat)
        # Reshape the output back to (samples, horizon, features)
        # Assuming horizon=1 and features=2 (CPU, Mem)
        return y_pred_flat.reshape(X.shape[0], 1, 2)

    def evaluate(self, X_test, y_test):
        logger.info(f"Evaluating {self.model_type}...")
        y_pred = self.predict(X_test)
        
        y_test_flat = y_test.reshape(y_test.shape[0], -1)
        y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
        
        mae = mean_absolute_error(y_test_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
        
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        return mae, rmse

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
