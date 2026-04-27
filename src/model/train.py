import os
import sys
import logging
import numpy as np

# Adjust path to import src modules if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.data_preprocessing import ResourceDataProcessor
from src.model.baseline_model import BaselineModels
from src.model.advanced_model import AdvancedModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def temporal_train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split time-series data into Train, Validation, and Test sets sequentially.
    Never shuffle time-series data!
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    logger.info("Starting Model Training Pipeline...")

    # 1. Load and Preprocess Data
    processor = ResourceDataProcessor()
    
    alibaba_file_path = "container_usage.csv"
    try:
        # The downloader method handles missing files
        ResourceDataProcessor.download_alibaba_sample(alibaba_file_path)
        df_raw = processor.load_alibaba_trace(alibaba_file_path)
    except FileNotFoundError:
        logger.error(f"Cannot find dataset {alibaba_file_path}. Exiting.")
        return

    # Fit scaler and preprocess
    df_clean = processor.preprocess(df_raw, fit_scaler=True)
    
    # Save the scaler for the prediction engine later!
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(processor.scaler, 'models/scaler.pkl')
    logger.info("Saved data scaler to models/scaler.pkl")

    # Create Sliding Windows
    window_size = 10
    predict_horizon = 1
    X, y = processor.create_sliding_windows(df_clean, window_size, predict_horizon)
    logger.info(f"Total dataset shape: X={X.shape}, y={y.shape}")

    # 2. Train/Validation/Test Split
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_train_val_test_split(X, y)
    logger.info(f"Train split: {X_train.shape[0]} samples")
    logger.info(f"Val split: {X_val.shape[0]} samples")
    logger.info(f"Test split: {X_test.shape[0]} samples")

    # 3. Train Baseline Model (Random Forest or Linear Regression)
    baseline = BaselineModels(model_type='random_forest')
    # Combine train and val for traditional ML baseline if wanted, or just use train.
    baseline.train(X_train, y_train)
    mae_base, rmse_base = baseline.evaluate(X_test, y_test)
    baseline.save('models/baseline_rf.pkl')

    # 4. Train Advanced Model (LSTM with PyTorch)
    lstm_trainer = AdvancedModelTrainer(input_size=2, hidden_size=64, num_layers=2, output_size=2, learning_rate=0.001)
    
    # LSTM uses the validation set for Early Stopping / Best Weight saving
    lstm_trainer.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=128)
    
    mae_lstm, rmse_lstm, _ = lstm_trainer.evaluate(X_test, y_test)
    lstm_trainer.save('models/advanced_lstm.pt')
    
    logger.info("=========================================")
    logger.info("FINAL EVALUATION ON TEST SET (Unseen Data):")
    logger.info(f"Random Forest - MAE: {mae_base:.4f}, RMSE: {rmse_base:.4f}")
    logger.info(f"LSTM          - MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}")
    logger.info("Artifacts saved in models/ directory.")
    logger.info("Pipeline Complete!")

if __name__ == "__main__":
    main()
