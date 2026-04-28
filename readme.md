# Resource Usage Prediction Engine

A machine learning-powered prediction engine for forecasting Kubernetes/container resource usage (CPU and Memory) for the next 5-10 minutes.

## Overview

The prediction engine provides multiple interfaces for resource usage forecasting:

- **Python API** (`predictor.py`): Direct Python library for integration
- **REST API** (`api.py`): HTTP endpoints for web services
- **CLI Tool** (`cli.py`): Command-line interface for quick predictions

### Key Features

✅ **Multiple Models**:
  - **LSTM (Advanced)**: Deep learning model with temporal dependencies
  - **Random Forest (Baseline)**: Traditional ML approach for comparison
  - **Ensemble**: Average predictions from both models with confidence intervals

✅ **Flexible Input**: Accepts recent time-series data (N timesteps of CPU/Memory)

✅ **Multi-Horizon Predictions**: Forecast for next 1, 5, or 10 minutes

✅ **Batch Processing**: Process multiple data sequences at once

✅ **Automatic Normalization**: Uses fitted scaler for data preprocessing

---

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- Required packages (see `requirements.txt`)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: For REST API
pip install flask
```

---

## Input/Output Specification

### Input Format

**Shape**: (N_timesteps, 2)
- First column: CPU usage (0-100%)
- Second column: Memory usage (0-100%)

**Example**:
```python
recent_data = [
    [30.5, 45.2],  # timestep 1
    [31.2, 46.1],  # timestep 2
    [32.1, 47.0],  # timestep 3
    ...
    [37.1, 52.0]   # timestep N
]
```

### Output Format

**Single Horizon Prediction**:
```json
{
  "cpu": 35.50,
  "memory": 52.30
}
```

**Multi-Horizon Prediction**:
```json
{
  "horizon_1min": {"cpu": 35.50, "memory": 52.30},
  "horizon_5min": {"cpu": 36.20, "memory": 53.10},
  "horizon_10min": {"cpu": 37.00, "memory": 54.00}
}
```

**Ensemble Prediction (with confidence)**:
```json
{
  "cpu": 35.75,
  "memory": 52.55,
  "cpu_std": 2.15,
  "memory_std": 1.25
}
```

---

## Usage Interfaces

### 1. Python API (Direct Library)

```python
import numpy as np
from src.model.predictor import ResourcePredictor

# Initialize predictor
predictor = ResourcePredictor(model_dir='models', window_size=10)

# Prepare recent data (10 timesteps)
recent_data = np.array([
    [30.5, 45.2], [31.2, 46.1], [32.1, 47.0], [32.8, 47.5],
    [33.5, 48.2], [34.2, 49.1], [35.0, 49.8], [35.7, 50.5],
    [36.4, 51.2], [37.1, 52.0]
])

# LSTM Prediction
pred_lstm = predictor.predict_lstm(recent_data)
# Output: {'cpu': 13.97, 'memory': 77.55}

# Baseline Prediction
pred_baseline = predictor.predict_baseline(recent_data)
# Output: {'cpu': 20.53, 'memory': 78.02}

# Ensemble Prediction (with confidence)
pred_ensemble = predictor.predict_with_confidence(recent_data)
# Output: {
#   'cpu': 17.25, 'cpu_std': 3.28,
#   'memory': 77.79, 'memory_std': 0.24
# }

# Multi-Horizon Predictions
multi_pred = predictor.predict_multiple_horizons(
    recent_data,
    horizons=[1, 5, 10],
    use_lstm=True
)
# Output: {
#   'horizon_1min': {'cpu': 13.97, 'memory': 77.55},
#   'horizon_5min': {'cpu': 13.97, 'memory': 77.55},
#   'horizon_10min': {'cpu': 13.97, 'memory': 77.55}
# }

# Batch Processing
data_batch = [recent_data, recent_data]  # 2 sequences
predictions = predictor.predict_batch(data_batch, use_lstm=True)
# Output: [pred1, pred2]
```

---

### 2. REST API

#### Starting the API Server

```bash
cd src/model
python api.py
```

Server runs on `http://localhost:5000`

#### API Endpoints

**Health Check**
```bash
GET /health
```

**LSTM Prediction**
```bash
curl -X POST http://localhost:5000/api/predict/lstm \
  -H "Content-Type: application/json" \
  -d '{
    "recent_data": [[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]
  }'
```

**Baseline Prediction**
```bash
curl -X POST http://localhost:5000/api/predict/baseline \
  -H "Content-Type: application/json" \
  -d '{
    "recent_data": [[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]
  }'
```

**Ensemble Prediction**
```bash
curl -X POST http://localhost:5000/api/predict/ensemble \
  -H "Content-Type: application/json" \
  -d '{
    "recent_data": [[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]
  }'
```

**Multi-Horizon Prediction**
```bash
curl -X POST http://localhost:5000/api/predict/multi-horizon \
  -H "Content-Type: application/json" \
  -d '{
    "recent_data": [[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]],
    "horizons": [1, 5, 10],
    "model": "lstm"
  }'
```

**Model Info**
```bash
curl http://localhost:5000/api/models/info
```

---

### 3. Command-Line Interface (CLI)

#### Basic Usage

```bash
# Single LSTM prediction
python src/model/cli.py --data "[[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]" --model lstm

# Ensemble prediction
python src/model/cli.py --data "[[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]" --model ensemble

# Multi-horizon prediction
python src/model/cli.py --data "[[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]" --model multi --horizons "1,5,10"

# All models
python src/model/cli.py --data "[[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]" --model all
```

#### Load Data from File

```bash
# Create data file
echo '[[30.5, 45.2], [31.2, 46.1], [32.1, 47.0]]' > recent_data.json

# Predict using file
python src/model/cli.py --file recent_data.json --model ensemble
```

#### Interactive Mode

```bash
python src/model/cli.py --interactive
```

Interactive mode allows you to:
1. Enter data interactively
2. Choose prediction model
3. Select multiple horizons
4. Run multiple predictions

#### CLI Options

```bash
python src/model/cli.py --help

Options:
  -i, --interactive       Run in interactive mode
  -d, --data DATA         JSON data as string
  -f, --file FILE         Load data from JSON file
  -m, --model MODEL       Model choice: lstm|baseline|ensemble|multi|all
  --horizons HORIZONS     Time horizons (default: 1,5,10)
  --model-dir MODEL_DIR   Directory with trained models
```

---

## Model Architecture

### LSTM Model
- **Type**: Deep Learning (Recurrent Neural Network)
- **Layers**: 2 stacked LSTM layers (64 hidden units each)
- **Output**: 1 prediction step ahead
- **Training**: 30 epochs with validation-based early stopping
- **Device**: CPU/GPU (auto-detected)

### Baseline Model
- **Type**: Random Forest Regressor
- **Trees**: 50 decision trees
- **Features**: Flattened time-series window
- **Output**: 1 prediction step ahead

### Ensemble
- **Method**: Average of LSTM and Baseline predictions
- **Confidence**: Standard deviation across models

---

## Data Requirements

- **Window Size**: 10 timesteps (matches training configuration)
- **Minimum Data**: At least 1 timestep
- **Features**: 2 columns (CPU%, Memory%)
- **Range**: 0-100 (percentage values)
- **Normalization**: Automatic (using fitted MinMaxScaler)

---

## Performance Metrics

From test set evaluation:

| Model | MAE | RMSE |
|-------|-----|------|
| Baseline (RF) | 0.1381 | 0.1926 |
| LSTM | 0.1332 | 0.1900 |

- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)

---

## Example Workflows

### Workflow 1: Real-time Monitoring System

```python
from src.model.predictor import ResourcePredictor
import numpy as np
from collections import deque

# Initialize
predictor = ResourcePredictor()
window = deque(maxlen=10)

# In monitoring loop:
def check_resource_spike(cpu, memory):
    window.append([cpu, memory])
    
    if len(window) == 10:
        recent_data = np.array(list(window))
        pred = predictor.predict_with_confidence(recent_data)
        
        if pred['cpu'] > 80 or pred['memory'] > 85:
            alert(f"High resource usage predicted: CPU={pred['cpu']:.1f}%, Mem={pred['memory']:.1f}%")
```

### Workflow 2: Capacity Planning

```python
from src.model.predictor import ResourcePredictor

predictor = ResourcePredictor()

# Get predictions for different horizons
recent_data = get_recent_data()  # Last 10 minutes
multi_pred = predictor.predict_multiple_horizons(
    recent_data,
    horizons=[1, 5, 10],
    use_lstm=True
)

# Log predictions for trend analysis
for horizon, pred in multi_pred.items():
    log_metrics(horizon, pred['cpu'], pred['memory'])
```

### Workflow 3: Web Service Integration

```python
from flask import Flask, request, jsonify
from src.model.predictor import ResourcePredictor

app = Flask(__name__)
predictor = ResourcePredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['recent_data']
    prediction = predictor.predict_with_confidence(data)
    return jsonify(prediction)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**: Run from project root:
```bash
cd /path/to/Resource-Usage-Predictor
python src/model/cli.py ...
```

### Issue: "Scaler not found"

**Solution**: Ensure `models/scaler.pkl` exists. Run training:
```bash
python src/model/train.py
```

### Issue: Model predictions seem off

**Possible causes**:
1. Input data not in 0-100% range
2. Data shape mismatch (should be N x 2)
3. Window size mismatch (should be 10 timesteps)

**Debug**:
```python
recent_data = np.array(your_data)
print(f"Shape: {recent_data.shape}")  # Should be (N, 2)
print(f"Min/Max: {recent_data.min()}/{recent_data.max()}")  # Should be ~0-100
```

---

## File Structure

```
src/model/
├── train.py              # Training pipeline
├── advanced_model.py     # LSTM model definition
├── baseline_model.py     # Baseline model definition
├── predictor.py          # Prediction engine (main)
├── api.py               # REST API interface
└── cli.py               # Command-line interface

models/
├── scaler.pkl           # Data normalization scaler
├── baseline_rf.pkl      # Trained Random Forest model
└── advanced_lstm.pt     # Trained LSTM model weights
```

---

## License

MIT License

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review example workflows
3. Test with sample data provided in each tool
