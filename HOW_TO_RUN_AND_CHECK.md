# How to Run and Check the Resource Usage Prediction Engine

This guide provides step-by-step instructions to run the application and verify all components are working correctly.

---

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Full Setup with Prometheus](#full-setup-with-prometheus)
3. [Component Checks](#component-checks)
4. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Step 1: Install Dependencies

```bash
cd Resource-Usage-Predictor
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

**Check**: You should see no errors. Verify Python version:
```bash
python --version  # Should be 3.10+
```

### Step 2: Run the FastAPI Application

```bash
python -m src.model.fastapi_app
```

**Expected output**:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
2026-05-01 XX:XX:XX - src.model.predictor - INFO - Loaded scaler from models\scaler.pkl
2026-05-01 XX:XX:XX - src.model.predictor - INFO - Loaded baseline model from models\baseline_rf.pkl
2026-05-01 XX:XX:XX - src.model.predictor - INFO - Loaded LSTM model from models\advanced_lstm.pt
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Note**: If port 8000 is busy, the app auto-selects port 8001. Check output for actual port.

### Step 3: Test Health Endpoint

Open a new terminal (keep the app running) and run:

```bash
# Windows PowerShell
(Invoke-WebRequest -UseBasicParsing http://localhost:8000/health).Content

# Linux/macOS curl
curl http://localhost:8000/health
```

**Expected response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-05-01T12:00:00.000000",
  "predictor_loaded": true,
  "recommender_loaded": true,
  "ingestion_running": true
}
```

✅ **Check passed**: All services initialized.

---

## Full Setup with Prometheus

### Option A: Docker Compose (Recommended)

**Prerequisites**: Docker and Docker Compose installed

#### Step 1: Create Docker Compose File

Create `docker-compose.yml` in the project root:

```yaml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    privileged: true

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - CONTAINERS_SELECTOR=pod!=""
      - QUERY_INTERVAL_SECONDS=60
    depends_on:
      - prometheus
      - cadvisor

volumes:
  prometheus_data:
```

#### Step 2: Create Prometheus Config

Create `prometheus.yml` in the project root:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

#### Step 3: Start All Services

```bash
docker-compose up -d
```

**Verify services started**:
```bash
docker-compose ps
```

You should see 3 containers running: `prometheus`, `cadvisor`, `app`.

#### Step 4: Wait for Initialization

Give services ~30 seconds to start. Check logs:

```bash
docker-compose logs app
```

Look for: `Metric ingestion manager started successfully`

#### Step 5: Test All Endpoints

**Prometheus health**:
```bash
curl http://localhost:8000/metrics/prometheus/health
```

Expected: `{"prometheus_url": "http://prometheus:9090", "is_healthy": true}`

**Get containers**:
```bash
curl http://localhost:8000/metrics/containers
```

Expected: JSON with list of monitored containers

**Ingestion status**:
```bash
curl http://localhost:8000/metrics/ingestion/status
```

Expected: JSON showing query interval, stats, and buffer stats

---

### Option B: Manual Setup (Advanced)

#### Step 1: Start Prometheus

```bash
# Pull and run Prometheus
docker run -d \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest

# Verify:
curl http://localhost:9090/-/healthy
```

Expected: HTTP 200

#### Step 2: Start cAdvisor

```bash
docker run -d \
  -p 8080:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  gcr.io/cadvisor/cadvisor:latest

# Verify:
curl http://localhost:8080/api/v1.3/docker
```

Expected: JSON with container info

#### Step 3: Start App

```bash
# Set environment variable for Prometheus URL
export PROMETHEUS_URL=http://localhost:9090

# Run app
python -m src.model.fastapi_app
```

---

## Component Checks

### 1. Data Preprocessing

**Test the preprocessing pipeline**:

```bash
python -c "
from src.data.data_preprocessing import ResourceDataProcessor
import numpy as np

processor = ResourceDataProcessor()

# Create sample data
import pandas as pd
df = pd.DataFrame({
    'timestamp': pd.date_range('2026-05-01', periods=100, freq='1min'),
    'cpu_usage': np.random.uniform(20, 80, 100),
    'memory_usage': np.random.uniform(30, 70, 100)
})

# Preprocess
df_clean = processor.preprocess(df, fit_scaler=True)
print(f'✓ Preprocessing successful. Shape: {df_clean.shape}')

# Create sliding windows
X, y = processor.create_sliding_windows(df_clean, window_size=10, predict_horizon=1)
print(f'✓ Sliding windows created. X shape: {X.shape}, y shape: {y.shape}')
"
```

**Expected output**:
```
✓ Preprocessing successful. Shape: (100, 3)
✓ Sliding windows created. X shape: (90, 10, 2), y shape: (90, 1, 2)
```

### 2. Model Predictions

**Test all prediction methods**:

```bash
python src/model/predictor.py
```

**Expected output**:
```
=== LSTM MODEL PREDICTION (Advanced) ===
CPU Usage (next timestep):    13.97%
Memory Usage (next timestep): 77.55%

=== BASELINE MODEL PREDICTION (Random Forest) ===
CPU Usage (next timestep):    20.53%
Memory Usage (next timestep): 78.02%

=== ENSEMBLE PREDICTION (Baseline + LSTM Average) ===
CPU Usage (next timestep):    17.25% ± 3.28
Memory Usage (next timestep): 77.79% ± 0.24

=== MULTI-HORIZON PREDICTIONS ===
horizon_1min:
  CPU:     13.97%
  Memory:  77.55%
horizon_5min:
  CPU:     13.97%
  Memory:  77.55%
horizon_10min:
  CPU:     13.97%
  Memory:  77.55%
```

✅ **Check**: All prediction methods working

### 3. Recommendation Engine

**Test scaling recommendations**:

```bash
python -c "
from src.model.recommender import ResourceRecommender
import numpy as np

recommender = ResourceRecommender(model_dir='models', window_size=10)

# Test with high resource usage
high_usage_data = np.array([
    [75.0, 80.0], [76.5, 81.2], [78.0, 82.5], [80.0, 84.0], [81.5, 85.5],
    [83.0, 87.0], [84.5, 88.5], [86.0, 90.0], [87.5, 91.5], [89.0, 93.0]
])

decision = recommender.recommend(high_usage_data, use_model='ensemble')

print(f'Action: {decision.action.value}')
print(f'Reason: {decision.reason}')
print(f'Confidence: {decision.confidence_score:.1%}')
print(f'CPU Request: {decision.recommendation.cpu_request_millicores}m')
print(f'Memory Limit: {decision.recommendation.memory_limit_mi}Mi')
"
```

**Expected output**:
```
Action: scale_up
Reason: Scale UP: CPU prediction 85.32% > 80.0%, Memory prediction 87.32% > 80.0%
Confidence: 92.3%
CPU Request: 1290m
Memory Limit: 1024Mi
```

✅ **Check**: Recommendation engine responding to high usage

### 4. API Prediction Endpoint

**Send prediction request**:

```bash
# Windows PowerShell
$body = @{
    recent_data = @(
        @(30.5, 45.2), @(31.2, 46.1), @(32.1, 47.0), @(32.8, 47.5),
        @(33.5, 48.2), @(34.2, 49.1), @(35.0, 49.8), @(35.7, 50.5),
        @(36.4, 51.2), @(37.1, 52.0)
    )
    use_model = "ensemble"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body -UseBasicParsing | Select-Object -ExpandProperty Content

# Linux/macOS curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "recent_data": [
      [30.5, 45.2], [31.2, 46.1], [32.1, 47.0], [32.8, 47.5],
      [33.5, 48.2], [34.2, 49.1], [35.0, 49.8], [35.7, 50.5],
      [36.4, 51.2], [37.1, 52.0]
    ],
    "use_model": "ensemble"
  }'
```

**Expected response**:
```json
{
  "cpu_prediction": "650m",
  "memory_prediction": "900Mi",
  "recommendation": {
    "cpu_request": "700m",
    "cpu_limit": "1Gi",
    "memory_request": "900Mi",
    "memory_limit": "1.2Gi",
    "scaling_action": "scale_up",
    "reason": "Scale UP: CPU prediction 85.32% > 80.0%, ..."
  },
  "timestamp": "2026-05-01T12:00:00.000000",
  "confidence_score": 0.923
}
```

✅ **Check**: API returning predictions with recommendations

### 5. CLI Tool

**Interactive prediction**:

```bash
python src/model/cli.py --interactive
```

Follow prompts to enter data and select prediction model.

**Non-interactive prediction**:

```bash
python src/model/cli.py \
  --data "[[30.5, 45.2], [31.2, 46.1], [32.1, 47.0], [33.5, 48.2], [34.2, 49.1], [35.0, 49.8], [35.7, 50.5], [36.4, 51.2], [37.1, 52.0], [37.9, 52.9]]" \
  --model ensemble
```

**Expected output**:
```
============================================================
ENSEMBLE PREDICTION (Baseline + LSTM Average)
============================================================
CPU Usage (next timestep):    17.25% ± 3.28
Memory Usage (next timestep): 77.79% ± 0.24
```

✅ **Check**: CLI tool working for all modes

### 6. Dockerfile Build

**Build and test Docker image**:

```bash
# Build
docker build -t resource-predictor:latest .

# Run container
docker run -p 8000:8000 \
  -e PROMETHEUS_URL=http://host.docker.internal:9090 \
  resource-predictor:latest

# In another terminal, test health
curl http://localhost:8000/health
```

**Expected**: Healthy response, container running

✅ **Check**: Docker image builds and runs

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution**: Ensure you're running commands from the project root directory:
```bash
cd Resource-Usage-Predictor
python -m src.model.fastapi_app
```

### Issue: Port 8000 Already in Use

**Solution**: The app automatically falls back to the next available port:
```bash
# Check output for actual port
# Or set custom port
export APP_PORT=8001
python -m src.model.fastapi_app
```

### Issue: Models Not Found

**Solution**: Ensure trained models exist in `models/` directory:
```bash
ls models/
# Should show: advanced_lstm.pt, baseline_rf.pkl, scaler.pkl
```

If missing, retrain:
```bash
python src/model/train.py
```

### Issue: Prometheus Health Check Fails

**Solution**: Ensure Prometheus is running on the correct URL:
```bash
curl http://localhost:9090/-/healthy

# If not running, start it
docker run -p 9090:9090 prom/prometheus:latest
```

### Issue: No Containers in `/metrics/containers` Endpoint

**Solution**: cAdvisor needs time to collect metrics. Wait 30-60 seconds and try again:
```bash
sleep 60
curl http://localhost:8000/metrics/containers
```

---

## Performance Validation

### Check Prediction Latency

```bash
python -c "
import time
import numpy as np
from src.model.predictor import ResourcePredictor

predictor = ResourcePredictor(model_dir='models', window_size=10)
data = np.random.uniform(20, 80, (10, 2))

# Warm-up
_ = predictor.predict_with_confidence(data)

# Measure
times = []
for _ in range(10):
    t = time.perf_counter()
    _ = predictor.predict_with_confidence(data)
    times.append((time.perf_counter() - t) * 1000)

avg_latency = np.mean(times)
print(f'Average latency: {avg_latency:.2f}ms (target: <5000ms)')
print(f'Min: {np.min(times):.2f}ms, Max: {np.max(times):.2f}ms')
"
```

**Expected**: ~50-150ms average (well under 5-second SLA)

---

## Next Steps

1. **Integrate with Kubernetes**: Use the `/predict` endpoint to drive autoscaling decisions
2. **Monitor Metrics**: Set up dashboards using Prometheus + Grafana
3. **Retrain Models**: As new data accumulates, periodically retrain for better accuracy
4. **Tune Scaling Policy**: Adjust thresholds in `ScalingPolicy` based on observed patterns

---

## Support

For issues:
1. Check logs: `docker-compose logs app` or terminal output
2. Verify prerequisites are installed and correct versions
3. Test individual components using the check commands above
4. Review [readme.md](readme.md) for architecture details
