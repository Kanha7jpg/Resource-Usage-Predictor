"""FastAPI service for resource usage prediction and recommendation."""

import json
import logging
import os
import socket
import sys
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.predictor import ResourcePredictor
from src.model.recommender import ResourceRecommender
from src.model.metric_ingestion import MetricIngestionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecentDataRequest(BaseModel):
    """Request body for prediction endpoint."""

    recent_data: list[list[float]] = Field(
        ...,
        description="Recent time-series data as [[cpu_percent, memory_percent], ...]"
    )
    use_model: str = Field(default="ensemble", description="Prediction model to use")
    current_cpu: int | None = Field(default=None, description="Current CPU request in millicores")
    current_memory: int | None = Field(default=None, description="Current memory request in Mi")

    @field_validator("recent_data")
    @classmethod
    def validate_recent_data(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) < 1:
            raise ValueError("recent_data must contain at least one timestep")
        try:
            arr = np.asarray(value, dtype=float)
        except Exception as exc:  # pragma: no cover - defensive validation
            raise ValueError(f"recent_data must be numeric: {exc}") from exc
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("recent_data must have shape (N, 2) with CPU and memory values")
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("recent_data cannot contain NaN or infinite values")
        return value


class PredictionResponse(BaseModel):
    """Response body for the prediction endpoint."""

    cpu_prediction: str
    memory_prediction: str
    recommendation: dict
    timestamp: str
    confidence_score: float | None = None


predictor: ResourcePredictor | None = None
recommender: ResourceRecommender | None = None
ingestion_manager: MetricIngestionManager | None = None


def initialize_services() -> None:
    """Load the trained prediction and recommendation models."""
    global predictor, recommender, ingestion_manager

    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    predictor = ResourcePredictor(model_dir=model_dir, window_size=10)
    recommender = ResourceRecommender(model_dir=model_dir, window_size=10)
    logger.info("FastAPI prediction and recommendation services initialized")

    # Initialize metric ingestion manager
    prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    query_interval = int(os.getenv("QUERY_INTERVAL_SECONDS", "60"))
    container_selector = os.getenv("CONTAINERS_SELECTOR", "")

    ingestion_manager = MetricIngestionManager(
        prometheus_url=prometheus_url,
        query_interval_seconds=query_interval,
        container_selector=container_selector,
    )

    # Register callbacks
    ingestion_manager.set_prediction_callback(_on_prediction_made)
    ingestion_manager.set_error_callback(_on_ingestion_error)

    logger.info("Metric ingestion manager initialized")


def _on_prediction_made(result: dict) -> None:
    """Callback when prediction is made."""
    logger.info(f"Prediction callback: {result['container_id']} - CPU: {result['prediction'].get('cpu', 'N/A')}")


def _on_ingestion_error(error: Exception) -> None:
    """Callback when ingestion error occurs."""
    logger.error(f"Ingestion error callback: {error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_services()
    # Start metric ingestion if Prometheus is available
    if ingestion_manager:
        try:
            ingestion_manager.start()
        except Exception as e:
            logger.warning(f"Failed to start metric ingestion: {e}")
    yield
    # Cleanup on shutdown
    if ingestion_manager:
        ingestion_manager.stop()
        ingestion_manager.close()


app = FastAPI(
    title="Resource Usage Prediction API",
    description="FastAPI service for CPU and memory predictions with recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Return service status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "predictor_loaded": predictor is not None,
        "recommender_loaded": recommender is not None,
        "ingestion_running": ingestion_manager.is_running if ingestion_manager else False,
    }


def _format_cpu_prediction(value: float) -> str:
    """Format a CPU prediction as millicores."""
    return f"{max(1, int(round(value * 10)))}m"


def _format_memory_prediction(value: float) -> str:
    """Format a memory prediction as Mi or Gi."""
    memory_mi = max(1, int(round(value * 10.24)))
    if memory_mi >= 1024:
        whole_gi, remainder_mi = divmod(memory_mi, 1024)
        if remainder_mi == 0:
            return f"{whole_gi}Gi"
        return f"{whole_gi}.{int(round(remainder_mi / 102.4)):01d}Gi"
    return f"{memory_mi}Mi"


@app.post("/predict", response_model=PredictionResponse)
def predict(request: RecentDataRequest) -> PredictionResponse:
    """Return predictions and a resource recommendation."""
    if predictor is None or recommender is None:
        raise HTTPException(status_code=503, detail="Prediction services are not initialized")

    recent_data = np.asarray(request.recent_data, dtype=float)

    if request.use_model not in {"lstm", "baseline", "ensemble", "multi_horizon"}:
        raise HTTPException(
            status_code=400,
            detail="use_model must be one of: lstm, baseline, ensemble, multi_horizon",
        )

    if request.use_model == "baseline":
        prediction = predictor.predict_baseline(recent_data)
    elif request.use_model == "lstm":
        prediction = predictor.predict_lstm(recent_data)
    else:
        prediction = predictor.predict_with_confidence(recent_data)

    decision = recommender.recommend(recent_data, use_model=request.use_model if request.use_model != "multi_horizon" else "ensemble")

    cpu_prediction_value = prediction["cpu"] if isinstance(prediction, dict) else prediction[0]
    memory_prediction_value = prediction["memory"] if isinstance(prediction, dict) else prediction[1]

    cpu_prediction = _format_cpu_prediction(cpu_prediction_value)
    memory_prediction = _format_memory_prediction(memory_prediction_value)

    recommendation = {
        "cpu_request": f"{decision.recommendation.cpu_request_millicores}m",
        "cpu_limit": f"{decision.recommendation.cpu_limit_millicores}m",
        "memory_request": f"{decision.recommendation.memory_request_mi}Mi",
        "memory_limit": f"{decision.recommendation.memory_limit_mi}Mi",
        "scaling_action": decision.action.value,
        "reason": decision.reason,
    }

    return PredictionResponse(
        cpu_prediction=cpu_prediction,
        memory_prediction=memory_prediction,
        recommendation=recommendation,
        timestamp=datetime.utcnow().isoformat(),
        confidence_score=round(decision.confidence_score, 3),
    )


# ============================================================================
# Prometheus Metrics Endpoints
# ============================================================================


@app.get("/metrics/prometheus/health")
def prometheus_health() -> dict:
    """Check Prometheus connectivity."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")

    is_healthy = ingestion_manager.prometheus_client.health_check()
    return {
        "prometheus_url": ingestion_manager.prometheus_url,
        "is_healthy": is_healthy,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics/containers")
def get_containers() -> dict:
    """Get all monitored containers with latest metrics."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")

    containers = ingestion_manager.buffer.get_all_containers()
    buffer_stats = ingestion_manager.buffer.get_stats()

    return {
        "containers": containers,
        "count": len(containers),
        "buffer_stats": buffer_stats,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics/containers/{container_id}/history")
def get_container_history(container_id: str, range_minutes: int = 60) -> dict:
    """Get historical metrics for a container."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")

    if container_id not in ingestion_manager.buffer.buffers:
        raise HTTPException(status_code=404, detail=f"Container {container_id} not found")

    df = ingestion_manager.buffer.buffers[container_id]

    if df.empty:
        return {
            "container_id": container_id,
            "data_points": 0,
            "history": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

    # Convert to list of dicts
    history = df[["timestamp", "cpu_percent", "memory_percent"]].to_dict(orient="records")

    return {
        "container_id": container_id,
        "data_points": len(df),
        "history": history,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics/containers/{container_id}/predict")
def get_container_prediction(container_id: str) -> dict:
    """Get prediction for specific container."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")

    if not ingestion_manager.buffer.is_ready_for_prediction(container_id):
        raise HTTPException(
            status_code=400,
            detail=f"Container {container_id} does not have enough data for prediction",
        )

    prediction = ingestion_manager.get_container_predictions(container_id)

    if prediction is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {
        "container_id": container_id,
        "prediction": prediction,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics/ingestion/status")
def get_ingestion_status() -> dict:
    """Get current ingestion status."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")

    return ingestion_manager.get_status()


@app.post("/metrics/ingestion/start")
def start_ingestion() -> dict:
    """Start automated ingestion."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")

    if ingestion_manager.is_running:
        return {"status": "already_running", "message": "Ingestion already started"}

    ingestion_manager.start()
    return {"status": "started", "message": "Ingestion started successfully"}


@app.post("/metrics/ingestion/stop")
def stop_ingestion() -> dict:
    """Stop automated ingestion."""
    if not ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")

    if not ingestion_manager.is_running:
        return {"status": "not_running", "message": "Ingestion not running"}

    ingestion_manager.stop()
    return {"status": "stopped", "message": "Ingestion stopped successfully"}


def _is_port_available(host: str, port: int) -> bool:
    """Return True when the TCP port can be bound on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _find_available_port(host: str, preferred_port: int, max_tries: int = 20) -> int:
    """Find the first available port, starting from preferred_port."""
    for offset in range(max_tries):
        candidate = preferred_port + offset
        if _is_port_available(host, candidate):
            return candidate
    raise RuntimeError(
        f"No free port found in range {preferred_port}-{preferred_port + max_tries - 1}"
    )


def main() -> None:
    """Run the FastAPI app with Uvicorn."""
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    preferred_port = int(os.getenv("APP_PORT", os.getenv("PORT", "8000")))
    auto_fallback = os.getenv("AUTO_PORT_FALLBACK", "true").lower() in {"1", "true", "yes"}

    if auto_fallback and not _is_port_available(host, preferred_port):
        selected_port = _find_available_port(host, preferred_port + 1)
        logger.warning(
            "Port %s is already in use. Starting server on port %s instead.",
            preferred_port,
            selected_port,
        )
    else:
        selected_port = preferred_port

    uvicorn.run("src.model.fastapi_app:app", host=host, port=selected_port, reload=False)


if __name__ == "__main__":
    main()
