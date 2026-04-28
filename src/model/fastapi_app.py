"""FastAPI service for resource usage prediction and recommendation."""

import json
import logging
import os
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


def initialize_services() -> None:
    """Load the trained prediction and recommendation models."""
    global predictor, recommender

    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    predictor = ResourcePredictor(model_dir=model_dir, window_size=10)
    recommender = ResourceRecommender(model_dir=model_dir, window_size=10)
    logger.info("FastAPI prediction and recommendation services initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_services()
    yield


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


def main() -> None:
    """Run the FastAPI app with Uvicorn."""
    import uvicorn

    uvicorn.run("src.model.fastapi_app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()