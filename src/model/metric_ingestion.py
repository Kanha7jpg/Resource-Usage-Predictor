"""Automated metric ingestion scheduler using APScheduler."""

import logging
import os
from datetime import datetime
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.data.prometheus_client import PrometheusConnector
from src.data.metric_transformer import MetricTransformer, TimeSeriesBuffer
from src.model.predictor import ResourcePredictor

logger = logging.getLogger(__name__)


class MetricIngestionManager:
    """Manages automated metric collection and ingestion."""

    def __init__(
        self,
        prometheus_url: str = None,
        query_interval_seconds: int = 60,
        buffer_size_minutes: int = 60,
        min_data_points: int = 10,
        cpu_cores: int = 1,
        memory_mb: int = 1024,
        container_selector: str = "",
    ):
        """
        Initialize metric ingestion manager.

        Args:
            prometheus_url: Prometheus server URL
            query_interval_seconds: Interval between queries (default 60s)
            buffer_size_minutes: Time window to keep metrics (default 60 min)
            min_data_points: Minimum data points before prediction (default 10)
            cpu_cores: Number of CPU cores for normalization
            memory_mb: Total memory in MB for normalization
            container_selector: Prometheus label selector for filtering containers
        """
        self.prometheus_url = prometheus_url or os.getenv(
            "PROMETHEUS_URL", "http://localhost:9090"
        )
        self.query_interval_seconds = query_interval_seconds
        self.container_selector = container_selector or os.getenv("CONTAINERS_SELECTOR", "")
        self.cpu_cores = cpu_cores
        self.memory_mb = memory_mb

        # Initialize components
        self.prometheus_client = PrometheusConnector(prometheus_url=self.prometheus_url)
        self.transformer = MetricTransformer(
            cpu_cores=cpu_cores, memory_mb=memory_mb, interpolation_method="linear"
        )
        self.buffer = TimeSeriesBuffer(
            max_age_minutes=buffer_size_minutes, min_data_points=min_data_points
        )
        self.predictor = ResourcePredictor()

        # Scheduler
        self.scheduler = BackgroundScheduler()
        self.is_running = False

        # Callbacks
        self.on_prediction_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None

        # Statistics
        self.stats = {
            "queries_total": 0,
            "queries_successful": 0,
            "queries_failed": 0,
            "predictions_made": 0,
            "last_query_time": None,
            "last_error": None,
        }

    def start(self):
        """Start the ingestion scheduler."""
        if self.is_running:
            logger.warning("Ingestion manager already running")
            return

        logger.info(f"Starting metric ingestion manager (interval: {self.query_interval_seconds}s)")

        # Add job
        self.scheduler.add_job(
            self._ingest_metrics,
            trigger=IntervalTrigger(seconds=self.query_interval_seconds),
            id="prometheus_ingest",
            name="Prometheus Metric Ingestion",
            replace_existing=True,
        )

        self.scheduler.start()
        self.is_running = True
        logger.info("Metric ingestion manager started successfully")

    def stop(self):
        """Stop the ingestion scheduler."""
        if not self.is_running:
            logger.warning("Ingestion manager not running")
            return

        logger.info("Stopping metric ingestion manager")
        self.scheduler.shutdown(wait=True)
        self.is_running = False
        logger.info("Metric ingestion manager stopped")

    def _ingest_metrics(self):
        """Execute metric ingestion cycle."""
        try:
            self.stats["queries_total"] += 1

            # Check Prometheus health
            if not self.prometheus_client.health_check():
                logger.warning("Prometheus health check failed")
                self.stats["queries_failed"] += 1
                return

            # Query CPU and memory metrics
            cpu_result = self.prometheus_client.query_container_cpu(
                container_selector=self.container_selector
            )
            memory_result = self.prometheus_client.query_container_memory(
                container_selector=self.container_selector
            )

            if not cpu_result or not memory_result:
                logger.warning("Failed to query metrics from Prometheus")
                self.stats["queries_failed"] += 1
                return

            self.stats["queries_successful"] += 1
            self.stats["last_query_time"] = datetime.now()

            # Process results for each container
            self._process_query_results(cpu_result, memory_result)

        except Exception as e:
            logger.error(f"Error during metric ingestion: {e}")
            self.stats["queries_failed"] += 1
            self.stats["last_error"] = str(e)
            if self.on_error_callback:
                self.on_error_callback(e)

    def _process_query_results(self, cpu_result: dict, memory_result: dict):
        """Process Prometheus query results and trigger predictions."""
        try:
            cpu_metrics = cpu_result.get("result", [])
            memory_metrics = memory_result.get("result", [])

            if not cpu_metrics or not memory_metrics:
                logger.debug("No metrics returned from Prometheus")
                return

            # Create mapping of container labels to metrics
            cpu_map = {self._get_container_id(m): m for m in cpu_metrics}
            memory_map = {self._get_container_id(m): m for m in memory_metrics}

            # Process each container
            for container_id in cpu_map.keys():
                if container_id not in memory_map:
                    continue

                try:
                    cpu_value = float(cpu_map[container_id]["value"][1])
                    memory_value = float(memory_map[container_id]["value"][1])
                    timestamp = int(cpu_map[container_id]["value"][0])

                    # Normalize values
                    cpu_percent = self.transformer.normalize_cpu_usage(cpu_value, interval_seconds=60)
                    memory_percent = self.transformer.normalize_memory_usage(memory_value)

                    # Add to buffer
                    self.buffer.add_metrics(container_id, timestamp, cpu_percent, memory_percent)

                    # Check if ready for prediction
                    if self.buffer.is_ready_for_prediction(container_id):
                        self._make_prediction(container_id)

                except (ValueError, KeyError, IndexError) as e:
                    logger.warning(f"Error processing metrics for {container_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing query results: {e}")

    def _make_prediction(self, container_id: str):
        """Make prediction for container using buffered data."""
        try:
            model_input = self.buffer.get_buffer(container_id)
            if model_input is None:
                return

            # Get prediction
            prediction = self.predictor.predict_with_confidence(model_input)

            self.stats["predictions_made"] += 1

            result = {
                "container_id": container_id,
                "timestamp": datetime.now(),
                "prediction": prediction,
                "buffer_size": len(model_input),
            }

            logger.info(
                f"Prediction for {container_id}: CPU={prediction['cpu']:.1f}%, "
                f"Memory={prediction['memory']:.1f}%"
            )

            # Call callback if registered
            if self.on_prediction_callback:
                self.on_prediction_callback(result)

        except Exception as e:
            logger.error(f"Error making prediction for {container_id}: {e}")

    @staticmethod
    def _get_container_id(metric: dict) -> str:
        """Extract container ID from metric labels."""
        labels = metric.get("metric", {})
        # Try multiple possible label names
        for label in ["container", "container_name", "pod", "pod_name", "instance"]:
            if label in labels:
                return labels[label]
        # Fallback
        return labels.get("instance", "unknown")

    def set_prediction_callback(self, callback: Callable):
        """Set callback function to call when prediction is made."""
        self.on_prediction_callback = callback

    def set_error_callback(self, callback: Callable):
        """Set callback function to call when error occurs."""
        self.on_error_callback = callback

    def get_status(self) -> dict:
        """Get current status of ingestion manager."""
        return {
            "is_running": self.is_running,
            "query_interval_seconds": self.query_interval_seconds,
            "prometheus_url": self.prometheus_url,
            "container_selector": self.container_selector,
            "stats": self.stats,
            "buffer_stats": self.buffer.get_stats(),
            "monitored_containers": self.buffer.get_all_containers(),
        }

    def get_container_predictions(self, container_id: str) -> Optional[dict]:
        """Get latest predictions for a container."""
        if not self.buffer.is_ready_for_prediction(container_id):
            return None

        model_input = self.buffer.get_buffer(container_id)
        if model_input is None:
            return None

        try:
            return self.predictor.predict_with_confidence(model_input)
        except Exception as e:
            logger.error(f"Error predicting for {container_id}: {e}")
            return None

    def close(self):
        """Clean up resources."""
        if self.is_running:
            self.stop()
        self.prometheus_client.close()
