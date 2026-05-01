"""Demo script for Prometheus metric ingestion integration."""

import logging
import os
import sys
import time
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.prometheus_client import PrometheusConnector
from src.data.metric_transformer import MetricTransformer, TimeSeriesBuffer
from src.model.metric_ingestion import MetricIngestionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_prometheus_client():
    """Demo 1: Test Prometheus client connectivity."""
    logger.info("=" * 70)
    logger.info("DEMO 1: Testing Prometheus Client")
    logger.info("=" * 70)

    prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    logger.info(f"Connecting to Prometheus at: {prometheus_url}")

    with PrometheusConnector(prometheus_url=prometheus_url) as client:
        # Check health
        logger.info("Checking Prometheus health...")
        is_healthy = client.health_check()
        logger.info(f"Prometheus health: {'✓ HEALTHY' if is_healthy else '✗ UNHEALTHY'}")

        if not is_healthy:
            logger.warning("Prometheus is not reachable. Skipping query tests.")
            logger.info("Make sure Prometheus is running at: http://localhost:9090")
            logger.info("Or set PROMETHEUS_URL environment variable")
            return

        # Get available metrics
        logger.info("Fetching available metrics...")
        metrics = client.get_metrics()
        if metrics:
            metric_list = metrics.get("metrics", [])
            logger.info(f"Found {len(metric_list)} available metrics")
            # Show first 5
            for metric in metric_list[:5]:
                logger.info(f"  - {metric}")
            if len(metric_list) > 5:
                logger.info(f"  ... and {len(metric_list) - 5} more")

        # Query container CPU
        logger.info("Querying container CPU metrics...")
        cpu_result = client.query_container_cpu()
        if cpu_result:
            cpu_metrics = cpu_result.get("result", [])
            logger.info(f"Found {len(cpu_metrics)} containers with CPU metrics")
            for metric in cpu_metrics[:3]:
                labels = metric.get("metric", {})
                value = metric.get("value", [0, 0])[1]
                logger.info(f"  - {labels.get('container', 'unknown')}: {value}")

        # Query container Memory
        logger.info("Querying container memory metrics...")
        memory_result = client.query_container_memory()
        if memory_result:
            memory_metrics = memory_result.get("result", [])
            logger.info(f"Found {len(memory_metrics)} containers with memory metrics")


def demo_metric_transformer():
    """Demo 2: Test metric transformation."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 2: Testing Metric Transformer")
    logger.info("=" * 70)

    transformer = MetricTransformer(cpu_cores=2, memory_mb=2048)

    # Test CPU normalization
    logger.info("Testing CPU normalization...")
    cpu_seconds = 1.2  # 1.2 CPU seconds over 60s interval
    cpu_percent = transformer.normalize_cpu_usage(cpu_seconds, interval_seconds=60)
    logger.info(f"  CPU seconds: {cpu_seconds} → {cpu_percent:.2f}%")

    # Test memory normalization
    logger.info("Testing memory normalization...")
    memory_bytes = 1024 * 1024 * 512  # 512 MB
    memory_percent = transformer.normalize_memory_usage(memory_bytes)
    logger.info(f"  Memory bytes: {memory_bytes:,} → {memory_percent:.2f}%")

    # Test Prometheus query result transformation
    logger.info("Testing Prometheus query result transformation...")
    mock_cpu_result = {
        "resultType": "instant",
        "result": [
            {
                "metric": {"container": "app-pod-1"},
                "value": [int(time.time()), "1.2"],
            }
        ],
    }
    cpu_metrics = transformer.transform_prometheus_query_result(mock_cpu_result, "cpu")
    logger.info(f"  Transformed {len(cpu_metrics)} metrics: {cpu_metrics}")


def demo_time_series_buffer():
    """Demo 3: Test time-series buffer."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 3: Testing Time-Series Buffer")
    logger.info("=" * 70)

    buffer = TimeSeriesBuffer(max_age_minutes=60, min_data_points=3)

    # Add sample metrics
    logger.info("Adding sample metrics to buffer...")
    timestamp = int(time.time())
    for i in range(5):
        buffer.add_metrics(
            "app-pod-1",
            timestamp - (5 - i) * 60,  # Add 5 data points, 1 minute apart
            cpu_percent=30.0 + i * 2,
            memory_percent=45.0 + i * 1.5,
        )

    # Check status
    logger.info("Buffer statistics:")
    stats = buffer.get_stats()
    for container_id, stat in stats.items():
        logger.info(f"  Container: {container_id}")
        logger.info(f"    - Data points: {stat['data_points']}")
        logger.info(f"    - Age (seconds): {stat['age_seconds']}")
        logger.info(f"    - Ready for prediction: {stat['is_ready']}")

    # Get buffer data
    logger.info("Retrieving buffer data for prediction...")
    model_input = buffer.get_buffer("app-pod-1")
    if model_input is not None:
        logger.info(f"  Model input shape: {model_input.shape}")
        logger.info(f"  Data:\n{model_input}")


def demo_metric_ingestion_manager():
    """Demo 4: Test metric ingestion manager."""
    logger.info("\n" + "=" * 70)
    logger.info("DEMO 4: Testing Metric Ingestion Manager")
    logger.info("=" * 70)

    prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    logger.info(f"Initializing manager with Prometheus: {prometheus_url}")

    manager = MetricIngestionManager(
        prometheus_url=prometheus_url,
        query_interval_seconds=5,  # Short interval for demo
        buffer_size_minutes=60,
        min_data_points=2,  # Lower threshold for demo
    )

    # Register callbacks
    def on_prediction(result):
        logger.info(f"✓ Prediction made: {result['container_id']} - CPU: {result['prediction'].get('cpu', 'N/A')}")

    def on_error(error):
        logger.error(f"✗ Error: {error}")

    manager.set_prediction_callback(on_prediction)
    manager.set_error_callback(on_error)

    # Get initial status
    logger.info("Initial status:")
    status = manager.get_status()
    logger.info(f"  - Is running: {status['is_running']}")
    logger.info(f"  - Query interval: {status['query_interval_seconds']}s")

    # Check Prometheus connectivity
    logger.info("Checking Prometheus connectivity...")
    if not manager.prometheus_client.health_check():
        logger.warning("✗ Prometheus is not reachable")
        logger.info("To use metric ingestion, start Prometheus:")
        logger.info("  docker run -p 9090:9090 prom/prometheus")
        return

    logger.info("✓ Prometheus is reachable")

    # Start ingestion
    logger.info("Starting metric ingestion (will run for 30 seconds)...")
    manager.start()

    try:
        # Run for 30 seconds
        for i in range(6):
            time.sleep(5)
            logger.info(f"  [{i+1}/6] Status check...")
            status = manager.get_status()
            logger.info(f"      - Queries successful: {status['stats']['queries_successful']}")
            logger.info(f"      - Predictions made: {status['stats']['predictions_made']}")
            logger.info(f"      - Monitored containers: {len(status['monitored_containers'])}")

    finally:
        logger.info("Stopping metric ingestion...")
        manager.stop()
        manager.close()

    # Final stats
    final_status = manager.get_status()
    logger.info("Final statistics:")
    logger.info(f"  - Total queries: {final_status['stats']['queries_total']}")
    logger.info(f"  - Successful: {final_status['stats']['queries_successful']}")
    logger.info(f"  - Failed: {final_status['stats']['queries_failed']}")
    logger.info(f"  - Predictions made: {final_status['stats']['predictions_made']}")


def main():
    """Run all demos."""
    logger.info("\n" + "=" * 70)
    logger.info("PROMETHEUS INTEGRATION DEMO")
    logger.info("=" * 70)

    try:
        # Demo 1: Prometheus client
        demo_prometheus_client()

        # Demo 2: Metric transformer
        demo_metric_transformer()

        # Demo 3: Time-series buffer
        demo_time_series_buffer()

        # Demo 4: Metric ingestion manager (requires Prometheus)
        demo_metric_ingestion_manager()

        logger.info("\n" + "=" * 70)
        logger.info("✓ DEMO COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"✗ Demo failed with error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
