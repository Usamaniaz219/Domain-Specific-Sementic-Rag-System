from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi import Response
from typing import Callable
import time

# Metrics
request_counter = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

def setup_metrics(app):
    """Setup Prometheus metrics endpoint"""
    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(REGISTRY), media_type="text/plain")

def monitor_requests(func: Callable) -> Callable:
    """Decorator to monitor function execution"""
    def wrapper(*args, **kwargs):
        # Record start time
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Record duration
        duration = time.time() - start_time
        
        # Update metrics (you would need to customize labels based on function)
        request_duration.labels(
            method="function",
            endpoint=func.__name__
        ).observe(duration)
        
        return result
    return wrapper