from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
from config.settings import settings
from utils.logger import get_logger
from utils.monitoring import request_counter, request_duration

logger = get_logger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API authentication"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks
        if request.url.path in ["/health", "/metrics", "/"]:
            return await call_next(request)
        
        # Check API key
        api_key = request.headers.get("API-Key")
        if not api_key or api_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        return await call_next(request)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and monitoring"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timer
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log request
            logger.info(
                f"Request {request_id} {request.method} {request.url.path} "
                f"completed in {duration:.2f}s with status {response.status_code}"
            )
            
            # Update metrics
            request_counter.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            print("################################")
            print("duration ###########",duration)
            logger.error(
                f"Request {request_id} {request.method} {request.url.path} "
                f"failed after {duration:.2f}s: {str(e)}"
            )
            
            # Update metrics for errors
            request_counter.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).inc()
            
            raise e