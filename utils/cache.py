import redis
import json
from typing import Any, Optional
from functools import wraps

from config.settings import settings

# Redis client (if configured)
redis_client = None
if settings.REDIS_URL:
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        redis_client.ping()  # Test connection
    except redis.ConnectionError:
        redis_client = None
        print("Redis connection failed, using in-memory cache")

# Fallback in-memory cache
memory_cache = {}

def get_cache(key: str) -> Optional[Any]:
    """Get value from cache"""
    if redis_client:
        try:
            value = redis_client.get(key)
            return json.loads(value) if value else None
        except redis.RedisError:
            return memory_cache.get(key)
    else:
        return memory_cache.get(key)

def set_cache(key: str, value: Any, ttl: int = None) -> bool:
    """Set value in cache"""
    if redis_client:
        try:
            if ttl:
                redis_client.setex(key, ttl, json.dumps(value))
            else:
                redis_client.set(key, json.dumps(value))
            return True
        except redis.RedisError:
            memory_cache[key] = value
            return False
    else:
        memory_cache[key] = value
        return True

def cache_decorator(ttl: int = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            cached = get_cache(cache_key)
            if cached is not None:
                return cached
            
            # Call function and cache result
            result = func(*args, **kwargs)
            set_cache(cache_key, result, ttl)
            return result
        return wrapper
    return decorator