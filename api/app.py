from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import datetime
import time

# from ..config.settings import settings
# from ..utils.monitoring import setup_metrics, monitor_requests
# from .middleware import AuthMiddleware, LoggingMiddleware
# from .routes import router as api_router

from config.settings import settings
from utils.monitoring import setup_metrics, monitor_requests
from api.middleware import AuthMiddleware, LoggingMiddleware
from api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application"""
    # Startup
    print("Starting RAG System API...")
    setup_metrics(app)
    yield
    # Shutdown
    print("Shutting down RAG System API...")

app = FastAPI(
    title=settings.APP_NAME,
    description="Domain-Specific Semantic RAG System API",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)

# Routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Domain-Specific Semantic RAG System API"}

@app.get("/health")
async def health():
    from core.rag_system import SemanticRAGSystem
    rag_system = SemanticRAGSystem()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "model": settings.LLM_MODEL,
        "vector_store": settings.VECTOR_STORE_TYPE
    }