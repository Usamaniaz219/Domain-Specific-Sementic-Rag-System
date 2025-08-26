import uvicorn
from pathlib import Path
import argparse

# Add the current directory to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Domain-Specific Semantic RAG System")
    parser.add_argument("--host", default=settings.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=settings.API_WORKERS, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Run the application directly, not as a module
    uvicorn.run(
        "api.app:app",  # This should point to your app instance
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload and settings.ENVIRONMENT == "development"
    )

if __name__ == "__main__":
    main()