from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
import time
import datetime
import config.settings as settings
from models.schemas import (
    QueryRequest, QueryResponse, FeedbackRequest, 
    IngestRequest, IngestResponse, HealthResponse
)
from core.rag_system import SemanticRAGSystem


router = APIRouter()

# Initialize RAG system
rag_system = SemanticRAGSystem()

# @router.post("/query", response_model=QueryResponse)
# async def query_endpoint(request: QueryRequest):
#     """Process a user query through the RAG system"""
#     try:
#         response = rag_system.query(
#             request.question,
#             request.user_id,
#             request.conversation_history
#         )
#         return response
#     except Exception as e:
#         print("query response error")
#         print(f"Response was: {response}")
#         raise HTTPException(status_code=500, detail=str(e))



@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a user query through the RAG system"""
    try:
        # Process the query
        response = rag_system.query(
            request.question,
            request.user_id,
            request.conversation_history
        )
        
        # Validate the response matches our expected format
        required_fields = ["question", "answer", "contexts", "scores", 
                          "processing_time", "strategy_used", "query_variations"]
        
        for field in required_fields:
            if field not in response:
                print(f"⚠️  Missing field in response: {field}")
                response[field] = "" if field == "answer" else [] if field.endswith("s") else 0.0
        
        # Ensure types are correct
        if not isinstance(response.get("scores", []), list):
            response["scores"] = []
        if not isinstance(response.get("contexts", []), list):
            response["contexts"] = []
        if not isinstance(response.get("query_variations", []), list):
            response["query_variations"] = [request.question]
        
        return response
        
    except Exception as e:
        print(f"Query processing error: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a proper error response that matches QueryResponse model
        error_response = QueryResponse(
            question=request.question,
            answer=f"Error processing query: {str(e)[:100]}...",
            contexts=[],
            scores=[],
            processing_time=0.0,
            strategy_used="error",
            query_variations=[request.question],
            model="error",
            usage=None
        )
        
        return error_response

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest documents into the system"""
    try:
        # Run ingestion in background
        background_tasks.add_task(
            rag_system.ingest_documents,
            request.file_paths,
            request.user_id
        )
        
        return IngestResponse(
            success=True,
            message="Document ingestion started in background",
            documents_processed=0,  # Will be updated async
            chunks_created=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Submit feedback for a query response"""
    try:
        rag_system.record_feedback(
            request.query_id,
            request.rating,
            request.feedback,
            request.user_id
        )
        return {"success": True, "message": "Feedback recorded"}
    except Exception as e:
        print("##########################################")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "model": settings.LLM_MODEL,
        "vector_store": settings.VECTOR_STORE_TYPE
    }