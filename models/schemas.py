from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[Dict[str, Any]]
    scores: List[float]
    processing_time: float
    strategy_used: str
    query_variations: List[str]
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

class FeedbackRequest(BaseModel):
    query_id: int
    rating: int
    feedback: Optional[str] = None
    user_id: Optional[str] = None

class IngestRequest(BaseModel):
    file_paths: List[str]
    # print("file ingested path",file_paths)
    user_id: Optional[str] = None

class IngestResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    chunks_created: int

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    model: str
    vector_store: str