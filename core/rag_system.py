import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# from ..config.settings import settings
# from ..config.constants import RetrievalStrategy
# from ..utils.logger import get_logger
# from ..utils.monitoring import monitor_requests

# from .document_processor import DocumentProcessor
# from .embedding_engine import EmbeddingEngine
# from .vector_store import VectorStore
# from .query_processor import QueryProcessor
# from .reranker import Reranker
# from .llm_generator import LLMGenerator


from config.settings import settings
from config.constants import RetrievalStrategy
from utils.logger import get_logger
from utils.monitoring import monitor_requests

from core.document_processor import DocumentProcessor
from core.embedding_engine import EmbeddingEngine
from core.vector_store import VectorStore
from core.query_processor import QueryProcessor
from core.reranker import Reranker
from core.llm_generator import LLMGenerator


logger = get_logger(__name__)

class SemanticRAGSystem:
    """Main RAG system that integrates all components"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.query_processor = QueryProcessor()
        self.reranker = Reranker()
        self.llm_generator = LLMGenerator()
        
        self.is_initialized = False
        logger.info("Semantic RAG system initialized")
    
    @monitor_requests
    def ingest_documents(self, file_paths: List[str], user_id: Optional[str] = None):
        """Ingest documents into the system"""
        logger.info(f"Starting ingestion of {len(file_paths)} documents")
        
        # Convert to Path objects
        file_paths = [Path(fp) for fp in file_paths]
        print("file_paths",file_paths)
        
        # Load documents
        documents = self.document_processor.load_documents(file_paths)
        print("documents",documents)
        
        # Chunk documents
        chunks = self.document_processor.chunk_documents(documents)
        
        # Generate embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_engine.batch_generate_embeddings(texts)
        
        # Prepare metadata for vector store
        metadata_list = []
        for chunk in chunks:
            metadata_list.append({
                'text': chunk['text'],
                'chunk_id': chunk['id'],
                'source': chunk['metadata']['source'],
                'file_type': chunk['metadata']['file_type'].value,
                'file_hash': chunk['metadata']['file_hash'],
                'chunk_hash': chunk['metadata']['chunk_hash'],
                'chunk_length': chunk['metadata']['chunk_length']
            })
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, metadata_list)
        
        self.is_initialized = True
        logger.info("Document ingestion completed")
        
        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks)
        }
    
    @monitor_requests
    def query(self, question: str, user_id: Optional[str] = None, 
             conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a user query through the full RAG pipeline"""
        start_time = datetime.now()
        
        if not self.is_initialized:
            raise ValueError("System not initialized. Please ingest documents first.")
        
        # Step 1: Query transformation
        transformed_queries = self.query_processor.transform_query(question)
        
        # Step 2: Semantic routing
        strategy = self.query_processor.route_query(question)
        
        # Step 3: Retrieve documents for each query variation
        all_results = []
        for query_variant in transformed_queries:
            results = self.vector_store.search(
                self.embedding_engine.generate_embedding(query_variant),
                settings.TOP_K_RETRIEVAL,
                strategy
            )
            all_results.extend(results)
        
        # Remove duplicates by chunk ID
        seen_chunks = set()
        unique_results = []
        for score, metadata in all_results:
            chunk_id = metadata.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append((score, metadata))
        
        # Step 4: Rerank results
        reranked_results = self.reranker.rerank(question, unique_results)
        
        # Step 5: Generate answer
        # context = "\n\n".join([metadata['text'] for _, metadata in reranked_results])
        context = "\n\n".join([metadata.get('text', '') for _, metadata in reranked_results])
        # print("###########################################")
        print("context",context)
        # print("#####################")
        llm_response = self.llm_generator.generate_answer(question, context, conversation_history)
        # print("##############################")
        # print("llm response",llm_response)
        # print("#########################################")
        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "question": question,
            "answer": llm_response["answer"],
            "contexts": [metadata for _, metadata in reranked_results],
            "scores": [score for score, _ in reranked_results],
            "processing_time": processing_time,
            "strategy_used": strategy.value,
            "query_variations": transformed_queries,
            "model": llm_response.get("model"),
            "usage": llm_response.get("usage")
        }
        
        # Record query in history
        self._record_query_history(question, strategy.value, len(reranked_results), 
                                 processing_time, user_id)
        
        logger.info(f"Query processed in {processing_time:.2f}s. Strategy: {strategy.value}")
        return response
    
    def _record_query_history(self, query: str, strategy: str, results_count: int,
                            processing_time: float, user_id: Optional[str] = None):
        """Record query in history database"""
        from models.database import get_db_session, QueryHistory
        
        session = get_db_session()
        try:
            history = QueryHistory(
                query=query,
                strategy=strategy,
                results_count=results_count,
                processing_time=processing_time,
                user_id=user_id
            )
            session.add(history)
            session.commit()
            return history.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording query history: {str(e)}")
        finally:
            session.close()
    
    def record_feedback(self, query_id: int, rating: int, 
                       feedback: Optional[str] = None, user_id: Optional[str] = None):
        """Record user feedback for a query"""
        from ..models.database import get_db_session, UserFeedback
        
        session = get_db_session()
        try:
            feedback_record = UserFeedback(
                query_id=query_id,
                rating=rating,
                feedback=feedback,
                user_id=user_id
            )
            session.add(feedback_record)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording feedback: {str(e)}")
        finally:
            session.close()
    
    def batch_query(self, questions: List[str], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process multiple queries in batch for efficiency"""
        if not self.is_initialized:
            raise ValueError("System not initialized. Please ingest documents first.")
        
        # Process each query
        results = []
        for question in questions:
            try:
                result = self.query(question, user_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query: {question[:50]}...: {str(e)}")
                results.append({
                    "question": question,
                    "answer": "Sorry, I encountered an error while processing your query.",
                    "error": str(e)
                })
        
        return results