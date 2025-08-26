import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os

from config.settings import settings
from config.constants import RetrievalStrategy
from utils.logger import get_logger
from models.database import get_db_session, DocumentChunk
import datetime

logger = get_logger(__name__)

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(self):
        self.vector_store_type = settings.VECTOR_STORE_TYPE
        self.collection_name = settings.COLLECTION_NAME
        
        if self.vector_store_type == "chroma":
            self._init_chroma()
        elif self.vector_store_type == "pinecone":
            self._init_pinecone()
        elif self.vector_store_type == "weaviate":
            self._init_weaviate()
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")
        
        logger.info(f"Vector store initialized with {self.vector_store_type}")
    
    def _init_chroma(self):
        """Initialize ChromaDB vector store with new configuration"""
        try:
            import chromadb
            
            # NEW: Use PersistentClient instead of the old Client with ChromaSettings
            self.client = chromadb.PersistentClient(
                path=str(settings.CACHE_DIR / "chroma")
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            logger.error("chromadb is required for Chroma vector store")
            raise
    
    def _init_pinecone(self):
        """Initialize Pinecone vector store"""
        try:
            import pinecone
            
            pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
            
            if self.collection_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.collection_name,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine"
                )
            
            self.index = pinecone.Index(self.collection_name)
        except ImportError:
            logger.error("pinecone-client is required for Pinecone vector store")
            raise
    
    def _init_weaviate(self):
        """Initialize Weaviate vector store"""
        try:
            import weaviate
            from weaviate import Client
            
            self.client = Client(
                url=os.getenv("WEAVIATE_URL", "http://localhost:8080")
            )
            
            # Create schema if it doesn't exist
            if not self.client.schema.contains({"class": self.collection_name}):
                schema = {
                    "class": self.collection_name,
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"]
                        },
                        {
                            "name": "source",
                            "dataType": ["text"]
                        },
                        {
                            "name": "file_type",
                            "dataType": ["text"]
                        }
                    ]
                }
                self.client.schema.create_class(schema)
        except ImportError:
            logger.error("weaviate-client is required for Weaviate vector store")
            raise
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add vectors to the store"""
        if len(vectors) != len(metadata_list):
            raise ValueError("Vectors and metadata must have the same length")
        
        if self.vector_store_type == "chroma":
            self._add_to_chroma(vectors, metadata_list)
        elif self.vector_store_type == "pinecone":
            self._add_to_pinecone(vectors, metadata_list)
        elif self.vector_store_type == "weaviate":
            self._add_to_weaviate(vectors, metadata_list)
        
        # Also store in database for metadata management
        self._store_in_database(metadata_list)
        
        logger.info(f"Added {len(vectors)} vectors to store")
    
    def _add_to_chroma(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add vectors to ChromaDB"""
        ids = [meta['chunk_id'] for meta in metadata_list]
        documents = [meta['text'] for meta in metadata_list]
        metadatas = [{k: v for k, v in meta.items() if k != 'text'} for meta in metadata_list]
        
        self.collection.add(
            embeddings=vectors.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def _add_to_pinecone(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add vectors to Pinecone"""
        vectors_with_ids = []
        for i, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
            vectors_with_ids.append((
                metadata['chunk_id'],
                vector.tolist(),
                {k: v for k, v in metadata.items() if k != 'text'}
            ))
        
        # Upsert in batches
        for i in range(0, len(vectors_with_ids), 100):
            batch = vectors_with_ids[i:i+100]
            self.index.upsert(vectors=batch)
    
    def _add_to_weaviate(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add vectors to Weaviate"""
        with self.client.batch as batch:
            for i, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
                properties = {
                    "text": metadata['text'],
                    "source": metadata.get('source', ''),
                    "file_type": metadata.get('file_type', '')
                }
                
                self.client.data_object.create(
                    data_object=properties,
                    class_name=self.collection_name,
                    vector=vector.tolist()
                )
    
    def _store_in_database(self, metadata_list: List[Dict[str, Any]]):
        """Store metadata in relational database"""
        session = None
        try:
            session = get_db_session()
            
            for metadata in metadata_list:
                # Create DocumentChunk object from metadata
                chunk = DocumentChunk(
                    chunk_id=metadata.get('chunk_id'),
                    text=metadata.get('text', ''),
                    source=metadata.get('source', ''),
                    file_type=metadata.get('file_type', ''),
                    file_hash=metadata.get('file_hash', ''),
                    chunk_hash=metadata.get('chunk_hash', ''),
                    chunk_length=metadata.get('chunk_length', 0),
                    meta_data=metadata  # Store the full metadata
                )
                session.merge(chunk)  # Use merge to handle updates
            
            session.commit()
            logger.info(f"Stored {len(metadata_list)} chunks in database")
            
        except Exception as e:
            # if session:
            #     session.rollback()
            logger.error(f"Error storing metadata in database: {str(e)}")
            # Don't re-raise if you want to continue despite database errors
            # raise
        finally:
            if session:
                session.close()

    # def _store_in_database(self, metadata_list: List[Dict[str, Any]]):
    #     """Store metadata in relational database"""
    #     session = get_db_session()
        
    #     try:
    #         for metadata in metadata_list:
    #             chunk = DocumentChunk(
    #                 chunk_id=metadata['chunk_id'],
    #                 text=metadata['text'],
    #                 source=metadata.get('source', ''),
    #                 file_type=metadata.get('file_type', ''),
    #                 file_hash=metadata.get('file_hash', ''),
    #                 chunk_hash=metadata.get('chunk_hash', ''),
    #                 chunk_length=metadata.get('chunk_length', 0),
    #                 meta_data=metadata  # Use the new field name
    #             )
    #             session.merge(chunk)  # Use merge to handle updates
            
    #         session.commit()
    #         logger.info(f"Stored {len(metadata_list)} chunks in database")
            
    #     except Exception as e:
    #         session.rollback()
    #         logger.error(f"Error storing metadata in database: {str(e)}")
    #         raise
    #     finally:
    #         session.close()
    
    def search(self, query_vector: np.ndarray, top_k: int = 5, 
               strategy: RetrievalStrategy = RetrievalStrategy.DENSE) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar vectors"""
        if strategy == RetrievalStrategy.DENSE:
            return self._dense_search(query_vector, top_k)
        elif strategy == RetrievalStrategy.SPARSE:
            return self._sparse_search(query_vector, top_k)
        else:
            # Hybrid approach
            dense_results = self._dense_search(query_vector, top_k)
            sparse_results = self._sparse_search(query_vector, top_k)
            return self._combine_results(dense_results, sparse_results, top_k)
    
    def _dense_search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Dense vector similarity search"""
        if self.vector_store_type == "chroma":
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                include=["metadatas", "distances", "documents"]
            )
            
            return [
                # (1 - distance, metadata[0])  # Convert distance to similarity
                (1 - distance, metadata)
                for distance, metadata in zip(results["distances"][0], results["metadatas"][0])
            ]
        
        elif self.vector_store_type == "pinecone":
            results = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            return [
                (match.score, match.metadata)
                for match in results.matches
            ]
        
        elif self.vector_store_type == "weaviate":
            results = self.client.query\
                .get(self.collection_name, ["text", "source", "file_type"])\
                .with_near_vector({"vector": query_vector.tolist()})\
                .with_limit(top_k)\
                .do()
            
            return [
                (item["certainty"], {
                    "text": item["text"],
                    "source": item["source"],
                    "file_type": item["file_type"]
                })
                for item in results["data"]["Get"][self.collection_name]
            ]
    
    def _sparse_search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Sparse keyword-based search (simulated)"""
        # In production, this would use Elasticsearch or similar
        # For now, we'll simulate by getting all chunks and doing text search
        
        session = get_db_session()
        try:
            # Get all chunks (this is inefficient, use proper search in production)
            chunks = session.query(DocumentChunk).all()
            
            # Simple keyword matching (would use BM25 in production)
            results = []
            for chunk in chunks:
                # Simple scoring based on keyword matches
                score = self._calculate_keyword_score(chunk.text, query_vector)
                if score > settings.SIMILARITY_THRESHOLD:
                    results.append((score, {
                        "text": chunk.text,
                        "source": chunk.source,
                        "file_type": chunk.file_type,
                        "chunk_id": chunk.chunk_id
                    }))
            
            # Sort and return top results
            results.sort(key=lambda x: x[0], reverse=True)
            return results[:top_k]
        finally:
            session.close()
    
    def _calculate_keyword_score(self, text: str, query_vector: np.ndarray) -> float:
        """Calculate keyword matching score (simulated)"""
        # In production, this would use proper keyword scoring
        return np.random.random()  # Placeholder
    
    def _combine_results(self, dense_results: List[Tuple[float, Dict[str, Any]]], 
                        sparse_results: List[Tuple[float, Dict[str, Any]]], 
                        top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """Combine results from dense and sparse retrieval using RRF"""
        # Reciprocal Rank Fusion
        all_results = {}
        
        # Add dense results
        for rank, (score, metadata) in enumerate(dense_results):
            chunk_id = metadata.get('chunk_id')
            if chunk_id not in all_results:
                all_results[chunk_id] = {
                    'metadata': metadata,
                    'dense_rank': rank + 1,
                    'sparse_rank': None
                }
            else:
                all_results[chunk_id]['dense_rank'] = rank + 1
        
        # Add sparse results
        for rank, (score, metadata) in enumerate(sparse_results):
            chunk_id = metadata.get('chunk_id')
            if chunk_id not in all_results:
                all_results[chunk_id] = {
                    'metadata': metadata,
                    'dense_rank': None,
                    'sparse_rank': rank + 1
                }
            else:
                all_results[chunk_id]['sparse_rank'] = rank + 1
        
        # Calculate RRF scores
        rrf_results = []
        for chunk_id, data in all_results.items():
            rrf_score = 0
            if data['dense_rank'] is not None:
                rrf_score += 1 / (60 + data['dense_rank'])
            if data['sparse_rank'] is not None:
                rrf_score += 1 / (60 + data['sparse_rank'])
            
            rrf_results.append((rrf_score, data['metadata']))
        
        # Sort by RRF score and return top results
        rrf_results.sort(key=lambda x: x[0], reverse=True)
        return rrf_results[:top_k]