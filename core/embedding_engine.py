import logging
import numpy as np
from typing import List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from utils.logger import get_logger
from utils.cache import get_cache, set_cache
import hashlib




logger = get_logger(__name__)

class EmbeddingEngine:
    """Handles text embedding generation using various models"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        
        # Initialize model based on settings
        if self.model_name.startswith("text-embedding-"):
            self.embedding_type = "openai"
        elif self.model_name.startswith("sentence-transformers/"):
            self.embedding_type = "sentence_transformers"
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.error("sentence-transformers is required for local embedding models")
                raise
        else:
            self.embedding_type = "huggingface"
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            except ImportError:
                logger.error("transformers is required for HuggingFace models")
                raise
        
        logger.info(f"Embedding engine initialized with model: {self.model_name}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        # Check cache first
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        cached = get_cache(cache_key)
        if cached:
            return np.array(cached)
        
        if self.embedding_type == "openai":
            embedding = self._get_openai_embedding(text)
        elif self.embedding_type == "sentence_transformers":
            embedding = self._get_sentence_transformer_embedding([text])[0]
        else:
            embedding = self._get_huggingface_embedding([text])[0]
        
        # Cache the result
        set_cache(cache_key, embedding.tolist(), settings.CACHE_TTL)
        return embedding
    
    def batch_generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently"""
        if not texts:
            return np.array([])
        
        # Check cache for all texts
        cache_keys = [f"embedding:{hashlib.md5(text.encode()).hexdigest()}" for text in texts]
        cached_results = [get_cache(key) for key in cache_keys]
        
        # Find texts that need embedding
        texts_to_embed = []
        indices_to_embed = []
        embeddings = np.zeros((len(texts), self.dimension))
        
        for i, (cached, text) in enumerate(zip(cached_results, texts)):
            if cached:
                embeddings[i] = cached
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            if self.embedding_type == "openai":
                new_embeddings = self._get_openai_embedding_batch(texts_to_embed)
            elif self.embedding_type == "sentence_transformers":
                new_embeddings = self._get_sentence_transformer_embedding(texts_to_embed)
            else:
                new_embeddings = self._get_huggingface_embedding(texts_to_embed)
            
            # Store new embeddings and cache them
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = embedding
                set_cache(cache_keys[idx], embedding.tolist(), settings.CACHE_TTL)
        
        return embeddings
    
    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API"""
        import openai
        
        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    def _get_openai_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts from OpenAI API"""
        import openai
        
        try:
            response = openai.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return np.array([item.embedding for item in response.data])
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {str(e)}")
            raise
    
    def _get_sentence_transformer_embedding(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using sentence-transformers"""
        return self.model.encode(texts, batch_size=self.batch_size)
    
    def _get_huggingface_embedding(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using HuggingFace transformers"""
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling
        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        return embeddings.numpy()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for transformer models"""
        import torch
        
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)