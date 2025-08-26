import logging
import re
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from config.constants import RetrievalStrategy
from utils.logger import get_logger
from utils.cache import get_cache, set_cache
import hashlib

logger = get_logger(__name__)

class QueryProcessor:
    """Handles query transformation and routing"""
    
    def __init__(self):
        self.query_expansion_enabled = settings.QUERY_EXPANSION_ENABLED
        logger.info("Query processor initialized")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def transform_query(self, query: str) -> List[str]:
        """Transform the query to handle ambiguity and complexity"""
        # Check cache first
        cache_key = f"query_transform:{hashlib.md5(query.encode()).hexdigest()}"
        cached = get_cache(cache_key)
        if cached:
            return cached
        
        transformations = []
        
        # 1. Original query
        transformations.append(query)
        
        # 2. Query decomposition for complex queries
        if len(query.split()) > 8:  # If query is long, decompose
            decomposed = self._decompose_query(query)
            transformations.extend(decomposed)
        
        # 3. Query expansion with synonyms
        if self.query_expansion_enabled:
            expanded = self._expand_query(query)
            transformations.extend(expanded)
        
        # 4. Spelling correction
        corrected = self._correct_spelling(query)
        if corrected and corrected != query:
            transformations.append(corrected)
        
        # Remove duplicates and empty strings
        transformations = list(set([t for t in transformations if t.strip()]))
        
        # Cache the results
        set_cache(cache_key, transformations, settings.CACHE_TTL)
        
        logger.info(f"Query transformed into {len(transformations)} variations: {transformations}")
        return transformations
    
    def route_query(self, query: str) -> RetrievalStrategy:
        """Determine whether to use dense or sparse retrieval"""
        # Check cache first
        cache_key = f"query_route:{hashlib.md5(query.encode()).hexdigest()}"
        cached = get_cache(cache_key)
        if cached:
            return RetrievalStrategy(cached)
        
        query_lower = query.lower()
        
        # Rule-based routing - in production would use ML model
        sparse_indicators = [
            len(query.split()) < 4,  # Very short queries
            any(word in query_lower for word in ["definition", "define", "what is", "who is"]),
            bool(re.search(r'\d{4}', query)),  # Contains year
            any(char in query for char in ['-', '+', '"']),  # Contains specific characters
            any(term in query_lower for term in ["code", "law", "regulation", "statute"]),  # Legal terms
        ]
        
        dense_indicators = [
            len(query.split()) > 6,  # Longer, more descriptive queries
            any(word in query_lower for word in ["explain", "describe", "how to", "why"]),
            "compare" in query_lower,
            "difference between" in query_lower,
        ]
        
        # Count indicators
        sparse_count = sum(sparse_indicators)
        dense_count = sum(dense_indicators)
        
        # Determine strategy
        if sparse_count > dense_count:
            strategy = RetrievalStrategy.SPARSE
        elif dense_count > sparse_count:
            strategy = RetrievalStrategy.DENSE
        else:
            strategy = RetrievalStrategy.HYBRID
        
        # Cache the result
        set_cache(cache_key, strategy.value, settings.CACHE_TTL)
        
        logger.info(f"Query routed to {strategy.value} retrieval: {query}")
        return strategy
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into simpler sub-queries"""
        # In production, this would use an LLM
        # For now, use simple rule-based decomposition
        
        decomposed = []
        
        # Split by conjunctions
        for separator in [" and ", " or ", " but ", " however "]:
            if separator in query.lower():
                parts = re.split(separator, query, flags=re.IGNORECASE)
                decomposed.extend([part.strip() for part in parts if part.strip()])
        
        # Handle comparison queries
        if " vs " in query.lower() or " versus " in query.lower():
            pattern = r"\s+(vs|versus)\s+"
            parts = re.split(pattern, query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                decomposed.append(f"What is {parts[0].strip()}?")
                decomposed.append(f"What is {parts[-1].strip()}?")
                decomposed.append(f"Compare {parts[0].strip()} and {parts[-1].strip()}")
        
        return decomposed
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        # In production, use WordNet or similar
        synonym_map = {
            "legal": ["law", "juridical", "judicial", "legislative"],
            "medical": ["health", "clinical", "healthcare", "medical"],
            "financial": ["economic", "fiscal", "monetary", "banking"],
            "document": ["file", "record", "paper", "report"],
            "analysis": ["examination", "study", "review", "assessment"],
        }
        
        expanded = []
        query_lower = query.lower()
        
        for term, synonyms in synonym_map.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    if expanded_query != query_lower:  # Only add if changed
                        expanded.append(expanded_query)
        
        return expanded
    
    def _correct_spelling(self, query: str) -> Optional[str]:
        """Correct spelling errors in query"""
        # In production, use a spell checker library
        # For now, implement simple common corrections
        common_errors = {
            "recieve": "receive",
            "seperate": "separate",
            "definately": "definitely",
            "goverment": "government",
            "occured": "occurred",
            "accomodate": "accommodate",
            "arguement": "argument",
            "comittee": "committee",
            "embarass": "embarrass",
            "harassment": "harassment",
        }
        
        corrected = query
        for error, correction in common_errors.items():
            corrected = re.sub(rf"\b{error}\b", correction, corrected, flags=re.IGNORECASE)
        
        return corrected if corrected != query else None