import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class Reranker:
    """Reranks retrieved results using cross-encoder models"""
    
    def __init__(self):
        self.model_name = settings.RERANKER_MODEL
        self.top_k = settings.TOP_K_RERANK
        
        # Initialize model
        if self.model_name.startswith("cross-encoder/"):
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(self.model_name)
            except ImportError:
                logger.error("sentence-transformers is required for cross-encoder models")
                raise
        else:
            # Custom model implementation would go here
            self.model = None
        
        logger.info(f"Reranker initialized with model: {self.model_name}")
    
    def rerank(self, query: str, results: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
        """Rerank results based on relevance to query"""
        if not results or not self.model:
            return results[:self.top_k] if results else []
        
        print("results  ###########",results)
        # Prepare query-document pairs for cross-encoder
        # pairs = [(query, result[1]['text']) for result in results]
        # pairs = [(query, result[1]['content']) for result in results]
        pairs = [(query, result[1].get('text') or result[1].get('content') or '') for result in results]

        
        # Get scores from cross-encoder
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"Error in cross-encoder prediction: {str(e)}")
            return results[:self.top_k]
        
        # Combine original scores with cross-encoder scores
        reranked_results = []
        for i, (original_score, metadata) in enumerate(results):
            if i < len(scores):
                # Combine scores (weighted average)
                combined_score = 0.7 * scores[i] + 0.3 * original_score
                reranked_results.append((combined_score, metadata))
            else:
                reranked_results.append((original_score, metadata))
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"Reranked {len(reranked_results)} results, returning top {self.top_k}")
        return reranked_results[:self.top_k]
    
    def batch_rerank(self, queries: List[str], results_list: List[List[Tuple[float, Dict[str, Any]]]]) -> List[List[Tuple[float, Dict[str, Any]]]]:
        """Batch rerank multiple queries for efficiency"""
        if not self.model:
            return [results[:self.top_k] for results in results_list]
        
        all_pairs = []
        pair_indices = []
        
        # Prepare all query-document pairs
        for query_idx, (query, results) in enumerate(zip(queries, results_list)):
            for result_idx, (score, metadata) in enumerate(results):
                all_pairs.append((query, metadata['text']))
                pair_indices.append((query_idx, result_idx))
        
        # Get scores in batch
        try:
            all_scores = self.model.predict(all_pairs)
        except Exception as e:
            logger.error(f"Error in batch cross-encoder prediction: {str(e)}")
            return [results[:self.top_k] for results in results_list]
        
        # Reconstruct results
        reranked_results_list = [[] for _ in range(len(queries))]
        
        for (query_idx, result_idx), score in zip(pair_indices, all_scores):
            original_score, metadata = results_list[query_idx][result_idx]
            combined_score = 0.7 * score + 0.3 * original_score
            reranked_results_list[query_idx].append((combined_score, metadata))
        
        # Sort each query's results and return top k
        final_results = []
        for results in reranked_results_list:
            results.sort(key=lambda x: x[0], reverse=True)
            final_results.append(results[:self.top_k])
        
        logger.info(f"Batch reranked {len(queries)} queries")
        return final_results