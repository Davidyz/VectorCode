import os
import sys
import json
import requests
from typing import List, Dict, Any

from .base import RerankerBase, register_reranker


@register_reranker("llamacpp")
class LlamaCppReranker(RerankerBase):
    """A reranker that uses a Llama.cpp server for reranking.
    
    This reranker makes API calls to a llama.cpp reranking endpoint to
    reorder documents based on relevance to the query.
    """
    
    def __init__(self, model_name=None, **kwargs):
        """Initialize the LlamaCppReranker.
        
        Args:
            model_name: The model name or API URL for the reranker.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        # Handle both positional and keyword model_name to avoid TypeError
        self.api_url = model_name or kwargs.get('model_name', 
                      os.environ.get('VECTORCODE_RERANKING_API_URL', 
                                    'http://localhost:8085/v1/reranking'))
        self.timeout = kwargs.get('timeout', 30)  # Default timeout of 30 seconds
        print(f'LlamaCppReranker initialized with API URL: {self.api_url}', file=sys.stderr)
    
    def rerank(self, results: Dict[str, Any]) -> List[str]:
        """Rerank the query results using the llama.cpp reranking API.
        
        Args:
            results: The query results from ChromaDB.
            
        Returns:
            A list of document IDs reordered based on relevance scores.
        """
        if not results.get("ids") or not results.get("documents"):
            print("Warning: Empty results or missing fields", file=sys.stderr)
            return []
            
        query_idx = 0  # Use the first query result set
        ids = results["ids"][query_idx]
        documents = results["documents"][query_idx]
        
        if not ids or not documents:
            print("Warning: No documents to rerank", file=sys.stderr)
            return []
            
        print(f'LlamaCppReranker.rerank called with {len(ids)} results', file=sys.stderr)
        
        # Use the query from configuration if available
        query = ""
        if "query_text" in self.kwargs:
            query = self.kwargs["query_text"]
        elif hasattr(self, "configs") and hasattr(self.configs, "query_message"):
            query = self.configs.query_message
        
        try:
            # Prepare the payload for the reranking API
            payload = {
                "query": query,
                "documents": documents
            }
            
            print(f"Sending reranking request to {self.api_url}", file=sys.stderr)
            
            # Make the API call
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response
                response_data = response.json()
                
                # Get the results with relevance scores
                if "results" in response_data:
                    # Create a list of (index, score) tuples
                    scores = [(r["index"], r["relevance_score"]) for r in response_data["results"]]
                    
                    # Sort by score (higher is better)
                    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
                    
                    # Get the original IDs in the new order
                    reordered_ids = [ids[idx] for idx, _ in sorted_scores]
                    
                    print(f"Reranked results based on relevance scores", file=sys.stderr)
                    return reordered_ids
            
            # If we get here, something went wrong with the API call
            print(f"Reranking API returned status {response.status_code}: {response.text}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error during reranking API call: {str(e)}", file=sys.stderr)
        
        # Fall back to original order if anything fails
        print("Falling back to original document order", file=sys.stderr)
        return ids
