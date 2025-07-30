
import asyncio
import os
import time
from typing import List, Dict, Any
import voyageai
from backend.src.utils.logger import logger
from backend.src.utils.exception import LegalRAGException
from backend.src.config_entity.config_params import RerankerConfig


# --- 3. Reranker Service ---
class RerankerService:
    """Service for reranking documents using Voyage AI's reranker."""
    def __init__(self, config: RerankerConfig):
        try:
            if not isinstance(config, RerankerConfig):
                raise TypeError("config must be an instance of RerankerConfig")
            self.config = config
            self.client = voyageai.Client(api_key=os.environ.get('VOYAGE_API_KEY'))
            if not self.client.api_key:
                raise ValueError("VOYAGE_API_KEY not found in environment")
            logger.info(f"RerankerService initialized with model '{self.config.model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize Voyage AI reranker: {e}")
            raise LegalRAGException(e)

    async def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks documents based on relevance to a query using Voyage AI's reranker.

        Args:
            query: A non-empty string query.
            documents: List of document dictionaries with 'text' or 'content' key.

        Returns:
            List of top-n reranked document dictionaries with 'rerank_score'.
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            if not documents:
                logger.warning("No documents provided for reranking")
                return []

            start_time = time.time()
            # Extract text, falling back to 'content' or a placeholder if missing
            doc_texts = [doc.get('text', doc.get('content', 'No text available')) for doc in documents]
            if not any(doc_texts):
                logger.warning("No valid text fields found in documents")
                return []

            # Call Voyage AI reranker
            reranking = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.rerank(
                    query=query,
                    documents=doc_texts,
                    model=self.config.model_name,
                    top_k=self.config.top_n,
                    truncation=True
                )
            )

            # Map reranking results back to original documents
            reranked_docs = []
            for result in reranking.results:
                original_doc = documents[result.index]
                original_doc['rerank_score'] = result.relevance_score
                reranked_docs.append(original_doc)

            elapsed_time = time.time() - start_time
            logger.info(f"Reranked {len(documents)} documents in {elapsed_time:.2f} seconds, kept top {len(reranked_docs)}")
            return reranked_docs
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise LegalRAGException(e)

