import asyncio
import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from backend.src.utils.exception import LegalRAGException
from backend.src.utils.logger import logger
from backend.src.config_entity.config_params import RetrievalConfig

# --- 3. Retrieval Service ---
class RetrievalService:
    """
    Service for dense retrieval from a Pinecone vector index.
    This component is independent and does not handle embedding generation.
    """

    def __init__(self, config: RetrievalConfig):
        if not isinstance(config, RetrievalConfig):
            raise TypeError("config must be an instance of RetrievalConfig")
        self.config = config
        self.index = self._initialize_pinecone()
        logger.info("RetrievalService initialized successfully.")

    def _initialize_pinecone(self) -> Pinecone.Index:
        """Initializes and validates the connection to the Pinecone index."""
        try:
            logger.info(f"Connecting to Pinecone in region '{self.config.region}'...")
            pc = Pinecone(api_key=self.config.pinecone_api_key)

            index_name = self.config.pinecone_index_name
            if index_name not in pc.list_indexes().names():
                raise ValueError(f"Index '{index_name}' not found in your Pinecone project.")

            index = pc.Index(index_name)
            logger.info(f"Successfully connected to Pinecone index: '{index_name}'")
            return index
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise LegalRAGException(e)

    async def retrieve(self, query_embedding: List[float], namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k documents from Pinecone using a pre-computed embedding vector.

        Args:
            query_embedding: A list of floats representing the query vector.
            namespace: The Pinecone namespace to search within (optional).

        Returns:
            A list of metadata dictionaries for the retrieved documents.
        """
        if not query_embedding or not isinstance(query_embedding, list):
            raise ValueError("query_embedding must be a non-empty list.")

        try:
            start_time = time.time()
            # Run the synchronous Pinecone query in an async-friendly executor
            loop = asyncio.get_event_loop()
            query_response = await loop.run_in_executor(
                None,  # Use the default thread pool executor
                lambda: self.index.query(
                    vector=query_embedding,
                    top_k=self.config.retrieval_top_k,
                    include_metadata=True,
                    namespace=namespace
                )
            )
            elapsed_time = time.time() - start_time

            candidates = [match.metadata for match in query_response.matches if match.metadata]

            logger.info(
                f"Retrieved {len(candidates)} documents from namespace '{namespace or 'default'}' in {elapsed_time:.2f}s.")
            if not candidates:
                logger.warning("No candidates were found for the given query embedding.")

            return candidates
        except Exception as e:
            logger.error(f"An error occurred during the retrieval operation: {e}")
            raise LegalRAGException(e)

