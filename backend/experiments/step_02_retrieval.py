import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pinecone import Pinecone

from backend.src.utils.common import read_yaml
from backend.src.constants import CONFIG_FILEPATH
from backend.src.utils.exception import LegalRAGException
from backend.src.utils.logger import logger

# Load environment variables from a .env file
load_dotenv()


# --- 1. Data Class for Configuration ---
@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for the Pinecone vector store and retrieval parameters."""
    pinecone_index_name: str
    pinecone_api_key: str
    retrieval_top_k: int
    region: str


# --- 2. Configuration Manager ---
class ConfigManager:
    """Manages loading and validation of configuration from a YAML file."""

    def __init__(self, config_path: Path = CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_path)
            # Ensure required sections exist in the config file
            if 'vector_store' not in self.config or 'retrieval' not in self.config:
                raise ValueError("Configuration error: 'vector_store' or 'retrieval' section is missing.")
            logger.info("Retrieval configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigManager: {e}")
            raise LegalRAGException(e)

    def get_retrieval_config(self) -> RetrievalConfig:
        """Extracts and validates retrieval configuration from YAML and environment."""
        try:
            vector_store_config = self.config['vector_store']
            retrieval_config = self.config['retrieval']
            api_key = os.environ.get('PINECONE_API_KEY')

            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables.")

            return RetrievalConfig(
                pinecone_index_name=vector_store_config.index_name,
                pinecone_api_key=api_key,
                retrieval_top_k=retrieval_config.top_k_candidates,
                region=vector_store_config.region
            )
        except (KeyError, AttributeError) as e:
            logger.error(f"Configuration file is missing a required key: {e}")
            raise LegalRAGException(e)


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


# --- 4. Independent Test Function ---
async def main():
    """Tests the RetrievalService with a hardcoded dummy embedding."""
    try:
        # 1. Initialize configuration and the service
        logger.info("--- Starting Independent Retrieval Test ---")
        config_manager = ConfigManager()
        retrieval_config = config_manager.get_retrieval_config()
        retriever = RetrievalService(config=retrieval_config)

        # 2. **CRITICAL STEP FOR INDEPENDENT TESTING**
        # Create a dummy embedding vector. This simulates receiving an embedding
        # from a separate embedding service. The dimension must match your index.
        # For 'voyage-law-2', the dimension is 1024.
        DUMMY_EMBEDDING_DIM = 1024
        dummy_embedding = [0.1] * DUMMY_EMBEDDING_DIM
        namespace_to_test = "companies-act-2015-v1"

        logger.info(
            f"Testing retrieval with a dummy embedding of dim={len(dummy_embedding)} in namespace '{namespace_to_test}'.")

        # 3. Call the retrieve method with the dummy embedding
        results = await retriever.retrieve(dummy_embedding, namespace=namespace_to_test)

        # 4. Log the results
        logger.info(f"\n--- Retrieval Results ---")
        logger.info(f"Successfully retrieved {len(results)} documents.")
        for i, doc in enumerate(results, 1):
            logger.info(f"\n--- Document {i} ---")
            logger.info(f"  Section: {doc.get('section', 'N/A')}")
            logger.info(f"  Title: {doc.get('title', 'N/A')}")
            text_content = doc.get('text', 'No text available')
            logger.info(f"  Excerpt: {text_content[:150]}...")
            logger.info(f"  Source: {doc.get('source_url', 'N/A')}")
        logger.info("\n--- Test Finished ---")

    except LegalRAGException as e:
        logger.error(f"A critical test failure occurred: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)


# --- 5. Script Entry Point ---
if __name__ == "__main__":
    asyncio.run(main())