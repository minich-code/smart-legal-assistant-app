import os
import asyncio
import time
from dataclasses import dataclass
from typing import List
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from pathlib import Path
from box import ConfigBox
from backend.src.utils.common import read_yaml
from backend.src.utils.logger import logger
from backend.src.utils.exception import LegalRAGException
from backend.src.constants import CONFIG_FILEPATH

load_dotenv()


# --- 1. Configuration Dataclass ---
@dataclass
class EmbeddingConfig:
    """Configuration for embedding model and API credentials."""
    api_key: str
    embedding_model_name: str


# --- 2. Configuration Manager Class ---
class ConfigurationManager:
    """Manages loading and validation of configuration from YAML and environment variables."""
    def __init__(self, config_path: str = CONFIG_FILEPATH):
        try:
            # Simplified logging for demo
            self.config_path = Path(config_path)
            self.config = read_yaml(self.config_path)
            if 'embedding_models' not in self.config:
                raise ValueError("Configuration error: 'embedding_models' section missing")
        except Exception as e:
            logger.error(f"Config init error: {e}")
            raise LegalRAGException(e)

    def get_embedding_config(self) -> EmbeddingConfig:
        """Extracts and validates embedding configuration."""
        try:
            embedding_config = self.config['embedding_models']
            api_key = os.environ.get('VOYAGE_API_KEY')
            if not api_key:
                raise ValueError("VOYAGE_API_KEY not found in environment")
            if not hasattr(embedding_config, 'embedding_model_name'):
                raise ValueError("Embedding model name not found")
            return EmbeddingConfig(
                api_key=api_key,
                embedding_model_name=embedding_config.embedding_model_name
            )
        except Exception as e:
            logger.error(f"Embedding config error: {e}")
            raise LegalRAGException(e)


# --- 3. The Embedding Service Class ---
class EmbeddingService:
    """Service for generating embeddings using Voyage AI."""
    def __init__(self, config: EmbeddingConfig):
        try:
            if not isinstance(config, EmbeddingConfig):
                raise ValueError("Invalid configuration: must be EmbeddingConfig")
            self.config = config
            self.client = voyageai.Client(api_key=self.config.api_key)
            # Reduced logging verbosity
            logger.info(f"Initialized with model '{self.config.embedding_model_name}'")
        except Exception as e:
            logger.error(f"Failed to init Voyage AI client: {e}")
            raise LegalRAGException(e)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))  # Tightened retry for demo
    async def embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        """
        Embeds a list of texts using Voyage AI.

        Args:
            texts: A list of strings to embed (non-empty)
            input_type: Either "query" or "document"

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            # Streamlined input validation
            if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
                raise ValueError("Input must be a non-empty list of non-empty strings")
            if input_type not in ["query", "document"]:
                raise ValueError("input_type must be 'query' or 'document'")

            start_time = time.time()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.embed(
                    texts=texts,
                    model=self.config.embedding_model_name,
                    input_type=input_type,
                    truncation=True
                )
            )
            elapsed_time = time.time() - start_time
            # Log only for debugging in demo
            logger.debug(f"Embedded {len(texts)} texts in {elapsed_time:.2f}s")
            return result.embeddings
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise LegalRAGException(e)


# --- 4. Main Function for Testing ---
async def main():
    """Tests embedding functionality with sample legal query and documents."""
    try:
        config_manager = ConfigurationManager()
        embedding_config = config_manager.get_embedding_config()
        embedder = EmbeddingService(embedding_config)

        query = "What are the responsibilities of shareholders?"
        query_embedding = await embedder.embed([query], input_type="query")
        logger.info(f"Query embedded. Vector length: {len(query_embedding[0])}")

        docs = ["This is the first legal document.", "This is the second legal document."]
        doc_embeddings = await embedder.embed(docs, input_type="document")
        logger.info(f"Embedded {len(doc_embeddings)} documents")
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        raise LegalRAGException(e)


# --- 5. Run the Test ---
if __name__ == "__main__":
    asyncio.run(main())