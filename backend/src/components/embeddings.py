
import asyncio
import time
from typing import List
import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

from backend.src.utils.logger import logger
from backend.src.utils.exception import LegalRAGException

from backend.src.config_entity.config_params import EmbeddingConfig

load_dotenv()

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


