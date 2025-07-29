
import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import voyageai
from backend.src.utils.logger import logger
from backend.src.utils.exception import LegalRAGException
from backend.src.utils.common import read_yaml
from backend.src.constants import CONFIG_FILEPATH

# Load .env file
load_dotenv()

# --- 1. Reranker Config ---
@dataclass
class RerankerConfig:
    """Configuration for reranking model and parameters."""
    model_name: str
    top_n: int


# --- 2. Config Manager ---
class ConfigManager:
    """Manages loading and validation of configuration from YAML and environment variables."""
    def __init__(self, config_path: str = CONFIG_FILEPATH):
        try:
            logger.info(f"Initializing ConfigurationManager with config file: {config_path}")
            self.config_path = Path(config_path)
            self.config = read_yaml(self.config_path)
            if not self.config or 'reranker' not in self.config:
                raise ValueError("Configuration error: 'reranker' section missing in config file")
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise LegalRAGException(e)

    def get_reranking_config(self) -> RerankerConfig:
        """Extracts and validates reranking configuration from YAML."""
        try:
            reranking_config = self.config['reranker']
            if not hasattr(reranking_config, 'model_name'):
                raise ValueError("Reranking model name not found in configuration.")
            if not hasattr(reranking_config, 'top_n'):
                raise ValueError("Reranked top_n candidates not found in configuration.")
            return RerankerConfig(
                model_name="rerank-2.5-lite",  # Override YAML to use Voyage reranker
                top_n=reranking_config.top_n
            )
        except Exception as e:
            logger.error(f"Error loading reranking configuration: {e}")
            raise LegalRAGException(e)


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


# --- 4. Main Test Function ---
async def main():
    """Tests reranking with a sample query and dummy legal documents."""
    try:
        logger.info("Loading reranking configuration")
        config_manager = ConfigManager()
        config = config_manager.get_reranking_config()
        logger.info("Reranking configuration loaded")

        logger.info("Initializing RerankerService")
        reranker = RerankerService(config)

        query = "What is the statement of guarantee for company registration?"
        dummy_documents = [
            {
                "section": "15",
                "title": "Statement of guarantee",
                "text": "15: Statement of guarantee 15 (1): The applicant for registration of a company to be limited by guarantee shall ensure...",
                "source_url": "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2024-12-27"
            },
            {
                "section": "86",
                "title": "Requirements for registration of unlimited company as private limited company",
                "text": "86: Requirements for registration of unlimited company as private limited company...",
                "source_url": "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2024-12-27"
            },
            {
                "section": "7",
                "title": "Companies limited by guarantee",
                "text": "7: Companies limited by guarantee...",
                "source_url": "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2024-12-27"
            },
            {
                "section": "13",
                "title": "Registration documents",
                "text": "13: Registration documents...",
                "source_url": "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2024-12-27"
            },
            {
                "section": "14",
                "title": "Statement of capital and initial shareholdings",
                "text": "14: Statement of capital and initial shareholdings...",
                "source_url": "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2024-12-27"
            }
        ]

        logger.info(f"Reranking {len(dummy_documents)} documents for query: {query}")
        reranked_docs = await reranker.rerank(query, dummy_documents)

        if not reranked_docs:
            logger.warning("No reranked documents returned")
            return

        logger.info("Reranked Documents:")
        for i, doc in enumerate(reranked_docs, 1):
            logger.info(f"Document {i}:")
            logger.info(f"Section: {doc.get('section', 'N/A')}")
            logger.info(f"Title: {doc.get('title', 'N/A')}")
            logger.info(f"Text: {doc.get('text', doc.get('content', 'N/A'))[:150]}...")
            logger.info(f"Source: {doc.get('source_url', 'N/A')}")
            logger.info(f"Rerank Score: {doc.get('rerank_score', 'N/A'):.4f}")
    except Exception as e:
        logger.error(f"Error in reranking process: {e}")
        raise LegalRAGException(e)


# --- 5. Entry Point ---
if __name__ == "__main__":
    asyncio.run(main())