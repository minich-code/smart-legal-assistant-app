import asyncio
import time
from typing import List, Dict, Any

from backend.src.config_settings.config_manager import ConfigurationManager
from backend.src.components.embeddings import EmbeddingService
from backend.src.components.retriever import RetrievalService
from backend.src.components.reranker import RerankerService
from backend.src.components.llm_service import LLMService, LLMConfig
# This class was refactored, and now the pipeline must adapt to it.
from backend.src.components.response import ResponseService
from backend.src.utils.exception import LegalRAGException
from backend.src.utils.logger import logger


class LegalRAGPipeline:
    """Orchestrates the full RAG pipeline from query to final answer."""

    def __init__(self, config_manager: ConfigurationManager):
        try:
            logger.info("Initializing LegalRAGPipeline...")

            # --- Service Initialization ---
            embedding_config = config_manager.get_embedding_config()
            self.embedding_service = EmbeddingService(embedding_config)

            retrieval_config = config_manager.get_retrieval_config()
            self.retrieval_service = RetrievalService(retrieval_config)

            reranker_config = config_manager.get_reranking_config()
            self.reranker_service = RerankerService(reranker_config)

            llm_config = config_manager.get_llm_config()
            self.llm_service = LLMService(llm_config) # The pipeline holds the LLM service

            response_config = config_manager.get_response_config()
            # --- FIX 1: Initialize ResponseService the new, stateless way. ---
            # It no longer needs the llm_service during initialization.
            self.response_service = ResponseService(response_config)

            # Store the highest-priority model for easy access
            self.default_model_id = self._get_default_model_id(llm_config)

            logger.info(f"LegalRAGPipeline initialized successfully. Default model: '{self.default_model_id}'")
        except Exception as e:
            logger.error(f"Failed to initialize LegalRAGPipeline: {e}")
            raise LegalRAGException(e)

    def _get_default_model_id(self, llm_config: 'LLMConfig') -> str:
        """Finds the model with the highest priority (lowest priority number)."""
        if not llm_config.providers:
            raise ValueError("No LLM providers configured.")
        highest_priority_provider = min(llm_config.providers, key=lambda p: p.get('priority', 999))
        return highest_priority_provider['model_name']

    async def process_query(self, query: str, namespace: str, model_id: str = None) -> Dict[str, Any]:
        """
        Processes a legal query through the full RAG pipeline.
        """
        try:
            start_time = time.time()
            logger.info(f"Processing query: '{query[:50]}...' in namespace: '{namespace}'")

            # Step 1: Generate Query Embedding
            logger.info("Step 1: Generating query embedding...")
            query_embedding_list = await self.embedding_service.embed([query], input_type="query")

            # Step 2: Retrieve Documents
            logger.info("Step 2: Retrieving documents...")
            retrieved_docs = await self.retrieval_service.retrieve(
                query_embedding=query_embedding_list[0],
                namespace=namespace
            )
            if not retrieved_docs:
                logger.warning("No documents were retrieved. Aborting further processing.")
                return {"answer": "I could not find any relevant documents to answer your question.", "citations": []}

            # Step 3: Rerank Documents
            logger.info("Step 3: Reranking retrieved documents...")
            reranked_docs = await self.reranker_service.rerank(query, retrieved_docs)

            # Step 4: Generate Final Response
            final_model_id = model_id or self.default_model_id
            logger.info(f"Step 4: Generating final response with model '{final_model_id}'...")

            # --- FIX 2: Pass the llm_service instance into the generate_response method. ---
            # This is where the dependency is now needed.
            response = await self.response_service.generate_response(
                query=query,
                documents=reranked_docs,
                model_id=final_model_id,
                llm_service=self.llm_service  # Pass the pipeline's LLM service here
            )

            elapsed_time = time.time() - start_time
            logger.info(f"Query processed successfully in {elapsed_time:.2f} seconds.")
            return response

        except Exception as e:
            logger.error(f"A critical error occurred during query processing: {e}")
            raise LegalRAGException(e)


# --- Test Function (This remains correct) ---
async def main():
    """Tests the full LegalRAGPipeline with a sample legal query."""
    try:
        logger.info("--- Starting Full Pipeline Test ---")
        config_manager = ConfigurationManager()
        pipeline = LegalRAGPipeline(config_manager)

        query = "What is the statement of guarantee for company registration in Kenya?"
        namespace = "companies-act-2015-v1"

        response = await pipeline.process_query(query=query, namespace=namespace)

        logger.info("\n--- PIPELINE RESPONSE ---")
        logger.info(f"Answer: {response['answer']}")
        logger.info("\nCitations:")
        if response['citations']:
            for i, citation in enumerate(response['citations'], 1):
                logger.info(f"  {i}. Section: {citation.get('section', 'N/A')}, "
                            f"Title: '{citation.get('title', 'N/A')}', "
                            f"Source: {citation.get('source_url', 'N/A')}")
        else:
            logger.info("  No citations were provided.")
        logger.info("------------------------")

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())