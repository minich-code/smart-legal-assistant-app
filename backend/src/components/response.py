import time
from typing import List, Dict, Any

from backend.src.config_entity.config_params import ResponseConfig
from backend.src.components.llm_service import LLMService
from backend.src.utils.exception import LegalRAGException
from backend.src.utils.logger import logger


class ResponseService:
    """
    Service for orchestrating the final response generation.
    It formats the prompt and uses the LLMService to get the generated text.
    """

    def __init__(self, response_config: ResponseConfig, llm_service: LLMService):
        if not isinstance(response_config, ResponseConfig):
            raise TypeError("response_config must be an instance of ResponseConfig")
        if not isinstance(llm_service, LLMService):
            raise TypeError("llm_service must be an instance of LLMService")

        self.config = response_config
        self.llm_service = llm_service
        logger.info("ResponseService initialized successfully.")

    async def generate_response(self, query: str, documents: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
        """
        Generates a final, citation-backed response.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")
        if not documents:
            logger.warning("No documents provided for response generation.")
            documents = []

        try:
            # --- Step 1: Prepare documents and citations ---
            # This logic is correct. It uses self.config.max_citations to limit
            # the number of documents shown to the LLM.
            formatted_docs = []
            citations = []
            for i, doc in enumerate(documents[:self.config.max_citations], 1):
                text = doc.get('text', 'No text available')
                section = doc.get('section', 'N/A')
                title = doc.get('title', 'N/A')
                source_url = doc.get('source_url', 'N/A')

                formatted_docs.append(
                    f"Document {i} (Section: {section}, Source: {source_url}):\n```{text[:700]}...```"
                )
                citations.append({"section": section, "title": title, "source_url": source_url})

            documents_str = "\n\n".join(formatted_docs) if formatted_docs else "No documents were provided."

            # --- Step 2: Format the final prompt for the LLM ---
            system_prompt = "You are a legal assistant specializing in Kenyan corporate law. Provide a clear and accurate answer based *only* on the provided documents."

            # --- THIS CODE IS NOW CORRECT ---
            # It no longer passes 'max_citations', matching the fixed template.
            user_prompt = self.config.cot_template.format(
                query=query,
                documents=documents_str
            )

            # --- Step 3: Call the independent LLMService ---
            generated_text = await self.llm_service.generate(
                model_id=model_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # --- Step 4: Package the final response ---
            return {
                "answer": generated_text,
                "citations": citations
            }
        except Exception as e:
            logger.error(f"Error generating final response: {e}", exc_info=True)
            raise LegalRAGException(e)