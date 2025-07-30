
import time
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.src.utils.logger import logger
from backend.src.utils.exception import LegalRAGException
from backend.src.config_entity.config_params import ResponseConfig
from backend.src.components.llm_service import LLMService

# --- 3. Response Service (Refactored) ---
class ResponseService:
    """
    Service for generating final responses.
    This service is now stateless regarding the LLM provider.
    """

    # --- CHANGE 1: __init__ no longer requires llm_service ---
    def __init__(self, response_config: ResponseConfig):
        """Initializes the service with just the response configuration."""
        try:
            if not isinstance(response_config, ResponseConfig):
                raise TypeError("response_config must be an instance of ResponseConfig")
            self.config = response_config
            logger.info("ResponseService initialized successfully (stateless).")
        except Exception as e:
            logger.error(f"Failed to initialize ResponseService: {e}")
            raise LegalRAGException(e)

    # --- CHANGE 2: generate_response now accepts the llm_service as a parameter ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_response(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        model_id: str,
        llm_service: LLMService  # The LLM dependency is passed here
    ) -> Dict[str, Any]:
        """
        Generates a final response with citations using the CoT template.
        """
        try:
            # --- Input validation
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            if not isinstance(llm_service, LLMService):
                raise TypeError("A valid LLMService instance must be provided.")
            if not documents:
                logger.warning("No documents provided for response generation")
                documents = []

            start_time = time.time()

            # --- Prepare documents and citations
            formatted_docs = []
            citations = []
            for i, doc in enumerate(documents, 1):
                text = doc.get('text', doc.get('content', 'No text available'))
                section = doc.get('section', 'N/A')
                source_url = doc.get('source_url', 'N/A')
                formatted_docs.append(
                    f"Document {i}: {text[:700]}...\nSection: {section}\nSource: {source_url}"
                )
                citations.append({
                    "section": section, "title": doc.get('title', 'N/A'), "source_url": source_url
                })
            documents_str = "\n\n".join(formatted_docs) or "No documents available."

            # --- Prepare prompts
            system_prompt = "You are a legal assistant specializing in Kenyan corporate law. Provide a clear, accurate answer based only on the text provided in the documents."
            user_prompt = self.config.cot_template.format(query=query, documents=documents_str)

            # --- Call the provided LLMService instance
            llm_answer_text = await llm_service.generate(
                model_id=model_id, system_prompt=system_prompt, user_prompt=user_prompt
            )

            if not llm_answer_text or not isinstance(llm_answer_text, str):
                raise ValueError("LLMService returned an invalid or empty response string.")

            elapsed_time = time.time() - start_time
            logger.info(f"Generated response in {elapsed_time:.2f} seconds.")

            # --- Construct the final dictionary
            return {"answer": llm_answer_text, "citations": citations}

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            raise LegalRAGException(e)

