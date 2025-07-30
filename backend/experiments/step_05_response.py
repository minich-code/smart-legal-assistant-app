# backend/experiments/step_05_response.py

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from importlib import import_module
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.src.utils.common import read_yaml
from backend.src.utils.logger import logger
from backend.src.utils.exception import LegalRAGException
from backend.src.constants import CONFIG_FILEPATH
# The import is still needed here to type-hint the method parameter
from backend.experiments.step_04_llms import LLMService, ConfigManager as LLMConfigManager

# Load .env file
load_dotenv()


# --- 1. Response Config (No changes needed) ---
@dataclass
class ResponseConfig:
    """Configuration for response generation and citation parameters."""
    cot_template: str


# --- 2. Config Manager (No changes needed) ---
class ConfigManager:
    """Manages loading and validation of configuration from YAML and environment variables."""

    def __init__(self, config_path: str = CONFIG_FILEPATH):
        try:
            logger.info(f"Initializing ConfigurationManager with config file: {config_path}")
            self.config_path = Path(config_path)
            self.config = read_yaml(self.config_path)
            if 'response' not in self.config:
                raise ValueError("Configuration error: 'response' section missing in config file")
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise LegalRAGException(e)

    def get_response_config(self) -> ResponseConfig:
        """Extracts and validates response configuration from YAML."""
        try:
            response_config = self.config['response']
            template_path_str = response_config.get('cot_template_path')
            if not template_path_str:
                raise ValueError("CoT template path not found in configuration.")

            template_path = template_path_str.replace('/', '.').replace('.py', '')
            module = import_module(template_path)
            cot_template = getattr(module, 'COT_PROMPT_TEMPLATE', None)
            if not cot_template:
                raise ValueError(f"COT_PROMPT_TEMPLATE not found in {template_path_str}")

            return ResponseConfig(cot_template=cot_template)
        except Exception as e:
            logger.error(f"Error loading response configuration: {e}")
            raise LegalRAGException(e)


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


# --- 4. Main Test Function (Adapted for the refactored service) ---
async def main():
    """Tests response generation with a legal query and dummy documents."""
    try:
        logger.info("--- Starting Response Service Test (Decoupled) ---")

        # 1. Setup all dependencies first
        logger.info("Loading LLM configuration and initializing LLMService")
        llm_config_manager = LLMConfigManager()
        llm_config = llm_config_manager.get_llm_config()
        llm_service = LLMService(llm_config)

        logger.info("Loading response configuration")
        response_config_manager = ConfigManager()
        response_config = response_config_manager.get_response_config()

        # --- CHANGE 3: Initialize ResponseService without the llm_service ---
        logger.info("Initializing ResponseService")
        response_service = ResponseService(response_config)

        # 2. Prepare data for the test
        if not llm_config.providers:
            raise ValueError("No LLM providers found in the configuration.")
        model_id = llm_config.providers[0]['model_name']
        logger.info(f"Using model: {model_id}")

        query = "What is the statement of guarantee for company registration in Kenya?"
        dummy_documents = [
            {"section": "15", "title": "Statement of guarantee", "text": "The application for registration of a company that is to be a company limited by guarantee shall be accompanied by a statement of guarantee.", "source_url": "http://kenyalaw.org/kl/fileadmin/pdfdownloads/Acts/CompaniesAct_No17of2015.pdf"},
            {"section": "7", "title": "Companies limited by guarantee", "text": "A company is a company limited by guarantee if the liability of its members is limited by the company's articles to such amount as the members may respectively undertake to contribute to the assets of the company in the event of its being wound up.", "source_url": "http://kenyalaw.org/kl/fileadmin/pdfdownloads/Acts/CompaniesAct_No17of2015.pdf"}
        ]

        logger.info(f"Generating response for query: '{query}'")
        # --- CHANGE 4: Pass the llm_service instance into the generate_response method ---
        result = await response_service.generate_response(
            query=query,
            documents=dummy_documents,
            model_id=model_id,
            llm_service=llm_service  # Dependency is passed here
        )

        # 3. Log results
        logger.info("--- TEST SUCCEEDED ---")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Citations (found {len(result['citations'])}):")
        for i, citation in enumerate(result['citations'], 1):
            logger.info(f"{i}. Section: {citation['section']}, Title: {citation['title']}")

    except Exception as e:
        logger.error(f"Error in response test: {e}", exc_info=True)


# --- 5. Entry Point ---
if __name__ == "__main__":
    asyncio.run(main())