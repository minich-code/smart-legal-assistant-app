import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from importlib import import_module
from backend.src.utils.common import read_yaml
from backend.src.utils.logger import logger
from backend.src.utils.exception import LegalRAGException
from backend.src.constants import CONFIG_FILEPATH
from backend.experiments.step_04_llms import LLMService
from tenacity import retry, stop_after_attempt, wait_exponential

# Load .env file
load_dotenv()


# --- 1. Response Config ---
@dataclass
class ResponseConfig:
    """Configuration for response generation and citation parameters."""
    max_citations: int
    cot_template: str


# --- 2. Config Manager ---
class ConfigManager:
    """Manages loading and validation of configuration from YAML and environment variables."""
    def __init__(self, config_path: str = CONFIG_FILEPATH):
        try:
            logger.info(f"Initializing ConfigurationManager with config file: {config_path}")
            self.config_path = Path(config_path)
            self.config = read_yaml(self.config_path)
            if not self.config or 'response' not in self.config:
                raise ValueError("Configuration error: 'response' section missing in config file")
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing ConfigurationManager: {e}")
            raise LegalRAGException(e)

    def get_response_config(self) -> ResponseConfig:
        """Extracts and validates response configuration from YAML."""
        try:
            response_config = self.config['response']
            if not hasattr(response_config, 'max_citations'):
                raise ValueError("Max citations not found in configuration.")
            if not hasattr(response_config, 'cot_template_path'):
                raise ValueError("CoT template path not found in configuration.")

            # Load CoT template
            template_path = response_config.cot_template_path.replace('/', '.').replace('.py', '')
            module = import_module(template_path)
            cot_template = getattr(module, 'COT_PROMPT_TEMPLATE', None)
            if not cot_template:
                raise ValueError(f"Invalid CoT template at {response_config.cot_template_path}")

            return ResponseConfig(
                max_citations=response_config.max_citations,
                cot_template=cot_template
            )
        except Exception as e:
            logger.error(f"Error loading response configuration: {e}")
            raise LegalRAGException(e)


# --- 3. Response Service ---
class ResponseService:
    """Service for generating final responses with citations using a Chain of Thought template."""
    def __init__(self, response_config: ResponseConfig, llm_service: LLMService):
        try:
            if not isinstance(response_config, ResponseConfig):
                raise TypeError("response_config must be an instance of ResponseConfig")
            if not isinstance(llm_service, LLMService):
                raise TypeError("llm_service must be an instance of LLMService")
            self.config = response_config
            self.llm_service = llm_service
            logger.info("ResponseService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ResponseService: {e}")
            raise LegalRAGException(e)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_response(self, query: str, documents: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
        """
        Generates a final response with citations using the CoT template.

        Args:
            query: The user query string.
            documents: List of document dictionaries with metadata (section, title, text, source_url).
            model_id: The LLM model to use (e.g., 'llama-3.3-70b-versatile').

        Returns:
            Dictionary with 'answer' and 'citations' (list of citation dictionaries).
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            if not documents:
                logger.warning("No documents provided for response generation")
                documents = []

            start_time = time.time()
            # Format documents for the CoT prompt
            formatted_docs = []
            citations = []
            for i, doc in enumerate(documents[:self.config.max_citations], 1):
                text = doc.get('text', doc.get('content', 'No text available'))
                section = doc.get('section', 'N/A')
                source_url = doc.get('source_url', 'N/A')
                formatted_docs.append(
                    f"Document {i}: {text[:500]}...\nSection: {section}\nSource: {source_url}"
                )
                citations.append({
                    "section": section,
                    "title": doc.get('title', 'N/A'),
                    "source_url": source_url
                })
            documents_str = "\n\n".join(formatted_docs) or "No documents available."

            # Format the CoT prompt
            system_prompt = "You are a legal assistant specializing in Kenyan corporate law."
            user_prompt = self.config.cot_template.format(
                query=query,
                documents=documents_str,
                max_citations=self.config.max_citations
            )
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Call LLMService
            result = await self.llm_service.generate(
                model_id=model_id,
                query=full_prompt,
                documents=documents
            )

            # Extract response and citations from result
            if not isinstance(result, dict) or 'response' not in result:
                raise ValueError("LLMService.generate did not return a dictionary with a 'response' key")
            response = result['response']
            result_citations = result.get('citations', citations)  # Use LLMService citations if available, else keep local citations

            elapsed_time = time.time() - start_time
            logger.info(f"Generated response in {elapsed_time:.2f} seconds: {response[:100]}...")

            return {
                "answer": response,
                "citations": result_citations[:self.config.max_citations]
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise LegalRAGException(e)


# --- 4. Main Test Function ---
async def main():
    """Tests response generation with a legal query and dummy documents."""
    try:
        logger.info("Loading response configuration")
        config_manager = ConfigManager()
        response_config = config_manager.get_response_config()

        # Assume LLMService is initialized externally
        from backend.experiments.step_04_llms import LLMService, ConfigManager as LLMConfigManager
        logger.info("Loading LLM configuration")
        llm_config_manager = LLMConfigManager()
        llm_config = llm_config_manager.get_llm_config()
        llm_service = LLMService(llm_config)

        logger.info("Initializing ResponseService")
        response_service = ResponseService(response_config, llm_service)

        query = "What is the statement of guarantee for company registration in Kenya?"
        dummy_documents = [
            {
                "section": "15",
                "title": "Statement of guarantee",
                "text": "15: Statement of guarantee 15 (1): The applicant for registration of a company to be limited by guarantee shall ensure the statement of guarantee is delivered to the Registrar, specifying the intent to form a company limited by guarantee and the amount each member undertakes to contribute.",
                "source_url": "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2024-12-27",
                "rerank_score": 0.95
            },
            {
                "section": "86",
                "title": "Requirements for registration of unlimited company as private limited company",
                "text": "86: Requirements for registration of unlimited company as private limited company include compliance with the Companies Act provisions for registration documents.",
                "source_url": "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2024-12-27",
                "rerank_score": 0.85
            },
            {
                "section": "7",
                "title": "Companies limited by guarantee",
                "text": "7: Companies limited by guarantee are formed on the principle that members' liability is limited to the amount they undertake to contribute in the event of winding up.",
                "source_url": "https://new.kenyalaw.org/akn/ke/act/2015/17/eng@2024-12-27",
                "rerank_score": 0.80
            }
        ]
        model_id = llm_service.providers[0]['model_name']  # Use first available model

        logger.info(f"Generating response for query: {query}")
        result = await response_service.generate_response(query, dummy_documents, model_id)

        logger.info("Response:")
        logger.info(f"Answer: {result['answer'][:200]}...")
        logger.info("Citations:")
        for i, citation in enumerate(result['citations'], 1):
            logger.info(f"{i}. Section: {citation['section']}, Title: {citation['title']}, Source: {citation['source_url']}")
    except Exception as e:
        logger.error(f"Error in response test: {e}")
        raise LegalRAGException(e)


# --- 5. Entry Point ---
if __name__ == "__main__":
    asyncio.run(main())