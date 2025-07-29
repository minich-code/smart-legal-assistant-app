import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together

from backend.src.constants import CONFIG_FILEPATH
from backend.src.utils.common import read_yaml
from backend.src.utils.exception import LegalRAGException
from backend.src.utils.logger import logger

# Load environment variables from .env file
load_dotenv()


# --- 1. Consolidated LLM Configuration ---
@dataclass(frozen=True)
class LLMConfig:
    """
    A single configuration class for LLM providers and generation parameters.
    The 'providers' list directly maps to the YAML structure.
    """
    max_tokens: int
    providers: List[Dict[str, Any]] = field(default_factory=list)


# --- 2. Simplified Configuration Manager ---
class ConfigManager:
    """Manages loading and validation of generation-specific configuration."""

    def __init__(self, config_path: Path = CONFIG_FILEPATH):
        try:
            self.config = read_yaml(config_path)
            if 'generation' not in self.config:
                raise ValueError("Config error: 'generation' section is missing.")
            logger.info("Generation configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error initializing ConfigManager for LLM: {e}")
            raise LegalRAGException(e)

    def get_llm_config(self) -> LLMConfig:
        """Extracts and validates LLM configuration directly from YAML."""
        try:
            gen_config = self.config['generation']
            providers_data = gen_config.get('providers', [])

            if not providers_data:
                raise ValueError("The 'providers' list is missing or empty.")

            # Optional: Validate priorities still exist, even in the dictionary
            for p in providers_data:
                p.setdefault('priority', 999)

            return LLMConfig(
                max_tokens=gen_config.max_tokens,
                providers=providers_data
            )
        except (KeyError, AttributeError) as e:
            logger.error(f"Error parsing LLM configuration: {e}")
            raise LegalRAGException(e)


# --- 3. LLM Service (Adapted for Dictionary Config) ---
class LLMService:
    """
    A generic service for generating text, adapted to use a dictionary-based provider config.
    """

    def __init__(self, config: LLMConfig):
        if not isinstance(config, LLMConfig):
            raise TypeError("config must be an instance of LLMConfig")
        self.config = config
        self.clients = self._initialize_clients()
        if not self.clients:
            raise LegalRAGException("No LLM clients initialized. Check API keys.")
        logger.info(f"LLMService initialized with {len(self.clients)} active clients.")

    def _initialize_clients(self) -> Dict[str, Any]:
        """Initializes API clients for all unique providers in the config."""
        clients = {}
        unique_providers = {p.get('provider') for p in self.config.providers if p.get('provider')}

        for provider_name in unique_providers:
            api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
            if not api_key:
                logger.warning(f"Skipping {provider_name} due to missing API key.")
                continue
            try:
                if provider_name == 'groq':
                    clients[provider_name] = AsyncGroq(api_key=api_key)
                elif provider_name == 'together':
                    clients[provider_name] = Together(api_key=api_key)
                logger.info(f"Initialized client for provider: '{provider_name}'")
            except Exception as e:
                logger.error(f"Failed to init client for {provider_name}: {e}")
        return clients

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(self, model_id: str, system_prompt: str, user_prompt: str) -> str:
        """
        Generates a text response using the best available provider for a given model.
        """
        available_providers = sorted(
            [p for p in self.config.providers if p.get('model_name') == model_id],
            key=lambda p: p.get('priority', 999)
        )
        if not available_providers:
            raise ValueError(f"No provider found in configuration for model: {model_id}")

        for provider_config in available_providers:
            provider_name = provider_config.get('provider')
            client = self.clients.get(provider_name)
            if not client:
                logger.warning(f"Skipping {provider_name} as its client is not available.")
                continue

            try:
                logger.info(f"Attempting generation with {provider_name.upper()} for model '{model_id}'...")
                start_time = time.time()
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

                if provider_name == 'groq':
                    completion = await client.chat.completions.create(
                        model=model_id, messages=messages, temperature=0.1, max_tokens=self.config.max_tokens
                    )
                    response_text = completion.choices[0].message.content

                elif provider_name == 'together':
                    loop = asyncio.get_event_loop()
                    completion = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                        model=model_id, messages=messages, temperature=0.1, max_tokens=self.config.max_tokens
                    ))
                    response_text = completion.choices[0].message.content

                else:
                    logger.warning(f"Unsupported provider '{provider_name}' encountered during generation.")
                    continue

                elapsed = time.time() - start_time
                logger.info(f"Successfully generated response from {provider_name.upper()} in {elapsed:.2f}s.")
                if not response_text:
                    raise ValueError("LLM returned an empty response.")
                return response_text

            except Exception as e:
                logger.error(f"Generation with {provider_name.upper()} failed: {e}")
                # Tenacity will handle the retry logic based on this exception

        raise LegalRAGException(f"All providers for model '{model_id}' failed.")


# --- 4. Main Test Function (Remains the same, logic is sound) ---
async def main():
    """Tests the independent LLMService."""

    def format_legal_rag_prompt(query: str, documents: List[Dict[str, Any]]) -> str:
        """Helper function to create the RAG prompt."""
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            text = doc.get('text', 'No text available')
            doc_texts.append(f"Document {i} (Section: {doc.get('section', 'N/A')}):\n```{text[:700]}...```")

        documents_str = "\n\n".join(doc_texts)
        return f"Based *only* on the documents provided, answer the query.\n\nQuery: {query}\n\nDocuments:\n{documents_str}"

    try:
        logger.info("--- Starting Independent LLM Service Test ---")
        config_manager = ConfigManager()
        llm_config = config_manager.get_llm_config()
        llm_service = LLMService(llm_config)

        system_prompt = "You are a legal assistant specializing in Kenyan corporate law. Provide clear, accurate answers based only on the text provided."
        query = "What is the statement of guarantee for company registration?"
        dummy_documents = [
            {"section": "15",
             "text": "15 (1): The applicant for registration of a company to be limited by guarantee shall ensure the statement of guarantee is delivered to the Registrar..."},
            {"section": "7",
             "text": "A company is a company limited by guarantee if members' liability is limited to the amount they undertake to contribute..."}
        ]

        user_prompt = format_legal_rag_prompt(query, dummy_documents)

        # Get all unique model IDs from the configuration
        model_ids = {p.get('model_name') for p in llm_config.providers if p.get('model_name')}

        for model_id in model_ids:
            logger.info(f"\n--- Testing Model: {model_id} ---")
            response = await llm_service.generate(model_id, system_prompt, user_prompt)
            logger.info(f"Query: {query}")
            logger.info(f"Generated Response: {response}")
            logger.info("--------------------")

    except LegalRAGException as e:
        logger.error(f"A critical error occurred in the LLM test: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


# --- 5. Entry Point ---
if __name__ == "__main__":
    asyncio.run(main())