
import asyncio
import os
import time
from typing import Dict, Any
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from together import Together
from backend.src.utils.exception import LegalRAGException
from backend.src.utils.logger import logger
from backend.src.config_entity.config_params import LLMConfig

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


