# import os
# from dotenv import load_dotenv
# from pathlib import Path
# from backend.src.utils.common import read_yaml
# from backend.src.utils.logger import logger
# from backend.src.utils.exception import LegalRAGException
# from backend.src.constants import CONFIG_FILEPATH
# from backend.src.config_entity.config_params import EmbeddingConfig, RetrievalConfig, RerankerConfig, LLMConfig, ResponseConfig
# 
# load_dotenv()
# 
# 
# # --- 2. Configuration Manager Class ---
# class ConfigurationManager:
#     """Manages loading and validation of configuration from YAML and environment variables."""
#     def __init__(self, config_path: str = CONFIG_FILEPATH):
#         try:
#             # Simplified logging for demo
#             self.config_path = Path(config_path)
#             self.config = read_yaml(self.config_path)
#             if 'embedding_models' not in self.config:
#                 raise ValueError("Configuration error: 'embedding_models' section missing")
#             if 'vector_store' not in self.config or 'retrieval' not in self.config:
#                 raise ValueError("Configuration error: 'vector_store' or 'retrieval' section is missing.")
#             if not self.config or 'reranker' not in self.config:
#                 raise ValueError("Configuration error: 'reranker' section missing in config file")
#             if not self.config or 'generation' not in self.config:
#                 raise ValueError("Configuration error: 'generation' section missing in config file")
#             if not self.config or 'response' not in self.config:
#                 raise ValueError("Configuration error: 'response' section missing in config file")
#         except Exception as e:
#             logger.error(f"Config init error: {e}")
#             raise LegalRAGException(e)
# 
# #------Embedding Config----------------
# 
#     def get_embedding_config(self) -> EmbeddingConfig:
#         """Extracts and validates embedding configuration."""
#         try:
#             embedding_config = self.config['embedding_models']
#             api_key = os.environ.get('VOYAGE_API_KEY')
#             if not api_key:
#                 raise ValueError("VOYAGE_API_KEY not found in environment")
#             if not hasattr(embedding_config, 'embedding_model_name'):
#                 raise ValueError("Embedding model name not found")
#             return EmbeddingConfig(
#                 api_key=api_key,
#                 embedding_model_name=embedding_config.embedding_model_name
#             )
#         except Exception as e:
#             logger.error(f"Embedding config error: {e}")
#             raise LegalRAGException(e)
# 
# 
# # Retriever
#     def get_retrieval_config(self) -> RetrievalConfig:
#         """Extracts and validates retrieval configuration from YAML and environment."""
#         try:
#             vector_store_config = self.config['vector_store']
#             retrieval_config = self.config['retrieval']
#             api_key = os.environ.get('PINECONE_API_KEY')
# 
#             if not api_key:
#                 raise ValueError("PINECONE_API_KEY not found in environment variables.")
# 
#             return RetrievalConfig(
#                 pinecone_index_name=vector_store_config.index_name,
#                 pinecone_api_key=api_key,
#                 retrieval_top_k=retrieval_config.top_k_candidates,
#                 region=vector_store_config.region
#             )
#         except (KeyError, AttributeError) as e:
#             logger.error(f"Configuration file is missing a required key: {e}")
#             raise LegalRAGException(e)
# 
# # Reranker
# 
#     def get_reranking_config(self) -> RerankerConfig:
#         """Extracts and validates reranking configuration from YAML."""
#         try:
#             reranking_config = self.config['reranker']
#             if not hasattr(reranking_config, 'model_name'):
#                 raise ValueError("Reranking model name not found in configuration.")
#             if not hasattr(reranking_config, 'top_n'):
#                 raise ValueError("Reranked top_n candidates not found in configuration.")
#             return RerankerConfig(
#                 model_name="rerank-2.5-lite",  # Override YAML to use Voyage reranker
#                 top_n=reranking_config.top_n
#             )
#         except Exception as e:
#             logger.error(f"Error loading reranking configuration: {e}")
#             raise LegalRAGException(e)
# 
# 
# # LLM
# 
#     def get_llm_config(self) -> LLMConfig:
#         """Extracts and validates LLM configuration from YAML."""
#         try:
#             generation_config = self.config['generation']
#             response_config = self.config.get('response', {})
#             if not hasattr(generation_config, 'max_tokens'):
#                 raise ValueError("Max tokens not found in configuration.")
#             if not hasattr(generation_config, 'providers'):
#                 raise ValueError("Providers not found in configuration.")
# 
#             # Filter for Groq and Together providers only
#             providers = [
#                 p for p in generation_config.providers
#                 if p.get('provider') in ['groq', 'together']
#             ]
#             if not providers:
#                 raise ValueError("No valid providers (groq, together) found in configuration.")
# 
#             # Validate provider priorities
#             for provider in providers:
#                 if not provider.get('priority'):
#                     logger.warning(f"Priority not set for {provider.get('model_name')}, defaulting to 999")
#                     provider['priority'] = 999
# 
#             return LLMConfig(
#                 max_tokens=generation_config.max_tokens,
#                 max_citations=response_config.get('max_citations', 3),
#                 providers=providers
#             )
#         except Exception as e:
#             logger.error(f"Error loading LLM configuration: {e}")
#             raise LegalRAGException(e)
# 
# #---------------Response
# 
#     def get_response_config(self) -> ResponseConfig:
#         """Extracts and validates response configuration from YAML."""
#         try:
#             response_config = self.config['response']
#             if not hasattr(response_config, 'max_citations'):
#                 raise ValueError("Max citations not found in configuration.")
#             if not hasattr(response_config, 'cot_template_path'):
#                 raise ValueError("CoT template path not found in configuration.")
# 
#             # Load CoT template
#             template_path = response_config.cot_template_path.replace('/', '.').replace('.py', '')
#             module = import_module(template_path)
#             cot_template = getattr(module, 'COT_PROMPT_TEMPLATE', None)
#             if not cot_template:
#                 raise ValueError(f"Invalid CoT template at {response_config.cot_template_path}")
# 
#             return ResponseConfig(
#                 max_citations=response_config.max_citations,
#                 cot_template=cot_template
#             )
#         except Exception as e:
#             logger.error(f"Error loading response configuration: {e}")
#             raise LegalRAGException(e)

import os
from functools import wraps
from importlib import import_module
from dotenv import load_dotenv
from pathlib import Path
from typing import Callable, Any

from backend.src.utils.common import read_yaml
from backend.src.utils.logger import logger
from backend.src.utils.exception import LegalRAGException
from backend.src.constants import CONFIG_FILEPATH
from backend.src.config_entity.config_params import (
    EmbeddingConfig,
    RetrievalConfig,
    RerankerConfig,
    LLMConfig,
    ResponseConfig
)

# Load environment variables from .env file at the start
load_dotenv()


def _config_loader(component_name: str) -> Callable:
    """
    A decorator to handle common exceptions for config-loading methods.
    It wraps a function, catches potential errors, logs them, and raises
    a LegalRAGException while preserving the original error's traceback.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: 'ConfigurationManager', *args, **kwargs) -> Any:
            try:
                return func(self, *args, **kwargs)
            except (KeyError, AttributeError, ValueError, ImportError) as e:
                error_msg = f"Failed to load {component_name} configuration due to invalid or missing key: {e}"
                logger.error(error_msg)
                # Use 'raise from' to chain exceptions for better debugging
                raise LegalRAGException(error_msg) from e
            except Exception as e:
                error_msg = f"An unexpected error occurred while loading {component_name} configuration: {e}"
                logger.error(error_msg, exc_info=True)
                raise LegalRAGException(error_msg) from e
        return wrapper
    return decorator


class ConfigurationManager:
    """
    A centralized manager for loading, validating, and providing all RAG pipeline
    configurations from a single YAML file and environment variables.
    """
    REQUIRED_SECTIONS = [
        'embedding_models', 'vector_store', 'retrieval',
        'reranker', 'generation', 'response'
    ]

    def __init__(self, config_path: Path = CONFIG_FILEPATH):
        """
        Initializes the manager by loading the YAML config, validating its structure,
        and fetching all necessary API keys from the environment.
        """
        try:
            self.config = read_yaml(config_path)
            for section in self.REQUIRED_SECTIONS:
                if section not in self.config:
                    raise ValueError(f"Configuration error: Required section '{section}' is missing.")

            self.voyage_api_key = os.environ.get('VOYAGE_API_KEY')
            self.pinecone_api_key = os.environ.get('PINECONE_API_KEY')

            if not self.voyage_api_key:
                raise ValueError("VOYAGE_API_KEY not found in environment variables.")
            if not self.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables.")

            logger.info("ConfigurationManager initialized successfully. All sections and API keys are present.")
        except Exception as e:
            logger.error(f"Failed to initialize ConfigurationManager: {e}")
            raise LegalRAGException(e)

    @_config_loader("EMBEDDING")
    def get_embedding_config(self) -> EmbeddingConfig:
        """Extracts and validates embedding configuration."""
        embedding_config = self.config['embedding_models']
        return EmbeddingConfig(
            api_key=self.voyage_api_key,
            embedding_model_name=embedding_config.embedding_model_name
        )

    @_config_loader("RETRIEVAL")
    def get_retrieval_config(self) -> RetrievalConfig:
        """Extracts and validates retrieval configuration."""
        vector_store_config = self.config['vector_store']
        retrieval_config = self.config['retrieval']
        return RetrievalConfig(
            pinecone_index_name=vector_store_config.index_name,
            pinecone_api_key=self.pinecone_api_key,
            retrieval_top_k=retrieval_config.top_k_candidates,
            region=vector_store_config.region
        )

    @_config_loader("RERANKER")
    def get_reranking_config(self) -> RerankerConfig:
        """Extracts and validates reranking configuration."""
        reranking_config = self.config['reranker']
        return RerankerConfig(
            model_name=reranking_config.model_name,
            top_n=reranking_config.top_n
        )

    @_config_loader("LLM")
    def get_llm_config(self) -> LLMConfig:
        """Extracts and validates LLM configuration, now consistent with other methods."""
        # **FIX:** The internal try/except block has been removed.
        # The decorator now handles all errors for this method.
        gen_config = self.config['generation']
        providers_data = gen_config.get('providers', [])

        if not providers_data:
            raise ValueError("The 'providers' list is missing or empty.")

        for p in providers_data:
            p.setdefault('priority', 999)

        return LLMConfig(
            max_tokens=gen_config.max_tokens,
            providers=providers_data
        )

    @_config_loader("RESPONSE")
    def get_response_config(self) -> ResponseConfig:
        """Extracts and validates response generation configuration."""
        response_config = self.config['response']

        # Dynamically load the Chain of Thought (CoT) prompt template
        template_path_str = response_config.cot_template_path
        template_module_path = template_path_str.replace('/', '.').replace('.py', '')
        module = import_module(template_module_path)
        cot_template = getattr(module, 'COT_PROMPT_TEMPLATE')

        # Use .get() to safely access 'max_citations' with a default fallback value.
        # This makes the application more resilient to minor configuration errors.
        max_citations_value = response_config.get('max_citations', 3)

        logger.info(f"ResponseService will use a max of {max_citations_value} citations.")

        return ResponseConfig(
            max_citations=max_citations_value,
            cot_template=cot_template
        )