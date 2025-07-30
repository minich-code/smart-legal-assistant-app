from dataclasses import dataclass, field
from typing import List, Dict, Any

# --- Embedding Config ---
@dataclass(frozen=True)  # FIX: Make immutable for consistency and safety
class EmbeddingConfig:
    """Configuration for embedding model and API credentials."""
    api_key: str
    embedding_model_name: str


# --- Retrieval Config ---
@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for the Pinecone vector store and retrieval parameters."""
    pinecone_index_name: str
    pinecone_api_key: str
    retrieval_top_k: int
    region: str


# --- Reranker Config ---
@dataclass(frozen=True)  # FIX: Make immutable for consistency and safety
class RerankerConfig:
    """Configuration for reranking model and parameters."""
    model_name: str
    top_n: int


# --- LLM Config ---
@dataclass(frozen=True)
class LLMConfig:
    """A single configuration class for LLM providers and generation parameters."""
    max_tokens: int
    providers: List[Dict[str, Any]] = field(default_factory=list)


# --- Response Config ---
@dataclass(frozen=True)
class ResponseConfig:
    """Configuration for response generation and citation parameters."""
    cot_template: str