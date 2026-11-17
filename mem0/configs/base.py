import os
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from mem0.configs.rerankers.config import RerankerConfig
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig

# Set up the directory path
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text data")
    memory: str = Field(
        ..., description="The memory deduced from the text data"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the memory")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text data")
    score: Optional[float] = Field(None, description="The score associated with the text data")
    created_at: Optional[str] = Field(None, description="The timestamp when the memory was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the memory was updated")


class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    history_db_path: str = Field(
        description="Path to the history database",
        default=os.path.join(mem0_dir, "history.db"),
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Configuration for the reranker",
        default=None,
    )
    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    custom_instructions: Optional[str] = Field(
        description="Custom instructions for fact extraction",
        default=None,
    )
    multilingual: bool = Field(
        description="Extract memories in the same language as the input messages (injects the language requirement into the extraction prompt)",
        default=False,
    )
    entity_source: Literal["spacy", "llm"] = Field(
        description="Entity source for ingestion and update relinking: 'spacy' (legacy NER) or 'llm' (extraction-output entities with aliases)",
        default="spacy",
    )
    near_duplicate: bool = Field(
        description="Lossless near-duplicate tagging on write (top-1 ANN >= profile threshold) plus read-time cluster folding",
        default=False,
    )
    rerank_top_n: int = Field(
        description="Maximum number of top candidates passed to the reranker when rerank=True",
        default=20,
    )
    rerank_trigger: Literal["always", "conditional"] = Field(
        description="'always': rerank whenever rerank=True (legacy semantics, top-N scoped). "
        "'conditional': rerank only on query/candidate language mismatch or low semantic scores.",
        default="always",
    )
    rerank_semantic_threshold: Optional[float] = Field(
        description="With rerank_trigger='conditional', trigger reranking when the best candidate's "
        "semantic score is below this value. None disables the low-score trigger.",
        default=None,
    )
    profiles: Optional[Dict[str, Dict[str, Any]]] = Field(
        description="Per-model ModelProfile overrides keyed by embedding model id (e.g. {'bge-m3': {'semantic_threshold': 0.2}})",
        default=None,
    )


class AzureConfig(BaseModel):
    """
    Configuration settings for Azure.

    Args:
        api_key (str): The API key used for authenticating with the Azure service.
        azure_deployment (str): The name of the Azure deployment.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_version (str): The version of the Azure API being used.
        default_headers (Dict[str, str]): Headers to include in requests to the Azure API.
    """

    api_key: str = Field(
        description="The API key used for authenticating with the Azure service.",
        default=None,
    )
    azure_deployment: str = Field(description="The name of the Azure deployment.", default=None)
    azure_endpoint: str = Field(description="The endpoint URL for the Azure service.", default=None)
    api_version: str = Field(description="The version of the Azure API being used.", default=None)
    default_headers: Optional[Dict[str, str]] = Field(
        description="Headers to include in requests to the Azure API.", default=None
    )
