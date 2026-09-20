# Validated, environment-driven configuration for the RAG chatbot
import os

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=os.path.join(_PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data & caching
    data_path: str = Field(
        default=os.path.join(_PROJECT_ROOT, "data", "cat-facts.txt"),
        description="Path to the knowledge base text file.",
    )
    cache_dir: str = Field(
        default=os.path.join(_PROJECT_ROOT, ".cache"),
        description="Directory used to cache computed embeddings.",
    )

    # Ollama models
    embedding_model: str = Field(
        default="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf",
        description="Ollama model used to embed text.",
        min_length=1,
    )
    language_model: str = Field(
        default="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
        description="Ollama model used to generate answers.",
        min_length=1,
    )

    # Retrieval
    top_n: int = Field(default=3, gt=0, description="Number of chunks to retrieve per query.")

    # Chunking
    chunk_size: int = Field(
        default=200, gt=0, description="Max chunk size in words before a paragraph is split."
    )
    chunk_overlap: int = Field(
        default=20, ge=0, description="Number of words shared between consecutive chunks."
    )

    # Logging
    log_level: str = Field(default="INFO", description="Python logging level name.")

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}), or chunks would never advance."
            )
        return self

    @property
    def embedding_cache_path(self) -> str:
        return os.path.join(self.cache_dir, "embeddings_cache.json")

# Single shared instance imported throughout the app
settings = Settings()