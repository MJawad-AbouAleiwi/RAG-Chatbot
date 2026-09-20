# Vector database and retrieval logic
import hashlib
import json
import logging
import os

import numpy as np
import ollama

from src.config import settings

logger = logging.getLogger(__name__)

# In-memory vector database: parallel lists kept in sync
_CHUNKS: list[str] = []
_EMBEDDINGS: np.ndarray | None = None


def _dataset_fingerprint(dataset: list[str]) -> str:
    # Hash the dataset + embedding model so the cache invalidates when either changes
    hasher = hashlib.sha256()
    hasher.update(settings.embedding_model.encode("utf-8"))
    for chunk in dataset:
        hasher.update(chunk.encode("utf-8"))
    return hasher.hexdigest()

def _embed(text: str) -> list[float]:
    # Call Ollama's embedding endpoint with a clear error if Ollama isn't available
    try:
        response = ollama.embed(model=settings.embedding_model, input=text)
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama to embed text using model '{settings.embedding_model}'. "
            "Is the Ollama server running, and has the model been pulled "
            f"(`ollama pull {settings.embedding_model}`)? Original error: {exc}"
        ) from exc
    return response["embeddings"][0]

def _load_cache(fingerprint: str) -> tuple[list[str], np.ndarray] | None:
    if not os.path.exists(settings.embedding_cache_path):
        return None
    try:
        with open(settings.embedding_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable embedding cache: %s", exc)
        return None

    if cached.get("fingerprint") != fingerprint:
        return None

    chunks = cached["chunks"]
    embeddings = np.array(cached["embeddings"], dtype=np.float32)
    logger.info("Loaded %d cached embeddings from %s", len(chunks), settings.embedding_cache_path)
    return chunks, embeddings

def _save_cache(fingerprint: str, chunks: list[str], embeddings: np.ndarray) -> None:
    os.makedirs(os.path.dirname(settings.embedding_cache_path), exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "chunks": chunks,
        "embeddings": embeddings.tolist(),
    }
    with open(settings.embedding_cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info("Cached %d embeddings to %s", len(chunks), settings.embedding_cache_path)

def build_database(dataset: list[str]) -> None:
    # Embed every entry in the dataset (or load from cache) and store it in memory
    global _CHUNKS, _EMBEDDINGS

    fingerprint = _dataset_fingerprint(dataset)
    cached = _load_cache(fingerprint)
    if cached is not None:
        _CHUNKS, _EMBEDDINGS = cached
        return

    logger.info("Embedding %d chunks (no valid cache found)...", len(dataset))
    embeddings = [_embed(chunk) for chunk in dataset]

    _CHUNKS = dataset
    _EMBEDDINGS = np.array(embeddings, dtype=np.float32)
    _save_cache(fingerprint, _CHUNKS, _EMBEDDINGS)

def retrieve(query: str, top_n: int = settings.top_n) -> list[tuple[str, float]]:
    # Return the top_n chunks most similar to the query
    if _EMBEDDINGS is None or len(_CHUNKS) == 0:
        raise RuntimeError("Vector database is empty. Call build_database() before retrieve().")

    query_embedding = np.array(_embed(query), dtype=np.float32)

    # Cosine similarity, vectorized across all stored chunks at once.
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(_EMBEDDINGS, axis=1)
    similarities = (_EMBEDDINGS @ query_embedding) / (chunk_norms * query_norm + 1e-10)

    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [(_CHUNKS[i], float(similarities[i])) for i in top_indices]