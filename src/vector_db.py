# Vector database and retrieval logic, backed by a persistent Chroma collection
import hashlib
import logging
import chromadb
import ollama

from src.config import settings

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

def _chunk_id(chunk: str) -> str:
    # Deterministic id for a chunk, used to detect new, unchanged, orremoved chunks
    return hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:24]

def _collection_name() -> str:
    # Bucket collections by embedding model
    model_hash = hashlib.sha256(settings.embedding_model.encode("utf-8")).hexdigest()[:12]
    return f"rag_chunks_{model_hash}"

def _get_collection() -> chromadb.Collection:
    # Lazily create the persistent collection on first use
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.vector_store_dir)
        _collection = _client.get_or_create_collection(
            name=_collection_name(),
            metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
        )
    return _collection

def _embed_many(texts: list[str]) -> list[list[float]]:
    # Embed a batch of texts in a single Ollama call
    if not texts:
        return []
    try:
        response = ollama.embed(model=settings.embedding_model, input=texts)
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama to embed text using model '{settings.embedding_model}'. "
            "Is the Ollama server running, and has the model been pulled "
            f"(`ollama pull {settings.embedding_model}`)? Original error: {exc}"
        ) from exc
    return response["embeddings"]

def _embed_one(text: str) -> list[float]:
    # Embed a single text string using the batch embedding function
    return _embed_many([text])[0]

def build_database(dataset: list[str]) -> None:
    # Sync the persistent vector store with the current dataset
    collection = _get_collection()

    incoming_ids = [_chunk_id(chunk) for chunk in dataset]
    id_to_chunk = dict(zip(incoming_ids, dataset))

    existing_ids = set(collection.get(include=[])["ids"])
    incoming_id_set = set(incoming_ids)

    stale_ids = list(existing_ids - incoming_id_set)
    if stale_ids:
        collection.delete(ids=stale_ids)
        logger.info("Removed %d stale chunks from the vector store", len(stale_ids))

    new_ids = [i for i in incoming_ids if i not in existing_ids]
    if not new_ids:
        logger.info("Vector store already up to date (%d chunks, nothing new)", len(dataset))
        return

    logger.info("Embedding %d new chunk(s) of %d total...", len(new_ids), len(dataset))
    new_chunks = [id_to_chunk[i] for i in new_ids]
    embeddings = _embed_many(new_chunks)
    collection.add(ids=new_ids, documents=new_chunks, embeddings=embeddings)
    logger.info("Vector store now holds %d chunks", collection.count())

def retrieve(query: str, top_n: int = settings.top_n) -> list[tuple[str, float]]:
    # Return the top_n chunks most similar to the query
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        raise RuntimeError("Vector database is empty. Call build_database() before retrieve().")

    query_embedding = _embed_one(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_n, count),
        include=["documents", "distances"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    
    return [(doc, 1.0 - dist) for doc, dist in zip(documents, distances)]