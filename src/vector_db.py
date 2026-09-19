# Vector database and retrieval logic
import ollama
from src.config import EMBEDDING_MODEL, TOP_N

# In-memory vector database
VECTOR_DB: list[tuple[str, list[float]]] = []

def add_chunk_to_database(chunk: str) -> None:
    # Embed a single chunk of text and append it to VECTOR_DB
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)["embeddings"][0]
    VECTOR_DB.append((chunk, embedding))

def build_database(dataset: list[str]) -> None:
    # Embed every entry in the dataset and store it in VECTOR_DB
    print("Adding chunks to the vector database...")
    for chunk in dataset:
        add_chunk_to_database(chunk)

def cosine_similarity(a: list[float], b: list[float]) -> float:
    # Compute the cosine similarity between two vectors
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(query: str, top_n: int = TOP_N) -> list[tuple[str, float]]:
    # Return the top_n chunks most similar to the query
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]