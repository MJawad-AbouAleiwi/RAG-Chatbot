# Loading and chunking the knowledge base
import logging

from src.chunking import chunk_text
from src.config import settings

logger = logging.getLogger(__name__)

def load_dataset(path: str | None = None) -> list[str]:
    # Read the knowledge base text file and split it into retrieval-sized chunks
    path = path or settings.data_path

    try:
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Knowledge base file not found at '{path}'. "
            "Check RAG_DATA_PATH (or the data_path setting in .env)."
        ) from exc

    if not text.strip():
        raise ValueError(f"Knowledge base file at '{path}' is empty.")

    dataset = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    if not dataset:
        raise ValueError(f"Knowledge base file at '{path}' produced no chunks after cleaning.")

    logger.info(
        "Loaded %d chunks from %s (chunk_size=%d, chunk_overlap=%d)",
        len(dataset),
        path,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return dataset