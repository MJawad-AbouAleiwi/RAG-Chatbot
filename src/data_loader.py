# Loading the dataset
import logging
from src.config import DATA_PATH

logger = logging.getLogger(__name__)

def load_dataset(path: str = DATA_PATH) -> list[str]:
    # Read the knowledge base text file and return a list of cleaned, non-empty lines
    try:
        with open(path, "r", encoding="utf-8") as file:
            raw_lines = file.readlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Knowledge base file not found at '{path}'. "
            "Check RAG_DATA_PATH or the default data/ folder."
        ) from exc

    # Strip whitespace and drop empty lines instead of embedding them as-is
    dataset = [line.strip() for line in raw_lines if line.strip()]

    if not dataset:
        raise ValueError(f"Knowledge base file at '{path}' is empty after cleaning.")

    logger.info("Loaded %d entries from %s", len(dataset), path)
    return dataset