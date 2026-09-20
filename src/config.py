# Configuration constants for the RAG chatbot
import os

# Folder that contains this file
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# Path to the text file
DATA_PATH = os.getenv("RAG_DATA_PATH", os.path.join(_PROJECT_ROOT, "data", "cat-facts.txt"))

# Where computed embeddings are cached so we don't re-embed on every run
CACHE_DIR = os.getenv("RAG_CACHE_DIR", os.path.join(_PROJECT_ROOT, ".cache"))
EMBEDDING_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.json")

# Ollama models
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf")
LANGUAGE_MODEL = os.getenv("RAG_LANGUAGE_MODEL", "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")

# Number of most relevant chunks to retrieve
TOP_N = int(os.getenv("RAG_TOP_N", "3"))