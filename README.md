# RAG Chatbot with Ollama

This project is a RAG chatbot built with Python and Ollama. It loads a text file of cat facts, converts each line into embeddings, retrieves the most relevant facts for a user question, and generates an answer using a local language model.

## How it Works

1. Loads the text file and splits it into chunks.
2. Embeds each new chunk and stores it in a persistent vector store on disk.
3. Retrieves the most relevant chunks for a query via Chroma's similarity search.
4. Sends the retrieved context to a language model.
5. Streams the chatbot response in real time.

## Project Structure

```
RAG-Chatbot/
├── data/
│   └── cat-facts.txt
├── src/
│   ├── config.py
│   ├── chunking.py
│   ├── data_loader.py
│   ├── vector_db.py
│   └── chatbot.py
├── main.py
├── .env.example
├── requirements.txt
└── README.md
```

## Chunking

1. Splits on blank lines into paragraphs. If the file has no blank lines at all,
   each non-empty line is treated as its own paragraph instead.
2. Keeps a paragraph as a single chunk if it fits within `RAG_CHUNK_SIZE` words.
3. Otherwise splits it into overlapping word-windows so context isn't lost at arbitrary cut points.

## Vector store

Chunks are embedded and stored in a persistent [Chroma](https://www.trychroma.com/)
collection on disk at `RAG_VECTOR_STORE_DIR`, using cosine similarity for retrieval.

- The collection is loaded from disk, not rebuilt from scratch, every time the app starts.
- Each chunk gets a stable id derived from a hash of its own text.
- Chroma indexes with HNSW instead of a linear scan over every stored vector.
- The collection name is derived from `RAG_EMBEDDING_MODEL`, so switching embedding models can't
  accidentally mix incompatible vectors in the same collection.

## Requirements

- Python.
- Ollama.
- Required models pulled in Ollama:
  - `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`
  - `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Pull the models:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Usage

```bash
python main.py
```

You'll be prompted with `Ask me a question:` - type your question about cats and press enter. The script will retrieve the most relevant facts and stream back an answer, then prompt you again for another question.

The dataset is loaded and embedded only once at startup, so each question after the first answers instantly without re-embedding.

To end the session, type `exit`, `quit`, `q`, or `bye` at the prompt.