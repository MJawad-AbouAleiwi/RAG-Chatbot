# Simple RAG Chatbot with Ollama

This project is a basic RAG chatbot built with Python and Ollama. It loads a text file of cat facts, converts each line into embeddings, retrieves the most relevant facts for a user question, and generates an answer using a local language model.

## How it works

1. Loads text data from `cat-facts.txt`.
2. Creates embeddings using `bge-base-en-v1.5-gguf`.
3. Stores embeddings in a simple in-memory vector database.
4. Retrieves the most relevant text using cosine similarity.
5. Sends the retrieved context to a language model `Llama-3.2-1B-Instruct-GGUF`.
6. Streams the chatbot response in real time.

## Requirements

- Python
- Ollama
- Required models pulled in Ollama:
  - `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`
  - `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`