# RAG Chatbot with Ollama

This project is a RAG chatbot built with Python and Ollama. It loads a text file of cat facts, converts each line into embeddings, retrieves the most relevant facts for a user question, and generates an answer using a local language model.

## How it Works

1. Loads text data.
2. Creates embeddings.
3. Stores embeddings in a simple in-memory vector database.
4. Retrieves the most relevant text using cosine similarity.
5. Sends the retrieved context to a language model.
6. Streams the chatbot response in real time.

## Project Structure

```
RAG-Chatbot/
├── data/
│   └── cat-facts.txt
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── vector_db.py
│   └── chatbot.py
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

- Python.
- Ollama.
- Required models pulled in Ollama:
  - `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`
  - `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`

Install the Python dependency:

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
