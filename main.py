# Simple RAG Chatbot with Ollama - entry point
from src.data_loader import load_dataset
from src.vector_db import build_database, retrieve
from src.chatbot import stream_answer

# Typing any of these ends the session
EXIT_KEYWORDS = {"exit", "quit", "q", "bye"}

def main():
    # Load the dataset
    dataset = load_dataset()

    # Build the vector database
    build_database(dataset)

    print("\nType 'exit' (or 'quit'/'bye') at any time to end the chat.\n")

    # Keep asking questions until the user wants to stop
    while True:
        input_query = input("Ask me a question: ").strip()

        if not input_query:
            continue

        if input_query.lower() in EXIT_KEYWORDS:
            print("Goodbye!")
            break

        retrieved_knowledge = retrieve(input_query)

        print("Retrieving knowledge...")
        for chunk, similarity in retrieved_knowledge:
            print(f"(similarity: {similarity:.2f}) {chunk}")

        stream_answer(input_query, retrieved_knowledge)
        print("\n")

if __name__ == "__main__":
    main()