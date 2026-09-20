# Simple RAG Chatbot with Ollama - entry point
import logging

from src.chatbot import stream_answer
from src.data_loader import load_dataset
from src.vector_db import build_database, retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Typing any of these ends the session
EXIT_KEYWORDS = {"exit", "quit", "q", "bye"}

def main():
    try:
        dataset = load_dataset()
        build_database(dataset)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Startup failed: %s", exc)
        return

    print("\nType 'exit' (or 'quit'/'bye') at any time to end the chat.\n")

    while True:
        try:
            input_query = input("Ask me a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not input_query:
            continue

        if input_query.lower() in EXIT_KEYWORDS:
            print("Goodbye!")
            break

        try:
            retrieved_knowledge = retrieve(input_query)

            print("Retrieving knowledge...")
            for chunk, similarity in retrieved_knowledge:
                print(f"(similarity: {similarity:.2f}) {chunk}")

            stream_answer(input_query, retrieved_knowledge)
            print("\n")
        except RuntimeError as exc:
            logger.error("%s", exc)

if __name__ == "__main__":
    main()