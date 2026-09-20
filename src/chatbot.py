# Generation phase
import logging
import ollama

from src.config import settings

logger = logging.getLogger(__name__)

def build_instruction_prompt(retrieved_knowledge: list[tuple[str, float]]) -> str:
    # Build the system prompt that gives the model the retrieved context
    context = "\n".join(f" - {chunk}" for chunk, _similarity in retrieved_knowledge)
    return (
        "You are a helpful chatbot!\n"
        "Use only the following pieces of context to answer the question. "
        "Don't make up any new information:\n"
        f"{context}"
    )

def stream_answer(input_query: str, retrieved_knowledge: list[tuple[str, float]]) -> str:
    # Send the prompt and question to the language model, printing and returning the answer
    instruction_prompt = build_instruction_prompt(retrieved_knowledge)

    try:
        stream = ollama.chat(
            model=settings.language_model,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": input_query},
            ],
            stream=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama to generate a response using model '{settings.language_model}'. "
            "Is the Ollama server running, and has the model been pulled "
            f"(`ollama pull {settings.language_model}`)? Original error: {exc}"
        ) from exc

    print("Chatbot responding...")
    full_response = []
    try:
        for chunk in stream:
            piece = chunk["message"]["content"]
            print(piece, end="", flush=True)
            full_response.append(piece)
    except Exception as exc:
        logger.error("Streaming interrupted: %s", exc)
        raise

    return "".join(full_response)