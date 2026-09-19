# Generation phase
import ollama
from src.config import LANGUAGE_MODEL

def build_instruction_prompt(retrieved_knowledge: list[tuple[str, float]]) -> str:
    # Build the system prompt that gives the model the retrieved context
    context = "\n".join(f" - {chunk}" for chunk, similarity in retrieved_knowledge)
    return f"""You are a helpful chatbot!
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {context}
        """

def stream_answer(input_query: str, retrieved_knowledge: list[tuple[str, float]]):
    # Send the prompt and the question to the language model
    instruction_prompt = build_instruction_prompt(retrieved_knowledge)

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": instruction_prompt},
            {"role": "user", "content": input_query},
        ],
        stream=True,
    )

    print("Chatbot responding...")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)