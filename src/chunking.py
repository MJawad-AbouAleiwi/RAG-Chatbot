# Text chunking strategy
import re

def split_into_paragraphs(text: str) -> list[str]:
    # Split raw text into paragraphs, with a sensible fallback for line-per-fact files
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paragraphs = re.split(r"\n\s*\n", normalized)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    if len(paragraphs) <= 1:
        # No blank-line-delimited paragraphs found
        paragraphs = [line.strip() for line in normalized.split("\n") if line.strip()]

    return paragraphs

def chunk_paragraph(paragraph: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # Split one paragraph into overlapping word-windows of at most chunk_size words
    words = paragraph.split()
    if len(words) <= chunk_size:
        return [paragraph]

    chunks = []
    step = max(chunk_size - chunk_overlap, 1)
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # Turn raw text into a list of retrieval-sized chunks
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[str] = []
    for paragraph in split_into_paragraphs(text):
        chunks.extend(chunk_paragraph(paragraph, chunk_size, chunk_overlap))
    return chunks