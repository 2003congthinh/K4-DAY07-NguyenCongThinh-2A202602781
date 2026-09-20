from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []

        for sentence in sentences:
            words = sentence.split()
            # Chunk the sentence by max_sentences_per_chunk
            for i in range(0, len(words), self.max_sentences_per_chunk):
                chunk = " ".join(words[i:i+self.max_sentences_per_chunk])
                chunks.append(chunk)
        return chunks
        # raise NotImplementedError("Implement SentenceChunker.chunk")


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            # No more separators, just force split
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        parts = current_text.split(sep) if sep else [current_text]
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) + len(sep) <= self.chunk_size:
                current_chunk += (sep if current_chunk else "") + part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, remaining_separators[1:]))
                    current_chunk = ""
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks


class SemanticChunker:
    """Chunk markdown-like text by heading sections, preserving context in each section."""

    def __init__(self, chunk_size: int = 500, overlap: int = 0) -> None:
        self.chunk_size = max(1, chunk_size)
        self.overlap = max(0, overlap)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        lines = text.splitlines()
        blocks: list[tuple[str | None, str]] = []
        current_heading: str | None = None
        buffer: list[str] = []

        for line in lines:
            heading_match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
            if heading_match:
                if current_heading is not None or buffer:
                    blocks.append((current_heading, "\n".join(buffer).strip()))
                current_heading = line.strip()
                buffer = []
            else:
                buffer.append(line)

        if current_heading is not None or buffer:
            blocks.append((current_heading, "\n".join(buffer).strip()))

        if not blocks:
            return RecursiveChunker(chunk_size=self.chunk_size).chunk(text.strip())

        chunks: list[str] = []
        for heading, body in blocks:
            section = body.strip()
            if not section and heading:
                section = heading
            if not section:
                continue

            if heading:
                base = f"{heading}\n{section}".strip()
            else:
                base = section

            if len(base) <= self.chunk_size:
                chunks.append(base)
                continue

            if heading:
                pieces = [p.strip() for p in re.split(r"\n\s*\n+", section) if p.strip()]
                if not pieces:
                    pieces = [section]
                current_block = heading
                for piece in pieces:
                    candidate = f"{heading}\n{piece}".strip()
                    if len(candidate) <= self.chunk_size:
                        if current_block == heading:
                            current_block = candidate
                        else:
                            current_block = f"{current_block}\n\n{piece}".strip()
                        continue

                    if current_block != heading:
                        chunks.append(current_block.strip())
                    if len(piece) <= self.chunk_size:
                        chunks.append(f"{heading}\n{piece}".strip())
                    else:
                        chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(f"{heading}\n{piece}"))
                    current_block = heading

                if current_block != heading:
                    chunks.append(current_block.strip())
            else:
                chunks.extend(RecursiveChunker(chunk_size=self.chunk_size).chunk(section))

        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_prod = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_prod / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        if not text:
            return {
                "fixed_size": {"count": 0, "avg_length": 0.0, "chunks": []},
                "by_sentences": {"count": 0, "avg_length": 0.0, "chunks": []},
                "recursive": {"count": 0, "avg_length": 0.0, "chunks": []},
            }

        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10)).chunk(text)
        sentence_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def summarize(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = sum(len(chunk) for chunk in chunks) / count if count else 0.0
            return {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }

        return {
            "fixed_size": summarize(fixed_chunks),
            "by_sentences": summarize(sentence_chunks),
            "recursive": summarize(recursive_chunks),
        }
