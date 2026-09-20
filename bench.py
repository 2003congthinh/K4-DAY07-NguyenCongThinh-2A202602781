from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import SemanticChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = Path("data/ecommerce")
QUERY_SET = [
    {
        "question": "Thời gian hỗ trợ đổi trả hàng tại Tiki là bao nhiêu ngày?",
        "metadata_filter": {"audience": "buyer"},
    },
    {
        "question": "Sản phẩm bảo hành gửi về Tiki thì mất bao lâu để nhận lại?",
        "metadata_filter": {"audience": "buyer"},
    },
    {
        "question": "Nhà Bán cần lưu trữ video đóng gói hàng hóa trong bao lâu?",
        "metadata_filter": {"audience": "seller"},
    },
    {
        "question": "Những nhóm sản phẩm nào không được đổi trả theo nhu cầu?",
        "metadata_filter": {"audience": "buyer"},
    },
    {
        "question": "Quy trình xử lý bảo hành của Nhà Bán gồm những bước nào?",
        "metadata_filter": {"audience": "seller"},
    },
]


def load_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter: dict[str, str] = {}
            for line in parts[1].splitlines():
                if not line.strip() or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                frontmatter[key.strip()] = value.strip().strip('"')
            return frontmatter, parts[2].strip()
    return {}, text.strip()


def chunk_markdown_file(path: Path, chunk_size: int = 500) -> list[Document]:
    frontmatter, content = load_frontmatter(path)
    metadata = dict(frontmatter)
    metadata.setdefault("doc_id", path.stem)
    metadata.setdefault("source_file", str(path))

    chunker = SemanticChunker(chunk_size=chunk_size)
    chunks = chunker.chunk(content)
    docs: list[Document] = []
    for idx, chunk in enumerate(chunks):
        docs.append(
            Document(
                id=f"{path.stem}#{idx}",
                content=chunk,
                metadata={**metadata, "doc_id": metadata.get("doc_id", path.stem)},
            )
        )
    return docs


def select_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    if provider == "gemini":
        try:
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return MockEmbedder()


def run_benchmark() -> None:
    markdown_files = sorted(DATA_DIR.glob("*.md"))
    documents: list[Document] = []
    for path in markdown_files:
        documents.extend(chunk_markdown_file(path))

    if not documents:
        print(f"No markdown files found in {DATA_DIR}")
        return

    embedder = select_embedder()
    store = EmbeddingStore(collection_name="bench_store", embedding_fn=embedder)
    store.add_documents(documents)

    print(f"Loaded {len(documents)} chunks from {len(markdown_files)} files")
    print(f"Corpus folder: {DATA_DIR}")
    for idx, item in enumerate(QUERY_SET, start=1):
        question = item["question"]
        metadata_filter = item.get("metadata_filter")
        results = store.search_with_filter(question, top_k=3, metadata_filter=metadata_filter)
        print(f"\nQ{idx}: {question}")
        print(f"  Filter: {metadata_filter}")
        for rank, result in enumerate(results, start=1):
            doc_id = result["metadata"].get("doc_id")
            content = result["content"][:140].replace("\n", " ")
            print(f"  {rank}. score={result['score']:.4f} doc_id={doc_id} :: {content}")


if __name__ == "__main__":
    run_benchmark()
