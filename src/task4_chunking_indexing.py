"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
import json
import sys
from typing import List

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Vì sao chọn 500? ...
CHUNK_OVERLAP = 50      # Vì sao chọn 50? ...
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Multilingual, tốt cho tiếng Việt
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents: List[dict] = []
    if not STANDARDIZED_DIR.exists():
        print(f"Standardized directory not found: {STANDARDIZED_DIR}")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = md_file.read_text(encoding="utf-8", errors="ignore")

        # determine type from path
        parts = str(md_file).split("/")
        doc_type = "legal" if "legal" in parts else (
            "news" if "news" in parts else "unknown")

        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type, "path": str(md_file)}
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    except Exception as e:
        print("Please install 'langchain-text-splitters' to use chunking:")
        print("pip install langchain-text-splitters")
        raise

    chunks: List[dict] = []

    if CHUNKING_METHOD == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif CHUNKING_METHOD == "markdown_header":
        splitter = MarkdownHeaderTextSplitter()
    else:
        # semantic chunking is advanced; fall back to recursive
        print("Unknown chunking method, falling back to 'recursive'")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    for doc_idx, doc in enumerate(documents):
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i, "doc_index": doc_idx}
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        print("Please install 'sentence-transformers' to compute embeddings:")
        print("pip install sentence-transformers")
        raise

    model_name = EMBEDDING_MODEL
    # If user hasn't configured HF_TOKEN, avoid downloading very large models
    hf_token = None
    try:
        import os
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_HUB_TOKEN")
    except Exception:
        hf_token = None

    # If model likely requires HF access and no token provided, use a lightweight fallback
    if ("/" in model_name or model_name.startswith("BAAI") or model_name.startswith("bge-")) and not hf_token:
        fallback = "sentence-transformers/all-MiniLM-L6-v2"
        print(
            f"No HF token detected — using lightweight fallback '{fallback}' instead of '{model_name}'")
        model = SentenceTransformer(fallback)
    else:
        try:
            model = SentenceTransformer(model_name)
        except Exception:
            fallback = "sentence-transformers/all-MiniLM-L6-v2"
            print(
                f"Failed to load '{model_name}', falling back to '{fallback}'")
            model = SentenceTransformer(fallback)

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        # ensure it's a plain list (not numpy)
        try:
            chunk["embedding"] = emb.tolist()
        except Exception:
            chunk["embedding"] = list(emb)

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    # By default write a local JSON index so the pipeline is runnable without
    # requiring a running Weaviate instance. If `weaviate` is requested and the
    # client is available and reachable, try to push data there.
    index_dir = Path(__file__).parent.parent / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    out_file = index_dir / "chunks_with_embeddings.json"

    if VECTOR_STORE == "weaviate":
        try:
            import weaviate
            client = weaviate.Client("http://localhost:8080")
            if client.is_ready():
                print(
                    "Weaviate detected at http://localhost:8080 — creating/importing objects")
                class_name = "Documents"
                # remove existing schema class if exists
                try:
                    client.schema.delete_class(class_name)
                except Exception:
                    pass

                schema = {
                    "class": class_name,
                    "properties": [
                        {"name": "content", "dataType": ["text"]},
                        {"name": "source", "dataType": ["text"]},
                        {"name": "doc_type", "dataType": ["text"]},
                        {"name": "chunk_index", "dataType": ["int"]}
                    ],
                    "vectorizer": "none"
                }
                client.schema.create_class(schema)

                with client.batch as batch:
                    for chunk in chunks:
                        props = {
                            "content": chunk["content"],
                            "source": chunk["metadata"].get("source"),
                            "doc_type": chunk["metadata"].get("type"),
                            "chunk_index": int(chunk["metadata"].get("chunk_index", 0)),
                        }
                        batch.add_data_object(
                            props, class_name, vector=chunk.get("embedding"))

                print("Imported chunks into Weaviate")
                return
            else:
                print(
                    "Weaviate client present but server not ready; falling back to JSON file")
        except Exception as e:
            print(
                "Weaviate not available or failed to connect; saving local JSON index instead.")

    # Fallback: save to local JSON
    serializable = []
    for c in chunks:
        serializable.append({
            "content": c["content"],
            "metadata": c.get("metadata", {}),
            "embedding": c.get("embedding")
        })

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2)

    print(f"Saved local index to: {out_file}")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(
        f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
