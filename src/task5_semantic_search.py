"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
import json
import os
from pathlib import Path
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """

    index_path = Path(__file__).parent.parent / "data" / \
        "index" / "chunks_with_embeddings.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Local index not found: {index_path}. Run Task 4 first.")

    with index_path.open("r", encoding="utf-8") as fh:
        chunks = json.load(fh)

    if SentenceTransformer is None:
        raise RuntimeError(
            "Please install 'sentence-transformers' in your venv: pip install sentence-transformers")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN")
    preferred = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

    if ("/" in preferred or preferred.startswith("BAAI") or preferred.startswith("bge-")) and not hf_token:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        model_name = preferred

    try:
        model = SentenceTransformer(model_name)
    except Exception:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    q_emb = model.encode(query)
    q_vec = np.array(q_emb, dtype=float)

    results: List[dict] = []
    for c in chunks:
        emb = c.get("embedding")
        if emb is None:
            continue
        vec = np.array(emb, dtype=float)
        denom = (np.linalg.norm(q_vec) * np.linalg.norm(vec))
        score = float(np.dot(q_vec, vec) / denom) if denom != 0 else 0.0
        results.append({
            "content": c.get("content"),
            "score": score,
            "metadata": c.get("metadata", {}),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
