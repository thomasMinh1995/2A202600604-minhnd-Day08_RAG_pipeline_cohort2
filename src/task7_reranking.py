"""
Task 7 — Reranking Module.

Implementation notes for MVP:
- RRF: useful when you have multiple retrieval results, e.g. vector search + BM25.
- MMR: useful when you want relevant but less duplicated chunks.
- Cross-encoder: optional Jina API if JINA_API_KEY is available; otherwise uses a
  lightweight lexical fallback so the demo can run without external services.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from typing import Any, Optional


Candidate = dict[str, Any]


# =============================================================================
# Shared helpers
# =============================================================================


def _validate_top_k(top_k: int) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    return top_k


def _tokenize(text: str) -> set[str]:
    """Simple multilingual-friendly tokenizer for fallback scoring."""
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """Return cosine similarity between two vectors.

    Raises:
        ValueError: if vectors are empty or have different dimensions.
    """
    if not vec_a or not vec_b:
        raise ValueError("Vectors must not be empty")
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_relevance(query: str, document: str) -> float:
    """Fallback relevance score based on token overlap.

    This is not a real cross-encoder. It only keeps the module runnable in local
    MVP environments where no reranker API/model is configured.
    """
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(document)

    if not query_tokens or not doc_tokens:
        return 0.0

    overlap = query_tokens & doc_tokens
    precision = len(overlap) / len(doc_tokens)
    recall = len(overlap) / len(query_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _candidate_key(item: Candidate) -> str:
    """Stable key used for merging duplicated results across ranked lists."""
    metadata = item.get("metadata") or {}
    for key in ("id", "chunk_id", "doc_id", "source"):
        if metadata.get(key) is not None:
            return f"{key}:{metadata[key]}"
    return f"content:{item.get('content', '')}"


# =============================================================================
# Cross-encoder reranking
# =============================================================================


def rerank_cross_encoder(
    query: str,
    candidates: list[Candidate],
    top_k: int = 5,
    jina_api_key: Optional[str] = None,
) -> list[Candidate]:
    """
    Rerank candidates using a cross-encoder reranker when available.

    Priority:
    1. Use Jina Reranker API if JINA_API_KEY or jina_api_key is provided.
    2. Fallback to lexical scoring so local MVP tests still run.

    Args:
        query: Câu truy vấn.
        candidates: List of {'content': str, 'score': float, 'metadata': dict}.
        top_k: Số lượng kết quả sau rerank.
        jina_api_key: Optional API key. If omitted, reads env JINA_API_KEY.

    Returns:
        List of top_k candidates with rerank_score and score, sorted descending.
    """
    _validate_top_k(top_k)
    if not candidates:
        return []

    api_key = jina_api_key or os.getenv("JINA_API_KEY")

    if api_key:
        try:
            import requests  # type: ignore

            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c.get("content", "") for c in candidates],
                    "top_n": min(top_k, len(candidates)),
                },
                timeout=30,
            )
            response.raise_for_status()
            reranked = response.json().get("results", [])

            results: list[Candidate] = []
            for row in reranked:
                idx = row["index"]
                score = float(row["relevance_score"])
                item = candidates[idx].copy()
                item["rerank_score"] = score
                item["score"] = score
                item["rerank_method"] = "jina-reranker-v2-base-multilingual"
                results.append(item)
            return results
        except Exception:
            # Keep MVP robust. In production, replace this with structured logging.
            pass

    scored: list[Candidate] = []
    for item in candidates:
        score = _lexical_relevance(query, item.get("content", ""))
        new_item = item.copy()
        new_item["rerank_score"] = score
        new_item["score"] = score
        new_item["rerank_method"] = "lexical_fallback"
        scored.append(new_item)

    return sorted(scored, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# =============================================================================
# MMR reranking
# =============================================================================


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[Candidate],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[Candidate]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Formula:
        MMR = λ * sim(query, doc) - (1 - λ) * max(sim(doc, selected_docs))

    Meaning:
    - λ gần 1.0: ưu tiên relevance với query.
    - λ gần 0.0: ưu tiên diversity, giảm trùng lặp giữa các chunk.
    """
    _validate_top_k(top_k)
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")
    if not candidates:
        return []

    for i, candidate in enumerate(candidates):
        if "embedding" not in candidate:
            raise ValueError(f"Candidate at index {i} is missing 'embedding'")

    selected: list[int] = []
    remaining = set(range(len(candidates)))
    relevance_cache = {
        idx: cosine_sim(query_embedding, candidates[idx]["embedding"])
        for idx in remaining
    }

    while remaining and len(selected) < top_k:
        best_idx: Optional[int] = None
        best_mmr_score = float("-inf")
        best_relevance = 0.0
        best_diversity_penalty = 0.0

        for idx in remaining:
            relevance = relevance_cache[idx]
            max_sim_to_selected = 0.0

            if selected:
                max_sim_to_selected = max(
                    cosine_sim(candidates[idx]["embedding"],
                               candidates[sel_idx]["embedding"])
                    for sel_idx in selected
                )

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_mmr_score:
                best_idx = idx
                best_mmr_score = mmr_score
                best_relevance = relevance
                best_diversity_penalty = max_sim_to_selected

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        candidates[best_idx] = candidates[best_idx].copy()
        candidates[best_idx]["mmr_score"] = best_mmr_score
        candidates[best_idx]["relevance_score"] = best_relevance
        candidates[best_idx]["diversity_penalty"] = best_diversity_penalty
        candidates[best_idx]["score"] = best_mmr_score
        candidates[best_idx]["rerank_method"] = "mmr"

    return [candidates[i] for i in selected]


# =============================================================================
# RRF reranking
# =============================================================================


def rerank_rrf(
    ranked_lists: list[list[Candidate]], top_k: int = 5, k: int = 60
) -> list[Candidate]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Formula:
        RRF(d) = Σ 1 / (k + rank_r(d))

    Meaning:
    - Một document xuất hiện thứ hạng cao ở nhiều retriever sẽ có điểm cao.
    - k càng lớn thì giảm chênh lệch giữa rank đầu và rank sau.
    - Phù hợp khi fuse BM25 + vector search + metadata search.
    """
    _validate_top_k(top_k)
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not ranked_lists:
        return []

    rrf_scores: defaultdict[str, float] = defaultdict(float)
    content_map: dict[str, Candidate] = {}
    rank_details: defaultdict[str, list[dict[str, int]]] = defaultdict(list)

    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank, item in enumerate(ranked_list, start=1):
            key = _candidate_key(item)
            rrf_scores[key] += 1 / (k + rank)
            rank_details[key].append({"list_index": list_idx, "rank": rank})

            # Keep the best original item by input score if duplicated.
            previous = content_map.get(key)
            if previous is None or item.get("score", 0) > previous.get("score", 0):
                content_map[key] = item

    sorted_keys = sorted(
        rrf_scores, key=lambda key: rrf_scores[key], reverse=True)

    results: list[Candidate] = []
    for key in sorted_keys[:top_k]:
        item = content_map[key].copy()
        item["rrf_score"] = rrf_scores[key]
        item["score"] = rrf_scores[key]
        item["rank_details"] = rank_details[key]
        item["rerank_method"] = "rrf"
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================


def rerank(
    query: str,
    candidates: list[Candidate],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "rrf"
) -> list[Candidate]:
    """
    Unified reranking interface for simple retrieval pipelines.

    Args:
        query: Câu truy vấn.
        candidates: Danh sách candidates từ retrieval.
        top_k: Số lượng kết quả sau rerank.
        method: "cross_encoder" or "rrf".

    Notes:
        - MMR cần query_embedding nên gọi trực tiếp rerank_mmr(...).
        - RRF thường cần nhiều ranked lists; nếu truyền một candidates list thì hàm này
          coi đó là một ranked list duy nhất.
    """
    method = method.lower()

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "mmr":
        raise ValueError(
            "MMR requires query_embedding. Call rerank_mmr(...) directly.")

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test 1: cross_encoder API if configured, otherwise lexical fallback.
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý",
            "score": 0.8, "metadata": {"id": "a"}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý",
            "score": 0.7, "metadata": {"id": "b"}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ",
            "score": 0.6, "metadata": {"id": "c"}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    print("Cross-encoder / lexical fallback:")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")

    # Test 2: RRF fusion from two rankers.
    bm25_results = [dummy_candidates[2],
                    dummy_candidates[0], dummy_candidates[1]]
    vector_results = [dummy_candidates[0],
                      dummy_candidates[2], dummy_candidates[1]]
    print("\nRRF:")
    for r in rerank_rrf([bm25_results, vector_results], top_k=2):
        print(f"[{r['score']:.4f}] {r['content']}")

    # Test 3: MMR requires embeddings.
    embedded_candidates = [
        {**dummy_candidates[0], "embedding": [0.9, 0.1, 0.1]},
        {**dummy_candidates[1], "embedding": [0.4, 0.8, 0.1]},
        {**dummy_candidates[2], "embedding": [0.8, 0.2, 0.2]},
    ]
    print("\nMMR:")
    for r in rerank_mmr([1.0, 0.0, 0.0], embedded_candidates, top_k=2):
        print(f"[{r['score']:.4f}] {r['content']}")
