"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

Design notes cho MVP:
    - Ưu tiên chạy ổn định trong môi trường local/demo.
    - Nếu semantic/lexical/rerank/PageIndex module chưa sẵn sàng, pipeline không crash;
      nó ghi warning và tiếp tục với phần còn lại.
    - RRF được implement nội bộ để không phụ thuộc task7 khi task7 chưa hoàn thành.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional


# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================
# Khi chạy trong package: python -m src.task9_retrieval_pipeline
# Khi chạy trực tiếp file: python task9_retrieval_pipeline.py
try:  # pragma: no cover - depends on project layout
    from .task5_semantic_search import semantic_search as _semantic_search
except Exception:  # Module task5 có thể chưa tồn tại trong bài demo local
    _semantic_search = None

try:  # pragma: no cover - depends on project layout
    from .task6_lexical_search import lexical_search as _lexical_search
except Exception:
    try:
        from task6_lexical_search import lexical_search as _lexical_search
    except Exception:
        _lexical_search = None

try:  # pragma: no cover - depends on project layout
    from .task7_reranking import rerank as _external_rerank
except Exception:
    try:
        from task7_reranking import rerank as _external_rerank
    except Exception:
        _external_rerank = None

try:  # pragma: no cover - depends on project layout
    from .task8_pageindex_vectorless import pageindex_search as _pageindex_search
except Exception:
    try:
        from task8_pageindex_vectorless import pageindex_search as _pageindex_search
    except Exception:
        _pageindex_search = None


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
DEFAULT_RETRIEVAL_MULTIPLIER = 3
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "rrf" | fallback lexical
RRF_K = 60


# =============================================================================
# NORMALIZATION / UTILS
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """Tokenizer đơn giản, đủ tốt cho fallback lexical rerank tiếng Việt/English."""
    return re.findall(r"[\wÀ-ỹ]+", (text or "").lower(), flags=re.UNICODE)


def _doc_key(item: dict) -> str:
    """
    Tạo key chống duplicate. Ưu tiên metadata id/source/page nếu có,
    fallback về content đã strip.
    """
    metadata = item.get("metadata") or {}
    for key in ("id", "chunk_id", "doc_id", "source_id"):
        if metadata.get(key):
            return f"{key}:{metadata[key]}"

    source = metadata.get("source") or metadata.get(
        "filename") or metadata.get("file") or ""
    page = metadata.get("page") or metadata.get("page_number") or ""
    content = (item.get("content") or "").strip()
    return f"{source}:{page}:{content[:300]}"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return default
        return number
    except Exception:
        return default


def _normalize_result(item: dict, source: str, rank: Optional[int] = None) -> dict:
    """Chuẩn hoá output từ semantic/BM25/PageIndex về cùng schema."""
    metadata = dict(item.get("metadata") or {})
    if rank is not None:
        metadata.setdefault(f"{source}_rank", rank)

    return {
        "content": str(item.get("content") or item.get("text") or ""),
        "score": _safe_float(item.get("score"), 0.0),
        "metadata": metadata,
        "source": source,
    }


def _run_search_safely(
    name: str,
    fn: Optional[Callable[..., list[dict]]],
    query: str,
    top_k: int,
) -> list[dict]:
    """Gọi search function nhưng không để một nhánh lỗi làm sập pipeline."""
    if fn is None:
        print(f"  ⚠ Skip {name}: module/function chưa sẵn sàng")
        return []

    try:
        raw_results = fn(query, top_k=top_k) or []
    except NotImplementedError as exc:
        print(f"  ⚠ Skip {name}: {exc}")
        return []
    except Exception as exc:
        print(f"  ⚠ {name} failed: {exc}")
        return []

    return [
        _normalize_result(item, source=name, rank=rank)
        for rank, item in enumerate(raw_results, start=1)
        if item and (item.get("content") or item.get("text"))
    ]


# =============================================================================
# FUSION / RERANKING
# =============================================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    top_k: int = DEFAULT_TOP_K,
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge nhiều ranked lists bằng Reciprocal Rank Fusion.

    Công thức:
        RRF(d) = Σ 1 / (k + rank_r(d))

    Lý do dùng RRF cho MVP:
        - Không cần normalize score giữa dense và BM25.
        - Bền vững khi mỗi retriever dùng scale điểm khác nhau.
        - Dễ giải thích trong demo.
    """
    scores: dict[str, float] = {}
    merged_items: dict[str, dict] = {}
    sources: dict[str, set[str]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = _doc_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            sources.setdefault(key, set()).add(
                str(item.get("source", "unknown")))

            if key not in merged_items:
                merged_items[key] = dict(item)
            else:
                # Giữ metadata đầy đủ nhất và raw score cao nhất để debug.
                current = merged_items[key]
                current["score"] = max(_safe_float(current.get(
                    "score")), _safe_float(item.get("score")))
                current.setdefault("metadata", {}).update(
                    item.get("metadata") or {})

    fused: list[dict] = []
    for key, rrf_score in scores.items():
        item = dict(merged_items[key])
        metadata = dict(item.get("metadata") or {})
        metadata["rrf_score"] = rrf_score
        metadata["retrieval_sources"] = sorted(sources.get(key, []))
        item["metadata"] = metadata
        item["score"] = rrf_score
        item["source"] = "hybrid"
        fused.append(item)

    fused.sort(key=lambda x: _safe_float(x.get("score")), reverse=True)
    return fused[:top_k]


def _lexical_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Fallback rerank không cần model/API.
    Điểm = overlap query terms + fused score gốc.
    """
    query_terms = set(_tokenize(query))
    if not query_terms:
        return candidates[:top_k]

    reranked: list[dict] = []
    for item in candidates:
        doc_terms = set(_tokenize(item.get("content", "")))
        overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        original_score = _safe_float(item.get("score"))
        rerank_score = 0.75 * overlap + 0.25 * original_score

        new_item = dict(item)
        metadata = dict(new_item.get("metadata") or {})
        metadata["pre_rerank_score"] = original_score
        metadata["lexical_overlap"] = overlap
        new_item["metadata"] = metadata
        new_item["score"] = rerank_score
        new_item["source"] = "hybrid"
        reranked.append(new_item)

    reranked.sort(key=lambda x: _safe_float(x.get("score")), reverse=True)
    return reranked[:top_k]


def _rerank_safely(
    query: str,
    candidates: list[dict],
    top_k: int,
    method: str = RERANK_METHOD,
) -> list[dict]:
    """Ưu tiên reranker từ task7; nếu lỗi/chưa implement thì fallback lexical."""
    if not candidates:
        return []

    if method == "rrf":
        return candidates[:top_k]

    if _external_rerank is not None:
        try:
            reranked = _external_rerank(
                query, candidates, top_k=top_k, method=method) or []
            normalized = [_normalize_result(item, "hybrid")
                          for item in reranked]
            return normalized[:top_k]
        except NotImplementedError as exc:
            print(
                f"  ⚠ Reranker chưa sẵn sàng ({exc}). Dùng lexical fallback.")
        except Exception as exc:
            print(f"  ⚠ Reranker failed ({exc}). Dùng lexical fallback.")

    return _lexical_rerank(query, candidates, top_k=top_k)


def _fallback_pageindex(query: str, top_k: int) -> list[dict]:
    """Fallback sang PageIndex nếu có; nếu không có thì trả []."""
    results = _run_search_safely("pageindex", _pageindex_search, query, top_k)
    for item in results:
        item["source"] = "pageindex"
    return results[:top_k]


# =============================================================================
# MAIN RETRIEVAL PIPELINE
# =============================================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → final_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn.
        top_k: Số lượng kết quả cuối cùng.
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results.
        use_reranking: Có áp dụng reranking hay không.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = max(int(top_k), 1)
    retrieval_k = max(top_k * DEFAULT_RETRIEVAL_MULTIPLIER, top_k)

    # Step 1: chạy semantic + lexical song song.
    search_jobs: dict[str, Optional[Callable[..., list[dict]]]] = {
        "semantic": _semantic_search,
        "lexical": _lexical_search,
    }
    ranked_lists: list[list[dict]] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_run_search_safely, name, fn, query, retrieval_k): name
            for name, fn in search_jobs.items()
        }
        for future in as_completed(futures):
            results = future.result()
            if results:
                ranked_lists.append(results)

    # Step 2: Merge bằng RRF.
    merged = reciprocal_rank_fusion(
        ranked_lists, top_k=retrieval_k) if ranked_lists else []

    # Step 3: Rerank.
    if use_reranking and merged:
        final_results = _rerank_safely(
            query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Step 4: threshold check → fallback PageIndex.
    best_score = _safe_float(final_results[0].get(
        "score")) if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid score ({best_score:.3f}) < threshold "
            f"({score_threshold:.3f}). Fallback → PageIndex"
        )
        fallback_results = _fallback_pageindex(query, top_k=top_k)
        if fallback_results:
            return fallback_results

        # Không có PageIndex thì vẫn trả hybrid best-effort để debug/demo không rỗng.
        return final_results[:top_k]

    return final_results[:top_k]


# =============================================================================
# DEMO HELPERS
# =============================================================================

def _install_dummy_retrievers_for_demo() -> None:
    """
    Chỉ dùng khi chạy trực tiếp file mà project chưa có task5/task6/task8.
    Không ảnh hưởng khi import module trong app thật.
    """
    global _semantic_search, _lexical_search, _pageindex_search

    dummy_docs = [
        {
            "content": "Điều 249 Bộ luật Hình sự quy định tội tàng trữ trái phép chất ma tuý, hình phạt có thể từ 1 năm đến tù chung thân tuỳ khối lượng.",
            "score": 0.92,
            "metadata": {"filename": "bo_luat_hinh_su.md", "page": 12},
        },
        {
            "content": "Luật Phòng, chống ma tuý 2021 quy định về quản lý người sử dụng trái phép chất ma tuý và cai nghiện ma tuý.",
            "score": 0.88,
            "metadata": {"filename": "luat_phong_chong_ma_tuy_2021.md", "page": 4},
        },
        {
            "content": "Một số tin tức năm 2024 đề cập nghệ sĩ bị bắt vì sử dụng ma tuý, nhưng đây là nguồn báo chí chứ không phải văn bản pháp luật.",
            "score": 0.54,
            "metadata": {"filename": "news_2024.md", "page": 1},
        },
    ]

    def simple_search(query: str, top_k: int = 5) -> list[dict]:
        q_terms = set(_tokenize(query))
        scored = []
        for doc in dummy_docs:
            d_terms = set(_tokenize(doc["content"]))
            overlap = len(q_terms & d_terms) / max(len(q_terms), 1)
            item = dict(doc)
            item["score"] = overlap
            scored.append(item)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    if _semantic_search is None:
        _semantic_search = simple_search
    if _lexical_search is None:
        _lexical_search = simple_search
    if _pageindex_search is None:
        _pageindex_search = simple_search


if __name__ == "__main__":
    _install_dummy_retrievers_for_demo()

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
