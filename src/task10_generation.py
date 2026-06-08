"""
Task 10 — Generation Có Citation.

Mục tiêu:
    1. Chọn top_k, top_p phù hợp và giải thích được trong demo.
    2. Sắp xếp lại chunks sau reranking để giảm "lost in the middle".
    3. Inject context vào prompt.
    4. Yêu cầu LLM trả lời có citation.
    5. Nếu không đủ evidence → "Tôi không thể xác minh thông tin này từ nguồn hiện có".

Ghi chú mentor thực chiến:
    - File này chạy được cả khi chưa cấu hình OPENAI_API_KEY.
    - Khi có OPENAI_API_KEY, generate_with_citation() sẽ gọi OpenAI.
    - Khi chưa có OPENAI_API_KEY hoặc package openai chưa cài, module fallback sang
      extractive answer đơn giản để vẫn demo được pipeline end-to-end.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv là optional trong môi trường demo/test.
    pass

try:  # Cho phép chạy như package: python -m src.task10_generation
    from .task9_retrieval_pipeline import retrieve  # type: ignore
except Exception:  # Cho phép chạy trực tiếp: python task10_generation.py
    try:
        from task9_retrieval_pipeline import retrieve  # type: ignore
    except Exception:
        retrieve = None  # type: ignore


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: số chunks đưa vào context.
# Chọn 5 vì thường đủ evidence cho câu hỏi pháp lý/tin tức ngắn, nhưng chưa quá dài
# khiến context bị loãng hoặc gặp "lost in the middle".
TOP_K = 5

# top_p: nucleus sampling.
# Chọn 0.9 để model vẫn linh hoạt trong diễn đạt tiếng Việt, nhưng không mở quá rộng
# không gian token như 0.95-1.0 trong bài toán cần factual/citation.
TOP_P = 0.9

# temperature: độ ngẫu nhiên.
# Chọn 0.3 vì RAG cần bám evidence, ít sáng tạo/hallucination.
TEMPERATURE = 0.3

DEFAULT_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
INSUFFICIENT_EVIDENCE_MESSAGE = "Tôi không thể xác minh thông tin này từ nguồn hiện có"
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = f"""Bạn là trợ lý RAG trả lời bằng tiếng Việt.

Nhiệm vụ:
- Chỉ sử dụng thông tin trong Context được cung cấp.
- Mỗi nhận định factual phải có citation ngay sau câu/ý, dùng đúng nhãn nguồn trong Context, ví dụ [D1] hoặc [D2].
- Nếu Context không có bằng chứng rõ ràng, trả lời: '{INSUFFICIENT_EVIDENCE_MESSAGE}'.
- Không suy đoán, không dùng kiến thức ngoài Context.
- Trình bày rõ ràng, ưu tiên bullet ngắn khi phù hợp.
"""


# =============================================================================
# HELPERS
# =============================================================================

def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _metadata(chunk: dict) -> dict:
    metadata = chunk.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def build_citation_label(chunk: dict, index: int) -> str:
    """Tạo citation label ngắn, ổn định để LLM dễ trích dẫn."""
    metadata = _metadata(chunk)
    explicit = (
        metadata.get("citation")
        or metadata.get("source")
        or metadata.get("filename")
        or metadata.get("file_name")
        or metadata.get("title")
        or metadata.get("document")
        or metadata.get("url")
    )

    if explicit:
        label = _safe_text(explicit)
        # Giữ label dễ đọc, tránh quá dài trong prompt.
        return label[:120]

    return f"D{index}"


def _tokenize(text: str) -> list[str]:
    """Tokenizer nhẹ cho fallback extractive answer."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def _sentence_split(text: str) -> list[str]:
    """Tách câu đơn giản, đủ dùng cho fallback không LLM."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
    return [p.strip(" -\t") for p in parts if p.strip(" -\t")]


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để giảm "lost in the middle".

    Input giả định đã sorted theo score giảm dần: [1, 2, 3, 4, 5]
    Output: [1, 3, 5, 4, 2]

    Ý tưởng:
        - Chunk tốt nhất để đầu context.
        - Chunk tốt thứ hai đưa về cuối context.
        - Các chunk còn lại nằm giữa theo pattern xen kẽ.
    """
    if not chunks:
        return []
    if len(chunks) <= 2:
        return list(chunks)

    reordered: list[dict] = []

    # Index chẵn theo zero-based: 0, 2, 4... giữ ở đầu/middle.
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    # Index lẻ theo zero-based: ..., 3, 1 đưa về cuối, để item rank #2 nằm cuối cùng.
    last_odd_index = len(chunks) - 1 if (len(chunks) -
                                         1) % 2 == 1 else len(chunks) - 2
    for i in range(last_odd_index, 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.

    Mỗi chunk có nhãn [D1], [D2]... và source/citation rõ ràng để LLM cite.
    """
    if not chunks:
        return ""

    context_parts: list[str] = []
    used_chars = 0

    for i, chunk in enumerate(chunks, 1):
        content = _safe_text(chunk.get("content"))
        if not content:
            continue

        metadata = _metadata(chunk)
        label = f"D{i}"
        source_label = build_citation_label(chunk, i)
        doc_type = _safe_text(metadata.get(
            "type") or metadata.get("doc_type"), "unknown")
        score = chunk.get("score", "")
        score_text = f" | Score: {float(score):.4f}" if isinstance(
            score, (int, float)) else ""

        header = f"[{label}] Source: {source_label} | Type: {doc_type}{score_text}"
        part = f"{header}\n{content}"

        # Giới hạn context để tránh prompt quá dài khi chunk content lớn.
        remaining = MAX_CONTEXT_CHARS - used_chars
        if remaining <= 0:
            break
        if len(part) > remaining:
            part = part[: max(0, remaining - 20)].rstrip() + "\n...[truncated]"

        context_parts.append(part)
        used_chars += len(part)

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION CLIENTS / FALLBACKS
# =============================================================================

@dataclass
class GenerationConfig:
    model: str = DEFAULT_MODEL
    temperature: float = TEMPERATURE
    top_p: float = TOP_P


def call_openai_llm(
    *,
    system_prompt: str,
    user_message: str,
    config: GenerationConfig,
) -> str:
    """Gọi OpenAI Chat Completions. Tách riêng để dễ mock/test."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=config.temperature,
        top_p=config.top_p,
    )
    answer = response.choices[0].message.content
    return answer.strip() if answer else INSUFFICIENT_EVIDENCE_MESSAGE


def generate_extractive_fallback(query: str, chunks: list[dict], max_sentences: int = 5) -> str:
    """
    Fallback khi chưa có LLM/API key.

    Cơ chế:
        - Chọn câu trong chunks có overlap token cao với query.
        - Mỗi câu gắn citation [D{i}].
        - Nếu không có overlap/evidence → trả message không xác minh.
    """
    if not chunks:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    query_terms = set(_tokenize(query))
    if not query_terms:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    scored_sentences: list[tuple[float, int, str]] = []

    for doc_index, chunk in enumerate(chunks, 1):
        content = _safe_text(chunk.get("content"))
        for sentence in _sentence_split(content):
            terms = set(_tokenize(sentence))
            if not terms:
                continue
            overlap = len(query_terms & terms)
            if overlap == 0:
                continue
            # Ưu tiên câu có overlap cao, chunk score cao và câu không quá dài.
            retrieval_score = chunk.get("score", 0.0)
            try:
                retrieval_score = float(retrieval_score)
            except Exception:
                retrieval_score = 0.0
            length_penalty = min(len(sentence) / 600, 1.0) * 0.05
            score = overlap + 0.1 * retrieval_score - length_penalty
            scored_sentences.append((score, doc_index, sentence))

    if not scored_sentences:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    selected: list[str] = []
    seen: set[str] = set()
    for _, doc_index, sentence in scored_sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(f"{sentence} [D{doc_index}]")
        if len(selected) >= max_sentences:
            break

    if not selected:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    return "\n".join(f"- {item}" for item in selected)


def _dummy_retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Demo fallback để file chạy trực tiếp khi Task 9 hoặc corpus chưa sẵn sàng."""
    demo_chunks = [
        {
            "content": (
                "Điều 249 Bộ luật Hình sự 2015, sửa đổi 2017, quy định tội tàng trữ "
                "trái phép chất ma túy có thể bị phạt tù tùy theo loại và khối lượng chất ma túy."
            ),
            "score": 0.82,
            "metadata": {"source": "Demo Bộ luật Hình sự", "type": "law"},
            "source": "demo",
        },
        {
            "content": (
                "Luật Phòng, chống ma túy 2021 quy định về quản lý người sử dụng trái phép "
                "chất ma túy, cai nghiện ma túy tự nguyện và cai nghiện ma túy bắt buộc."
            ),
            "score": 0.76,
            "metadata": {"source": "Demo Luật Phòng chống ma túy 2021", "type": "law"},
            "source": "demo",
        },
        {
            "content": (
                "Thông tin về nghệ sĩ bị bắt vì liên quan tới ma túy cần nguồn tin báo chí cụ thể; "
                "demo corpus này không có danh sách nghệ sĩ hay thời điểm vụ việc."
            ),
            "score": 0.33,
            "metadata": {"source": "Demo note", "type": "note"},
            "source": "demo",
        },
    ]
    return demo_chunks[:top_k]


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    *,
    llm_caller: Optional[Callable[..., str]] = None,
    retrieval_fn: Optional[Callable[..., list[dict]]] = None,
    config: Optional[GenerationConfig] = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt
        5. Call LLM nếu có API key; nếu không, fallback extractive answer
        6. Return answer + sources + prompt metadata
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    config = config or GenerationConfig()
    retrieval_fn = retrieval_fn or retrieve or _dummy_retrieve  # type: ignore

    try:
        chunks = retrieval_fn(query, top_k=top_k)  # type: ignore[misc]
    except Exception as exc:
        # Trong MVP, pipeline generation vẫn nên demo được dù retrieval chưa cấu hình.
        print(f"⚠ Retrieval failed ({exc}). Using demo fallback chunks.")
        chunks = _dummy_retrieve(query, top_k=top_k)

    chunks = chunks or []

    if not chunks:
        return {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "retrieval_source": "none",
            "context": "",
            "used_llm": False,
        }

    # Đảm bảo input cho LLM đã theo score giảm dần trước khi reorder.
    sorted_chunks = sorted(
        chunks,
        key=lambda c: float(c.get("score", 0.0)) if isinstance(
            c.get("score", 0.0), (int, float)) else 0.0,
        reverse=True,
    )
    reordered = reorder_for_llm(sorted_chunks[:top_k])
    context = format_context(reordered)

    user_message = f"""Context:
{context}

---

Question: {query}

Yêu cầu trả lời:
- Trả lời bằng tiếng Việt.
- Chỉ dùng Context ở trên.
- Mỗi ý factual phải có citation dạng [D1], [D2]...
- Nếu thiếu bằng chứng, ghi đúng câu: {INSUFFICIENT_EVIDENCE_MESSAGE}.
"""

    used_llm = False
    llm_caller = llm_caller or call_openai_llm

    try:
        answer = llm_caller(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            config=config,
        )
        used_llm = True
    except Exception as exc:
        print(
            f"⚠ LLM generation unavailable ({exc}). Using extractive fallback.")
        answer = generate_extractive_fallback(query, reordered)

    retrieval_source = "none"
    if chunks:
        retrieval_source = _safe_text(
            chunks[0].get("source"), "hybrid") or "hybrid"

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
        "context": context,
        "used_llm": used_llm,
        "generation_config": {
            "top_k": top_k,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "model": config.model,
        },
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA:\n{result['answer']}")
        print(
            f"\n[Sources: {len(result['sources'])} chunks | "
            f"via {result['retrieval_source']} | used_llm={result['used_llm']}]"
        )
