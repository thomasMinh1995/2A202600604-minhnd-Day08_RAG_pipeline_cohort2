"""
Task 8 — PageIndex Vectorless RAG.

PageIndex là hướng Vectorless RAG: không bắt buộc dùng embedding/vector store,
mà tận dụng cấu trúc tài liệu (heading/section/page tree) để tìm vùng nội dung phù hợp.

MVP trong file này gồm 2 lớp:
1. PageIndex cloud adapter: dùng SDK `pageindex` nếu đã cài và có PAGEINDEX_API_KEY.
2. Local structural fallback: chạy được ngay với Markdown trong data/standardized/,
   mô phỏng vectorless retrieval bằng cấu trúc heading + lexical score.

Cài đặt cloud SDK nếu cần:
    pip install pageindex python-dotenv

.env:
    PAGEINDEX_API_KEY=...
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "").strip()
STANDARDIZED_DIR = Path(__file__).resolve(
).parent.parent / "data" / "standardized"


# =============================================================================
# Data model
# =============================================================================

@dataclass
class StructuralNode:
    """Một node cấu trúc trong document Markdown."""

    title: str
    content: str
    level: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.content}".strip()


# =============================================================================
# Utilities
# =============================================================================

_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, hỗ trợ tiếng Việt ở mức MVP."""
    return _TOKEN_RE.findall(text.lower())


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def discover_markdown_files(directory: Path = STANDARDIZED_DIR) -> list[Path]:
    """Tìm toàn bộ Markdown files trong thư mục standardized."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.md") if p.is_file())


def parse_markdown_to_nodes(md_text: str, metadata: Optional[dict[str, Any]] = None) -> list[StructuralNode]:
    """
    Parse Markdown thành các section nodes theo heading.

    Nếu document không có heading, toàn bộ nội dung được xem như 1 node.
    Đây là fallback vectorless local: truy xuất dựa trên cấu trúc section thay vì chunk cố định.
    """
    metadata = metadata or {}
    lines = md_text.splitlines()
    nodes: list[StructuralNode] = []

    current_title = metadata.get("filename", "Document")
    current_level = 1
    current_lines: list[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index, current_lines
        content = "\n".join(current_lines).strip()
        if not content and not current_title:
            return
        node_meta = dict(metadata)
        node_meta.update({"section_index": section_index,
                         "heading_level": current_level})
        nodes.append(
            StructuralNode(
                title=str(current_title).strip() or "Untitled section",
                content=content,
                level=current_level,
                metadata=node_meta,
            )
        )
        section_index += 1
        current_lines = []

    saw_heading = False
    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            if saw_heading or current_lines:
                flush()
            saw_heading = True
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines or not nodes:
        flush()

    return [node for node in nodes if node.searchable_text.strip()]


def load_structural_corpus(directory: Path = STANDARDIZED_DIR) -> list[StructuralNode]:
    """Load Markdown corpus thành các structural nodes."""
    corpus: list[StructuralNode] = []
    for md_file in discover_markdown_files(directory):
        rel_path = str(md_file.relative_to(directory)
                       ) if directory.exists() else md_file.name
        metadata = {
            "filename": md_file.name,
            "path": str(md_file),
            "relative_path": rel_path,
            "type": md_file.parent.name,
        }
        corpus.extend(parse_markdown_to_nodes(
            _safe_read_text(md_file), metadata))
    return corpus


# =============================================================================
# Local structural retrieval fallback
# =============================================================================

class LocalStructuralPageIndex:
    """
    Lightweight local fallback cho demo.

    Cơ chế:
    - Không dùng embedding/vector DB.
    - Parse Markdown theo heading thành node cấu trúc.
    - Score query với node bằng BM25-like lexical scoring.
    - Boost nếu query term xuất hiện trong title/heading.
    """

    def __init__(self, nodes: list[StructuralNode]):
        self.nodes = nodes
        self._tokenized_docs = [
            tokenize(node.searchable_text) for node in nodes]
        self._doc_freq = self._build_doc_freq(self._tokenized_docs)
        self._avgdl = (
            sum(len(tokens)
                for tokens in self._tokenized_docs) / len(self._tokenized_docs)
            if self._tokenized_docs
            else 0.0
        )

    @staticmethod
    def _build_doc_freq(tokenized_docs: list[list[str]]) -> dict[str, int]:
        df: dict[str, int] = defaultdict(int)
        for tokens in tokenized_docs:
            for token in set(tokens):
                df[token] += 1
        return dict(df)

    def _bm25_score(self, query_tokens: list[str], doc_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> float:
        if not query_tokens or not doc_tokens or not self.nodes:
            return 0.0

        tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        total_docs = len(self.nodes)
        score = 0.0

        for token in query_tokens:
            if token not in tf:
                continue
            df = self._doc_freq.get(token, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denom = tf[token] + k1 * \
                (1 - b + b * doc_len / (self._avgdl or 1.0))
            score += idf * (tf[token] * (k1 + 1)) / denom

        return score

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = tokenize(query)
        scored: list[tuple[float, StructuralNode]] = []

        for node, doc_tokens in zip(self.nodes, self._tokenized_docs):
            score = self._bm25_score(query_tokens, doc_tokens)

            title_tokens = set(tokenize(node.title))
            title_hits = sum(
                1 for token in query_tokens if token in title_tokens)
            if title_hits:
                score += 0.25 * title_hits

            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "content": node.content or node.title,
                "score": float(score),
                "metadata": {**node.metadata, "title": node.title, "level": node.level},
                "source": "pageindex_local_structural",
            }
            for score, node in scored[: max(top_k, 0)]
        ]


# =============================================================================
# PageIndex cloud adapter
# =============================================================================

def _get_pageindex_client(api_key: Optional[str] = None) -> Any:
    """
    Tạo PageIndex client nếu SDK đã được cài.

    SDK/API có thể thay đổi, nên adapter này thử các class phổ biến thay vì hard-code
    một implementation duy nhất. Nếu không có SDK/key, raise lỗi rõ ràng.
    """
    resolved_key = (api_key or PAGEINDEX_API_KEY).strip()
    if not resolved_key:
        raise RuntimeError(
            "Missing PAGEINDEX_API_KEY. Set it in .env or pass api_key.")

    try:
        import pageindex  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: pip install pageindex") from exc

    client_cls = getattr(pageindex, "PageIndex", None) or getattr(
        pageindex, "Client", None)
    if client_cls is None:
        raise RuntimeError(
            "Cannot find PageIndex client class in installed `pageindex` package.")

    try:
        return client_cls(api_key=resolved_key)
    except TypeError:
        return client_cls(resolved_key)


def _call_first_available(obj: Any, method_names: Iterable[str], **kwargs: Any) -> Any:
    """Call method đầu tiên tồn tại trên SDK client."""
    last_error: Optional[Exception] = None
    for name in method_names:
        method = getattr(obj, name, None)
        if method is None:
            continue
        try:
            return method(**kwargs)
        except TypeError as exc:
            last_error = exc
            # Một số SDK dùng positional args cho content/query.
            try:
                if "content" in kwargs:
                    return method(kwargs["content"], metadata=kwargs.get("metadata"))
                if "query" in kwargs:
                    return method(kwargs["query"], top_k=kwargs.get("top_k"))
            except TypeError as positional_exc:
                last_error = positional_exc
    if last_error:
        raise last_error
    raise RuntimeError(
        f"No compatible method found. Tried: {', '.join(method_names)}")


def upload_documents(directory: Path = STANDARDIZED_DIR, api_key: Optional[str] = None, dry_run: bool = False) -> list[dict]:
    """
    Upload toàn bộ Markdown documents lên PageIndex.

    Args:
        directory: Thư mục chứa standardized Markdown files.
        api_key: Override PAGEINDEX_API_KEY nếu muốn.
        dry_run: True để chỉ preview danh sách file, không gọi API.

    Returns:
        List trạng thái upload theo từng file.
    """
    md_files = discover_markdown_files(directory)
    if not md_files:
        return []

    client = None if dry_run else _get_pageindex_client(api_key)
    uploaded: list[dict] = []

    for md_file in md_files:
        content = _safe_read_text(md_file)
        metadata = {
            "filename": md_file.name,
            "path": str(md_file),
            "relative_path": str(md_file.relative_to(directory)),
            "type": md_file.parent.name,
        }

        response = None
        if not dry_run:
            response = _call_first_available(
                client,
                ("upload", "upload_document", "add_document", "index_document"),
                content=content,
                metadata=metadata,
            )

        uploaded.append({"file": str(md_file), "metadata": metadata,
                        "response": response, "dry_run": dry_run})
        print(f"  ✓ {'Found' if dry_run else 'Uploaded'}: {md_file.name}")

    return uploaded


def _normalize_pageindex_result(result: Any) -> dict:
    """Normalize result object/dict từ SDK về schema chung của pipeline."""
    if isinstance(result, dict):
        content = result.get("content") or result.get(
            "text") or result.get("document") or ""
        score = result.get("score") or result.get(
            "relevance_score") or result.get("similarity") or 0.0
        metadata = result.get("metadata") or {}
    else:
        content = getattr(result, "content", None) or getattr(
            result, "text", None) or getattr(result, "document", "")
        score = getattr(result, "score", None) or getattr(
            result, "relevance_score", None) or 0.0
        metadata = getattr(result, "metadata", None) or {}

    return {
        "content": str(content),
        "score": float(score or 0.0),
        "metadata": metadata if isinstance(metadata, dict) else {"raw_metadata": metadata},
        "source": "pageindex",
    }


def pageindex_search(
    query: str,
    top_k: int = 5,
    *,
    api_key: Optional[str] = None,
    use_cloud: bool = True,
    fallback_to_local: bool = True,
    directory: Path = STANDARDIZED_DIR,
) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex cloud; fallback local structural nếu chưa cấu hình API.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': str}
    """
    if top_k <= 0:
        return []

    if use_cloud:
        try:
            client = _get_pageindex_client(api_key)
            raw_results = _call_first_available(
                client,
                ("query", "search", "retrieve"),
                query=query,
                top_k=top_k,
            )
            if isinstance(raw_results, dict):
                raw_results = raw_results.get(
                    "results", raw_results.get("data", []))
            return [_normalize_pageindex_result(item) for item in list(raw_results)[:top_k]]
        except Exception as exc:
            if not fallback_to_local:
                raise
            print(
                f"⚠ PageIndex cloud unavailable, fallback local structural search. Reason: {exc}")

    local_index = LocalStructuralPageIndex(load_structural_corpus(directory))
    return local_index.search(query=query, top_k=top_k)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Chưa set PAGEINDEX_API_KEY. Sẽ chạy local structural fallback.")
        print("  Khi cần cloud API: đăng ký tại https://pageindex.ai/ và thêm PAGEINDEX_API_KEY vào .env")

    nodes = load_structural_corpus(STANDARDIZED_DIR)
    if not nodes:
        print(f"⚠ Không tìm thấy Markdown corpus tại: {STANDARDIZED_DIR}")
        print("  Demo bằng dummy Markdown data...\n")
        demo_md = """
# Luật ma túy
## Điều 248: Tàng trữ trái phép chất ma túy
Người nào tàng trữ trái phép chất ma túy thì tùy khối lượng và tính chất có thể bị xử lý hình sự.
## Tin tức nghệ sĩ
Nghệ sĩ X bị bắt vì sử dụng ma túy trong một vụ việc giải trí.
## Hình phạt
Một số trường hợp có thể bị phạt tù từ 2 năm đến 7 năm.
"""
        demo_nodes = parse_markdown_to_nodes(
            demo_md, {"filename": "demo.md", "type": "demo"})
        local_index = LocalStructuralPageIndex(demo_nodes)
        results = local_index.search(
            "hình phạt tàng trữ trái phép chất ma túy", top_k=3)
    else:
        print(f"Loaded {len(nodes)} structural nodes from {STANDARDIZED_DIR}")
        results = pageindex_search(
            "hình phạt sử dụng ma tuý", top_k=3, fallback_to_local=True)

    print("\nTest query results:")
    for r in results:
        title = r.get("metadata", {}).get("title", "")
        print(f"[{r['score']:.3f}] {title} — {r['content'][:100]}...")
