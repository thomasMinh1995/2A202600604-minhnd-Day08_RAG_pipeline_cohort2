"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

CORPUS: list[dict] = []
bm25_model = None


def build_bm25_index():
    """
    Xây dựng BM25 index từ corpus lưu trong ChromaDB.
    """
    global CORPUS, bm25_model
    if CORPUS:
        return  # Đã load rồi

    print("Đang tải dữ liệu từ ChromaDB để xây dựng index BM25...")
    client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = client.get_collection("DrugLawDocs")

    # Lấy toàn bộ dữ liệu ra
    data = collection.get(include=["documents", "metadatas"])

    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    if not docs:
        print("Cảnh báo: Không tìm thấy dữ liệu trong ChromaDB!")
        return

    for d, m in zip(docs, metas):
        CORPUS.append({"content": d, "metadata": m})

    print(f"Bắt đầu tokenize {len(CORPUS)} chunks...")
    # Tokenize đơn giản bằng khoảng trắng (có thể dùng underthesea để xịn hơn)
    tokenized_corpus = [doc["content"].lower().split() for doc in CORPUS]

    print("Khởi tạo mô hình BM25Okapi...")
    bm25_model = BM25Okapi(tokenized_corpus)
    print("Xây dựng BM25 Index thành công!")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.
    """
    build_bm25_index()

    if bm25_model is None:
        return []

    tokenized_query = query.lower().split()
    scores = bm25_model.get_scores(tokenized_query)

    # Lấy top_k kết quả có điểm cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search(
        "Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
