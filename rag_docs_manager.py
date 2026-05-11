# rag_docs_manager.py
"""
RAG tiêu chuẩn dùng ChromaDB + sentence-transformers.
- Mỗi document = 1 file nguyên vẹn (không chunk).
- Hỗ trợ .txt, .md, .py
- Tự động index khi file mới được thêm vào thư mục rag_docs/
- Dùng song song với ragdata/ hiện tại: kết quả inject vào prompt qua <rag_docs_context>
"""

import os
import time
import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import Optional

# ── Cấu hình ──────────────────────────────────────────────────────────────────
RAG_DOCS_DIR = "./rag_docs"
CHROMA_DB_DIR = "./rag_docs/.chromadb"
COLLECTION_NAME = "rag_docs_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80MB, nhanh, tốt cho tiếng Anh + code
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py"}
TOP_K = 3  # Số file trả về tối đa


# ── Singleton resources (load 1 lần) ──────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None
_last_rag_docs_fingerprint: Optional[str] = None
# Skip sync_index() body (filesystem scan + Chroma checks) between queries when index is warm.
_rag_sync_skip_until: float = 0.0
_RAG_SYNC_COOLDOWN_SEC = 60.0


def _rag_docs_fingerprint() -> str:
    """Stable signature of supported files under rag_docs/ (name + mtime) for sync skipping."""
    if not os.path.isdir(RAG_DOCS_DIR):
        return ""
    parts: list[str] = []
    try:
        names = sorted(os.listdir(RAG_DOCS_DIR))
    except OSError:
        return ""
    for name in names:
        fp = os.path.join(RAG_DOCS_DIR, name)
        if not os.path.isfile(fp):
            continue
        if os.path.splitext(name)[1] not in SUPPORTED_EXTENSIONS:
            continue
        try:
            parts.append(f"{name}:{os.path.getmtime(fp):.6f}")
        except OSError:
            parts.append(name)
    return "|".join(parts)


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("[RAGDocs] Loading embedding model...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def warmup_rag_embedder() -> None:
    """
    Load SentenceTransformer + Chroma client once at app startup so the first
    RAG query does not pay the full model cold-start cost.
    """
    _get_embedder()
    _get_collection()


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Helpers ───────────────────────────────────────────────────────────────────
def _file_hash(content: str) -> str:
    """MD5 của nội dung file — dùng để phát hiện thay đổi."""
    return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()


def _file_id(filepath: str) -> str:
    """ID duy nhất dựa trên tên file (relative path)."""
    return os.path.relpath(filepath, RAG_DOCS_DIR).replace(os.sep, "/")


def _read_file(filepath: str) -> Optional[str]:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        print(f"[RAGDocs] Cannot read {filepath}: {e}")
        return None


# ── Core API ──────────────────────────────────────────────────────────────────
def sync_index():
    """Duyệt thư mục rag_docs/ và cập nhật database với logic CHUNKING."""
    global _last_rag_docs_fingerprint, _rag_sync_skip_until
    fp_now = _rag_docs_fingerprint()
    collection = _get_collection()
    # Avoid re-scanning every file + Chroma lookups on each user query when nothing changed.
    if (
        _last_rag_docs_fingerprint == fp_now
        and fp_now != ""
        and collection.count() > 0
    ):
        _rag_sync_skip_until = time.monotonic() + _RAG_SYNC_COOLDOWN_SEC
        return
    embedder = _get_embedder()

    # Lấy danh sách file hiện có
    files = [
        f
        for f in os.listdir(RAG_DOCS_DIR)
        if os.path.isfile(os.path.join(RAG_DOCS_DIR, f))
        and os.path.splitext(f)[1] in SUPPORTED_EXTENSIONS
    ]

    for filename in files:
        file_path = os.path.join(RAG_DOCS_DIR, filename)
        file_id = hashlib.md5(filename.encode()).hexdigest()
        _, ext = os.path.splitext(filename)

        # Kiểm tra file đã được index chưa (có thể dùng hash nội dung để tối ưu hơn)
        existing = collection.get(where={"file_id": file_id})
        if existing and existing["ids"]:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # THỰC HIỆN CHUNKING
        chunks = _chunk_text(content)
        
        chunk_ids = [f"{file_id}_ch_{i}" for i in range(len(chunks))]
        chunk_metadatas = [
            {
                "file_id": file_id,
                "filename": filename,
                "extension": ext,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        # Sinh embeddings cho từng chunk
        embeddings = embedder.encode(chunks).tolist()

        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadatas,
        )
        print(f"[RAGDocs] Indexed {filename} split into {len(chunks)} chunks.")

    _last_rag_docs_fingerprint = fp_now
    _rag_sync_skip_until = time.monotonic() + _RAG_SYNC_COOLDOWN_SEC


def query_docs(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Tìm các file liên quan nhất với câu hỏi.
    Trả về list[dict] với keys: file_id, filename, content, score.
    Mỗi phần tử = 1 file nguyên vẹn (không chunk).
    """
    global _rag_sync_skip_until
    collection = _get_collection()
    now = time.monotonic()
    # Hot path: avoid sync_index (listdir + per-file Chroma checks) on every chat turn.
    if not (collection.count() > 0 and now < _rag_sync_skip_until):
        sync_index()
        collection = _get_collection()

    if collection.count() == 0:
        return []

    embedder = _get_embedder()
    query_embedding = embedder.encode(question, show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)  # cosine distance → similarity
        output.append(
            {
                "file_id": meta["file_id"],
                "filename": meta["filename"],
                "extension": meta.get("extension", ""),
                "content": doc,
                "similarity": similarity,
            }
        )

    return output


def build_rag_context(
    question: str, top_k: int = TOP_K, min_similarity: float = 0.25
) -> str:
    """
    Truy vấn RAG và trả về chuỗi context sẵn sàng inject vào prompt.
    Lọc bỏ kết quả có similarity < min_similarity.
    Trả về "" nếu không tìm thấy gì.
    """
    docs = query_docs(question, top_k=top_k)
    filtered_docs = [d for d in docs if d["similarity"] >= min_similarity]

    # Fallback: nếu query quá ngắn/chung chung làm similarity thấp,
    # vẫn trả về doc gần nhất để LLM có ngữ cảnh thay vì trả rỗng.
    if not filtered_docs and docs:
        filtered_docs = [docs[0]]
        print(
            f"[RAGDocs] No docs above similarity {min_similarity}. "
            f"Fallback to top-1 ({docs[0]['filename']}, sim={docs[0]['similarity']})."
        )

    if not filtered_docs:
        return ""

    parts = []
    for i, d in enumerate(filtered_docs, 1):
        parts.append(
            f"<rag_doc_{i} file=\"{d['file_id']}\" similarity=\"{d['similarity']}\">\n"
            f"{d['content']}\n"
            f"</rag_doc_{i}>"
        )

    context_block = "\n\n".join(parts)
    return f"\n<rag_docs_context>\n{context_block}\n</rag_docs_context>\n"


def list_indexed_files() -> list[dict]:
    """Trả về danh sách tất cả file đã index trong DB."""
    collection = _get_collection()
    result = collection.get(include=["metadatas"])
    return [
        {
            "file_id": m["file_id"],
            "filename": m["filename"],
            "extension": m.get("extension", ""),
        }
        for m in (result["metadatas"] or [])
    ]


def delete_file_from_index(file_id: str) -> bool:
    """Xóa 1 file khỏi index theo file_id."""
    collection = _get_collection()
    try:
        collection.delete(ids=[file_id])
        print(f"[RAGDocs] Deleted from index: {file_id}")
        return True
    except Exception as e:
        print(f"[RAGDocs] Error deleting {file_id}: {e}")
        return False

def _chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Chia nhỏ văn bản thành các đoạn (chunks) để embedding chính xác hơn.
    - chunk_size: Độ dài tối đa mỗi đoạn.
    - chunk_overlap: Đoạn gối đầu để không mất ngữ cảnh giữa 2 chunk.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks