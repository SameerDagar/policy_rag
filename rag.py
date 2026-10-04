"""Core RAG pipeline: load -> chunk -> embed -> FAISS -> retrieve -> generate."""
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
INDEX_DIR = Path("index")
CHUNK_WORDS, CHUNK_OVERLAP = 150, 30
MIN_SCORE = 0.20  # below this cosine similarity, we refuse without calling the LLM

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are a company policy assistant. Answer ONLY using the numbered context \
passages provided. Rules:
1. If the context does not contain the answer, reply exactly: "{NO_ANSWER}"
2. Never use outside knowledge, guess, or fill gaps.
3. Cite the passages you used like [1] or [2][3] after each claim.
4. Be concise and quote numbers/limits exactly as written."""

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ---------- 1. Document input ----------
def load_document(path):
    """Return a list of (page_label, text) for .txt/.md/.pdf files."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [(f"p.{i+1}", p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
    return [("", path.read_text(encoding="utf-8", errors="ignore"))]


# ---------- 2. Chunking ----------
def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step) if words[i:i + size]]


def build_chunks(paths):
    chunks = []
    for path in paths:
        for page, text in load_document(path):
            for c in chunk_text(text):
                chunks.append({"text": c, "source": Path(path).name, "page": page})
    return chunks


# ---------- 3+4. Embeddings + FAISS ----------
def embed(texts):
    vecs = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def build_index(paths):
    chunks = build_chunks(paths)
    if not chunks:
        raise ValueError("No text found in the supplied documents.")
    index = faiss.IndexFlatIP(embed(["x"]).shape[1])  # inner product == cosine (normalized)
    index.add(embed([c["text"] for c in chunks]))
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    return index, chunks


def load_index():
    if not (INDEX_DIR / "faiss.index").exists():
        return None, None
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        return faiss.read_index(str(INDEX_DIR / "faiss.index")), pickle.load(f)


# ---------- 5. Semantic retrieval ----------
def retrieve(query, index, chunks, k=4):
    scores, ids = index.search(embed([query]), k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i != -1]


# ---------- 6+7. Strict-prompt generation ----------
def answer(query, index, chunks, k=4):
    hits = retrieve(query, index, chunks, k)
    if not hits or hits[0]["score"] < MIN_SCORE:
        return NO_ANSWER, hits
    context = "\n\n".join(
        f"[{i}] ({h['source']} {h['page']})\n{h['text']}" for i, h in enumerate(hits, 1)
    )
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return resp.content[0].text.strip(), hits
