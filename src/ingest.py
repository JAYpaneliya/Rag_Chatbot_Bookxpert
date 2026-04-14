"""
ingest.py — Document Ingestion, Chunking, Embedding, and Vector Store Indexing
Supports: PDF, TXT, DOCX
Uses: sentence-transformers (free local embeddings) + ChromaDB (free local vector store)
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import List, Dict

# Document loaders
import fitz  # PyMuPDF for PDF
import docx  # python-docx for DOCX

# Embeddings
from sentence_transformers import SentenceTransformer

# Vector store
import chromadb
from chromadb.config import Settings

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent.parent / "data"
VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Fast, free, 384-dim
CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 80    # overlap characters
BATCH_SIZE      = 64    # embed in batches

# ─── Text Extraction ─────────────────────────────────────────────────────────

def extract_text_pdf(path: Path) -> List[Dict]:
    """Extract text page-by-page from PDF."""
    pages = []
    doc = fitz.open(str(path))
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        # Strip common header/footer noise (short lines at top/bottom)
        lines = text.split("\n")
        lines = [l.strip() for l in lines if len(l.strip()) > 3]
        clean = "\n".join(lines)
        if clean:
            pages.append({"text": clean, "source": path.name, "page": page_num})
    doc.close()
    return pages


def extract_text_docx(path: Path) -> List[Dict]:
    """Extract text paragraph-by-paragraph from DOCX."""
    doc = docx.Document(str(path))
    # Group paragraphs into pseudo-pages (~40 paragraphs each)
    pages, buffer, page_num = [], [], 1
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            buffer.append(t)
        if len(buffer) >= 40:
            pages.append({"text": "\n".join(buffer), "source": path.name, "page": page_num})
            buffer = []
            page_num += 1
    if buffer:
        pages.append({"text": "\n".join(buffer), "source": path.name, "page": page_num})
    return pages


def extract_text_txt(path: Path) -> List[Dict]:
    """Extract text from plain TXT, split into 40-line pseudo-pages."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    pages, page_num = [], 1
    for i in range(0, len(lines), 40):
        chunk = "\n".join(lines[i:i+40]).strip()
        if chunk:
            pages.append({"text": chunk, "source": path.name, "page": page_num})
            page_num += 1
    return pages


def load_document(path: Path) -> List[Dict]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    elif ext == ".docx":
        return extract_text_docx(path)
    elif ext == ".txt":
        return extract_text_txt(path)
    else:
        print(f"  [SKIP] Unsupported format: {path.name}")
        return []

# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str, page: int) -> List[Dict]:
    """
    Strategy: Fixed-size character chunking with overlap.
    Rationale: Simple, predictable chunk sizes work well for sentence-transformer
    models with 512-token limits. Overlap preserves context at boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if len(chunk.strip()) > 30:   # skip near-empty chunks
            chunk_id = hashlib.md5(f"{source}-{page}-{start}".encode()).hexdigest()
            chunks.append({
                "id":     chunk_id,
                "text":   chunk.strip(),
                "source": source,
                "page":   page,
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ─── Main Indexing Pipeline ───────────────────────────────────────────────────

def index_documents():
    print("\n=== RAG Document Indexer ===\n")

    # 1. Find documents
    docs_paths = list(DATA_DIR.glob("**/*.pdf")) + \
                 list(DATA_DIR.glob("**/*.txt")) + \
                 list(DATA_DIR.glob("**/*.docx"))

    if not docs_paths:
        print(f"[ERROR] No documents found in {DATA_DIR}")
        print("  Add PDF/TXT/DOCX files to the /data folder and re-run.")
        sys.exit(1)

    print(f"Found {len(docs_paths)} document(s):")
    for p in docs_paths:
        print(f"  - {p.name}")

    # 2. Load + chunk all documents
    all_chunks = []
    for path in docs_paths:
        print(f"\nProcessing: {path.name}")
        pages = load_document(path)
        print(f"  Extracted {len(pages)} page(s)")
        for page_data in pages:
            chunks = chunk_text(page_data["text"], page_data["source"], page_data["page"])
            all_chunks.extend(chunks)
            print(f"  Page {page_data['page']}: {len(chunks)} chunk(s)")

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # 3. Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Model loaded.")

    # 4. Batch embed
    print(f"Embedding {len(all_chunks)} chunks in batches of {BATCH_SIZE}...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # 5. Store in ChromaDB (persisted to disk)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

    # Drop and recreate collection for clean re-index
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"\nDropped existing collection '{COLLECTION_NAME}' for fresh index.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches
    print("Inserting into ChromaDB...")
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i+BATCH_SIZE]
        batch_embs   = embeddings[i:i+BATCH_SIZE].tolist()
        collection.add(
            ids        = [c["id"]   for c in batch_chunks],
            documents  = [c["text"] for c in batch_chunks],
            embeddings = batch_embs,
            metadatas  = [{"source": c["source"], "page": c["page"]} for c in batch_chunks],
        )

    print(f"\n✓ Indexed {collection.count()} chunks into ChromaDB.")
    print(f"  Vector store saved to: {VECTORSTORE_DIR}")
    print("\nIndexing complete! You can now run query.py\n")


if __name__ == "__main__":
    index_documents()
