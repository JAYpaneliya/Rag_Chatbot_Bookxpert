"""
query.py — Interactive Q&A Bot (CLI)
Retrieves relevant chunks from ChromaDB and generates answers via Ollama (free, local LLM).
Falls back to a clean context-only extractive answer if Ollama is not running.
"""

import sys
import json
import textwrap
import requests
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

# ─── Config ─────────────────────────────────────────────────────────────────
VECTORSTORE_DIR  = Path(__file__).parent.parent / "vectorstore"
COLLECTION_NAME  = "rag_documents"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
TOP_K            = 5       # number of chunks to retrieve (configurable)
OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"    # change to "mistral", "phi3", etc. if preferred
MAX_CONTEXT_CHARS = 3000

# ─── LLM Backends ────────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    """Call local Ollama instance (completely free)."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return None   # Ollama not running
    except Exception as e:
        return None


def extractive_fallback(context: str, question: str) -> str:
    """
    Simple extractive fallback when no LLM is available.
    Scores sentences by keyword overlap with the question.
    """
    q_words = set(question.lower().split())
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 20]
    scored = []
    for s in sentences:
        s_words = set(s.lower().split())
        score = len(q_words & s_words)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:5]]
    if not top:
        return "I could not find a relevant answer in the provided documents."
    return ". ".join(top) + "."


def build_prompt(question: str, context_chunks: list) -> str:
    context_text = "\n\n---\n\n".join(
        [f"[Source: {c['source']} | Page {c['page']}]\n{c['text']}" for c in context_chunks]
    )
    # Trim to max context length
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n...[truncated]"

    prompt = f"""You are a helpful document assistant. Answer the question below using ONLY the provided context.
If the answer is not present in the context, say: "I don't have enough information in the provided documents to answer this."
Do NOT use any outside knowledge.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""
    return prompt

# ─── Retrieval ───────────────────────────────────────────────────────────────

def retrieve(collection, model, question: str, top_k: int = TOP_K):
    q_embedding = model.encode([question], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":     results["documents"][0][i],
            "source":   results["metadatas"][0][i]["source"],
            "page":     results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i],
        })
    return chunks

# ─── Display ─────────────────────────────────────────────────────────────────

def print_answer(question: str, answer: str, chunks: list, used_llm: str):
    print("\n" + "═" * 70)
    print(f"Q: {question}")
    print("─" * 70)
    print("ANSWER:")
    print(textwrap.fill(answer, width=70))
    print("\nSOURCES:")
    seen = set()
    for c in chunks:
        key = f"{c['source']} | Page {c['page']}"
        if key not in seen:
            seen.add(key)
            relevance = 1 - c["distance"]   # cosine similarity
            print(f"  • {key}  (relevance: {relevance:.2f})")
    print(f"\n[Backend: {used_llm}]")
    print("═" * 70 + "\n")

# ─── Main Loop ───────────────────────────────────────────────────────────────

def main():
    print("\n=== RAG Document Q&A Bot ===")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")

    # Load vector store
    if not VECTORSTORE_DIR.exists():
        print("[ERROR] Vector store not found. Run ingest.py first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"[ERROR] Collection '{COLLECTION_NAME}' not found. Run ingest.py first.")
        sys.exit(1)

    print(f"Loaded vector store: {collection.count()} chunks indexed.")

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Ready!\n")

    # Check if Ollama is running
    ollama_available = call_ollama("Hello") is not None
    if ollama_available:
        print(f"✓ Ollama detected — using model: {OLLAMA_MODEL}")
    else:
        print("⚠ Ollama not running — using extractive fallback (no LLM).")
        print("  To enable full LLM answers: install Ollama and run `ollama run llama3`\n")

    # Interactive loop
    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Retrieve
        chunks = retrieve(collection, model, question)

        # Generate answer
        if ollama_available:
            prompt = build_prompt(question, chunks)
            answer = call_ollama(prompt)
            if answer is None:
                answer = extractive_fallback(
                    "\n".join(c["text"] for c in chunks), question
                )
                used_llm = "Extractive Fallback (Ollama disconnected)"
            else:
                used_llm = f"Ollama ({OLLAMA_MODEL})"
        else:
            context = "\n".join(c["text"] for c in chunks)
            answer = extractive_fallback(context, question)
            used_llm = "Extractive Fallback"

        print_answer(question, answer, chunks, used_llm)


if __name__ == "__main__":
    main()
