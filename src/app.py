"""
app.py — Streamlit Web UI for the RAG Q&A Bot (Bonus)
Run with: streamlit run src/app.py
"""

import requests
import textwrap
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

# ─── Config ─────────────────────────────────────────────────────────────────
VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini" 
TOP_K           = 5
MAX_CONTEXT_CHARS = 3000

# ─── Cached Resources ─────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return client.get_collection(COLLECTION_NAME)

# ─── Helpers ────────────────────────────────────────────────────────────────

def retrieve(collection, model, question: str, top_k: int):
    q_emb = model.encode([question], normalize_embeddings=True)[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
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


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']} | Page {c['page']}]\n{c['text']}" for c in chunks]
    )
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
    return f"""You are a helpful document assistant. Answer the question below using ONLY the provided context.
If the answer is not present in the context, say: "I don't have enough information in the provided documents to answer this."
Do NOT use any outside knowledge.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def call_ollama(prompt: str) -> str | None:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return None


def extractive_fallback(context: str, question: str) -> str:
    q_words = set(question.lower().split())
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 20]
    scored = sorted(sentences, key=lambda s: len(set(s.lower().split()) & q_words), reverse=True)
    top = scored[:5]
    return (". ".join(top) + ".") if top else "I could not find a relevant answer in the documents."

# ─── UI ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Document Q&A Bot", page_icon="📚", layout="wide")
st.title("📚 RAG Document Q&A Bot")
st.caption("Ask questions about your uploaded documents. Answers are grounded in the source material.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=TOP_K)
    st.markdown("---")
    st.markdown("**LLM Backend**")
    ollama_test = call_ollama("Hi")
    if ollama_test is not None:
        st.success(f"✅ Ollama running ({OLLAMA_MODEL})")
    else:
        st.warning("⚠️ Ollama not detected\nUsing extractive fallback")
    st.markdown("---")
    st.markdown("**How to use:**\n1. Run `ingest.py` first\n2. Ask questions below")

# Load resources
try:
    model      = load_model()
    collection = load_collection()
    st.info(f"✅ {collection.count()} chunks loaded from vector store.")
except Exception as e:
    st.error(f"❌ Could not load vector store: {e}\nPlease run `python src/ingest.py` first.")
    st.stop()

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Question input
with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Your question:", placeholder="What does the document say about...?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Searching documents..."):
        chunks = retrieve(collection, model, question, top_k)

    with st.spinner("Generating answer..."):
        prompt = build_prompt(question, chunks)
        answer = call_ollama(prompt)
        used_llm = f"Ollama ({OLLAMA_MODEL})"
        if answer is None:
            context = "\n".join(c["text"] for c in chunks)
            answer  = extractive_fallback(context, question)
            used_llm = "Extractive Fallback"

    st.session_state.history.append({
        "question": question,
        "answer":   answer,
        "chunks":   chunks,
        "backend":  used_llm,
    })

# Display history (newest first)
for entry in reversed(st.session_state.history):
    st.markdown(f"### ❓ {entry['question']}")
    st.success(entry["answer"])

    with st.expander("📄 Source Chunks & Citations"):
        for c in entry["chunks"]:
            relevance = 1 - c["distance"]
            st.markdown(f"**📌 {c['source']} — Page {c['page']}** *(relevance: {relevance:.2f})*")
            st.text(textwrap.fill(c["text"], width=80))
            st.markdown("---")

    st.caption(f"Backend: {entry['backend']}")
    st.markdown("---")
