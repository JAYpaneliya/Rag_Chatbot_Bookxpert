# 📚 RAG Document Q&A Bot

A fully local, **100% free** Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions against your own collection of documents and receive accurate, grounded answers with source citations — all running on your machine with no paid APIs.

---

## 🛠 Tech Stack

| Library / Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core language |
| PyMuPDF (`fitz`) | 1.24.5 | PDF text extraction with page metadata |
| python-docx | 1.1.2 | DOCX file parsing |
| sentence-transformers | 3.0.1 | Local embedding model (no API key needed) |
| torch | 2.3.1 | Backend for sentence-transformers |
| chromadb | 0.5.3 | Local persistent vector database |
| streamlit | 1.36.0 | Web UI (bonus interface) |
| requests | 2.32.3 | HTTP calls to local Ollama LLM |
| Ollama | latest | Free local LLM runner (phi3:mini / llama3.2) |
| Pillow | 10.x | Required by Streamlit for image handling |

---

## 🏗 Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        INDEXING PIPELINE                         │
│                        (run once via ingest.py)                  │
│                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌────────┐  │
│  │Documents │───▶│   Load    │───▶│   Chunk    │───▶│ Embed  │  │
│  │PDF/TXT/  │    │ & Extract │    │ 500 chars  │    │MiniLM  │  │
│  │DOCX      │    │   text    │    │ 80 overlap │    │L6-v2   │  │
│  └──────────┘    └───────────┘    └────────────┘    └───┬────┘  │
│                                                          │       │
│                                                          ▼       │
│                                                   ┌──────────┐  │
│                                                   │ ChromaDB │  │
│                                                   │ (on disk)│  │
│                                                   └──────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                            │
│                   (query.py or app.py)                           │
│                                                                  │
│  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌────────┐  │
│  │  User    │───▶│  Embed    │───▶│  Retrieve  │───▶│  LLM   │  │
│  │ Question │    │  Query    │    │  top-k     │    │Ollama  │  │
│  │          │    │ MiniLM    │    │  chunks    │    │phi3    │  │
│  └──────────┘    └───────────┘    └────────────┘    └───┬────┘  │
│                                                          │       │
│                                                          ▼       │
│                                               ┌──────────────┐  │
│                                               │ Answer with  │  │
│                                               │  Citations   │  │
│                                               └──────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Pipeline Description

1. **Document Ingestion** — PDF, TXT, and DOCX files are loaded from the `/data` folder. Text is extracted page-by-page, stripping noise like page numbers and short header/footer lines.

2. **Text Chunking** — Each page's text is split into overlapping character-level chunks (500 chars, 80-char overlap) to keep chunk sizes predictable and within the embedding model's token limit.

3. **Embedding** — All chunks are embedded in batches of 64 using `all-MiniLM-L6-v2`, a free model that runs entirely on CPU. Embeddings are L2-normalized for cosine similarity.

4. **Vector Storage** — Chunk embeddings and metadata (source filename, page number) are stored in a ChromaDB collection that persists to disk under `/vectorstore`. No re-indexing is needed on subsequent runs.

5. **Retrieval** — At query time, the question is embedded with the same model and a cosine similarity search retrieves the top-k most relevant chunks from ChromaDB.

6. **Answer Generation** — Retrieved chunks are assembled into a grounded prompt and sent to a local Ollama LLM. The model is explicitly instructed not to use outside knowledge. If Ollama is not running, an extractive keyword-scoring fallback is used automatically.

---

## ✂️ Chunking Strategy

**Strategy: Fixed-size character chunking with overlap**

- **Chunk size**: 500 characters
- **Overlap**: 80 characters

**Why this strategy was chosen:**

Fixed-size character chunking produces predictable, bounded chunks that stay well within the 512-token context limit of the `all-MiniLM-L6-v2` embedding model. The 80-character overlap ensures that sentences or ideas that fall on a chunk boundary are still represented in both adjacent chunks, preserving retrieval context.

Sentence-based chunking was considered but rejected because it produces highly variable chunk sizes — a document with many short bullet points creates dozens of tiny chunks, while a document with long paragraphs creates oversized ones. Paragraph-based chunking was also considered but is too dependent on consistent document formatting. Fixed-size chunking works reliably across all document types and is the most common strategy in production RAG systems.

---

## 🔢 Embedding Model & Vector Database

### Embedding Model: `all-MiniLM-L6-v2`

- **Why**: Completely free, runs on CPU with no GPU required, produces 384-dimensional dense vectors, and achieves strong semantic similarity performance on standard retrieval benchmarks (MTEB). It is the most widely used free embedding model for RAG systems and has an excellent speed/quality trade-off for documents of this scale.
- **Alternative considered**: `all-mpnet-base-v2` (768 dimensions, higher quality) — rejected because it is approximately 3× slower on CPU with marginal quality gain for short Q&A tasks.

### Vector Database: ChromaDB

- **Why**: Free and fully open-source, persists embeddings to disk automatically (no re-indexing on restart), has a simple and clean Python API, and supports cosine similarity natively via HNSW indexing. It requires zero infrastructure setup — no Docker, no server, no configuration files.
- **Alternative considered**: FAISS — powerful and fast, but requires manual serialization/deserialization for persistence and has a lower-level API. ChromaDB provides the same core functionality with less boilerplate for a project of this scope.

---

## ⚡ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-qa-bot.git
cd rag-document-qa-bot
```

### 2. Create and activate a Python virtual environment

```bash
python -m venv venv

# Mac / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
pip install Pillow   # required by Streamlit on some systems
```

### 4. Add your documents to the `/data` folder

Place 4–5 documents (PDF, TXT, or DOCX) into the `/data` folder. At least one must be a PDF. Each document must be at least 500 words.

```
data/
├── document1.pdf
├── document2.txt
├── document3.docx
├── document4.pdf
└── document5.txt
```

### 5. Index your documents

```bash
python src/ingest.py
```

This loads all documents, chunks them, embeds them in batches, and saves the vector store to disk. Run this **once** — or re-run whenever you add or change documents.

### 6. Install Ollama for full LLM answers (free, local — recommended)

1. Download from [https://ollama.com](https://ollama.com) and install the Mac/Windows/Linux app.
2. Pull the recommended model (runs well on 8 GB RAM):

```bash
ollama run phi3:mini
```

3. Keep Ollama running in a **separate terminal tab**. If you skip this step, the bot will automatically use an extractive fallback — it still works, just without fluent LLM-generated answers.

### 7. Run the bot

**Option A — Command Line (interactive loop):**

```bash
python src/query.py
```

**Option B — Web UI via Streamlit (bonus):**

```bash
python -m streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> ⚠️ Always use `python -m streamlit` on Mac with Anaconda installed — this ensures the command uses your venv's Python, not the system Anaconda installation.

---

## 🔑 Environment Variables

This project requires **no API keys**. All components run locally and are completely free.

If you wish to extend the project with a paid LLM backend in the future, create a `.env` file in the project root:

```
# .env — never commit this file
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
```

Then load it in your script with:

```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

The `.env` file is already listed in `.gitignore`. **Never commit actual API keys to your repository.**

---

## 💬 Example Queries

The following sample questions are designed for a knowledge base covering topics such as artificial intelligence, climate change, business strategy, and public health. Adjust these to match your actual documents.

| # | Sample Question | Expected Answer Theme |
|---|---|---|
| 1 | "What is the main argument made in the first document?" | Core thesis or central claim of the document |
| 2 | "What are the key causes of climate change according to the report?" | Emissions, deforestation, industrial activity |
| 3 | "What recommendations does the author make in the conclusion?" | Actionable suggestions from the document's final section |
| 4 | "How does the document define machine learning?" | Technical or layperson definition from the source text |
| 5 | "What statistics or data points are mentioned about global temperatures?" | Specific figures cited in the document |
| 6 | "What is the capital city of Jupiter?" | *(Out-of-scope)* Bot should respond: "I don't have enough information in the provided documents to answer this." |

---

## ⚠️ Known Limitations

1. **No cross-document synthesis** — The retrieval step returns the top-k chunks independently ranked by similarity. The bot cannot reason across two documents that use different terminology to describe the same concept.

2. **Fixed context window** — Only 3,000 characters of retrieved context are passed to the LLM per query. For very complex multi-part questions, some relevant context may be cut off.

3. **Scanned / image-based PDFs not supported** — PDFs that consist entirely of scanned images (with no embedded text layer) will produce empty extraction. Pre-process such files with an OCR tool like Tesseract before ingesting.

4. **Table and list fragmentation** — Fixed-size character chunking may split tables or multi-line lists across chunk boundaries, causing retrieved chunks to contain incomplete structured data.

5. **Extractive fallback quality** — When Ollama is not running, answers are produced by keyword-overlap sentence scoring. This is accurate (grounded in the document) but not fluent — it does not paraphrase or synthesize, it extracts.

6. **No incremental indexing** — Running `ingest.py` drops and rebuilds the entire collection. Adding a single new document requires re-indexing all documents. For large collections, a document-level hash check would be a valuable improvement.

7. **Single-language support** — The `all-MiniLM-L6-v2` model performs best on English text. Retrieval quality degrades on documents in other languages.

---

## 📁 Project Structure

```
rag-document-qa-bot/
├── data/                  ← Your PDF / TXT / DOCX documents go here
├── vectorstore/           ← Auto-created by ingest.py (ChromaDB files)
├── src/
│   ├── ingest.py          ← Indexing pipeline: load → chunk → embed → store
│   ├── query.py           ← CLI interactive Q&A loop
│   └── app.py             ← Streamlit web UI (bonus)
├── requirements.txt       ← All Python dependencies with versions
├── .gitignore             ← Excludes venv/, vectorstore/, .env
└── README.md              ← This file
```

---

## 📜 License

MIT — free to use, modify, and distribute.
