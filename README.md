# Regulatory Compliance RAG Engine

A production-ready Python RAG pipeline that lets users upload regulatory PDFs and ask compliance questions, receiving answers **strictly grounded** in the uploaded documents with page-level citations and three deterministic anti-hallucination guards.

---

## Architecture Overview

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Ingestion                    │
│  SHA-256 dedup → PyPDFLoader → Chunk    │
│  → Assign chunk IDs → Embed → PGVector  │
└─────────────────────────┬───────────────┘
                          │ stored in PostgreSQL
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2 — Retrieval                                            │
│  Dense (HNSW cosine k=15) ─┐                                    │
│                             ├─ Union by chunk_id                │
│  Sparse (BM25 k=15)  ──────┘     │                             │
│                                  ▼                              │
│                           Guard 1: cosine ≥ 0.50 OR BM25 hit   │
│                                  │                              │
│                                  ▼                              │
│                    CrossEncoder rerank → top 6                  │
│                                  │                              │
│                           Guard 2: top score ≥ 0.0             │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────┐
│  Stage 3 — Augmentation                                          │
│  Format excerpts with Document + Page labels                     │
│  ChatPromptTemplate (system + <user_query> XML isolation)        │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────┐
│  Stage 4 — Generation                                            │
│  gemini-2.5-flash (temp=0) with fallbacks                        │
│  StrOutputParser                                                 │
│  Guard 3: Citation regex validator                               │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                            JSON Response
                    answer + citations[] + citations_verified
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Pipeline | LangChain (LCEL) |
| Embeddings | Google Gemini `text-embedding-004` (3072-dim) |
| LLM | `gemini-2.5-flash` → fallback chain |
| Vector store | PostgreSQL + pgvector (HNSW index) |
| Keyword retrieval | BM25 (rank-bm25) |
| Re-ranking | `BAAI/bge-reranker-v2-m3` (cross-encoder) |
| PDF parsing | PyPDFLoader (pypdf) |
| Chunking | tiktoken-aware RecursiveCharacterTextSplitter |

---

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with the **pgvector** extension installed
- A Google AI Studio API key with Gemini access

### Install pgvector on PostgreSQL

```sql
-- Run as superuser in psql
CREATE EXTENSION IF NOT EXISTS vector;
```

Or install from: https://github.com/pgvector/pgvector

---

## Setup

### 1. Clone and create a virtual environment

```powershell
cd "ERC Analyzer"
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
# CPU-only torch (saves ~2 GB vs CUDA build)
pip install torch==2.5.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
notepad .env   # fill in GOOGLE_API_KEY and POSTGRES_PASSWORD
```

Required variables:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio key |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `POSTGRES_DB` | Database name (default: `erc_analyzer`) |
| `POSTGRES_USER` | PostgreSQL user (default: `postgres`) |
| `POSTGRES_HOST` | Host (default: `localhost`) |

### 4. Create the PostgreSQL database

```powershell
# Run the setup script
psql -U postgres -f scripts/setup_db.sql
```

### 5. Run the server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `POST /documents/ingest`

Upload a regulatory PDF for ingestion.

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@gdpr.pdf"
```

**Response:**
```json
{
  "status": "INGESTED",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "file_name": "gdpr.pdf",
  "page_count": 88,
  "chunk_count": 312,
  "message": "Successfully ingested 'gdpr.pdf' — 88 pages, 312 chunks."
}
```

Returns `"status": "ALREADY_INGESTED"` if the same file (by SHA-256) was uploaded before — zero re-embedding cost.

---

### `POST /query`

Ask a compliance question.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the lawful bases for processing personal data?",
    "document_filter": ["gdpr.pdf"]
  }'
```

**Response:**
```json
{
  "answer": "Under Article 6, processing is lawful if the data subject has given consent [Document: gdpr.pdf, Page: 37]. Processing is also lawful for the performance of a contract [Document: gdpr.pdf, Page: 37].",
  "citations": [
    {
      "chunk_id": "3f2c8a...",
      "document_name": "gdpr.pdf",
      "page_number": 37,
      "similarity_score": 0.823,
      "excerpt": "Article 6 — Lawfulness of processing. Processing shall be lawful only if..."
    }
  ],
  "citations_verified": true,
  "query": "What are the lawful bases for processing personal data?"
}
```

**Error responses:**

| Code | Meaning |
|---|---|
| `404` | Guard 1 or Guard 2 fired — query has no relevant content in documents |
| `422` | Guard 3 fired — LLM answer contains unverifiable or zero citations |

---

### `GET /documents`

List all ingested documents.

```bash
curl http://localhost:8000/documents
```

---

### `DELETE /documents/{document_id}`

Remove a document and all its chunks from pgvector.

```bash
curl -X DELETE http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000
```

---

### `GET /health`

Liveness check. Returns database connectivity status.

---

## Anti-Hallucination Guards

The system has **three deterministic guards** — the LLM is never called if Guards 1 or 2 fail, and the answer is rejected if Guard 3 fails.

| Guard | Stage | Condition | HTTP code |
|---|---|---|---|
| **Guard 1** — Cosine Floor | After retrieval | Best cosine ≥ 0.50 OR any BM25 match | 404 |
| **Guard 2** — Cross-Encoder Floor | After reranking | Top cross-encoder score ≥ 0.0 | 404 |
| **Guard 3** — Citation Validator | After generation | ≥1 citation AND all citations in retrieved context | 422 |

---

## Running Tests

```powershell
pip install pytest pytest-asyncio
pytest tests/ -v
```

Tests are self-contained — no database or API keys required.

---

## Project Structure

```
ERC Analyzer/
├── app/
│   ├── main.py              # FastAPI app + lifespan (DB bootstrap)
│   ├── config.py            # Pydantic-settings (all knobs as env vars)
│   ├── db.py                # SQLAlchemy engine + documents table ORM
│   ├── models.py            # FastAPI request/response schemas
│   ├── exceptions.py        # Typed domain exceptions
│   ├── pipeline/
│   │   ├── ingestion.py     # Stage 1 — PDF → pgvector
│   │   └── rag_chain.py     # Stages 2–4 — Retrieval → Generation
│   └── routers/
│       ├── documents.py     # /documents endpoints
│       └── query.py         # /query endpoint
├── tests/
│   ├── test_ingestion.py    # Unit tests for ingestion logic
│   └── test_guards.py       # Unit tests for all three guards
├── scripts/
│   └── setup_db.sql         # One-time PostgreSQL bootstrap
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Tuning Parameters

All knobs are environment variables — no code changes needed:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1200 | Tiktoken chunk size (tokens) |
| `CHUNK_OVERLAP` | 200 | Overlap between adjacent chunks |
| `DENSE_TOP_K` | 15 | Candidates from HNSW cosine search |
| `SPARSE_TOP_K` | 15 | Candidates from BM25 |
| `RERANK_TOP_N` | 6 | Chunks kept after cross-encoder |
| `COSINE_FLOOR` | 0.50 | Guard 1 cosine threshold |
| `CROSS_ENCODER_FLOOR` | 0.0 | Guard 2 cross-encoder threshold |
| `LLM_PRIMARY` | gemini-2.5-flash | Primary Gemini model |
