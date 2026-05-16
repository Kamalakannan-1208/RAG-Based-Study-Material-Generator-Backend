# RAG Study Material Generator — FastAPI Backend

A production-ready RAG (Retrieval-Augmented Generation) backend that:

1. **Accepts PDF/DOCX uploads** (1–2 pages)
2. **Generates topics** from the document using LLaMA-3 8B via Groq
3. **Answers topic queries** using Qdrant vector search + LLM
4. **Exports a PDF report** with all topics, retrieved context, and LLM answers

---

## Stack

| Component       | Technology                        |
|-----------------|-----------------------------------|
| API Framework   | FastAPI                           |
| LLM             | LLaMA-3 8B (via Groq — ultra fast)|
| Embeddings      | `BAAI/bge-small-en-v1.5` (fastembed - ONNX, lightweight) |
| Vector DB       | Qdrant Cloud (clustered)          |
| PDF/DOCX Parse  | pdfplumber + python-docx          |
| PDF Report Gen  | ReportLab                         |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

Copy `.env` and fill in your keys:

```bash
cp .env .env.local
```

```dotenv
QDRANT_URL=https://...qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=Study_Material_Generator

# Free key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key
```

Load before running:
```bash
export $(cat .env | xargs)
```

### 3. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: **http://localhost:8000/docs**

---

## API Endpoints

### `POST /upload`

Upload a PDF or DOCX (max 2 pages). Returns topics and a session ID.

**Request:** `multipart/form-data`
- `file`: PDF or DOCX file

**Response:**
```json
{
  "session_id": "abc123...",
  "filename": "lecture.pdf",
  "topics": ["Topic 1", "Topic 2", ...],
  "total_chunks": 8
}
```

---

### `POST /query`

Query the uploaded document on a specific topic.

**Request:**
```json
{
  "session_id": "abc123...",
  "query": "Explain the water cycle",
  "top_k": 4
}
```

**Response:**
```json
{
  "session_id": "abc123...",
  "query": "Explain the water cycle",
  "context_chunks": ["chunk 1 text...", "chunk 2 text..."],
  "answer": "The water cycle consists of..."
}
```

---

### `POST /generate-report`

Run RAG for all topics and generate a downloadable PDF report.

**Request:**
```json
{
  "session_id": "abc123...",
  "topics": ["Topic 1", "Topic 2"],
  "document_title": "Biology Notes",
  "top_k": 4
}
```

**Response:**
```json
{
  "report_path": "/tmp/rag_reports/report_abc123_20241201_143000.pdf",
  "download_url": "/download-report/report_abc123_20241201_143000.pdf",
  "total_topics": 2
}
```

---

### `GET /download-report/{filename}`

Download the generated PDF report.

---

## Flow Diagram

```
Upload PDF/DOCX
      │
      ▼
Extract Text (pdfplumber / python-docx)
      │
      ├─► Chunk Text (sliding window, 400 chars, 80 overlap)
      │         │
      │         ▼
      │   Embed Chunks (BAAI/bge-small-en-v1.5 via fastembed)
      │         │
      │         ▼
      │   Upsert to Qdrant Cloud ──► session_id returned
      │
      └─► Generate Topics (Groq LLaMA-3 8B)
                │
                ▼
           Topics returned to client
                │
                ▼  (for each topic)
          Vector Search in Qdrant (filtered by session_id)
                │
                ▼
          Build Prompt with Context
                │
                ▼
          Groq LLaMA-3 8B → Answer
                │
                ▼
          Append to PDF Report (ReportLab)
                │
                ▼
          Download PDF
```

---

## LangSmith Tracing

This project integrates with [LangSmith](https://smith.langchain.com/) for observability and debugging of LLM calls.

### Enable LangSmith Tracing

Add the following to your `.env` file:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=study-material-generator
```

### What Gets Traced

- **Topic Generation**: LLM calls to extract topics from uploaded documents
- **RAG Queries**: Vector search retrieval and answer generation
- **Latency & Token Usage**: Performance metrics for all LLM operations
- **Error Tracking**: Automatic capture of exceptions and failures

### Viewing Traces

1. Get your API key from [LangSmith](https://smith.langchain.com/)
2. Set the environment variables as shown above
3. Run the application — traces will be automatically sent to LangSmith
4. View your traces at https://smith.langchain.com/

---

## Notes

- **Qdrant port:** The cloud URL uses port `6333` (REST). The original `.env` had `:633` — make sure to use `:6333`.
- **Groq model:** `llama3-8b-8192` — extremely fast (tokens/sec ~800+).
- **Embeddings run locally** on CPU — no external embedding API needed.
- Each upload creates a new `session_id` so documents don't mix in the vector store.
