# RAG Study Material Generator — Backend API

<div align="center">

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Red?style=for-the-badge&logo=qdrant&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Black?style=for-the-badge&logo=groq&logoColor=white)

**AI-powered study material generation using RAG (Retrieval-Augmented Generation)**

[Live Demo](https://rag-based-study-material-generator.vercel.app/) | [Report Issues](https://github.com/Kamalakannan-1208/RAG-Based-Study-Material-Generator-Backend/issues)

</div>

---

## 📖 Overview

The RAG Study Material Generator is an intelligent backend API that transforms educational documents into comprehensive study materials. Upload a syllabus document, and the system automatically generates topics, retrieves relevant information from a knowledge base, and produces detailed explanations — all powered by state-of-the-art AI.

### Key Features

- 📄 **Document Processing** — Upload PDF or DOCX files (syllabus)
- 🧠 **AI Topic Generation** — Automatically extract key topics using LLaMA-3 8B via Groq
- 🔍 **Semantic Search** — Retrieve relevant content using Qdrant vector database
- 📝 **Smart Q&A Generation** — Generate detailed answers for each topic
- 📑 **PDF Report Export** — Download comprehensive study materials as PDF
- 🔬 **LangSmith Integration** — Full observability and tracing for LLM operations
- ⚡ **High Performance** — Optimized embeddings with ONNX runtime

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Upload PDF    │───▶│  Extract Topics  │───▶│  Vector Search  │
│   (Syllabus)    │    │  (Groq LLaMA-3)  │    │   (Qdrant)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Download PDF   │◀───│  Generate Report │◀───│  Generate Q&A   │
│     Report      │    │   (ReportLab)    │    │  (Groq LLaMA-3) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Tech Stack

| Component       | Technology                              |
|-----------------|-----------------------------------------|
| **API Framework**   | FastAPI                                 |
| **LLM**             | LLaMA-3 8B (via Groq — ~800 tokens/sec) |
| **Embeddings**      | `BAAI/bge-small-en-v1.5` (fastembed - ONNX) |
| **Vector DB**       | Qdrant Cloud                            |
| **PDF/DOCX Parse**  | pdfplumber + python-docx                |
| **PDF Generation**  | ReportLab                               |
| **Observability**   | LangSmith                               |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uv for package management
- Qdrant Cloud account ([Get started free](https://cloud.qdrant.io/))
- Groq API key ([Get free key](https://console.groq.com/))

### 1. Clone the Repository

```bash
git clone https://github.com/Kamalakannan-1208/RAG-Based-Study-Material-Generator-Backend.git
cd RAG-Based-Study-Material-Generator-Backend
```

### 2. Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Or using uv (recommended for faster installs):
```bash
uv pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file and configure your keys:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```dotenv
# Qdrant Configuration
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=study_material_generator

# Groq Configuration
GROQ_API_KEY=your_groq_api_key

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# LangSmith Tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=study-material-generator
```

### 4. Run the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📡 API Reference

### Core Endpoints

#### 1. Upload & Generate Report (Single-Step)

```http
POST /upload-and-generate-report
Content-Type: multipart/form-data
```

**Description**: Complete pipeline — upload syllabus, generate topics, query knowledge base, and return study material.

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | PDF or DOCX syllabus (max 2 pages for topic extraction) |
| `document_title` | String | "Study Material" | Title for the generated Material |
| `top_k` | Integer | 4 | Number of context chunks to retrieve  |
| `return_pdf` | Boolean | false | If true, returns PDF directly; otherwise returns JSON |

**Response (JSON)**:
```json
{
  "filename": "syllabus.pdf",
  "document_title": "Computer Science 101",
  "topics": ["Machine Learning", "Data Structures", "Algorithms"],
  "total_chunks_extracted": 12,
  "total_topics": 3,
  "qa_summary": [
    {
      "topic": "Machine Learning",
      "answer_preview": "Machine learning is a subset of AI..."
    }
  ]
}
```

**Response (PDF)**: Returns `application/pdf` stream with the complete study material report.

---

#### 2. Upload to Knowledge Base

```http
POST /upload-to-knowledge-base
Content-Type: multipart/form-data
```

**Description**: Add documents to the knowledge base collection for RAG queries.

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | PDF document to add to knowledge base |
| `document_title` | String | (filename) | Optional title for the document |

**Response**:
```json
{
  "status": "success",
  "filename": "textbook.pdf",
  "document_title": "Introduction to AI",
  "total_chunks": 45,
  "total_points_uploaded": 45,
  "collection_stats": {
    "vectors_count": 1234,
    "status": "green"
  }
}
```

---

#### 3. Get Knowledge Base Stats

```http
GET /knowledge-base/stats
```

**Description**: Retrieve statistics about the vector database collection.

**Response**:
```json
{
  "vectors_count": 1234,
  "status": "green",
  "indexed_vectors_count": 1234
}
```

---

#### 4. Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "ok"
}
```

---

#### 5. Download Report

```http
GET /download-report/{filename}
```

**Description**: Download a previously generated PDF report.

---

## 🔧 Configuration

### Qdrant Setup

1. Create a free cluster at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Copy your cluster URL and API key
3. The collection is auto-created on first use

**Important**: Use port `6333` (not `633`) for Qdrant Cloud REST API.

### Groq Setup

1. Sign up at [Groq Console](https://console.groq.com/)
2. Create an API key
3. The system uses `llama3-8b` model by default

### LangSmith Observability

Enable detailed tracing for debugging and monitoring:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=study-material-generator
```

**What gets traced**:
- Topic generation LLM calls
- RAG query retrieval and answer generation
- Latency and token usage metrics
- Error tracking and exceptions

View traces at: https://smith.langchain.com/

---

## 📂 Project Structure

```
RAG-Based-Study-Material-Generator-Backend/
├── main.py                    # FastAPI application & API endpoints
├── services/
│   ├── __init__.py
│   ├── document_processor.py  # PDF/DOCX text extraction
│   ├── topic_generator.py     # LLM-based topic extraction
│   ├── rag_service.py         # RAG query engine
│   ├── vector_store.py        # Qdrant vector database operations
│   ├── pdf_report.py          # PDF report generation
│   └── langsmith_config.py    # LangSmith tracing configuration
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic data models
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project metadata
├── .env.example              # Environment template
└── README.md                 # This file
```

---

## 🧪 Testing the API

### Using cURL

**Upload and generate report**:
```bash
curl -X POST "http://localhost:8000/upload-and-generate-report" \
  -F "file=@syllabus.pdf" \
  -F "document_title=CS101" \
  -F "top_k=4" \
  -F "return_pdf=false"
```

**Upload to knowledge base**:
```bash
curl -X POST "http://localhost:8000/upload-to-knowledge-base" \
  -F "file=@textbook.pdf" \
  -F "document_title=AI Fundamentals"
```

### Using Python Requests

```python
import requests

# Upload and generate report
with open('syllabus.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/upload-and-generate-report',
        files={'file': f},
        data={
            'document_title': 'My Course',
            'top_k': 4,
            'return_pdf': False
        }
    )
    print(response.json())
```

---

## 🏎️ Performance

- **Topic Generation**: ~2-5 seconds (depending on syllabus length)
- **RAG Queries**: ~1-3 seconds per topic (parallel processing)
- **Embedding**: Runs locally on CPU (~50ms per chunk)
- **LLM Inference**: ~800 tokens/second via Groq

**Optimization Tips**:
- Keep syllabus uploads under 2 pages for faster topic extraction
- Adjust `top_k` (1-10) to balance speed vs. answer quality
- Use LangSmith to identify bottlenecks

---

## 🔒 Security Considerations

- CORS is enabled for all origins (configure for production)
- API keys should be stored securely in environment variables
- File uploads are validated (PDF/DOCX only)
- Temporary files are cleaned up after processing
- Consider adding rate limiting for production use

---


## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Groq](https://groq.com/) — Lightning-fast LLM inference
- [Qdrant](https://qdrant.tech/) — Vector similarity search engine
- [LangChain](https://python.langchain.com/) — LLM application framework
- [LangSmith](https://smith.langchain.com/) — LLM observability platform

---

## 📞 Support
- **Live Demo**: https://rag-based-study-material-generator.vercel.app/

---

<div align="center">

**Built with ❤️ using FastAPI, Groq, and Qdrant**

[⬆ Back to Top](#rag-study-material-generator---backend-api)

</div>