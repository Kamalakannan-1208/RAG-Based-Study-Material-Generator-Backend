from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn
from dotenv import load_dotenv
from datetime import datetime
from langsmith import traceable

load_dotenv()

from services.document_processor import DocumentProcessor
from services.topic_generator import TopicGenerator
from services.rag_service import RAGService
from services.pdf_report import PDFReportService
from services.vector_store import VectorStoreService
from services.langsmith_config import (
    configure_langsmith,
    traced_business_logic,
    track_resource_utilization,
    log_metric,
)
import tempfile, os
import uuid
import io
import time

# Configure LangSmith tracing
configure_langsmith()

app = FastAPI(
    title="RAG Study Material Generator",
    description="Upload a PDF/DOCX syllabus, generate topics, query against knowledge base, and export study material as PDF.",
    version="1.0.0",
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

doc_processor = DocumentProcessor()
topic_generator = TopicGenerator()
rag_service = RAGService()
pdf_report = PDFReportService()
vector_store = VectorStoreService()


@app.post("/upload-and-generate-report")
@traced_business_logic(
    name="report_generation_pipeline",
    tags=["api", "report_generation", "rag_pipeline"],
    metadata={"endpoint": "/upload-and-generate-report"}
)
async def upload_and_generate_report(
    file: UploadFile = File(...),
    document_title: str = Form(default="Study Material"),
    top_k: int = Form(default=4, ge=1, le=10),
    return_pdf: bool = Form(default=False)
):
    """
    Single-step endpoint: Upload a syllabus PDF/DOCX → Generate topics → Query each topic 
    against the knowledge base → Create PDF study material report.
    
    The uploaded document is ONLY used to extract topics. The actual content comes from 
    the pre-existing knowledge base in Qdrant.
    
    Args:
        file: PDF or DOCX syllabus file
        document_title: Title for the report (default: "Study Material")
        top_k: Number of context chunks to retrieve per topic (1-10, default: 4)
        return_pdf: If True, returns PDF file directly. If False, returns JSON with metadata.
    
    Returns:
        If return_pdf=True: PDF file as application/pdf stream
        If return_pdf=False: JSON with topics, qa_summary, and metadata
    """
    allowed = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    if file.content_type not in allowed and not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files are supported.")

    # Save uploaded file temporarily for processing
    suffix = ".pdf" if file.filename.endswith(".pdf") else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        pipeline_start = time.time()
        
        # 1. Process document - extract and clean text (syllabus limited to 2 pages)
        with track_resource_utilization("document_processing", tags=["pdf", "docx"]):
            text, chunks = doc_processor.process(tmp_path, suffix, max_pages=2)

        if not text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from document.")

        log_metric("extracted_text_length", len(text), unit="chars")
        log_metric("chunks_created", len(chunks), unit="count")

        # 2. Generate topics from the syllabus using LLM
        with track_resource_utilization("topic_generation", tags=["llm", "groq"]):
            topics = await topic_generator.generate_topics(text)

        if not topics:
            raise HTTPException(status_code=422, detail="Could not generate topics from document.")
        
        log_metric("topics_generated", len(topics), unit="count")

        # 3. Query each topic against the knowledge base (Qdrant) - IN PARALLEL
        with track_resource_utilization("parallel_rag_queries", tags=["async", "parallel"]):
            import asyncio
            tasks = [rag_service.query(query=topic, top_k=top_k) for topic in topics]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            qa_pairs = []
            successful_queries = 0
            failed_queries = 0
            
            for topic, result in zip(topics, results):
                if isinstance(result, Exception):
                    # Handle individual topic failures gracefully
                    failed_queries += 1
                    qa_pairs.append({
                        "topic": topic,
                        "query": topic,
                        "context_chunks": [],
                        "answer": f"Error processing topic: {str(result)}",
                    })
                else:
                    successful_queries += 1
                    qa_pairs.append({
                        "topic": topic,
                        "query": topic,
                        "context_chunks": result["context_chunks"],
                        "answer": result["answer"],
                    })
            
            log_metric("successful_rag_queries", successful_queries, unit="count")
            log_metric("failed_rag_queries", failed_queries, unit="count")

        # 4. Generate PDF report with all Q&A
        with track_resource_utilization("pdf_generation", tags=["reportlab", "pdf"]):
            session_id = str(uuid.uuid4())
        
        # Sanitize filename for Content-Disposition header
        safe_filename = file.filename.replace(" ", "_").encode("ascii", "ignore").decode()
        pdf_filename = f"report_{safe_filename}_{session_id[:8]}.pdf"
        
        if return_pdf:
            # Generate PDF in memory and return directly
            pdf_bytes = pdf_report.build_report_bytes(
                session_id=session_id,
                document_title=document_title,
                qa_pairs=qa_pairs,
            )
            
            log_metric("pdf_size_bytes", len(pdf_bytes), unit="bytes")
            
            # Log total pipeline time
            total_duration = time.time() - pipeline_start
            log_metric("total_pipeline_latency_ms", total_duration * 1000, unit="ms")
            
            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=\"{pdf_filename}\""
                }
            )
        else:
            # Log total pipeline time
            total_duration = time.time() - pipeline_start
            log_metric("total_pipeline_latency_ms", total_duration * 1000, unit="ms")
            
            # Return JSON response with metadata (original behavior)
            return {
                "filename": file.filename,
                "document_title": document_title,
                "topics": topics,
                "total_chunks_extracted": len(chunks),
                "total_topics": len(topics),
                "qa_summary": [
                    {"topic": qa["topic"], "answer_preview": qa["answer"][:100] + "..."}
                    for qa in qa_pairs
                ]
            }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/download-report/{filename}")
async def download_report(filename: str):
    """Download the generated PDF report."""
    report_dir = "/tmp/rag_reports"
    file_path = os.path.join(report_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename,
    )


@app.post("/upload-to-knowledge-base")
async def upload_to_knowledge_base(
    file: UploadFile = File(...),
    document_title: str = Form(default=None),
):
    """
    Upload a PDF document to the knowledge collection book.
    
    The document will be:
    1. Processed and text extracted
    2. Split into chunks (chunk_size=1000, overlap=200)
    3. Embedded using HuggingFace SentenceTransformer (all-MiniLM-L6-v2)
    4. Stored in Qdrant vector database
    
    Args:
        file: PDF document to upload
        document_title: Optional title for the document (defaults to filename)
        
    Returns:
        Upload result with document info and chunk count
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for knowledge base upload.")

    # Save uploaded file temporarily for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 1. Process document - extract text and create chunks
        text, chunks = doc_processor.process(tmp_path, ".pdf")

        if not text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from document.")

        if not chunks:
            raise HTTPException(status_code=422, detail="No chunks could be created from the document.")

        # 2. Prepare metadata
        title = document_title if document_title else os.path.splitext(file.filename)[0]
        metadata = {
            "filename": file.filename,
            "document_title": title,
            "upload_date": datetime.utcnow().isoformat(),
            "total_pages": len(chunks),  # Approximate page count
        }

        # 3. Upload chunks to vector store
        point_ids = vector_store.upload_documents(chunks, metadata=metadata)

        # 4. Get collection stats
        stats = vector_store.get_collection_stats()

        return {
            "status": "success",
            "filename": file.filename,
            "document_title": title,
            "total_chunks": len(chunks),
            "total_points_uploaded": len(point_ids),
            "collection_stats": stats,
            "message": f"Successfully uploaded {len(chunks)} chunks to knowledge base."
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/knowledge-base/stats")
async def get_knowledge_base_stats():
    """
    Get statistics about the knowledge base collection.
    
    Returns:
        Collection statistics including vector count and status
    """
    try:
        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)