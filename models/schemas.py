from pydantic import BaseModel, Field
from typing import List, Optional


class TopicGenerationResponse(BaseModel):
    session_id: str
    filename: str
    topics: List[str]
    total_chunks: int


class QueryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from /upload response")
    query: str = Field(..., description="Topic or question to query")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of context chunks to retrieve")


class QueryResponse(BaseModel):
    session_id: str
    query: str
    context_chunks: List[str]
    answer: str


class GenerateReportRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from /upload response")
    topics: List[str] = Field(..., description="List of topics to generate answers for")
    document_title: Optional[str] = Field(default="Study Material", description="Title for the PDF report")
    top_k: int = Field(default=4, ge=1, le=10)


class GenerateReportResponse(BaseModel):
    report_path: str
    download_url: str
    total_topics: int
