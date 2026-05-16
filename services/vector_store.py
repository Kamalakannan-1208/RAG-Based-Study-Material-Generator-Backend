import os
import time
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Distance,
    VectorParams,
    PointStruct,
)
from fastembed import TextEmbedding
import uuid

from services.langsmith_config import (
    configure_langsmith,
    traced_retriever,
    traced_api_call,
    traced_business_logic,
    track_resource_utilization,
    log_metric,
    track_error,
)


QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "")
VECTOR_DIM = 384   # BAAI/bge-small-en-v1.5 dimension
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed embedding model

# Configure LangSmith on module import
configure_langsmith()

# Singleton pattern for embedding model to avoid loading it multiple times
_embedding_model_instance = None

def get_embedding_model():
    """Get or create the singleton embedding model instance."""
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedding_model_instance


class VectorStoreService:
    """Service for managing the knowledge base in Qdrant.
    
    Supports both reading from and writing to Qdrant collections.
    Uses FastEmbed (ONNX-based) for lightweight, fast embeddings.
    """
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or COLLECTION_NAME
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
        )
        # Use singleton embedding model to avoid loading it multiple times
        self.encoder = get_embedding_model()
        self._ensure_collection()

    # ------------------------------------------------------------------
    def _ensure_collection(self):
        """Create the collection if it doesn't exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )

    # ------------------------------------------------------------------
    @traced_retriever(name="vector_search_operation", tags=["qdrant", "search", "embedding"])
    def search(self, query: str, top_k: int = 4) -> List[str]:
        """Search the knowledge base for relevant chunks with detailed tracing.
        
        Args:
            query: The topic or question to search for
            top_k: Number of results to return
            
        Returns:
            List of relevant text chunks from the knowledge base
        """
        start_time = time.time()
        
        # Generate embedding with tracing
        embedding_start = time.time()
        query_vec = list(self.encoder.embed([query]))[0].tolist()
        embedding_duration = time.time() - embedding_start
        
        log_metric("embedding_latency_ms", embedding_duration * 1000, unit="ms")
        
        # Search in Qdrant with tracing
        qdrant_start = time.time()
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        ).points
        qdrant_duration = time.time() - qdrant_start
        
        log_metric("qdrant_search_latency_ms", qdrant_duration * 1000, unit="ms")
        
        # Extract text from results
        texts = [hit.payload["text"] for hit in results]
        
        # Log search metrics
        total_duration = time.time() - start_time
        log_metric("total_search_latency_ms", total_duration * 1000, unit="ms")
        log_metric("search_results_count", len(texts), unit="count")
        
        # Log similarity scores if available
        if results:
            avg_score = sum(hit.score for hit in results) / len(results)
            log_metric("avg_similarity_score", avg_score, unit="score")
        
        return texts

    # ------------------------------------------------------------------
    def upload_documents(self, chunks: List[str], metadata: Dict[str, Any] = None) -> List[str]:
        """Upload document chunks to the knowledge base with detailed tracing.
        
        Args:
            chunks: List of text chunks to upload
            metadata: Optional metadata to attach to all chunks (e.g., filename, upload_date)
            
        Returns:
            List of point IDs for the uploaded chunks
        """
        if not chunks:
            return []
        
        start_time = time.time()
        
        # Log input metrics
        log_metric("chunks_to_upload", len(chunks), unit="count")
        total_chars = sum(len(c) for c in chunks)
        log_metric("total_upload_chars", total_chars, unit="chars")
        
        # Generate embeddings with tracing
        embedding_start = time.time()
        embeddings = [emb.tolist() for emb in self.encoder.embed(chunks)]
        embedding_duration = time.time() - embedding_start
        
        log_metric("batch_embedding_latency_ms", embedding_duration * 1000, unit="ms")
        log_metric("avg_embedding_per_chunk_ms", (embedding_duration * 1000) / len(chunks), unit="ms")
        
        # Create points for Qdrant
        points = []
        point_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            payload = {
                "text": chunk,
                "chunk_index": i,
            }
            if metadata:
                payload.update(metadata)
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upload to Qdrant with tracing
        qdrant_start = time.time()
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        qdrant_duration = time.time() - qdrant_start
        
        log_metric("qdrant_upsert_latency_ms", qdrant_duration * 1000, unit="ms")
        
        # Log total metrics
        total_duration = time.time() - start_time
        log_metric("total_upload_latency_ms", total_duration * 1000, unit="ms")
        
        return point_ids

    # ------------------------------------------------------------------
    def delete_by_metadata(self, filter_conditions: Dict[str, Any]) -> int:
        """Delete documents from the knowledge base by metadata filter.
        
        Args:
            filter_conditions: Dictionary of field-value pairs to match
            
        Returns:
            Number of points deleted
        """
        conditions = [
            FieldCondition(
                key=key,
                match=MatchValue(value=value),
            )
            for key, value in filter_conditions.items()
        ]
        
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=conditions
            ),
        )
        
        return result.deleted if result else 0

    # ------------------------------------------------------------------
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection information
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "points_count": info.points_count,
            "status": str(info.status),
        }
