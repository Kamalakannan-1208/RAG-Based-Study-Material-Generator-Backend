import os
import time
from groq import AsyncGroq
from services.vector_store import VectorStoreService
from services.langsmith_config import (
    configure_langsmith, 
    traced_llm, 
    traced_retriever,
    traced_api_call,
    traced_business_logic,
    track_resource_utilization,
    log_metric,
    track_error,
)
from typing import List, Dict

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

# Configure LangSmith on module import
configure_langsmith()


class RAGService:
    """RAG service for querying topics against the knowledge base with enhanced tracing."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=GROQ_API_KEY)
        self.vector_store = VectorStoreService()

    @traced_business_logic(
        name="rag_query_pipeline",
        tags=["rag", "query", "retrieval_augmented_generation"],
        metadata={"model": GROQ_MODEL}
    )
    async def query(self, query: str, top_k: int = 4) -> Dict:
        """
        Complete RAG pipeline:
        1. Retrieve top-k relevant chunks from the knowledge base in Qdrant.
        2. Build a prompt with retrieved context.
        3. Generate answer via Groq LLaMA-8B.
        
        Args:
            query: The topic or question to search for
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dictionary with query, context_chunks, and answer
        """
        start_time = time.time()
        
        # Step 1: Retrieve from knowledge base (traced)
        chunks = self._search_knowledge_base(query=query, top_k=top_k)
        
        if not chunks:
            track_error(
                error_type="no_results_found",
                error_message=f"No relevant content found for query: {query[:100]}...",
                context={"query_length": len(query), "top_k": top_k}
            )
            return {
                "query": query,
                "context_chunks": [],
                "answer": "No relevant content found in the knowledge base for this query.",
            }
        
        # Log retrieval metrics
        log_metric("retrieved_chunks", len(chunks), unit="count")
        
        # Step 2: Build context
        context = "\n\n---\n\n".join(
            [f"[Chunk {i+1}]:\n{chunk}" for i, chunk in enumerate(chunks)]
        )
        
        # Log context metrics
        log_metric("context_length", len(context), unit="chars")
        
        # Step 3: Generate answer via Groq (traced)
        answer = await self._generate_answer(query=query, context=context)
        
        # Log total pipeline metrics
        total_duration = time.time() - start_time
        log_metric("rag_total_latency_ms", total_duration * 1000, unit="ms")
        log_metric("answer_length", len(answer), unit="chars")
        
        return {
            "query": query,
            "context_chunks": chunks,
            "answer": answer,
        }

    @traced_retriever(name="vector_store_search", tags=["qdrant", "retrieval"])
    def _search_knowledge_base(self, query: str, top_k: int) -> List[str]:
        """Search the knowledge base for relevant chunks with detailed tracing."""
        with track_resource_utilization("vector_search", tags=["qdrant", "embedding"]):
            start_time = time.time()
            
            chunks = self.vector_store.search(query=query, top_k=top_k)
            
            duration = time.time() - start_time
            log_metric("vector_search_latency_ms", duration * 1000, unit="ms")
            
            return chunks

    @traced_business_logic(
        name="answer_generation",
        tags=["llm", "answer_generation", "rag"],
        metadata={"model": GROQ_MODEL}
    )
    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using Groq LLM based on the retrieved context."""
        prompt = f"""You are an expert educational content creator and subject matter expert. Your task is to create comprehensive, well-structured study material based on the provided context + General knowledge.

**Your Role:**
- You are creating study material for students who need to understand this topic thoroughly
- Your answers should be educational, accurate, and easy to understand

**Response Guidelines:**
1. **Accuracy First**: Only use information from the provided context + your knowledge. Do not make assumptions.
2. **Comprehensive Coverage**: Address all aspects of the query using the available context
3. **Clear Structure**: Organize your response with:
   - A clear introduction that directly answers the query
   - Main content organized with headings or bullet points where appropriate
   - Key takeaways or summary if the content is lengthy
4. **Academic Tone**: Use formal, educational language suitable for study materials
5. **Handle Missing Information**: If the context doesn't contain enough information to fully answer the query:
   - Try explain using your knowledge . donot give false assumetions.
**Formatting Instructions:**
- Use **bold** for key terms and concepts
- Use bullet points (•) for lists of items
- Use numbered lists for sequential or ranked information
- Use clear paragraph breaks between different ideas
- Keep paragraphs focused and concise (3-5 sentences each)

**Context Chunks:**
{context}

**Question/Topic to Address:**
{query}

**Your Comprehensive Study Material Response:**"""

        try:
            response = await self._call_groq_api(prompt)
            return response
        except Exception as e:
            track_error(
                error_type="answer_generation_failed",
                error_message=str(e),
                context={"query_length": len(query), "context_length": len(context)}
            )
            return f"Error generating answer: {str(e)}"

    @traced_api_call(
        name="groq_answer_generation",
        service="groq",
        tags=["llm", "answer_generation"],
        metadata={"model": GROQ_MODEL}
    )
    async def _call_groq_api(self, prompt: str) -> str:
        """Internal method to call Groq API for answer generation with tracing."""
        with track_resource_utilization("groq_api_call", tags=["groq"]):
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise and knowledgeable study assistant. "
                            "Always base your answers on the provided context. "
                            "Provide complete, comprehensive answers."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=2048,
            )
            
            duration = time.time() - start_time
            
            # Log performance metrics
            log_metric("groq_answer_latency_ms", duration * 1000, unit="ms")
            if response.usage:
                log_metric("answer_prompt_tokens", response.usage.prompt_tokens, unit="tokens")
                log_metric("answer_completion_tokens", response.usage.completion_tokens, unit="tokens")
                log_metric("answer_total_tokens", response.usage.total_tokens, unit="tokens")
            
            return response.choices[0].message.content.strip()
