"""
Enhanced LangSmith tracing configuration for the Study Material Generator.

This module sets up comprehensive LangSmith tracing to monitor and debug LLM calls,
business logic, resource utilization, and external API calls.

LangSmith provides observability for LLM applications by tracking:
- LLM inputs and outputs
- Token usage and latency
- Errors and exceptions
- Full execution traces
- Custom business logic spans
- Resource utilization metrics
- External API call monitoring
"""

import os
import time
import asyncio
import functools
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from contextlib import contextmanager

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


# Track if LangSmith has been configured to avoid duplicate initialization
_langsmith_configured = False

def configure_langsmith():
    """
    Configure LangSmith tracing based on environment variables.
    
    Required environment variables:
    - LANGSMITH_TRACING: Set to 'true' to enable tracing
    - LANGSMITH_ENDPOINT: LangSmith API endpoint
    - LANGSMITH_API_KEY: LangSmith API key
    - LANGSMITH_PROJECT: Project name in LangSmith
    """
    global _langsmith_configured
    
    # Skip if already configured in this process
    if _langsmith_configured:
        return
    
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
        os.environ["LANGSMITH_PROJECT"] = os.getenv(
            "LANGSMITH_PROJECT", "study-material-generator"
        )
        print("LangSmith tracing enabled for project: study-material-generator")
        _langsmith_configured = True
    else:
        print("LangSmith tracing disabled")
        _langsmith_configured = True


def get_langsmith_client() -> Optional[Client]:
    """
    Get a LangSmith client instance if tracing is enabled.
    
    Returns:
        Client instance or None if not configured
    """
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        try:
            return Client()
        except Exception as e:
            print(f"Failed to initialize LangSmith client: {e}")
    return None


# ============================================================================
# Tracing Decorators
# ============================================================================

def traced(*, name: Optional[str] = None, project_name: Optional[str] = None, 
           run_type: str = "chain", tags: Optional[List[str]] = None,
           metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace function execution with LangSmith.
    
    Args:
        name: Name of the run (defaults to function name)
        project_name: Project name in LangSmith
        run_type: Type of run (chain, llm, tool, retriever, embedding, prompt)
        tags: List of tags to add to the run
        metadata: Additional metadata to attach to the run
    
    Returns:
        Decorated function with tracing enabled
    
    Example:
        @traced(name="generate_topics", tags=["topic_generation"])
        async def generate_topics(text: str) -> List[str]:
            ...
    """
    def decorator(func):
        return traceable(
            name=name or func.__name__,
            project_name=project_name,
            run_type=run_type,
            tags=tags,
            metadata=metadata,
        )(func)
    return decorator


def traced_llm(*, name: Optional[str] = None, tags: Optional[List[str]] = None,
               metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator specifically for tracing LLM calls.
    
    Example:
        @traced_llm(name="groq_completion", tags=["groq", "llama"])
        async def call_llm(prompt: str) -> str:
            ...
    """
    return traced(
        name=name,
        project_name=None,
        run_type="llm",
        tags=tags,
        metadata=metadata,
    )


def traced_tool(*, name: Optional[str] = None, tags: Optional[List[str]] = None,
                metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator specifically for tracing tool calls.
    
    Example:
        @traced_tool(name="vector_search", tags=["retrieval"])
        def search(query: str) -> List[Document]:
            ...
    """
    return traced(
        name=name,
        project_name=None,
        run_type="tool",
        tags=tags,
        metadata=metadata,
    )


def traced_retriever(*, name: Optional[str] = None, tags: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator specifically for tracing retriever calls.
    
    Example:
        @traced_retriever(name="vector_store_search")
        def retrieve(query: str) -> List[Document]:
            ...
    """
    return traced(
        name=name,
        project_name=None,
        run_type="retriever",
        tags=tags,
        metadata=metadata,
    )


# ============================================================================
# Enhanced Business Logic Tracing (Simplified - just use traceable directly)
# ============================================================================

def traced_business_logic(*, name: Optional[str] = None, 
                          tags: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None):
    """
    Enhanced decorator for tracing business logic.
    This is now a simple wrapper around the traceable decorator.
    
    Args:
        name: Name of the business operation
        tags: List of tags for categorization
        metadata: Additional metadata to attach
    
    Example:
        @traced_business_logic(
            name="process_document",
            tags=["document_processing", "pdf"],
            metadata={"file_type": "pdf"}
        )
        async def process_document(file_path: str) -> str:
            ...
    """
    return traced(
        name=name,
        run_type="chain",
        tags=tags or ["business_logic"],
        metadata=metadata,
    )


# ============================================================================
# External API Call Tracing (Simplified - just use traceable directly)
# ============================================================================

def traced_api_call(*, name: Optional[str] = None, 
                    service: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for tracing external API calls (Groq, Qdrant, etc.).
    This is now a simple wrapper around the traceable decorator.
    
    Args:
        name: Name of the API operation
        service: Service name (e.g., "groq", "qdrant")
        tags: Additional tags
        metadata: Additional metadata
    
    Example:
        @traced_api_call(
            name="create_completion",
            service="groq",
            tags=["llm", "completion"]
        )
        async def call_groq_api(prompt: str) -> str:
            ...
    """
    return traced(
        name=name,
        run_type="tool",
        tags=(tags or []) + [f"api:{service}"] if service else tags,
        metadata=metadata,
    )


# ============================================================================
# Resource Utilization Tracking
# ============================================================================

@contextmanager
def track_resource_utilization(operation_name: str, 
                               track_memory: bool = True,
                               track_time: bool = True,
                               tags: Optional[List[str]] = None):
    """
    Context manager for tracking resource utilization during operations.
    
    Features:
    - Memory usage tracking (RSS, VMS)
    - CPU time tracking
    - Operation duration
    - Automatic LangSmith event logging
    
    Args:
        operation_name: Name of the operation being tracked
        track_memory: Whether to track memory usage
        track_time: Whether to track execution time
        tags: Tags for categorization
    
    Example:
        with track_resource_utilization("document_processing", tags=["pdf"]):
            result = process_document(file_path)
    """
    start_time = time.time()
    metrics = {
        "operation": operation_name,
        "start_time": datetime.utcnow().isoformat(),
    }
    
    # Track initial memory if requested
    if track_memory:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            metrics["start_memory_rss_mb"] = round(mem_info.rss / 1024 / 1024, 2)
            metrics["start_memory_vms_mb"] = round(mem_info.vms / 1024 / 1024, 2)
        except ImportError:
            metrics["memory_tracking"] = "psutil not installed"
        except Exception as e:
            metrics["memory_tracking_error"] = str(e)
    
    try:
        yield metrics
    finally:
        # Track final metrics
        if track_time:
            metrics["duration_seconds"] = round(time.time() - start_time, 3)
        
        if track_memory:
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                metrics["end_memory_rss_mb"] = round(mem_info.rss / 1024 / 1024, 2)
                metrics["end_memory_vms_mb"] = round(mem_info.vms / 1024 / 1024, 2)
                metrics["memory_delta_mb"] = round(
                    metrics.get("end_memory_rss_mb", 0) - metrics.get("start_memory_rss_mb", 0), 2
                )
            except:
                pass
        
        metrics["end_time"] = datetime.utcnow().isoformat()
        
        # Log to LangSmith if available
        try:
            if run_tree := get_current_run_tree():
                run_tree.add_event(
                    name=f"resource_metrics_{operation_name}",
                    metadata={
                        "metrics": metrics,
                        "tags": tags or [],
                    }
                )
        except:
            pass


# ============================================================================
# Pipeline Tracing
# ============================================================================

def trace_pipeline_step(step_name: str, 
                       inputs: Optional[Dict[str, Any]] = None,
                       tags: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for tracing individual steps in a pipeline.
    
    Features:
    - Step-level tracing within a larger pipeline
    - Input/output tracking
    - Error propagation
    - Pipeline context preservation
    
    Args:
        step_name: Name of the pipeline step
        inputs: Input parameters to track
        tags: Step-specific tags
        metadata: Additional metadata
    
    Example:
        @trace_pipeline_step(
            "topic_generation",
            tags=["nlp", "extraction"],
            metadata={"model": "llama-3.1-8b"}
        )
        async def generate_topics(text: str) -> List[str]:
            ...
    """
    return traced(
        name=f"pipeline:{step_name}",
        run_type="chain",
        tags=(tags or []) + ["pipeline_step"],
        metadata=metadata,
    )


# ============================================================================
# Error Tracking
# ============================================================================

def track_error(error_type: str, 
                error_message: str, 
                context: Optional[Dict[str, Any]] = None,
                severity: str = "error"):
    """
    Manually track an error in LangSmith.
    
    Args:
        error_type: Type/category of error
        error_message: Error message
        context: Additional context about the error
        severity: Error severity (error, warning, critical)
    """
    try:
        if run_tree := get_current_run_tree():
            error_data = {
                "error_type": error_type,
                "error_message": error_message,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                **(context or {}),
            }
            run_tree.add_event(name="error", metadata=error_data)
    except Exception as e:
        # Silently fail if tracing is not available
        logger.debug(f"Failed to track error in LangSmith: {e}")


# ============================================================================
# Metrics Tracking
# ============================================================================

def log_metric(name: str, 
               value: float, 
               unit: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None):
    """
    Log a metric to LangSmith.
    
    Args:
        name: Metric name
        value: Metric value
        unit: Unit of measurement
        metadata: Additional metadata
    """
    try:
        if run_tree := get_current_run_tree():
            metric_data = {
                "name": name,
                "value": value,
                "unit": unit,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }
            run_tree.add_event(name="metric", metadata=metric_data)
    except Exception as e:
        # Silently fail if tracing is not available
        logger.debug(f"Failed to log metric '{name}' in LangSmith: {e}")