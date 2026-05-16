import os
import json
import re
import time
from typing import List
from groq import AsyncGroq
from dotenv import load_dotenv
from services.langsmith_config import (
    configure_langsmith, 
    traced_llm, 
    traced_api_call,
    traced_business_logic,
    track_resource_utilization,
    log_metric,
)

GROQ_MODEL = "llama-3.1-8b-instant"

# Configure LangSmith on module import
configure_langsmith()


class TopicGenerator:
    def __init__(self):
        load_dotenv()
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", ""))

    @traced_business_logic(
        name="topic_generation_pipeline",
        tags=["topic_generation", "nlp", "extraction"],
        metadata={"model": GROQ_MODEL, "max_topics": 8}
    )
    async def generate_topics(self, text: str) -> List[str]:
        """
        Use Groq LLaMA-8B to extract 5–8 study topics from the document text.
        Returns a list of topic strings (e.g., ["Python programming", "Data types", "Operators"]).
        """
        prompt = f"""You are an expert educational content analyst and curriculum designer. Your task is to analyze the provided document and extract the most important study topics.

**Guidelines for topic extraction:**
1. Extract specific, well-defined topics that can be used as study headings
2. Order topics from foundational to advanced when possible
3. Keep the number of topics minimal (3-8 topics)

**CRITICAL OUTPUT REQUIREMENTS:**
- Output ONLY the topic names, one per line
- NO numbers, NO bullets, NO punctuation before topics
- NO introductory text, NO explanations, NO markdown formatting
- NO "Based on the document" or similar phrases
- Each topic should be 1-4 words long

**Example of correct output:**
Introduction
Data Types
Operators
Control Structures

**Document Content:**
\"\"\"
{text[:2500]}
\"\"\"
"""
        response = await self._call_groq_api(prompt)

        raw = response.choices[0].message.content.strip()

        # Try to parse as JSON first (in case LLM returns JSON array)
        try:
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            topics = json.loads(raw.strip())
            if isinstance(topics, list):
                # Extract just the topic names, stripping any numbers
                clean_topics = []
                for t in topics:
                    topic_str = str(t).strip()
                    # Strip leading numbers like "0 ", "1. ", etc.
                    topic_str = re.sub(r'^\d+[\.\)\s]+', '', topic_str).strip()
                    if topic_str:
                        clean_topics.append(topic_str)
                log_metric("topics_generated", len(clean_topics), unit="count")
                return clean_topics
        except (json.JSONDecodeError, ValueError):
            pass

        # Parse topic list from raw text
        lines = raw.splitlines()
        topics = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip introductory/explanatory lines
            lower_line = line.lower()
            if any(skip in lower_line for skip in [
                "based on", "here are", "following", "extracted", 
                "topics:", "study topics", "important"
            ]):
                continue
            
            # Match numbered format: "0 Topic", "1. Topic", "0) Topic", "1 2 Data Types"
            match = re.match(r'^(\d+)[\.\)\s]\s+(.+)$', line)
            if match:
                topic = match.group(2).strip()
                # Strip any leading numbers from the topic text itself (e.g., "2 Data Types" -> "Data Types")
                topic = re.sub(r'^\d+[\.\)\s]+', '', topic).strip()
                topics.append(topic)
            # Match bullet points: "- Topic", "• Topic"
            elif line.startswith(('-', '•')):
                topic = line.lstrip('-• ').strip()
                # Strip any leading numbers from the topic text
                topic = re.sub(r'^\d+[\.\)\s]+', '', topic).strip()
                if topic:
                    topics.append(topic)
            # Plain text line that looks like a topic (short, no sentences)
            elif len(line) < 100 and '.' not in line:
                # Strip any leading numbers
                topic = re.sub(r'^\d+[\.\)\s]+', '', line).strip()
                if topic:
                    topics.append(topic)
        
        # Return just the topic names without numbering
        result = topics[:8]
        
        if not result:
            result = ["General Overview"]
        
        # Log metrics
        log_metric("topics_generated", len(result), unit="count")
        
        return result

    @traced_api_call(
        name="groq_chat_completion",
        service="groq",
        tags=["llm", "chat_completion"],
        metadata={"model": GROQ_MODEL}
    )
    async def _call_groq_api(self, prompt: str):
        """Internal method to call Groq API with tracing."""
        with track_resource_utilization("groq_api_call", tags=["groq"]):
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            
            duration = time.time() - start_time
            
            # Log performance metrics
            log_metric("groq_latency_ms", duration * 1000, unit="ms")
            if response.usage:
                log_metric("prompt_tokens", response.usage.prompt_tokens, unit="tokens")
                log_metric("completion_tokens", response.usage.completion_tokens, unit="tokens")
                log_metric("total_tokens", response.usage.total_tokens, unit="tokens")
            
            return response
