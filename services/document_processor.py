import pdfplumber
import docx
import re
from typing import List, Tuple


class DocumentProcessor:
    """Extract text from PDF or DOCX and split into semantic chunks."""

    CHUNK_SIZE = 1000      # Target characters per chunk
    CHUNK_OVERLAP = 200    # overlap to preserve context

    def process(self, file_path: str, suffix: str, max_pages: int = None) -> Tuple[str, List[str]]:
        """
        Returns (full_text, list_of_chunks).
        
        Args:
            file_path: Path to the document
            suffix: File extension (.pdf or .docx)
            max_pages: Maximum number of pages to process (None for unlimited)
            
        Raises ValueError if the document exceeds max_pages (if specified).
        """
        if suffix == ".pdf":
            full_text, page_count = self._extract_pdf(file_path)
        else:
            full_text, page_count = self._extract_docx(file_path)

        if max_pages is not None and page_count > max_pages:
            raise ValueError(f"Document has {page_count} pages. Maximum allowed is {max_pages}.")

        chunks = self._semantic_chunk_text(full_text)
        return full_text, chunks

    # ------------------------------------------------------------------
    def _extract_pdf(self, path: str) -> Tuple[str, int]:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                t = page.extract_text() or ""
                text_parts.append(t)
        return "\n\n".join(text_parts), page_count

    def _extract_docx(self, path: str) -> Tuple[str, int]:
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)

        # Estimate page count: ~500 words per page
        word_count = len(full_text.split())
        estimated_pages = max(1, round(word_count / 500))

        return full_text, estimated_pages

    # ------------------------------------------------------------------
    def _semantic_chunk_text(self, text: str) -> List[str]:
        """
        Semantic chunking that respects sentence and paragraph boundaries.
        
        This approach:
        1. Splits text into sentences (preserving semantic meaning)
        2. Groups sentences into chunks that respect paragraph boundaries
        3. Maintains approximate chunk size while avoiding mid-sentence breaks
        4. Preserves context better than fixed-size character chunking
        """
        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        if not text:
            return []
        
        # Split into paragraphs first (preserves document structure)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, split by sentences
            if len(current_chunk) + len(paragraph) > self.CHUNK_SIZE and current_chunk:
                # Split paragraph into sentences
                sentences = self._split_into_sentences(paragraph)
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.CHUNK_SIZE and current_chunk:
                        # Current chunk is full, save it and start new one
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                # Handle overlap for context preservation
                if chunks and current_chunk:
                    overlap_text = self._get_overlap_text(chunks[-1], current_chunk)
                    if overlap_text:
                        current_chunk = overlap_text + " " + current_chunk
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Final pass: ensure chunks aren't too large (split if needed)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.CHUNK_SIZE * 1.5:  # Allow some flexibility
                # Split large chunks by sentences
                sentences = self._split_into_sentences(chunk)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) > self.CHUNK_SIZE and temp_chunk:
                        final_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        temp_chunk += " " + sentence if temp_chunk else sentence
                if temp_chunk.strip():
                    final_chunks.append(temp_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while handling common edge cases.
        
        Handles:
        - Standard punctuation (. ! ?)
        - Abbreviations (Mr., Dr., etc.)
        - Decimal numbers (3.14)
        - Quotations
        """
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Pattern to split on sentence boundaries
        # This handles most cases while avoiding splitting on abbreviations
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def _get_overlap_text(self, previous_chunk: str, current_chunk: str) -> str:
        """
        Get overlap text from the end of previous chunk to maintain context.
        Returns the last sentence(s) from previous chunk that fit within overlap size.
        """
        if not previous_chunk:
            return ""
        
        # Get last sentences from previous chunk
        sentences = self._split_into_sentences(previous_chunk)
        overlap_parts = []
        overlap_length = 0
        
        # Work backwards from the end
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) > self.CHUNK_OVERLAP:
                break
            overlap_parts.insert(0, sentence)
            overlap_length += len(sentence) + 1  # +1 for space
        
        return " ".join(overlap_parts) if overlap_parts else ""

    # Keep old method for backward compatibility if needed
    def _chunk_text(self, text: str) -> List[str]:
        """Legacy sliding window chunking with overlap (for backward compatibility)."""
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks