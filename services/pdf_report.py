import os
import io
import re
from typing import List, Dict, Tuple
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    PageBreak,
    Table,
    TableStyle,
)

REPORT_DIR = "/tmp/rag_reports"


class PDFReportService:
    def __init__(self):
        os.makedirs(REPORT_DIR, exist_ok=True)

    def build_report(
        self,
        session_id: str,
        document_title: str,
        qa_pairs: List[Dict],
    ) -> str:
        """
        Build a styled PDF report with all topics, queries, context chunks, and LLM answers.
        Returns the output file path.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{session_id[:8]}_{timestamp}.pdf"
        output_path = os.path.join(REPORT_DIR, filename)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        story = self._build_story(styles, document_title, qa_pairs, session_id)
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)

        return output_path

    def build_report_bytes(
        self,
        session_id: str,
        document_title: str,
        qa_pairs: List[Dict],
    ) -> bytes:
        """
        Build a styled PDF report and return it as bytes (in-memory).
        Returns the PDF content as bytes.
        """
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        story = self._build_story(styles, document_title, qa_pairs, session_id)
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)

        # Get the bytes and close the buffer
        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    # ------------------------------------------------------------------
    def _build_story(self, styles, document_title, qa_pairs, session_id):
        """Build PDF story for a clean, study-friendly report."""
        story = []

        # ---- Clean, Professional Styles ----
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=12,
            alignment=1,  # Center
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#666666"),
            spaceAfter=6,
            alignment=1,  # Center
        )
        heading1_style = ParagraphStyle(
            "TopicHeading",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#1a1a2e"),
            spaceBefore=20,
            spaceAfter=10,
            borderWidth=1,
            borderColor=colors.HexColor("#1a1a2e"),
            borderPadding=8,
            backColor=colors.HexColor("#f0f4f8"),
        )
        heading2_style = ParagraphStyle(
            "SubHeading",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#2c3e50"),
            spaceBefore=14,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=11,
            leading=17,
            textColor=colors.HexColor("#333333"),
            spaceAfter=10,
            justification=True,
        )
        subheading_style = ParagraphStyle(
            "SubHeading2",
            parent=body_style,
            fontSize=12,
            textColor=colors.HexColor("#2c3e50"),
            spaceBefore=12,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        )
        bullet_style = ParagraphStyle(
            "Bullet",
            parent=body_style,
            leftIndent=20,
            spaceAfter=4,
        )
        code_style = ParagraphStyle(
            "Code",
            parent=body_style,
            fontSize=10,
            textColor=colors.HexColor("#2d2d2d"),
            backColor=colors.HexColor("#f5f5f5"),
            leftIndent=10,
            rightIndent=10,
            spaceAfter=8,
            fontName="Courier",
        )

        # ---- Title Page ----
        story.append(Spacer(1, 3 * cm))
        story.append(Paragraph(self._escape_for_pdf(document_title), title_style))
        story.append(Paragraph(
            "Study Material",
            subtitle_style,
        ))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y')}",
            subtitle_style,
        ))
        story.append(Spacer(1, 2 * cm))
        
        # Table of Contents info
        toc_style = ParagraphStyle(
            "TOC",
            parent=body_style,
            fontSize=10,
            textColor=colors.HexColor("#666666"),
            alignment=1,
        )
        story.append(Paragraph(f"This document covers {len(qa_pairs)} key topics.", toc_style))
        story.append(PageBreak())

        # ---- Table of Contents ----
        story.append(Paragraph("Table of Contents", heading1_style))
        story.append(Spacer(1, 0.5 * cm))
        
        for idx, pair in enumerate(qa_pairs, start=1):
            topic = pair.get("topic", f"Topic {idx}")
            story.append(Paragraph(
                f"{idx}. {self._escape_for_pdf(topic)}",
                body_style,
            ))
        story.append(PageBreak())

        # ---- Topic Sections (Clean Study Content) ----
        for idx, pair in enumerate(qa_pairs, start=1):
            topic = pair.get("topic", f"Topic {idx}")
            answer = pair.get("answer", "")

            # Topic heading with number
            story.append(Paragraph(f"Chapter {idx}: {self._escape_for_pdf(topic)}", heading1_style))
            story.append(Spacer(1, 0.3 * cm))

            # Process answer content
            elements = self._parse_markdown_content(answer)
            for elem_type, content in elements:
                if elem_type == 'spacer':
                    story.append(Spacer(1, 0.2 * cm))
                elif elem_type == 'bullet':
                    story.append(Paragraph(
                        f"• {self._escape_for_pdf(content)}",
                        bullet_style,
                    ))
                elif elem_type == 'numbered':
                    story.append(Paragraph(
                        self._escape_for_pdf(content),
                        bullet_style,
                    ))
                elif elem_type == 'bold_heading':
                    story.append(Paragraph(
                        self._format_bold_text(content),
                        heading2_style,
                    ))
                elif elem_type == 'subheading':
                    story.append(Paragraph(
                        self._format_bold_text(content),
                        subheading_style,
                    ))
                elif elem_type == 'code_block':
                    # Code block - escape and use Courier font style
                    escaped = self._escape_for_pdf(content)
                    # Replace newlines with <br/> for Paragraph
                    formatted = escaped.replace('\n', '<br/>')
                    story.append(Paragraph(formatted, code_style))
                elif elem_type == 'paragraph':
                    story.append(Paragraph(
                        self._format_bold_text(content),
                        body_style,
                    ))

            story.append(Spacer(1, 0.5 * cm))
            
            # Page break between chapters (except last)
            if idx < len(qa_pairs):
                story.append(PageBreak())

        # ---- Back Cover ----
        story.append(PageBreak())
        back_style = ParagraphStyle(
            "BackCover",
            parent=body_style,
            fontSize=10,
            textColor=colors.HexColor("#888888"),
            alignment=1,
        )
        story.append(Spacer(1, 4 * cm))
        story.append(Paragraph("— End of Study Material —", back_style))
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "Generated by RAG Study Material Generator",
            back_style,
        ))

        return story

    def _parse_markdown_content(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse markdown content into structured elements.
        Returns list of (type, content) tuples.
        Types: 'spacer', 'bullet', 'numbered', 'bold_heading', 'subheading', 'code_block', 'paragraph'
        """
        elements = []
        lines = text.split("\n")
        in_code_block = False
        code_content = []
        
        for line in lines:
            stripped = line.strip()
            
            # Handle code blocks
            if stripped.startswith("```"):
                if in_code_block:
                    # End code block
                    elements.append(('code_block', "\n".join(code_content)))
                    code_content = []
                    in_code_block = False
                else:
                    # Start code block
                    in_code_block = True
                continue
            
            if in_code_block:
                code_content.append(stripped)
                continue
            
            if not stripped:
                elements.append(('spacer', ''))
                continue
            
            # Handle markdown headings (###, ##, #)
            if stripped.startswith("###"):
                content = stripped.lstrip("#").strip()
                elements.append(('subheading', content))
            elif stripped.startswith("##"):
                content = stripped.lstrip("#").strip()
                elements.append(('bold_heading', content))
            elif stripped.startswith("#"):
                content = stripped.lstrip("#").strip()
                elements.append(('bold_heading', content))
            # Handle bullet points (•, -, *)
            elif stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("*"):
                content = stripped[1:].strip()
                # Clean up any leading/trailing bold/italic markers
                content = self._clean_markdown_markers(content)
                elements.append(('bullet', content))
            # Handle numbered lists (e.g., "1. Item" or "1) Item")
            elif re.match(r'^\d+[\.\)]\s+', stripped):
                content = self._clean_markdown_markers(stripped)
                elements.append(('numbered', content))
            # Handle bold headings (line is just **bold text**)
            elif stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
                bold_text = stripped.strip("*")
                elements.append(('bold_heading', bold_text))
            # Regular paragraph (may contain inline bold)
            else:
                elements.append(('paragraph', stripped))
        
        # Handle unclosed code block
        if in_code_block and code_content:
            elements.append(('code_block', "\n".join(code_content)))
        
        return elements

    @staticmethod
    def _clean_markdown_markers(text: str) -> str:
        """Clean up markdown bold/italic markers from text.
        
        Handles cases like:
        - *text* (italic)
        - **text** (bold)
        - ***text*** (bold italic)
        - *text** (mismatched)
        - text* (trailing marker)
        """
        text = text.strip()
        
        # Remove trailing single asterisk (e.g., "Introduction*" -> "Introduction")
        if text.endswith("*") and not text.endswith("**"):
            text = text[:-1].strip()
        
        # Remove trailing double asterisk
        if text.endswith("**"):
            text = text[:-2].strip()
        
        # Now handle leading markers
        # Handle ***text*** (bold italic)
        if text.startswith("***") and len(text) > 3:
            text = text[3:].strip()
        # Handle **text** (bold)
        elif text.startswith("**") and len(text) > 2:
            text = text[2:].strip()
        # Handle *text* (italic)
        elif text.startswith("*") and len(text) > 1:
            text = text[1:].strip()
        
        return text.strip()

    @staticmethod
    def _format_bold_text(text: str) -> str:
        """Convert markdown-style **bold** to ReportLab <b>bold</b>.
        
        Handles unclosed bold markers and validates output.
        """
        if "**" not in text:
            return PDFReportService._escape_for_pdf(text)
        
        # Split by ** and process
        parts = text.split("**")
        result = []
        
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Bold parts (between ** markers)
                escaped = PDFReportService._escape_for_pdf(part)
                result.append(f"<b>{escaped}</b>")
            else:
                result.append(PDFReportService._escape_for_pdf(part))
        
        joined = "".join(result)
        
        # Validate that all <b> tags are properly closed
        open_count = joined.count("<b>")
        close_count = joined.count("</b>")
        if open_count != close_count:
            # Fallback: strip ** and escape without bold formatting
            return PDFReportService._escape_for_pdf(text.replace("**", ""))
        
        return joined

    # ------------------------------------------------------------------
    @staticmethod
    def _header_footer(canvas, doc):
        canvas.saveState()
        width, height = A4
        # Footer
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(2 * cm, 1.2 * cm, "RAG Study Material Generator")
        canvas.drawRightString(width - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    @staticmethod
    def _escape_for_pdf(text: str) -> str:
        """Escape special XML/HTML chars for ReportLab Paragraph.
        
        This method properly escapes characters that would be interpreted
        as HTML/XML tags by ReportLab's Paragraph parser.
        """
        if not text:
            return text
        
        text = str(text)
        # Order matters: & must be first to avoid double-escaping
        # Use explicit strings to build the HTML entities
        text = text.replace('&', '&' + 'amp;')
        text = text.replace('<', '&' + 'lt;')
        text = text.replace('>', '&' + 'gt;')
        
        return text

    def _preprocess_content(self, text: str) -> List[Tuple[str, str]]:
        """Alias for _parse_markdown_content for backwards compatibility."""
        return self._parse_markdown_content(text)