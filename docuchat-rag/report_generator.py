import os
import re
import io
from dotenv import load_dotenv

load_dotenv()

import fitz  # PyMuPDF
from langchain_ollama import ChatOllama
from typing import List, Dict

# ReportLab imports for generating styled PDFs
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Initialize local LLM
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_ctx=4096,
    num_predict=512
)

EXTRACTION_PROMPT = """
You are analyzing the document: "{filename}".
Extract all specific information matching these user requirements:
{requirements}

Document Content:
{text}

Provide a concise, direct summary of the extracted data for this document. If a requested criterion is not present, state 'Not mentioned'.
"""

SYNTHESIS_PROMPT = """
You are an expert report writer. Synthesize the findings from all uploaded documents into a single consolidated report.

User Requirements & Format Target:
{requirements}

Data Extracted Per Document:
{document_summaries}

Rules:
1. Follow the user's required report format and sections strictly.
2. Group or compare findings across documents where applicable.
3. Clearly cite the source document filename for each point.
"""

def extract_from_pdf(file_bytes: bytes, filename: str, requirements: str) -> Dict[str, str]:
    """Scans and extracts only the relevant sections to reduce token load."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    keywords = [w.lower().strip(".,:;!?") for w in requirements.split() if len(w) > 3][:8]
    matched_pages = []
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()
        if any(kw in text.lower() for kw in keywords) or page_num < 2:
            matched_pages.append(text)
            
    condensed_text = "\n".join(matched_pages)[:8000]
    
    if not condensed_text.strip():
        condensed_text = "\n".join([doc[i].get_text() for i in range(min(2, total_pages))])[:8000]

    prompt = EXTRACTION_PROMPT.format(
        filename=filename,
        requirements=requirements,
        text=condensed_text
    )
    
    response = llm.invoke(prompt)
    return {"filename": filename, "summary": response.content}

def generate_multi_doc_report(uploaded_files, requirements: str) -> str:
    """Processes uploaded files and generates the consolidated report."""
    extracted_data = []

    for file in uploaded_files:
        data = extract_from_pdf(file.getvalue(), file.name, requirements)
        extracted_data.append(data)

    combined_docs = "\n\n".join(
        [f"--- DOCUMENT: {item['filename']} ---\n{item['summary']}" for item in extracted_data]
    )

    final_prompt = SYNTHESIS_PROMPT.format(
        requirements=requirements,
        document_summaries=combined_docs
    )
    
    report_response = llm.invoke(final_prompt)
    return report_response.content

def create_styled_pdf(markdown_text: str) -> bytes:
    """Converts markdown report text into a styled Blue-and-White PDF document."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []
    lines = markdown_text.split('\n')

    for line in lines:
        clean = line.strip()
        if not clean:
            story.append(Spacer(1, 6))
            continue

        clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean)
        clean = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean)

        if clean.startswith('# '):
            story.append(Paragraph(clean[2:], title_style))
        elif clean.startswith('## '):
            story.append(Paragraph(clean[3:], h2_style))
        elif clean.startswith('### '):
            story.append(Paragraph(clean[4:], h2_style))
        elif clean.startswith(('-', '*')) and not clean.startswith('---'):
            bullet_text = f"&bull; {clean.lstrip('-*').strip()}"
            story.append(Paragraph(bullet_text, bullet_style))
        elif clean.startswith('|'):
            story.append(Paragraph(f"<code>{clean}</code>", body_style))
        else:
            story.append(Paragraph(clean, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()