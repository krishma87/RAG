import os
import re
import io
import fitz  # PyMuPDF
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import List, Dict

# ReportLab imports for generating styled PDFs
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

load_dotenv()

def get_groq_api_key():
    """Safely retrieves the Groq API key from environment or secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return None

def get_available_groq_model(api_key: str) -> str:
    """Queries Groq API dynamically to find an active model available to this account."""
    preferred_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            available_ids = [m["id"] for m in response.json().get("data", [])]
            for pref in preferred_models:
                if pref in available_ids:
                    return pref
            for m_id in available_ids:
                if "whisper" not in m_id and "guard" not in m_id:
                    return m_id
    except Exception:
        pass
    return "llama3-8b-8192"

def get_llm():
    """Initializes LLM on-demand using dynamically validated model name."""
    key = get_groq_api_key()
    if not key:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file or Streamlit Cloud Secrets.")
    
    model_name = get_available_groq_model(key)
    return ChatGroq(
        model=model_name,
        api_key=key,
        temperature=0
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
    llm = get_llm()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    keywords = [w.lower().strip(".,:;!?") for w in requirements.split() if len(w) > 3][:8]
    matched_pages = []
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()
        if any(kw in text.lower() for kw in keywords) or page_num < 2:
            matched_pages.append(text)
            
    condensed_text = "\n".join(matched_pages)[:12000]
    
    if not condensed_text.strip():
        condensed_text = "\n".join([doc[i].get_text() for i in range(min(2, total_pages))])[:12000]

    prompt = EXTRACTION_PROMPT.format(
        filename=filename,
        requirements=requirements,
        text=condensed_text
    )
    
    response = llm.invoke(prompt)
    return {"filename": filename, "summary": response.content}

def generate_multi_doc_report(uploaded_files, requirements: str) -> str:
    """Processes uploaded files and generates the consolidated report."""
    llm = get_llm()
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