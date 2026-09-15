# RAG

A lightweight PDF-based RAG chat application built with Streamlit, LangChain, Chroma, and Ollama.

## Features
- Upload a PDF
- Extract text and split it into chunks
- Generate embeddings locally with a sentence-transformer model
- Retrieve relevant chunks with a vector store
- Answer questions using an Ollama local LLM

## Project structure
- `docuchat-rag/app.py` – Streamlit frontend
- `docuchat-rag/rag_engine.py` – PDF processing and retrieval logic
- `docuchat-rag/requirements.txt` – Python dependencies

## Setup

1. Open a terminal in the project root.
2. Create and activate a virtual environment if needed.
3. Install dependencies:

```bash
pip install -r docuchat-rag/requirements.txt
```

4. Start Ollama locally and ensure the model exists:

```bash
ollama pull llama3.2:1b
```

5. Run the app:

```bash
streamlit run docuchat-rag/app.py
```

Then open the local URL shown by Streamlit in your browser.

## Notes
- This app expects a local Ollama installation.
- For best results, upload a PDF with selectable text.
