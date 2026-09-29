import os
import streamlit as st
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    """Finds an active, unrestricted production chat model on this Groq account."""
    preferred_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
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
                if not any(blocked in m_id.lower() for blocked in ["whisper", "guard", "orpheus", "canopylabs"]):
                    return m_id
    except Exception:
        pass
    return "openai/gpt-oss-120b"

def format_docs(docs):
    """Combines document chunks into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

def process_pdf(pdf_path: str):
    """Loads PDF, extracts text, and generates vector embeddings."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    total_text = "".join([doc.page_content.strip() for doc in documents])
    if not total_text:
        raise ValueError("The uploaded PDF contains no extractable text. Please use a selectable-text PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

def get_rag_chain(vectorstore):
    """Creates a RAG chain powered by Groq with an active model."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful knowledge assistant. Answer the user's question using ONLY the context provided below.
    If the context does not contain the answer, reply that you cannot find the answer in the provided document.

    <context>
    {context}
    </context>

    Question: {question}
    """)

    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file or Streamlit Cloud Secrets.")

    model_name = get_available_groq_model(groq_api_key)

    llm = ChatGroq(
        model=model_name,
        api_key=groq_api_key,
        temperature=0
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain