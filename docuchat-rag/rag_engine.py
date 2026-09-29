import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ordered list of supported models to try
CANDIDATE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192"
]

def get_groq_api_key():
    """Safely retrieves the Groq API key from environment variables or Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return None

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
    """Creates a RAG chain powered by Groq with model fallback."""
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

    # Find the first model available to this key
    active_llm = None
    for model_name in CANDIDATE_MODELS:
        try:
            test_llm = ChatGroq(
                model=model_name,
                api_key=groq_api_key,
                temperature=0
            )
            # Lightweight ping to verify model availability
            test_llm.invoke("hi")
            active_llm = test_llm
            break
        except Exception:
            continue

    if active_llm is None:
        # Fall back to primary instant model if network ping fails
        active_llm = ChatGroq(
            model="llama-3.1-8b-instant"  # ✅ Correct ('m')
            api_key=groq_api_key,
            temperature=0
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | active_llm
        | StrOutputParser()
    )
    return rag_chain