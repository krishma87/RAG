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
    """Creates a RAG chain powered by Groq (Llama 3.1 8B)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful knowledge assistant. Answer the user's question using ONLY the context provided below.
    If the context does not contain the answer, reply that you cannot find the answer in the provided document.

    <context>
    {context}
    </context>

    Question: {question}
    """)

    # Check both environment variable and Streamlit secrets for deployment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
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