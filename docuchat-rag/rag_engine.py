import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_docs(docs):
    """Combines document chunks into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

def process_pdf(pdf_path: str):
    """Loads PDF, extracts text, and generates local vector embeddings."""
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

def get_llm():
    """Use OpenAI in hosted environments, and Ollama for local development."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )


def get_rag_chain(vectorstore):
    """Creates a RAG chain using the configured LLM backend."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful knowledge assistant. Answer the user's question using ONLY the context provided below.
    If the context does not contain the answer, reply that you cannot find the answer in the provided document.

    <context>
    {context}
    </context>

    Question: {question}
    """)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return rag_chain