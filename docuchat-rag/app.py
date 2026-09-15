import streamlit as st
import tempfile
import os
from rag_engine import process_pdf, get_rag_chain

st.set_page_config(page_title="DocuChat AI - RAG Assistant")
st.title("Document Chat AI - PDF Assistant")

with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Process uploaded document
if uploaded_file and st.session_state.rag_chain is None:
    with st.spinner("Processing document and generating embeddings..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            vectorstore = process_pdf(tmp_path)
            st.session_state.rag_chain = get_rag_chain(vectorstore)
            st.sidebar.success("Document indexed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input handling
if user_query := st.chat_input("Ask a question about your document..."):
    if st.session_state.rag_chain is None:
        st.warning("Please upload a valid PDF document first in the sidebar!")
    else:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer..."):
                try:
                    answer = st.session_state.rag_chain.invoke(user_query)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error querying model: {e}")