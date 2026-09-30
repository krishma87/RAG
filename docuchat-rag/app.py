import streamlit as st
import tempfile
import os
from rag_engine import process_pdf, get_rag_chain
from report_generator import generate_multi_doc_report, create_styled_pdf

st.set_page_config(page_title="DocuChat AI - Assistant & Report Generator", layout="wide")

# Sidebar navigation mode
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Select Feature",
    ["Single PDF Q&A", "Multi-Doc Report Generator"]
)

# =========================================================
# FEATURE 1: Single PDF Q&A (Existing Workflow)
# =========================================================
if app_mode == "Single PDF Q&A":
    st.title("Document Chat AI - PDF Assistant")

    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="single_pdf")

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

# =========================================================
# FEATURE 2: Multi-Document Custom Report Generator
# =========================================================
elif app_mode == "Multi-Doc Report Generator":
    st.title("Multi-Document Custom Report Generator")
    st.caption("Upload multiple PDFs (e.g., 14 reports) and generate a unified summary tailored to your format requirements.")

    uploaded_files = st.file_uploader(
        "Upload PDF Reports (Max 200MB each)",
        type=["pdf"],
        accept_multiple_files=True,
        key="multi_pdf"
    )

    custom_requirements = st.text_area(
        "Define Your Report Structure & Extraction Requirements",
        height=180,
        placeholder="Example:\n"
                    "1. Executive Summary: Core takeaways across all documents.\n"
                    "2. Comparison Table: Columns for Document Name, Metric X, Metric Y, Status.\n"
                    "3. Key Risks & Action Items: Bulleted points extracted per file.\n"
                    "4. Final Strategic Recommendations."
    )

    # Keep only THIS single button:
    if st.button("Generate Consolidated Report", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF report.")
        elif not custom_requirements.strip():
            st.warning("Please enter your structure and extraction requirements.")
        else:
            with st.spinner("Analyzing documents and compiling report..."):
                try:
                    report_text = generate_multi_doc_report(uploaded_files, requirements)
                    st.session_state["generated_report"] = report_text
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

    # Editable area & download button appear once the report is generated
    if "generated_report" in st.session_state and st.session_state["generated_report"]:
        st.markdown("---")
        st.subheader("Edit & Finalize Report")

        # 1. Editable text area
        edited_report = st.text_area(
            "Modify the generated report before downloading:",
            value=st.session_state["generated_report"],
            height=350,
            key="edited_report_content"
        )
        st.session_state["generated_report"] = edited_report

        # 2. Download PDF button
        try:
            pdf_bytes = create_styled_pdf(edited_report)
            st.download_button(
                label="Download Final PDF Report",
                data=pdf_bytes,
                file_name="Consolidated_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error generating PDF file: {e}")