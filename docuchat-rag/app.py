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

    if st.button("Generate Consolidated Report", type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one PDF file.")
        elif not custom_requirements.strip():
            st.error("Please provide your report requirements/instructions.")
        else:
            with st.spinner(f"Reading, extracting, and synthesizing content from {len(uploaded_files)} files..."):
                try:
                    report = generate_multi_doc_report(uploaded_files, custom_requirements)
                    st.session_state["generated_report"] = report
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

    # Display results and download options
    if "generated_report" in st.session_state and st.session_state["generated_report"]:
        st.markdown("---")
        st.subheader("Generated Report")

        # Styled Blue-and-White Card Preview
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 6px solid #2563EB; padding: 20px; border-radius: 8px;">
                {st.session_state["generated_report"]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("") # spacing

        # Display results, allow live editing, and handle downloads
    if "generated_report" in st.session_state and st.session_state["generated_report"]:
        st.markdown("---")
        st.subheader("Edit & Finalize Report")
        st.caption("You can edit the text directly in the box below before downloading.")

        # 1. Editable Text Box loaded with the AI-generated report
        edited_report = st.text_area(
            label="Report Content (Markdown supported)",
            value=st.session_state["generated_report"],
            height=350,
            key="report_editor"
        )

        # 2. Live formatted preview of your edits
        with st.expander("Preview Formatted Output", expanded=False):
            with st.container(border=True):
                st.markdown(edited_report)

        st.write("")

        # 3. Generate PDF dynamically using the EDITED text
        pdf_bytes = create_styled_pdf(edited_report)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                label="Download Edited PDF",
                data=pdf_bytes,
                file_name="consolidated_summary_report.pdf",
                mime="application/pdf"
            )
        with col2:
            st.download_button(
                label="Download Edited Markdown (.md)",
                data=edited_report,
                file_name="consolidated_summary_report.md",
                mime="text/markdown"
            )