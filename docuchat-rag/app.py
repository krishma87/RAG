import streamlit as st
import tempfile
import os
from rag_engine import process_pdf, get_rag_chain
from report_generator import generate_multi_doc_report, create_styled_pdf

st.set_page_config(
    page_title="DocuChat AI - Assistant & Report Generator",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
feature = st.sidebar.radio(
    "Select Feature",
    ["Single PDF Q&A", "Multi-Doc Report Generator"]
)

# ---------------------------------------------------------
# Feature 1: Single PDF Q&A
# ---------------------------------------------------------
if feature == "Single PDF Q&A":
    st.title("Document Chat AI - PDF Assistant")

    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "rag_chain" not in st.session_state:
        st.session_state["rag_chain"] = None
    if "current_file_name" not in st.session_state:
        st.session_state["current_file_name"] = None

    if uploaded_file is not None:
        if st.session_state["current_file_name"] != uploaded_file.name:
            st.session_state["chat_history"] = []
            st.session_state["rag_chain"] = None
            st.session_state["current_file_name"] = uploaded_file.name

            with st.spinner(f"Indexing '{uploaded_file.name}' and building vector store..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    vectorstore = process_pdf(tmp_path)
                    st.session_state["rag_chain"] = get_rag_chain(vectorstore)
                    st.sidebar.success(f"Indexed: {uploaded_file.name}")
                except Exception as e:
                    st.sidebar.error(f"Error processing file: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    if uploaded_file and st.session_state["rag_chain"]:
        st.info(f" **Active Document:** `{uploaded_file.name}` ({round(uploaded_file.size / 1024, 1)} KB)")
        if not st.session_state["chat_history"]:
            st.markdown(
                """
                > **Suggested Queries:**
                > - *Summarize the key findings and conclusions of this document.*
                > - *Explain the technical architecture and core methodology.*
                > - *List all actionable recommendations or critical requirements.*
                """
            )
    elif not uploaded_file:
        st.info("Upload a PDF document in the sidebar to begin analysis.")

    for message in st.session_state["chat_history"]:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about your document..."):
        if st.session_state["rag_chain"] is None:
            st.warning("Please upload a valid PDF document in the sidebar first.")
        else:
            st.chat_message("user", avatar="👤").markdown(user_query)
            st.session_state["chat_history"].append({"role": "user", "content": user_query})

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching document & generating answer..."):
                    try:
                        answer = st.session_state["rag_chain"].invoke(user_query)
                        st.markdown(answer)
                        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error querying model: {e}")

# ---------------------------------------------------------
# Feature 2: Multi-Doc Custom Report Generator
# ---------------------------------------------------------
elif feature == "Multi-Doc Report Generator":
    st.title("Multi-Document Custom Report Generator")
    st.write("Upload multiple PDFs (e.g., 14 reports) and generate a unified summary tailored to your format requirements.")

    uploaded_files = st.file_uploader(
        "Upload PDF Reports (Max 200MB each)",
        type=["pdf"],
        accept_multiple_files=True
    )

    requirements = st.text_area(
        "Define Your Report Structure & Extraction Requirements",
        value=(
            "1. Overview of Scope:\n"
            "- What scope or domain does each document address?\n\n"
            "2. Detailed Criteria Extraction:\n"
            "- Technical architecture, threat vectors, or security measures.\n"
            "- Operational arrangements, schedules, rules, or requirements."
        ),
        height=180
    )

    if st.button("Generate Consolidated Report", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF report.")
        elif not requirements or not requirements.strip():
            st.warning("Please enter your structure and extraction requirements.")
        else:
            with st.spinner("Analyzing documents and compiling report..."):
                try:
                    report_text = generate_multi_doc_report(uploaded_files, requirements)
                    st.session_state["generated_report"] = report_text
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

    # Editable Area & Download PDF
    if "generated_report" in st.session_state and st.session_state["generated_report"]:
        st.markdown("---")
        st.subheader("Edit & Finalize Report")

        edited_report = st.text_area(
            "Modify the generated report before downloading:",
            value=st.session_state["generated_report"],
            height=350,
            key="edited_report_content"
        )
        st.session_state["generated_report"] = edited_report

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