import os

import streamlit as st

from ingestion.pipeline import IngestionPipeline
import config


def render_sidebar():
    """Render the sidebar with document upload and management."""
    with st.sidebar:
        st.title("Query Right")
        st.caption("Legal Document Q&A System")
        st.divider()

        # --- Document Upload ---
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or DOCX files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="doc_uploader",
        )

        if uploaded_files and st.button("Process Documents", type="primary"):
            pipeline = IngestionPipeline()
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}...")

                os.makedirs(config.UPLOAD_DIR, exist_ok=True)
                file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                result = pipeline.ingest_file(file_path)
                st.success(
                    f"**{result['file']}**: {result['pages_loaded']} pages, "
                    f"{result['chunks_created']} chunks indexed"
                )
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("All documents processed!")

        # --- Document Management ---
        st.divider()
        st.subheader("Indexed Documents")
        pipeline = IngestionPipeline()
        sources = pipeline.list_ingested_sources()
        chunk_count = pipeline.get_document_count()

        st.metric("Total Chunks", chunk_count)

        if sources:
            for source in sources:
                col1, col2 = st.columns([3, 1])
                col1.text(source)
                if col2.button("X", key=f"del_{source}"):
                    pipeline.delete_source(source)
                    st.rerun()
        else:
            st.info("No documents indexed yet. Upload files above.")

        # --- Tavily Key ---
        st.divider()
        tavily_key = st.text_input(
            "Tavily API Key",
            value=config.TAVILY_API_KEY,
            type="password",
            key="tavily_key",
            help="Required for web search fallback",
        )
        if tavily_key:
            config.TAVILY_API_KEY = tavily_key
