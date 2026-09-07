from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from database import (
    clear_all_facts,
    delete_facts_for_document,
    init_db,
    insert_facts,
    list_source_documents,
    search_facts,
)
from fact_extractor import FactExtractor, client_from_settings
from pdf_processor import process_pdf

load_dotenv()
init_db()

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")
st.title("Fact Knowledge Layer")
st.caption(
    "Upload PDFs, extract generic entity–attribute–value facts with source evidence, "
    "and search them in SQLite."
)

with st.sidebar:
    st.header("LLM settings")
    env_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
    entered_api_key = st.text_input(
        "API key",
        value="",
        type="password",
        placeholder="****" if env_api_key else "Enter an API key",
        help="Leave blank to use the environment key (never shown here). Enter your own key to override.",
    )
    api_key = entered_api_key.strip() or env_api_key
    model = st.text_input(
        "Model",
        value=os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini",
    )
    base_url = st.text_input(
        "Base URL (optional)",
        value=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "",
        help="Leave blank for OpenAI. Set this for Groq, Ollama, or other compatible APIs.",
    )
    replace_existing = st.checkbox(
        "Replace facts for re-uploaded documents",
        value=True,
    )
    st.divider()
    if st.button("Clear all stored facts", type="secondary"):
        clear_all_facts()
        st.success("All facts removed.")
        st.rerun()

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Extract facts", type="primary", disabled=not uploaded_files):
    try:
        llm = client_from_settings(
            api_key=api_key or None,
            model=model or None,
            base_url=base_url or None,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    extractor = FactExtractor(llm)
    progress = st.progress(0.0, text="Starting extraction…")
    status = st.empty()
    total_facts = 0

    try:
        for file_index, uploaded in enumerate(uploaded_files, start=1):
            source_name = uploaded.name
            status.info(f"Processing {source_name} ({file_index}/{len(uploaded_files)})")
            chunks = process_pdf(uploaded.getvalue(), source_name)
            if not chunks:
                st.warning(f"No extractable text in {source_name}. Scanned PDFs need OCR.")
                continue

            if replace_existing:
                delete_facts_for_document(source_name)

            document_facts = []
            for chunk_index, chunk in enumerate(chunks, start=1):
                progress.progress(
                    ((file_index - 1) + chunk_index / len(chunks)) / len(uploaded_files),
                    text=f"{source_name}: page {chunk.page_number} ({chunk_index}/{len(chunks)})",
                )
                document_facts.extend(extractor.extract_from_chunk(chunk))

            records = [fact.to_record() for fact in document_facts]
            insert_facts(records)
            total_facts += len(records)
            st.write(f"Stored {len(records)} facts from `{source_name}`.")

        progress.progress(1.0, text="Done")
        status.success(f"Extracted {total_facts} facts from {len(uploaded_files)} PDF(s).")
    except Exception as exc:
        status.error(f"Extraction failed: {exc}")

st.divider()
st.subheader("Facts")

documents = list_source_documents()
filter_cols = st.columns([2, 1])
with filter_cols[0]:
    query = st.text_input("Search facts", placeholder="entity, attribute, value, evidence…")
with filter_cols[1]:
    selected_document = st.selectbox(
        "Source document",
        options=["All documents"] + documents,
    )

source_filter = None if selected_document == "All documents" else selected_document
rows = search_facts(query=query, source_document=source_filter)

if not rows:
    st.info("No facts yet. Upload PDFs and run extraction.")
else:
    frame = pd.DataFrame(rows)
    display_columns = [
        "entity",
        "attribute",
        "value",
        "unit",
        "period",
        "confidence",
        "source_document",
        "page_number",
        "evidence_text",
        "id",
    ]
    st.dataframe(
        frame[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.NumberColumn(format="%.2f"),
            "page_number": st.column_config.NumberColumn(format="%d"),
            "evidence_text": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"{len(rows)} fact(s) shown. Every row includes evidence text and a page number.")
