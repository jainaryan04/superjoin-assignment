from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from database import (
    clear_all_facts,
    delete_facts_for_document,
    get_fact,
    init_db,
    insert_facts,
    list_source_documents,
    search_facts,
)
from fact_extractor import FactExtractor, client_from_settings
from pdf_processor import process_pdf
from provenance_service import link_fact_relationships
from fact_details import render_fact_details

load_dotenv()


def _bridge_streamlit_secrets() -> list[str]:
    """Copy anything key-shaped from st.secrets into the environment.

    Streamlit Cloud stores deployment secrets in ``st.secrets`` and does not
    reliably export them as env vars, so the whole app (which reads ``os.getenv``)
    misses them. This scans every top-level entry and every ``[section]`` for the
    settings we use, case-insensitively. Returns the names of the secrets it saw
    (never their values) for the diagnostic below.
    """
    wanted = {
        "openai_api_key": "OPENAI_API_KEY",
        "llm_api_key": "LLM_API_KEY",
        "api_key": "OPENAI_API_KEY",
        "openai_model": "OPENAI_MODEL",
        "llm_model": "LLM_MODEL",
        "model": "OPENAI_MODEL",
        "openai_base_url": "OPENAI_BASE_URL",
        "llm_base_url": "LLM_BASE_URL",
        "base_url": "OPENAI_BASE_URL",
    }
    seen: list[str] = []
    try:
        items = list(st.secrets.items())
    except Exception:
        return seen
    for key, value in items:
        if hasattr(value, "items"):  # a [section]
            for sub_key, sub_value in value.items():
                seen.append(f"{key}.{sub_key}")
                target = wanted.get(str(sub_key).lower())
                if target and sub_value and not os.getenv(target):
                    os.environ[target] = str(sub_value)
        else:
            seen.append(str(key))
            target = wanted.get(str(key).lower())
            if target and value and not os.getenv(target):
                os.environ[target] = str(value)
    return seen


_secret_names = _bridge_streamlit_secrets()

init_db()

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")
st.title("Fact Knowledge Layer")
st.caption(
    "Upload PDFs to extract entity–attribute–value facts, each grounded in a source snippet. "
    "Use the pages in the sidebar for canonical clusters, similar facts and relationships."
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
    if not api_key:
        st.warning("No API key found in env, .env, secrets, or the field above.")
        with st.expander("What the app can see"):
            st.write("st.secrets entries:", _secret_names or "(none)")
            st.write(
                "env OPENAI_API_KEY set:",
                bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")),
            )
            st.caption(
                "On Streamlit Cloud, put `OPENAI_API_KEY = \"sk-...\"` in "
                "Settings → Secrets (top level), then Reboot the app."
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
            stored = insert_facts(records)
            link_result = link_fact_relationships(llm=extractor.llm)
            total_facts += stored
            st.write(
                f"Stored {stored} canonical facts from `{source_name}` "
                f"and added {link_result['relationships_added']} relationship(s)."
            )

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

show_raw = st.toggle(
    "Show canonical & raw columns",
    value=False,
    help="Adds the pre-normalisation entity/attribute/value alongside the canonical form.",
)

source_filter = None if selected_document == "All documents" else selected_document
rows = search_facts(query=query, source_document=source_filter)

# One friendly header per column; the same map also drives which columns show.
COLUMN_CONFIG = {
    "entity": st.column_config.TextColumn("Entity"),
    "attribute": st.column_config.TextColumn("Attribute"),
    "value": st.column_config.TextColumn("Value"),
    "unit": st.column_config.TextColumn("Unit"),
    "period": st.column_config.TextColumn("Period"),
    "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
    "document_count": st.column_config.NumberColumn("Docs", format="%d"),
    "evidence_count": st.column_config.NumberColumn("Evidence", format="%d"),
    "relationship_count": st.column_config.NumberColumn("Links", format="%d"),
    "source_document": st.column_config.TextColumn("Primary source"),
    "page_number": st.column_config.NumberColumn("Page", format="%d"),
    "evidence_text": st.column_config.TextColumn("Evidence snippet", width="large"),
    "canonical_entity": st.column_config.TextColumn("Canonical entity"),
    "raw_attribute": st.column_config.TextColumn("Raw attribute"),
    "canonical_value": st.column_config.TextColumn("Canonical value"),
    "original_value": st.column_config.TextColumn("Original value"),
}
CORE_COLUMNS = [
    "entity", "attribute", "value", "unit", "period", "confidence",
    "document_count", "evidence_count", "relationship_count",
    "source_document", "page_number", "evidence_text",
]
RAW_COLUMNS = ["canonical_entity", "raw_attribute", "canonical_value", "original_value"]

if not rows:
    st.info("No facts yet. Upload PDFs and run extraction.")
else:
    if st.session_state.pop("returning_from_detail", False):
        st.session_state["facts_table_nonce"] = st.session_state.get("facts_table_nonce", 0) + 1

    frame = pd.DataFrame(rows)
    display_columns = CORE_COLUMNS + (RAW_COLUMNS if show_raw else [])
    display_columns = [column for column in display_columns if column in frame.columns]
    event = st.dataframe(
        frame[display_columns],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"facts_table_{st.session_state.get('facts_table_nonce', 0)}",
        column_config=COLUMN_CONFIG,
    )
    st.caption(
        f"{len(rows)} fact(s). *Evidence* counts source snippets, *Links* counts related facts. "
        "Select a row to see its evidence and relationships."
    )

    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        selected_id = str(frame.iloc[selected_rows[0]]["id"])
        st.session_state["selected_fact_id"] = selected_id
        st.query_params["fact_id"] = selected_id
        fact = get_fact(selected_id)
        if fact:
            st.divider()
            render_fact_details(fact)
            if st.button("Open full fact detail page"):
                st.switch_page("pages/7_Fact_Detail.py")
