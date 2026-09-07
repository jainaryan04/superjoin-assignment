from __future__ import annotations

import pandas as pd
import streamlit as st

from database import (
    get_fact,
    init_db,
    list_source_documents,
    search_facts,
    validate_relationships,
)
from fact_extractor import client_from_settings
from provenance_service import debug_rows, link_fact_relationships
from fact_details import render_fact_details

st.set_page_config(page_title="Provenance debug", layout="wide")
init_db()
st.title("Provenance debug")
st.caption("Evidence is source text. Relationships are connections between facts.")

documents = list_source_documents()
selected_document = st.selectbox(
    "Source document",
    options=["All documents"] + documents,
)
source_filter = None if selected_document == "All documents" else selected_document

actions = st.columns(2)
with actions[0]:
    if st.button("Re-run relationship linking"):
        try:
            llm = client_from_settings()
        except ValueError:
            llm = None
        result = link_fact_relationships(source_filter, llm=llm)
        st.success(
            f"Touched {result['facts_touched']} fact(s); "
            f"added {result['relationships_added']} relationship(s)."
        )
        st.rerun()
with actions[1]:
    report = validate_relationships()
    if report["ok"]:
        st.success("Validation passed: no self-links, no duplicates, counts match.")
    else:
        st.error(
            f"Validation failed — self-links: {len(report['self_links'])}, "
            f"duplicates: {len(report['duplicates'])}, "
            f"count mismatches: {len(report['count_mismatches'])}."
        )
        st.json(report)

rows = search_facts(source_document=source_filter)
debug = debug_rows(rows)
if not debug:
    st.info("No facts stored yet.")
    st.stop()

overview = pd.DataFrame(
    [
        {
            "Fact": row["fact"],
            "Evidence Count": row["evidence_count"],
            "Relationship Count": row["relationship_count"],
            "id": row["id"],
        }
        for row in debug
    ]
)
st.subheader("Facts")
event = st.dataframe(
    overview[["Fact", "Evidence Count", "Relationship Count", "id"]],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="provenance_debug_table",
)

selected_id = None
selected_rows = event.selection.rows if event.selection else []
if selected_rows:
    selected_id = str(overview.iloc[selected_rows[0]]["id"])
elif debug:
    selected_id = str(debug[0]["id"])
if not selected_id:
    st.stop()

selected = next((row for row in debug if row["id"] == selected_id), None)
fact = get_fact(selected_id)
if fact is None or selected is None:
    st.warning("Selected fact is no longer available.")
    st.stop()

st.divider()
render_fact_details(fact)

with st.expander("Evidence Records", expanded=True):
    if selected["evidence_records"]:
        st.dataframe(pd.DataFrame(selected["evidence_records"]), use_container_width=True, hide_index=True)
    else:
        st.write("None")

with st.expander("Relationship Records", expanded=True):
    records = selected["relationship_records"]
    if records:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "source": item.get("source_statement"),
                        "relationship_type": item.get("relationship_type"),
                        "target": item.get("target_statement"),
                        "confidence": item.get("confidence"),
                        "reasoning": item.get("reasoning"),
                    }
                    for item in records
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("None")
