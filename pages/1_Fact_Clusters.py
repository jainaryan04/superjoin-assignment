from __future__ import annotations

import pandas as pd
import streamlit as st

from database import init_db, list_fact_clusters, rebuild_fact_clusters, search_facts

st.set_page_config(page_title="Fact Clusters", layout="wide")
init_db()
st.title("Canonical Fact Clusters")
st.caption(
    "Facts that share a canonical attribute, canonical value, and period are grouped. "
    "Raw fact instances remain on the main page."
)

actions = st.columns(2)
with actions[0]:
    if st.button("Rebuild clusters"):
        count = rebuild_fact_clusters()
        st.success(f"Stored {count} cluster(s).")
        st.rerun()

clusters = list_fact_clusters()
if not clusters:
    rebuilt = rebuild_fact_clusters()
    clusters = list_fact_clusters()
    if not rebuilt:
        st.info("No facts stored yet.")
        st.stop()

st.subheader("Clusters")
for cluster in clusters:
    attribute = cluster.get("canonical_attribute") or "value"
    value = cluster.get("canonical_value") or ""
    period = cluster.get("period") or "unspecified period"
    documents = cluster.get("supporting_documents") or []
    with st.expander(
        f"{attribute} = {value} · {period} · {cluster.get('document_count') or 0} document(s)",
        expanded=int(cluster.get("document_count") or 0) > 1,
    ):
        st.markdown(f"**{attribute} = {value}**")
        st.markdown(f"**{period}**")
        st.markdown("Corroborated by:")
        if documents:
            for name in documents:
                st.markdown(f"- {name}")
        else:
            st.write("No source documents attached.")
        metrics = st.columns(2)
        metrics[0].metric("Documents", cluster.get("document_count") or 0)
        metrics[1].metric("Evidence", cluster.get("evidence_count") or 0)
        facts = [
            row
            for row in search_facts()
            if row["id"] in set(cluster.get("supporting_fact_ids") or [])
        ]
        if facts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "id": row["id"],
                            "value": row.get("value"),
                            "source_document": row.get("source_document"),
                            "page_number": row.get("page_number"),
                        }
                        for row in facts
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
