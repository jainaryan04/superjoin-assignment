from __future__ import annotations

import streamlit as st

from database import get_fact, get_relationship, init_db, list_all_relationships

st.set_page_config(page_title="Relationship detail", layout="wide")
init_db()
st.title("Relationship Detail")

rel_id = st.query_params.get("rel_id") or st.session_state.get("selected_relationship_id")
if not rel_id:
    options = list_all_relationships()
    if not options:
        st.info("No relationships stored yet.")
        st.stop()
    labels = {
        f"{item.get('relationship_type')} · {item.get('source_fact')} → {item.get('target_fact')}": item["id"]
        for item in options
    }
    selected = st.selectbox("Relationship", list(labels.keys()))
    rel_id = labels[selected]

item = get_relationship(str(rel_id))
if item is None:
    st.warning("That relationship was not found.")
    st.stop()

st.markdown(f"**{item.get('relationship_type')}**")
st.write(item.get("reasoning") or "No reasoning stored.")
st.metric("Confidence", float(item.get("confidence") or 0))

cols = st.columns(2)
with cols[0]:
    st.markdown("#### Source fact")
    source = get_fact(str(item["source_fact_id"]))
    if source:
        st.write(source.get("canonical_attribute"), "=", source.get("value"))
        st.caption(source.get("id"))
with cols[1]:
    st.markdown("#### Target fact")
    target = get_fact(str(item["target_fact_id"]))
    if target:
        st.write(target.get("canonical_attribute"), "=", target.get("value"))
        st.caption(target.get("id"))

st.markdown("#### Supporting fact IDs")
st.write(item.get("supporting_fact_ids") or [])
if item.get("computed_components"):
    st.markdown("#### Computed components")
    st.write(item.get("computed_components"))

st.markdown("#### Supporting evidence")
evidence_ids = set(item.get("supporting_evidence_ids") or [])
shown = False
for fact_id in item.get("supporting_fact_ids") or [item["source_fact_id"], item["target_fact_id"]]:
    fact = get_fact(str(fact_id))
    if not fact:
        continue
    for snippet in fact.get("evidence") or []:
        if evidence_ids and snippet.get("id") not in evidence_ids:
            continue
        shown = True
        with st.expander(
            f"{snippet.get('source_document')} · page {snippet.get('page_number')}",
            expanded=True,
        ):
            st.write(snippet.get("evidence_text"))
if not shown:
    st.info("No supporting evidence snippets were stored for this relationship.")
