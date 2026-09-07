"""Fact detail view: fact information, evidence snippets, related facts."""

from __future__ import annotations

import streamlit as st

from database import format_fact_statement, get_fact, init_db, provenance_dot
from provenance_service import evidence_tree_text


def render(embedded: bool = False) -> None:
    init_db()
    fact_id = st.query_params.get("fact_id") or st.session_state.get("selected_fact_id")

    if not embedded:
        if st.button("← Back to facts"):
            st.session_state["returning_from_detail"] = True
            st.query_params.clear()
            st.switch_page("app.py")

    if not fact_id:
        st.subheader("Fact Details")
        st.info("Select a fact to inspect evidence and related facts.")
        return

    fact = get_fact(str(fact_id))
    if fact is None:
        st.subheader("Fact Details")
        st.warning("That fact was not found. It may have been cleared.")
        return

    render_fact_details(fact)


def render_fact_details(fact: dict) -> None:
    statement = format_fact_statement(fact)
    st.subheader("Fact Details")
    st.markdown(f"**{statement}**")

    st.markdown("#### Fact Information")
    info = st.columns(6)
    info[0].markdown(f"**Entity**  \n{fact.get('entity') or '—'}")
    info[1].markdown(f"**Raw Attribute**  \n{fact.get('raw_attribute') or '—'}")
    info[2].markdown(f"**Canonical Attribute**  \n{fact.get('canonical_attribute') or fact.get('attribute') or '—'}")
    info[3].markdown(f"**Value**  \n{fact.get('value') or '—'}")
    info[4].markdown(f"**Period**  \n{fact.get('period') or '—'}")
    info[5].markdown(f"**Confidence**  \n{float(fact.get('confidence') or 0):.2f}")

    counts = st.columns(2)
    counts[0].metric("Evidence Count", fact.get("evidence_count") or 0)
    counts[1].metric("Relationship Count", fact.get("relationship_count") or 0)

    st.markdown("#### Evidence")
    evidence = fact.get("evidence") or []
    if not evidence:
        st.info("No evidence snippets are attached to this fact.")
    else:
        for item in evidence:
            snippet = item.get("evidence_text") or ""
            header = (
                f"{item.get('evidence_type', 'DIRECT')} · "
                f"{item.get('source_document')} · Page {item.get('page_number')}"
            )
            with st.expander(header, expanded=True):
                meta = st.columns(4)
                meta[0].markdown(f"**Evidence Type**  \n{item.get('evidence_type', 'DIRECT')}")
                meta[1].markdown(f"**Source Document**  \n{item.get('source_document')}")
                meta[2].markdown(f"**Page Number**  \n{item.get('page_number')}")
                meta[3].markdown(f"**Confidence**  \n{float(item.get('confidence') or 0):.2f}")
                st.markdown("**Evidence Text**")
                st.write(snippet)

    st.markdown("#### Provenance tree")
    st.code(evidence_tree_text(fact), language=None)
    try:
        st.graphviz_chart(provenance_dot(fact))
    except Exception:
        pass

    st.markdown("#### Relationship Details")
    relationships = fact.get("relationships") or []
    if not relationships:
        st.caption("No fact-to-fact relationships.")
        return
    for rel in relationships:
        arrow = rel.get("arrow") or (
            f"{rel.get('source_statement')} -> {rel.get('target_statement')}"
        )
        header = f"{rel.get('relationship_type')} · {arrow}"
        with st.expander(header, expanded=True):
            st.markdown(f"**Related Fact**  \n{rel.get('related_statement')}")
            cols = st.columns(3)
            cols[0].markdown(f"**Relationship Type**  \n`{rel.get('relationship_type')}`")
            cols[1].markdown(f"**Direction**  \n{rel.get('direction_label') or rel.get('direction')}")
            cols[2].markdown(f"**Confidence**  \n{float(rel.get('confidence') or 0):.2f}")
            st.markdown(f"**Source**  \n{rel.get('source_statement')}")
            st.markdown(f"**Target**  \n{rel.get('target_statement')}")
            st.markdown(f"**Arrow**  \n`{arrow}`")
            st.markdown("**Reasoning**")
            st.write(rel.get("reasoning") or "—")
