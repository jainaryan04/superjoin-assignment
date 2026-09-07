from __future__ import annotations

import pandas as pd
import streamlit as st

from database import (
    delete_duplicate_relationships,
    init_db,
    list_all_relationships,
    search_facts,
    validate_relationships,
)
from fact_extractor import client_from_settings
from provenance_service import (
    annotate_relationship_validations,
    load_candidate_debug_log,
    rebuild_relationships,
)

st.set_page_config(page_title="Relationship debug", layout="wide")
init_db()
st.title("Relationship debug")
st.caption("Inspect stored links and the last linking pass. Evidence is not shown here.")

actions = st.columns(3)
with actions[0]:
    if st.button("Rebuild relationships"):
        try:
            llm = client_from_settings()
        except ValueError:
            llm = None
        result = rebuild_relationships(llm=llm)
        st.success("Relationships rebuilt.")
        counts = st.columns(4)
        counts[0].metric("Raw Relationships Generated", result.get("raw_relationships_generated", 0))
        counts[1].metric("Duplicate Relationships Removed", result.get("duplicate_relationships_removed", 0))
        counts[2].metric("Self Relationships Removed", result.get("self_relationships_removed", 0))
        counts[3].metric("Final Relationships Stored", result.get("final_relationships_stored", 0))
with actions[1]:
    report = validate_relationships()
    if report["ok"]:
        st.success("Structural validation passed: no self-links, no duplicates, counts match.")
    else:
        st.error("Structural validation failed.")
        st.json(
            {
                "self_links": report["self_links"],
                "duplicates": report["duplicates"],
                "count_mismatches": report["count_mismatches"],
            }
        )
with actions[2]:
    if st.button("Remove duplicate relationships"):
        removed = delete_duplicate_relationships()
        st.success(f"Removed {removed} duplicate row(s).")
        st.rerun()

facts = search_facts()
overview = pd.DataFrame(
    [
        {
            "Fact": f"{row.get('canonical_attribute') or row['attribute']} = {row['value']}",
            "Raw Attribute": row.get("raw_attribute") or "",
            "Canonical Attribute": row.get("canonical_attribute") or row.get("attribute") or "",
            "Relationship Count": row["relationship_count"],
            "id": row["id"],
        }
        for row in facts
    ]
)
if overview.empty:
    st.info("No facts stored yet.")
    st.stop()

st.subheader("Facts")
event = st.dataframe(
    overview,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="relationship_debug_table",
)

all_rels = annotate_relationship_validations(list_all_relationships())
selected_rows = event.selection.rows if event.selection else []
selected_id = str(overview.iloc[selected_rows[0]]["id"]) if selected_rows else None

st.subheader("Stored relationships")
if selected_id:
    records = [
        item
        for item in all_rels
        if item["source_fact_id"] == selected_id or item["target_fact_id"] == selected_id
    ]
    st.caption("Showing relationships for the selected fact.")
else:
    records = all_rels
    st.caption("Select a fact row to filter. Showing every relationship.")

if records:
    failed = sum(1 for item in records if item.get("validation_result") == "FAILED")
    if failed:
        st.error(f"{failed} relationship(s) failed semantic validation.")
    else:
        st.success("All visible relationships passed semantic validation.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Source": item.get("source_fact") or item.get("source_statement"),
                    "Target": item.get("target_fact") or item.get("target_statement"),
                    "Source Canonical Attribute": item.get("source_canonical_attribute")
                    or (item.get("source_fact_record") or {}).get("canonical_attribute")
                    or item.get("source_attribute"),
                    "Target Canonical Attribute": item.get("target_canonical_attribute")
                    or (item.get("target_fact_record") or {}).get("canonical_attribute")
                    or item.get("target_attribute"),
                    "Relationship": item.get("relationship_type"),
                    "Accepted/Rejected": (
                        "Accepted" if item.get("validation_result") == "PASSED" else "Rejected"
                    ),
                    "Reason": item.get("validation_reason") or item.get("reasoning"),
                }
                for item in records
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("No relationship records.")

st.subheader("Last linking pass")
st.caption("Every proposed pair from the last rebuild, with accept/reject reasons.")
candidates = load_candidate_debug_log()
if not candidates:
    st.info("No candidate log yet. Extract facts or rebuild relationships.")
else:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Source": item.get("source_fact"),
                    "Target": item.get("target_fact"),
                    "Source Canonical Attribute": item.get("source_canonical_attribute"),
                    "Target Canonical Attribute": item.get("target_canonical_attribute"),
                    "Relationship": item.get("proposed_relationship"),
                    "Accepted/Rejected": "Accepted" if item.get("accepted") else "Rejected",
                    "Reason": item.get("reason"),
                }
                for item in candidates
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"{len(candidates)} candidate row(s).")
