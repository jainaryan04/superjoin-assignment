from __future__ import annotations

import pandas as pd
import streamlit as st

from database import format_fact_statement, get_fact, init_db, search_facts
from semantic_search_service import embedding_text, index_all_facts, similar_facts

st.set_page_config(page_title="Similar Facts", layout="wide")
init_db()

st.title("Similar Facts")
st.caption(
    "Select a stored fact to retrieve nearest neighbors from the FAISS index. "
    "Embeddings use canonical entity, attribute, value, and period."
)

rows = search_facts()
if not rows:
    st.info("No facts yet. Extract facts from the main page first.")
    st.stop()

if st.button("Rebuild embeddings"):
    updated = index_all_facts()
    st.success(f"Indexed {len(rows)} fact(s); re-encoded {updated}.")

options = {
    f"{format_fact_statement(row)}  ·  {row['id'][:8]}": row["id"] for row in rows
}
selected_label = st.selectbox("Fact", options=list(options.keys()))
k = st.slider("Neighbors", min_value=1, max_value=min(20, max(len(rows) - 1, 1)), value=min(5, max(len(rows) - 1, 1)))

fact_id = options[selected_label]
fact = get_fact(fact_id)
if fact is None:
    st.warning("That fact was not found.")
    st.stop()

st.markdown("#### Selected fact")
meta = st.columns(4)
meta[0].markdown(f"**Canonical entity**  \n{fact.get('canonical_entity') or '—'}")
meta[1].markdown(f"**Canonical attribute**  \n{fact.get('canonical_attribute') or '—'}")
meta[2].markdown(f"**Canonical value**  \n{fact.get('canonical_value') or '—'}")
meta[3].markdown(f"**Period**  \n{fact.get('period') or '—'}")
st.caption(f"Embedding text: `{embedding_text(fact)}`")

neighbors = similar_facts(fact_id, k=k)
st.markdown("#### Nearest neighbors")
if not neighbors:
    st.info("No neighboring facts in the index yet.")
else:
    frame = pd.DataFrame(
        [
            {
                "similarity": row["similarity"],
                "canonical_entity": row.get("canonical_entity"),
                "canonical_attribute": row.get("canonical_attribute"),
                "canonical_value": row.get("canonical_value"),
                "period": row.get("period"),
                "value": row.get("value"),
                "embedding_text": row.get("embedding_text"),
                "id": row.get("id"),
            }
            for row in neighbors
        ]
    )
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "similarity": st.column_config.NumberColumn(format="%.4f"),
        },
    )
