from __future__ import annotations

import pandas as pd
import streamlit as st

from database import init_db
from fact_integrity import list_fact_integrity

st.set_page_config(page_title="Fact integrity", layout="wide")
init_db()
st.title("Fact integrity")
st.caption(
    "Facts are immutable after extraction. This page flags stored rows whose "
    "attribute or value drifted from the original snapshot."
)
rows = list_fact_integrity()
if not rows:
    st.info("No facts stored yet.")
    st.stop()
failed = sum(1 for row in rows if row["integrity_status"] == "FAIL")
if failed:
    st.error(f"{failed} fact(s) failed integrity checks.")
else:
    st.success("All facts passed integrity checks.")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Fact ID": row["id"],
                "Raw Attribute": row["raw_attribute"],
                "Canonical Attribute": row["canonical_attribute"],
                "Value": row["value"],
                "Original Value": row["original_value"],
                "Integrity Status": row["integrity_status"],
                "Reason": row["reason"],
            }
            for row in rows
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
