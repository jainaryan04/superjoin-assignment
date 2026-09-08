"""Compare stored facts against the original extracted snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from canonicalization_service import canonicalize_fact, revenue_children
from database import search_facts

CHILD_REVENUE = revenue_children()


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).lower()


def integrity_status(fact: dict[str, Any], cohort: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    original_entity = fact.get("original_entity") or fact.get("entity")
    original_attribute = fact.get("original_attribute") or fact.get("raw_attribute") or fact.get("attribute")
    original_value = fact.get("original_value") if fact.get("original_value") not in (None, "") else fact.get("value")
    original_period = (
        fact.get("original_period")
        if fact.get("original_period") not in (None, "")
        else fact.get("period")
    )
    expected = canonicalize_fact(
        {
            "entity": original_entity,
            "attribute": original_attribute,
            "value": original_value,
            "period": original_period,
            "original_entity": original_entity,
            "original_attribute": original_attribute,
            "original_value": original_value,
            "original_period": original_period,
        }
    )
    stored_attr = str(fact.get("canonical_attribute") or fact.get("attribute") or "").strip()
    stored_value = str(fact.get("value") or "").strip()
    reasons: list[str] = []
    if _norm(stored_value) != _norm(expected.get("value")):
        reasons.append("Value changed after extraction.")
    if stored_attr != expected["canonical_attribute"]:
        reasons.append(
            f"Attribute changed unexpectedly ({original_attribute!r} -> {stored_attr!r})."
        )
    if cohort:
        total_values = {
            _norm(row.get("value"))
            for row in cohort
            if (row.get("canonical_attribute") or row.get("attribute")) == "total_revenue"
            and row.get("id") != fact.get("id")
        }
        if stored_attr in CHILD_REVENUE and _norm(stored_value) in total_values:
            reasons.append("Synthetic child fact: child value matches a total_revenue value.")
    status = "PASS" if not reasons else "FAIL"
    return {
        "id": fact.get("id"),
        "raw_attribute": fact.get("raw_attribute") or original_attribute,
        "canonical_attribute": stored_attr,
        "value": stored_value,
        "original_value": original_value,
        "integrity_status": status,
        "reason": " ".join(reasons) if reasons else "Stored fact matches original extraction.",
    }


def list_fact_integrity(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    rows = search_facts(db_path=db_path)
    return [integrity_status(row, rows) for row in rows]
