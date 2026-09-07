"""Strict CONTRADICTS rules; PART_OF uses a revenue hierarchy detector."""

from __future__ import annotations

import re
from typing import Any

from canonicalization_service import (
    infer_canonical_attribute,
    normalized_value_key,
    periods_mergeable,
    revenue_children,
)

PERIOD_RE = re.compile(
    r"\b(?:(fy|cy|ay)\s*)?(\d{4})(?:\s*[-/]\s*\d{2,4})?\b|\bq[1-4]\b",
    re.IGNORECASE,
)
REVENUE_RE = re.compile(r"\brevenues?\b", re.IGNORECASE)
COMPONENT_PREFIXES = (
    "product",
    "products",
    "service",
    "services",
    "goods",
    "other",
    "domestic",
    "international",
    "subscription",
    "licensing",
    "license",
)
PARENT_PREFIXES = (
    "total",
    "overall",
    "consolidated",
    "combined",
    "net",
    "group",
)
TOTAL_REVENUE = "total_revenue"
CHILD_REVENUE = revenue_children()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").split())


def strip_period_tokens(text: str) -> str:
    return PERIOD_RE.sub(" ", text)


def canonical_attribute(attribute: Any) -> str:
    """Attribute identity for CONTRADICTS: period stripped, parent prefixes removed."""
    text = strip_period_tokens(normalize_text(attribute))
    parts = text.split()
    while parts and parts[0] in PARENT_PREFIXES:
        parts = parts[1:]
    return " ".join(parts).strip()


def fact_canonical_attribute(fact: dict[str, Any] | None) -> str:
    if not fact:
        return ""
    stored = str(fact.get("canonical_attribute") or "").strip()
    if stored:
        return stored
    return infer_canonical_attribute(fact)


def attribute_core(attribute: Any) -> str:
    text = canonical_attribute(attribute)
    parts = text.split()
    if parts and parts[0] in COMPONENT_PREFIXES:
        return " ".join(parts[1:]).strip() or text
    return text


def component_prefix(attribute: Any) -> str | None:
    text = canonical_attribute(attribute)
    parts = text.split()
    if parts and parts[0] in COMPONENT_PREFIXES:
        return parts[0]
    return None


def is_parent_attribute(attribute: Any) -> bool:
    return component_prefix(attribute) is None


def same_entity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_entity = normalize_text(left.get("canonical_entity") or left.get("entity"))
    right_entity = normalize_text(right.get("canonical_entity") or right.get("entity"))
    return left_entity == right_entity


def canonical_period(fact: dict[str, Any]) -> str:
    raw = " ".join(
        [
            str(fact.get("period") or ""),
            str(fact.get("attribute") or ""),
            str(fact.get("raw_attribute") or ""),
        ]
    )
    match = PERIOD_RE.search(normalize_text(raw))
    if not match:
        return ""
    year = match.group(2) if match.lastindex and match.group(2) else match.group(0)
    return normalize_text(year)


def periods_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = canonical_period(left)
    b = canonical_period(right)
    if not a or not b:
        return True
    return a == b


def same_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_cat = revenue_category(left)
    right_cat = revenue_category(right)
    if left_cat or right_cat:
        return left_cat == right_cat
    return component_prefix(left.get("attribute")) == component_prefix(right.get("attribute"))


def revenue_category(fact_or_attribute: Any) -> str | None:
    """Detect total_revenue / product_revenue / services_revenue / ..."""
    if isinstance(fact_or_attribute, dict):
        stored = fact_canonical_attribute(fact_or_attribute)
        if stored == TOTAL_REVENUE or stored in CHILD_REVENUE:
            return stored
        attribute = fact_or_attribute.get("raw_attribute") or fact_or_attribute.get("attribute")
    else:
        attribute = fact_or_attribute
    text = canonical_attribute(attribute)
    if not REVENUE_RE.search(text) and "revenue" not in normalize_text(attribute):
        inferred = None
        if isinstance(fact_or_attribute, dict):
            inferred = infer_canonical_attribute(fact_or_attribute)
        if inferred == TOTAL_REVENUE or inferred in CHILD_REVENUE:
            return inferred
        return None
    prefix = component_prefix(attribute)
    if prefix in {"product", "products"}:
        return "product_revenue"
    if prefix in {"service", "services"}:
        return "services_revenue"
    if prefix in {"subscription"}:
        return "subscription_revenue"
    if prefix in {"licensing", "license"}:
        return "licensing_revenue"
    if prefix:
        return None
    return TOTAL_REVENUE


def is_revenue_child_of_total(child: dict[str, Any], parent: dict[str, Any]) -> bool:
    child_cat = revenue_category(child)
    parent_cat = revenue_category(parent)
    return child_cat in CHILD_REVENUE and parent_cat == TOTAL_REVENUE


def is_hierarchical_child(child: dict[str, Any], parent: dict[str, Any]) -> bool:
    """PART_OF hierarchy. Children may only point at total_revenue, never at another child."""
    if child.get("id") and child.get("id") == parent.get("id"):
        return False
    if fact_canonical_attribute(child) == fact_canonical_attribute(parent):
        return False
    if revenue_category(parent) in CHILD_REVENUE:
        return False
    if is_revenue_child_of_total(child, parent):
        return True
    if revenue_category(child) in CHILD_REVENUE:
        return False
    child_attr = child.get("raw_attribute") or child.get("attribute")
    parent_attr = parent.get("raw_attribute") or parent.get("attribute")
    child_core = attribute_core(child_attr)
    parent_core = attribute_core(parent_attr)
    if not child_core or child_core != parent_core:
        return False
    child_prefix = component_prefix(child_attr)
    return bool(child_prefix and is_parent_attribute(parent_attr))


def is_sibling(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_rev = revenue_category(left)
    right_rev = revenue_category(right)
    if left_rev in CHILD_REVENUE and right_rev in CHILD_REVENUE:
        return left_rev != right_rev
    if attribute_core(left.get("attribute")) != attribute_core(right.get("attribute")):
        return False
    left_p = component_prefix(left.get("attribute"))
    right_p = component_prefix(right.get("attribute"))
    return bool(left_p and right_p and left_p != right_p)


def semantically_same_fact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("id") or "") and str(left.get("id") or "") == str(right.get("id") or ""):
        return True
    if not same_entity(left, right):
        return False
    if fact_canonical_attribute(left) != fact_canonical_attribute(right):
        return False
    if normalized_value_key(left.get("value"), left.get("unit")) != normalized_value_key(
        right.get("value"), right.get("unit")
    ):
        return False
    return periods_mergeable(left.get("period"), right.get("period"))


def allowed_relationship_types(candidate: dict[str, Any], parent: dict[str, Any]) -> list[str]:
    """PART_OF and CONTRADICTS are evaluated independently."""
    if candidate.get("id") is not None and parent.get("id") is not None:
        if str(candidate.get("id")) == str(parent.get("id")):
            return []
    if semantically_same_fact(candidate, parent):
        return []

    allowed: list[str] = []
    if _part_of_pair_eligible(candidate, parent):
        allowed.append("PART_OF")

    if _contradiction_pair_eligible(candidate, parent):
        allowed.extend(["CONTRADICTS", "CORROBORATES", "RECONCILES"])

    return sorted(set(allowed))


def _part_of_pair_eligible(candidate: dict[str, Any], parent: dict[str, Any]) -> bool:
    if is_sibling(candidate, parent):
        return False
    return is_hierarchical_child(candidate, parent)


def _contradiction_pair_eligible(candidate: dict[str, Any], parent: dict[str, Any]) -> bool:
    if not same_entity(candidate, parent):
        return False
    if not periods_compatible(candidate, parent):
        return False
    if fact_canonical_attribute(candidate) != fact_canonical_attribute(parent):
        return False
    if not same_scope(candidate, parent):
        return False
    return True


def filter_relationship_candidates(
    parent: dict[str, Any],
    nearby: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (plausible candidates, rejection log rows)."""
    plausible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in nearby:
        allowed = allowed_relationship_types(candidate, parent)
        if allowed:
            plausible.append({"candidate": candidate, "allowed_types": allowed})
            continue
        rejected.append(
            _rejection_row(
                candidate,
                parent,
                _proposed_type(candidate, parent),
                _reject_reason(candidate, parent),
            )
        )
    return plausible, rejected


def _proposed_type(candidate: dict[str, Any], parent: dict[str, Any]) -> str:
    if is_sibling(candidate, parent):
        return "PART_OF"
    if is_hierarchical_child(candidate, parent) or is_hierarchical_child(parent, candidate):
        return "PART_OF"
    if fact_canonical_attribute(candidate) != fact_canonical_attribute(parent):
        return "CONTRADICTS"
    return "UNKNOWN"


def _reject_reason(candidate: dict[str, Any], parent: dict[str, Any]) -> str:
    if is_sibling(candidate, parent):
        return "Sibling facts cannot be PART_OF each other."
    if not same_entity(candidate, parent) and not is_hierarchical_child(candidate, parent):
        if fact_canonical_attribute(candidate) == fact_canonical_attribute(parent):
            return "Entities differ."
    if is_hierarchical_child(parent, candidate):
        return "Parent cannot be PART_OF its child."
    if fact_canonical_attribute(candidate) != fact_canonical_attribute(parent):
        if not is_hierarchical_child(candidate, parent):
            return "attribute mismatch"
    if not periods_compatible(candidate, parent):
        return "Periods are incompatible."
    return "Candidate pair is not eligible."


def _rejection_row(
    source: dict[str, Any],
    target: dict[str, Any],
    proposed: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "source_fact": _statement(source),
        "target_fact": _statement(target),
        "candidate": _statement(source),
        "proposed_relationship": proposed,
        "accepted": False,
        "reason": reason,
        "source_fact_id": source.get("id"),
        "target_fact_id": target.get("id"),
        "source_canonical_attribute": fact_canonical_attribute(source),
        "target_canonical_attribute": fact_canonical_attribute(target),
        "source_raw_attribute": source.get("raw_attribute") or source.get("attribute"),
        "target_raw_attribute": target.get("raw_attribute") or target.get("attribute"),
    }


def _statement(fact: dict[str, Any]) -> str:
    attribute = str(fact.get("canonical_attribute") or fact.get("attribute") or "").strip()
    value = str(fact.get("value") or "").strip()
    unit = str(fact.get("unit") or "").strip()
    if unit and unit.lower() not in value.lower():
        value = f"{value} {unit}".strip()
    return f"{attribute} = {value}".strip(" =")
