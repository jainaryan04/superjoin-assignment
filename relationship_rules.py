"""Strict CONTRADICTS rules; PART_OF uses a revenue hierarchy detector."""

from __future__ import annotations

import re
from typing import Any

from canonicalization_service import (
    dimensions_compatible,
    infer_canonical_attribute,
    normalized_value_key,
    periods_mergeable,
    revenue_children,
    value_dimension,
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


# Words that only qualify a metric, not the subject it belongs to. "Annual
# Revenue" and "Revenue" are the same subject; "Acme" and "Globex" are not.
_ENTITY_QUALIFIERS = {
    "annual", "total", "net", "gross", "consolidated", "combined", "overall",
    "group", "reported", "restated", "adjusted", "the", "a",
}


def _entity_identity_key(fact: dict[str, Any]) -> str:
    raw = normalize_text(
        fact.get("original_entity")
        or fact.get("canonical_entity")
        or fact.get("entity")
    )
    stripped = " ".join(p for p in raw.split() if p not in _ENTITY_QUALIFIERS)
    return stripped or raw


def same_entity_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Entity identity for cross-fact reasoning.

    Uses the pre-canonicalization ``original_entity`` (so two companies' CEOs,
    which both canonicalize to "CEO", are never linked), but ignores pure metric
    qualifiers so an extractor that writes "Annual Revenue" in one document and
    "Revenue" in another is still recognised as the same subject.
    """
    return _entity_identity_key(left) == _entity_identity_key(right)


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


def same_period(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = normalize_text(left.get("period"))
    b = normalize_text(right.get("period"))
    return bool(a and b and a == b)


def periods_differ(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a = normalize_text(left.get("period"))
    b = normalize_text(right.get("period"))
    if not a and not b:
        return True
    if not a or not b:
        return True
    return a != b


def values_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return normalized_value_key(
        left.get("canonical_value") or left.get("value"), left.get("unit")
    ) == normalized_value_key(
        right.get("canonical_value") or right.get("value"), right.get("unit")
    )


def numeric_amount(fact: dict[str, Any]) -> tuple[float | None, str]:
    key = normalized_value_key(fact.get("canonical_value") or fact.get("value"), fact.get("unit"))
    amount_text, _, unit = key.partition("|")
    try:
        return float(amount_text), unit
    except (TypeError, ValueError):
        return None, unit


def values_approximately_equivalent(
    left: dict[str, Any], right: dict[str, Any], *, tolerance: float = 0.05
) -> bool:
    left_n, left_u = numeric_amount(left)
    right_n, right_u = numeric_amount(right)
    if left_n is None or right_n is None:
        return False
    if left_u and right_u and left_u != right_u:
        return False
    denom = max(abs(left_n), abs(right_n), 1e-9)
    return abs(left_n - right_n) / denom <= tolerance


def fact_dimension(fact: dict[str, Any]) -> str:
    return value_dimension(
        fact.get("canonical_value") or fact.get("value"), fact.get("unit")
    )


def values_comparable(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Two facts are comparable only when their value dimensions line up.

    A money figure and a percentage are not in disagreement; they measure
    different things, so they must not produce CORROBORATES or CONTRADICTS.
    """
    return dimensions_compatible(
        left.get("canonical_value") or left.get("value"),
        left.get("unit"),
        right.get("canonical_value") or right.get("value"),
        right.get("unit"),
    )


def values_corroborate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return values_equivalent(left, right) or values_approximately_equivalent(left, right)


TRANSITION_RE = re.compile(
    r"\b(resign(?:ed|ation)?|appoint(?:ed|ment)?|succeeded|replaced|took over|outgoing|incoming|formerly)\b",
    re.IGNORECASE,
)


def _fact_texts(fact: dict[str, Any]) -> list[str]:
    texts = [
        str(fact.get("value") or ""),
        str(fact.get("canonical_value") or ""),
        str(fact.get("evidence_text") or ""),
        str(fact.get("entity") or ""),
    ]
    for item in fact.get("evidence") or []:
        texts.append(str(item.get("evidence_text") or ""))
        texts.append(str(item.get("source_document") or ""))
    return texts


def _fact_blob(fact: dict[str, Any]) -> str:
    """Lower-cased searchable text for a fact, memoised on the dict."""
    blob = fact.get("_text_blob")
    if blob is None:
        blob = " ".join(_fact_texts(fact)).lower()
        fact["_text_blob"] = blob
    return blob


def _fact_mentions_transition(fact: dict[str, Any]) -> bool:
    return bool(TRANSITION_RE.search(_fact_blob(fact)))


# Keyed by id() of the cohort list. link_fact_relationships holds one cohort for
# a whole run and calls reset_relationship_caches() before it starts, so this
# never serves a stale list.
_TRANSITION_CACHE: dict[int, tuple[int, list[dict[str, Any]]]] = {}


def reset_relationship_caches() -> None:
    _TRANSITION_CACHE.clear()


def transition_candidates(cohort: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Facts in the cohort whose text mentions a transition, computed once.

    Non-matching facts contribute nothing to ``find_transition_support`` (they
    are skipped immediately), so pre-filtering turns a full O(N) cohort scan per
    pair into an O(k) scan over the handful of transition facts.
    """
    if not cohort:
        return []
    key = id(cohort)
    cached = _TRANSITION_CACHE.get(key)
    if cached is not None and cached[0] == len(cohort):
        return cached[1]
    matches = [fact for fact in cohort if _fact_mentions_transition(fact)]
    _TRANSITION_CACHE[key] = (len(cohort), matches)
    return matches


def find_transition_support(
    left: dict[str, Any],
    right: dict[str, Any],
    cohort: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Return supporting fact/evidence IDs that mention a real transition."""
    names = {
        str(left.get("value") or "").strip().lower(),
        str(right.get("value") or "").strip().lower(),
    }
    names.discard("")
    fact_ids: list[str] = []
    evidence_ids: list[str] = []
    seen_facts: set[str] = set()
    candidates = list(transition_candidates(cohort))
    seen_ids = {id(fact) for fact in candidates}
    for fact in (left, right):
        if id(fact) not in seen_ids:
            candidates.append(fact)
    for fact in candidates:
        blob = _fact_blob(fact)
        if not TRANSITION_RE.search(blob):
            continue
        if names and not any(name and name in blob for name in names):
            continue
        fact_id = str(fact.get("id") or "")
        if fact_id and fact_id not in seen_facts:
            seen_facts.add(fact_id)
            fact_ids.append(fact_id)
        for item in fact.get("evidence") or []:
            snippet = str(item.get("evidence_text") or "").lower()
            if TRANSITION_RE.search(snippet) and item.get("id"):
                evidence_ids.append(str(item["id"]))
    return {"fact_ids": fact_ids, "evidence_ids": evidence_ids}


def pair_evidence_ids(*facts: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for fact in facts:
        for item in fact.get("evidence") or []:
            if item.get("id"):
                ids.append(str(item["id"]))
    return ids


def semantically_same_fact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("id") or "") and str(left.get("id") or "") == str(right.get("id") or ""):
        return True
    return False


def classify_relationship(
    left: dict[str, Any],
    right: dict[str, Any],
    cohort: list[dict[str, Any]] | None = None,
) -> str | None:
    detail = classify_relationship_detail(left, right, cohort=cohort)
    return None if detail is None else str(detail["relationship_type"])


def classify_relationship_detail(
    left: dict[str, Any],
    right: dict[str, Any],
    cohort: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the relationship payload for a distinct fact pair, if any."""
    if not left or not right:
        return None
    if str(left.get("id") or "") and str(left.get("id") or "") == str(right.get("id") or ""):
        return None
    if fact_canonical_attribute(left) != fact_canonical_attribute(right):
        return None
    if not same_entity_identity(left, right):
        return None
    if not values_comparable(left, right):
        return None
    attr = fact_canonical_attribute(left)
    support_ids = [str(left.get("id") or ""), str(right.get("id") or "")]
    evidence_ids = pair_evidence_ids(left, right)
    approx = values_approximately_equivalent(left, right) and not values_equivalent(left, right)
    if values_corroborate(left, right):
        reasoning = (
            "Values are approximately equivalent within 5% tolerance."
            if approx
            else "Same canonical attribute and normalized value."
        )
        return {
            "relationship_type": "CORROBORATES",
            "reasoning": reasoning,
            "supporting_fact_ids": support_ids,
            "supporting_evidence_ids": evidence_ids,
            "confidence": 0.84 if approx else 0.9,
        }
    if same_period(left, right):
        return {
            "relationship_type": "CONTRADICTS",
            "reasoning": "Same canonical attribute and period but different normalized values.",
            "supporting_fact_ids": support_ids,
            "supporting_evidence_ids": evidence_ids,
            "confidence": 0.9,
        }
    support = find_transition_support(left, right, cohort)
    if support["fact_ids"]:
        cited = ", ".join(f"#{item}" for item in support["fact_ids"])
        return {
            "relationship_type": "RECONCILES",
            "reasoning": f"Supported by resignation/appointment fact {cited}.",
            "supporting_fact_ids": support["fact_ids"],
            "supporting_evidence_ids": support["evidence_ids"] or evidence_ids,
            "confidence": 0.86,
        }
    rel_type = (
        "POTENTIAL_CONTRADICTION"
        if attr == "holder"
        else "UNRESOLVED_DIFFERENCE"
    )
    return {
        "relationship_type": rel_type,
        "reasoning": (
            "Different values for the same canonical attribute without extracted transition evidence."
        ),
        "supporting_fact_ids": support_ids,
        "supporting_evidence_ids": evidence_ids,
        "confidence": 0.7,
    }


def relationship_explanation(
    left: dict[str, Any],
    right: dict[str, Any],
    rel_type: str,
    cohort: list[dict[str, Any]] | None = None,
) -> str:
    detail = classify_relationship_detail(left, right, cohort=cohort)
    if detail and detail.get("relationship_type") == rel_type:
        return str(detail.get("reasoning") or "")
    if rel_type == "CORROBORATES":
        return "Same canonical attribute and normalized value."
    if rel_type == "CONTRADICTS":
        return "Same canonical attribute and period but different normalized values."
    if rel_type == "RECONCILES":
        return "Supported by extracted transition evidence."
    if rel_type == "COMPUTED_SUPPORT":
        return "Component values sum to the total."
    return ""


def allowed_relationship_types(candidate: dict[str, Any], parent: dict[str, Any]) -> list[str]:
    """PART_OF, CORROBORATES, CONTRADICTS, and RECONCILES are evaluated independently."""
    if candidate.get("id") is not None and parent.get("id") is not None:
        if str(candidate.get("id")) == str(parent.get("id")):
            return []

    allowed: list[str] = []
    if _part_of_pair_eligible(candidate, parent):
        allowed.append("PART_OF")

    classified = classify_relationship(candidate, parent)
    if classified:
        allowed.append(classified)

    return sorted(set(allowed))


def _part_of_pair_eligible(candidate: dict[str, Any], parent: dict[str, Any]) -> bool:
    if is_sibling(candidate, parent):
        return False
    return is_hierarchical_child(candidate, parent)


def _contradiction_pair_eligible(candidate: dict[str, Any], parent: dict[str, Any]) -> bool:
    return classify_relationship(candidate, parent) == "CONTRADICTS"


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
        proposed = _proposed_type(candidate, parent)
        if proposed not in {"PART_OF"}:
            continue
        rejected.append(
            _rejection_row(
                candidate,
                parent,
                proposed,
                _reject_reason(candidate, parent),
            )
        )
    return plausible, rejected


def _proposed_type(candidate: dict[str, Any], parent: dict[str, Any]) -> str:
    classified = classify_relationship(candidate, parent)
    if classified:
        return classified
    if is_sibling(candidate, parent):
        return "PART_OF"
    if is_hierarchical_child(candidate, parent) or is_hierarchical_child(parent, candidate):
        return "PART_OF"
    if fact_canonical_attribute(candidate) != fact_canonical_attribute(parent):
        return "UNKNOWN"
    return "UNKNOWN"


def _reject_reason(candidate: dict[str, Any], parent: dict[str, Any]) -> str:
    if is_sibling(candidate, parent):
        return "Sibling facts cannot be PART_OF each other."
    if is_hierarchical_child(parent, candidate):
        return "Parent cannot be PART_OF its child."
    if fact_canonical_attribute(candidate) != fact_canonical_attribute(parent):
        if not is_hierarchical_child(candidate, parent):
            return "attribute mismatch"
    classified = classify_relationship(candidate, parent)
    if classified is None and fact_canonical_attribute(candidate) == fact_canonical_attribute(parent):
        return "Pair does not corroborate, contradict, or reconcile."
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
