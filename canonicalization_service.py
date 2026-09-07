"""Normalize extracted facts: period vs attribute, units, dates, canonical names."""

from __future__ import annotations

import re
from typing import Any

PERIOD_RE = re.compile(
    r"\b(?:(fy|cy|ay)\s*)?(\d{4})(?:\s*[-/]\s*\d{2,4})?\b|\bq[1-4](?:\s*fy?\s*\d{2,4})?\b",
    re.IGNORECASE,
)
REVENUE_RE = re.compile(r"\brevenues?\b", re.IGNORECASE)

TEMPORAL_ATTRIBUTES = {
    "fiscal_year",
    "fiscalyear",
    "fiscal year",
    "fy",
    "year",
    "period",
    "calendar_year",
    "calendar year",
    "financial_year",
    "financial year",
    "date",
    "year_ended",
    "year ended",
}

REVENUE_CHILD_PREFIXES = (
    ("subscription", "subscription_revenue"),
    ("licensing", "licensing_revenue"),
    ("license", "licensing_revenue"),
    ("product", "product_revenue"),
    ("products", "product_revenue"),
    ("services", "services_revenue"),
    ("service", "services_revenue"),
)

REVENUE_HIERARCHY = {
    "total_revenue": (
        "product_revenue",
        "services_revenue",
        "subscription_revenue",
        "licensing_revenue",
    )
}

UNIT_ALIASES = {
    "crores": "crore",
    "crore": "crore",
    "crs": "crore",
    "cr": "crore",
    "lacs": "lakh",
    "lakh": "lakh",
    "lac": "lakh",
    "million": "million",
    "billion": "billion",
    "mn": "million",
    "bn": "billion",
    "percent": "%",
    "percentage": "%",
    "pct": "%",
}


def canonicalize_fact(fact: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with raw_* and canonical_* fields populated.

    Does not read evidence text. Page context must not change the metric.
    Value is never rewritten except for whitespace.
    """
    out = dict(fact)
    raw_attribute = str(
        out.get("original_attribute")
        or out.get("raw_attribute")
        or out.get("attribute")
        or ""
    ).strip()
    out["raw_attribute"] = str(out.get("raw_attribute") or raw_attribute).strip()
    source_value = out.get("original_value") if out.get("original_value") not in (None, "") else out.get("value")
    out["value"] = canonicalize_value(source_value)

    period = canonicalize_period(
        out.get("original_period") or out.get("period"),
        raw_attribute,
        out.get("entity"),
    )
    out["period"] = period or None

    unit = canonicalize_unit(out.get("unit"), out.get("value"))
    out["unit"] = unit or None

    canonical_attr = infer_canonical_attribute(out, raw_attribute)
    canonical_entity = canonicalize_entity(out.get("entity"), canonical_attr, raw_attribute)
    canonical_value = canonicalize_value(out.get("value"), unit)

    out["canonical_attribute"] = canonical_attr
    out["canonical_entity"] = canonical_entity
    out["canonical_value"] = canonical_value
    out["attribute"] = canonical_attr
    if canonical_entity and not str(out.get("entity") or "").strip():
        out["entity"] = canonical_entity
    return out


def infer_canonical_attribute(fact: dict[str, Any], raw_attribute: str | None = None) -> str:
    """Infer metric name from entity and attribute only — never from evidence."""
    raw = normalize_token(
        raw_attribute
        or fact.get("original_attribute")
        or fact.get("raw_attribute")
        or fact.get("attribute")
    )
    entity = normalize_token(fact.get("original_entity") or fact.get("entity"))
    metric_text = f"{raw} {entity}"

    if is_temporal_attribute(raw):
        return infer_metric_from_context(entity, metric_text)

    revenue_attr = infer_revenue_attribute(raw, entity)
    if revenue_attr:
        return revenue_attr

    cleaned = strip_period_tokens(raw).strip()
    cleaned = re.sub(r"\b(total|overall|consolidated|combined|net|group)\b", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    if not cleaned or is_temporal_attribute(cleaned):
        return infer_metric_from_context(entity, metric_text)
    return to_snake(cleaned)


def infer_revenue_attribute(raw: str, entity: str) -> str | None:
    text = strip_period_tokens(f"{raw} {entity}")
    if not REVENUE_RE.search(text):
        return None
    for prefix, canonical in REVENUE_CHILD_PREFIXES:
        if re.search(rf"\b{prefix}\b", text):
            return canonical
    return "total_revenue"


def infer_metric_from_context(entity: str, metric_text: str) -> str:
    revenue_attr = infer_revenue_attribute("", entity) or infer_revenue_attribute(metric_text, entity)
    if revenue_attr:
        return revenue_attr
    if entity and not is_temporal_attribute(entity):
        return to_snake(strip_period_tokens(entity)) or "value"
    return "value"


def is_temporal_attribute(attribute: Any) -> bool:
    text = normalize_token(attribute)
    return text in TEMPORAL_ATTRIBUTES or text.replace(" ", "_") in TEMPORAL_ATTRIBUTES


def canonicalize_entity(entity: Any, canonical_attribute: str, raw_attribute: str) -> str:
    text = str(entity or "").strip()
    if text and not is_temporal_attribute(text):
        if canonical_attribute.endswith("_revenue") and REVENUE_RE.search(text):
            return "Revenue"
        return text
    if canonical_attribute.endswith("_revenue"):
        return "Revenue"
    cleaned = strip_period_tokens(normalize_token(raw_attribute))
    return str(entity or cleaned or "Entity").strip() or "Entity"


def canonicalize_period(*parts: Any) -> str:
    blob = " ".join(str(part or "") for part in parts)
    match = PERIOD_RE.search(blob)
    if not match:
        return str(parts[0] or "").strip()
    if match.group(0).lower().startswith("q"):
        return match.group(0).upper().replace(" ", "")
    year = match.group(2) if match.lastindex and match.group(2) else match.group(0)
    prefix = (match.group(1) or "FY").upper()
    if prefix not in {"FY", "CY", "AY"}:
        prefix = "FY"
    return f"{prefix}{year}"


def canonicalize_unit(unit: Any, value: Any = None) -> str:
    text = f"{unit or ''} {value or ''}".lower()
    if "%" in text:
        return "%"
    for token, canonical in UNIT_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            return canonical
    return str(unit or "").strip()


def canonicalize_value(value: Any, unit: str | None = None) -> str:
    text = " ".join(str(value or "").split())
    return text


_AMOUNT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def normalized_value_key(value: Any, unit: Any = None) -> str:
    """Identity for numeric/text values, ignoring currency symbols and spacing."""
    text = f"{value or ''} {unit or ''}".replace(",", "")
    match = _AMOUNT_RE.search(text)
    unit_n = canonicalize_unit(unit, value)
    if match:
        amount = float(match.group())
        return f"{amount:g}|{unit_n}"
    return normalize_token(value)


def periods_mergeable(left: Any, right: Any) -> bool:
    a = normalize_token(left)
    b = normalize_token(right)
    if not a or not b:
        return True
    return a == b


def more_specific_period(left: Any, right: Any) -> str | None:
    a = str(left or "").strip()
    b = str(right or "").strip()
    return a or b or None


def strip_period_tokens(text: str) -> str:
    return PERIOD_RE.sub(" ", text or "")


def normalize_token(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").replace("-", " ").split())


def to_snake(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", normalize_token(text)).strip("_")
    return cleaned or "value"


def revenue_children() -> set[str]:
    children: set[str] = set()
    for items in REVENUE_HIERARCHY.values():
        children.update(items)
    return children
