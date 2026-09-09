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
    ("software", "product_revenue"),
    ("product", "product_revenue"),
    ("products", "product_revenue"),
    ("consulting", "services_revenue"),
    ("services", "services_revenue"),
    ("service", "services_revenue"),
)

EMPLOYEE_ALIASES = {
    "employees",
    "employee",
    "employee count",
    "employee_count",
    "headcount",
    "head count",
    "staff",
    "workforce",
    "workforce strength",
    "staff strength",
    "number of employees",
    "no of employees",
    "total employees",
    "total headcount",
    "permanent employees",
}

# A headcount attribute is a bare count phrase, optionally qualified.
HEADCOUNT_PHRASE_RE = re.compile(
    r"^(?:total\s+|average\s+|permanent\s+|contract\s+|number\s+of\s+|no\.?\s+of\s+)*"
    r"(?:employees?|headcount|head\s?count|workforce|staff)"
    r"(?:\s+(?:count|strength|number))?$"
)
# Fragments that make an attribute a money/ratio/derived metric, never a headcount.
NON_HEADCOUNT_RE = re.compile(
    r"\b(salary|salaries|benefit|benefits|compensation|option|options|share|shares|"
    r"percent|percentage|pct|cost|costs|contribution|expense|expenses|revenue|"
    r"payable|insurance|turnover|ratio|margin|rate|amount|value|esic|healthcare|"
    r"session|training|attrition|wage|wages|remuneration|bonus)\b"
)
# Ratio / derived metrics that must not collapse into a level metric such as
# total_revenue (e.g. percentage_of_total_revenue, CAGR, growth_rate).
DERIVED_METRIC_RE = re.compile(
    r"\b(percentage|percent|pct|ratio|margin|cagr|growth|yoy|y o y|increase|"
    r"decrease|proportion|share of|per share)\b"
)

CURRENCY_UNIT_WORDS = {
    "inr", "usd", "eur", "gbp", "rs", "rupee", "rupees", "dollar", "dollars",
    "euro", "euros", "pound", "pounds",
}

CEO_ALIASES = {
    "ceo",
    "chief executive",
    "chief executive officer",
    "managing director",
    "md",
}

HEDGE_RE = re.compile(
    r"\b(approximately|approx|about|around|nearly|roughly|circa|estimated|almost)\b|~",
    re.IGNORECASE,
)

REVENUE_HIERARCHY = {
    "total_revenue": (
        "product_revenue",
        "services_revenue",
        "subscription_revenue",
        "licensing_revenue",
    )
}

CURRENCY_SYMBOLS = (
    ("₹", "INR"),
    ("rs.", "INR"),
    ("rs ", "INR"),
    ("inr", "INR"),
    ("rupees", "INR"),
    ("rupee", "INR"),
    ("usd", "USD"),
    ("eur", "EUR"),
    ("gbp", "GBP"),
    ("$", "USD"),
    ("€", "EUR"),
    ("£", "GBP"),
)

MONTHS = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

ADDRESS_REPLACEMENTS = (
    (r"\bstreet\b", "street"),
    (r"\bst\.?(?=\W|$)", "street"),
    (r"\broad\b", "road"),
    (r"\brd\.?(?=\W|$)", "road"),
    (r"\bavenue\b", "avenue"),
    (r"\bave\.?(?=\W|$)", "avenue"),
    (r"\bboulevard\b", "boulevard"),
    (r"\bblvd\.?(?=\W|$)", "boulevard"),
    (r"\blane\b", "lane"),
    (r"\bln\.?(?=\W|$)", "lane"),
    (r"\bapartment\b", "apt"),
    (r"\bapt\.?(?=\W|$)", "apt"),
    (r"\bpincode\b", "pin"),
    (r"\bpin code\b", "pin"),
    (r"\bzip code\b", "zip"),
    (r"\bzipcode\b", "zip"),
)

ISO_DATE_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
DMY_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")
NAMED_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+([A-Za-z]{3,9}),?\s+(\d{4})\b|\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b"
)
MONTH_YEAR_RE = re.compile(r"\b([A-Za-z]{3,9})\s+(\d{4})\b")
_AMOUNT_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

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
    Display value is never rewritten except for whitespace; normalized form
    is stored in canonical_value.
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
    out["value"] = " ".join(str(source_value or "").split())

    period = canonicalize_period(
        out.get("original_period") or out.get("period"),
        raw_attribute,
        out.get("entity"),
    )
    out["period"] = period or None

    unit = canonicalize_unit(out.get("unit"), out.get("value"))
    out["unit"] = unit or None

    apply_entity_resolution(out, raw_attribute)
    raw_attribute = str(out.get("raw_attribute") or raw_attribute).strip()

    canonical_attr = infer_canonical_attribute(out, raw_attribute)
    canonical_value = canonicalize_value(out.get("value"), unit, canonical_attr)
    # Guard against value/attribute contradictions the attribute name alone can't
    # catch: a headcount that is money / a percentage / a scaled magnitude, or a
    # revenue *level* whose value is a ratio. Demote to a generic attribute so
    # the mislabel does not pollute a clean metric bucket.
    if _demote_on_value_conflict(canonical_attr, canonical_value, unit) != canonical_attr:
        fallback = to_snake(strip_period_tokens(raw_attribute)) or "value"
        # The fallback must not land back on a conflicting metric bucket.
        if (
            fallback in TEMPORAL_ATTRIBUTES
            or _demote_on_value_conflict(fallback, canonical_value, unit) != fallback
        ):
            fallback = "value"
        canonical_attr = fallback
        out["attribute"] = canonical_attr
        canonical_value = canonicalize_value(out.get("value"), unit, canonical_attr)
    canonical_entity = canonicalize_entity(out.get("entity"), canonical_attr, raw_attribute)

    out["canonical_attribute"] = canonical_attr
    out["canonical_entity"] = canonical_entity
    out["canonical_value"] = canonical_value
    out["currency"] = canonicalize_currency_code(out.get("value"), unit) or None
    out["attribute"] = canonical_attr
    # Only fill an empty entity; the observed subject is never overwritten so a
    # second canonicalization pass sees the same inputs as the first.
    if not str(out.get("entity") or "").strip():
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

    if is_ceo_role(raw, entity, fact.get("value")):
        return "holder"
    if is_employee_metric(raw, entity):
        return "employee_count"

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
    # A ratio or derived metric about revenue is not a revenue level.
    if DERIVED_METRIC_RE.search(normalize_token(text)):
        return None
    has_revenue = bool(REVENUE_RE.search(text) or re.search(r"\b(turnover|sales)\b", text))
    for prefix, canonical in REVENUE_CHILD_PREFIXES:
        if re.search(rf"\b{prefix}\b", text) and (has_revenue or prefix in {"software", "consulting"}):
            return canonical
    if has_revenue:
        return "total_revenue"
    return None


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
    if canonical_attribute == "holder":
        return "CEO"
    if canonical_attribute == "employee_count":
        return "Employees"
    if text and not is_temporal_attribute(text):
        if canonical_attribute.endswith("_revenue") and REVENUE_RE.search(text):
            return "Revenue"
        return text
    if canonical_attribute.endswith("_revenue"):
        return "Revenue"
    cleaned = strip_period_tokens(normalize_token(raw_attribute))
    return str(entity or cleaned or "Entity").strip() or "Entity"


def apply_entity_resolution(fact: dict[str, Any], raw_attribute: str) -> None:
    """Map employee/CEO surface forms onto stable attribute names.

    Never writes ``fact["entity"]``: the observed subject is preserved and the
    derived name lives in ``canonical_entity``. Mutating entity here made
    ``infer_canonical_attribute`` read its own output, so canonicalizing an
    already-canonicalized fact could change the result.
    """
    entity = str(fact.get("entity") or "").strip()
    value = str(fact.get("value") or "").strip()
    if is_ceo_role(raw_attribute, entity, value):
        person = extract_person_name(entity, raw_attribute, value)
        if person:
            fact["attribute"] = "holder"
            fact["raw_attribute"] = raw_attribute or "holder"
            fact["value"] = person
        return
    if is_employee_metric(raw_attribute):
        fact["attribute"] = "employee_count"


def is_employee_metric(attribute: Any, entity: Any = None) -> bool:
    """True only for genuine headcount attributes.

    Matches an exact alias or a bare headcount phrase; never an arbitrary
    substring, so ``salary_and_other_employee_benefits`` and
    ``percentage_of_female_employees`` are rejected. ``entity`` is accepted for
    call compatibility but deliberately ignored: entity is resolved downstream
    and reading it here made attribute inference depend on its own output.
    """
    text = normalize_token(attribute)
    if not text:
        return False
    if NON_HEADCOUNT_RE.search(text):
        return False
    if text in EMPLOYEE_ALIASES:
        return True
    return bool(HEADCOUNT_PHRASE_RE.match(text))


def is_ceo_role(attribute: Any, entity: Any = None, value: Any = None) -> bool:
    attr = normalize_token(attribute)
    ent = normalize_token(entity)
    val = normalize_token(value)
    if attr in CEO_ALIASES or ent in CEO_ALIASES:
        return True
    if attr in {"position", "title", "role", "designation"} and val in CEO_ALIASES:
        return True
    if val in CEO_ALIASES and looks_like_person_name(entity):
        return True
    blob = normalize_token(f"{attribute or ''} {entity or ''} {value or ''}")
    if re.search(r"\bposition\s*=\s*(ceo|chief executive)\b", blob):
        return True
    return False


def extract_person_name(entity: Any, attribute: Any, value: Any) -> str:
    entity_text = str(entity or "").strip()
    if "|" in entity_text:
        left = entity_text.split("|", 1)[0].strip()
        if looks_like_person_name(left):
            return left
    if looks_like_person_name(entity) and normalize_token(entity) not in CEO_ALIASES:
        return str(entity).strip()
    if looks_like_person_name(value) and normalize_token(value) not in CEO_ALIASES:
        return str(value).strip()
    blob = f"{entity or ''} {attribute or ''} {value or ''}"
    piped = re.search(
        r"([A-Za-z][A-Za-z.]+(?:\s+[A-Za-z][A-Za-z.]+)+)\s*\|\s*(?:position|title|role)\s*=\s*"
        r"(ceo|chief executive(?: officer)?)",
        blob,
        re.IGNORECASE,
    )
    if piped:
        return piped.group(1).strip()
    labeled = re.search(
        r"(?:ceo|chief executive(?: officer)?)\s*[:|-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        blob,
        re.IGNORECASE,
    )
    if labeled:
        return labeled.group(1).strip()
    return str(value or entity or "").strip()


def looks_like_person_name(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or normalize_token(text) in CEO_ALIASES:
        return False
    parts = [part for part in re.split(r"[\s|,]+", text) if part]
    if len(parts) < 2 or len(parts) > 4:
        return False
    return all(part.replace(".", "").isalpha() for part in parts)


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


def is_currency_token(text: Any) -> bool:
    """True when a bare unit string is a currency, not a magnitude scale."""
    raw = str(text or "").strip()
    if not raw:
        return False
    if raw in {"₹", "$", "€", "£"}:
        return True
    return normalize_token(raw).rstrip(".") in CURRENCY_UNIT_WORDS


def canonicalize_currency_code(value: Any, unit: Any = None) -> str:
    """Currency for a value/unit pair, or '' when the value is not monetary."""
    blob = f"{value or ''} {unit or ''}"
    lower = blob.lower()
    for token, code in CURRENCY_SYMBOLS:
        if token in lower or token in blob:
            return code
    return ""


def canonicalize_unit(unit: Any, value: Any = None) -> str:
    """Magnitude scale only. Currency is a separate dimension (see
    ``canonicalize_currency_code``) and must never be echoed back as a unit,
    otherwise ``INR``/``₹`` produce two identities for one value."""
    text = f"{unit or ''} {value or ''}".lower()
    if "%" in text:
        return "%"
    for token, canonical in UNIT_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            return canonical
    raw = str(unit or "").strip()
    if is_currency_token(raw):
        return ""
    return raw


def canonicalize_value(value: Any, unit: str | None = None, attribute: Any = None) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    date = canonicalize_date(text)
    if date:
        return date
    money = canonicalize_currency(text, unit)
    if money:
        return money
    if looks_like_address(text, attribute):
        return canonicalize_address(text)
    return text


def canonicalize_currency(value: Any, unit: Any = None) -> str | None:
    """Render a monetary value as ``amount [CURRENCY] [scale]``.

    Each slot is emitted at most once, so ``₹15.36`` with unit ``INR`` and the
    same value with unit ``₹`` both render ``15.36 INR`` instead of the previous
    ``15.36 INR INR`` / ``15.36 INR ₹``.
    """
    blob = f"{value or ''} {unit or ''}"
    compact = blob.replace(",", "")
    match = _AMOUNT_RE.search(compact)
    if match is None:
        return None
    amount = float(match.group().replace(",", ""))
    lower = blob.lower()
    currency = canonicalize_currency_code(value, unit)
    unit_n = canonicalize_unit(unit, value)
    if not currency and not unit_n and "₹" not in blob and "$" not in blob:
        if not re.search(r"\b(inr|usd|eur|gbp|rs|rupee|crore|lakh|million|billion)\b", lower):
            return None
    parts = [f"{amount:g}"]
    if currency:
        parts.append(currency)
    if unit_n and unit_n != currency:
        parts.append(unit_n)
    return " ".join(parts)


def canonicalize_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    iso = ISO_DATE_RE.search(text)
    if iso:
        year, month, day = iso.group(1), int(iso.group(2)), int(iso.group(3))
        return f"{year}-{month:02d}-{day:02d}"
    named = NAMED_DATE_RE.search(text)
    if named:
        if named.group(1):
            day, month_name, year = named.group(1), named.group(2), named.group(3)
        else:
            month_name, day, year = named.group(4), named.group(5), named.group(6)
        month = MONTHS.get(month_name.lower())
        if month:
            return f"{year}-{month}-{int(day):02d}"
    dmy = DMY_DATE_RE.search(text)
    if dmy and not _AMOUNT_RE.fullmatch(text.replace(",", "").replace(" ", "")):
        first, second, year = int(dmy.group(1)), int(dmy.group(2)), dmy.group(3)
        if first > 12:
            day, month = first, second
        else:
            day, month = first, second
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year}-{month:02d}-{day:02d}"
    month_year = MONTH_YEAR_RE.search(text)
    if month_year:
        month = MONTHS.get(month_year.group(1).lower())
        if month:
            return f"{month_year.group(2)}-{month}"
    return None


def looks_like_address(value: Any, attribute: Any = None) -> bool:
    attr = normalize_token(attribute)
    if any(token in attr for token in ("address", "location", "office", "hq", "headquarters")):
        return True
    text = normalize_token(value)
    return bool(
        re.search(r"\b(street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|pin|zip|apt)\b", text)
        and re.search(r"\d", text)
    )


def canonicalize_address(value: Any) -> str:
    text = " ".join(str(value or "").replace(",", " , ").split())
    lower = text.lower()
    for pattern, replacement in ADDRESS_REPLACEMENTS:
        lower = re.sub(pattern, replacement, lower)
    lower = re.sub(r"\s+,", ",", lower)
    lower = re.sub(r",\s*", ", ", lower)
    lower = re.sub(r"\s+\.", ".", lower)
    return " ".join(lower.split())


COUNT_UNIT_RE = re.compile(
    r"\b(shares?|equity shares?|options?|units?|parcels?|customers?|employees?|people|persons?)\b",
    re.IGNORECASE,
)
SCALE_RE = re.compile(r"\b(crore|lakh|million|billion|thousand)\b", re.IGNORECASE)

# Which value dimensions may be compared with each other. Comparing across
# dimensions (money vs ratio) is meaningless, not a disagreement.
DIMENSION_COMPATIBILITY: dict[str, set[str]] = {
    "ratio": {"ratio"},
    "count": {"count", "plain"},
    "money": {"money", "scaled", "plain"},
    "scaled": {"money", "scaled", "plain"},
    "plain": {"money", "scaled", "plain", "count"},
}


_HEADCOUNT_NOISE_RE = re.compile(
    r"[%₹$€£]|\b(share|shares|option|options|parcel|parcels|equity|percent|pct|"
    r"crore|lakh|million|billion|thousand|mn|bn|inr|usd|eur|gbp|rs)\b",
    re.IGNORECASE,
)


_LEVEL_ATTRIBUTES = {
    "total_revenue", "product_revenue", "services_revenue",
    "subscription_revenue", "licensing_revenue", "revenue",
}


def _demote_on_value_conflict(
    canonical_attr: str, canonical_value: Any, unit: Any
) -> str:
    """Return a replacement attribute when the value contradicts the metric.

    ``''`` means "demote, but the caller must choose the fallback name".
    Returning ``canonical_attr`` unchanged means the fact is consistent.
    """
    if canonical_attr == "employee_count" and not is_plausible_headcount(canonical_value, unit):
        return ""
    if canonical_attr in _LEVEL_ATTRIBUTES and value_dimension(canonical_value, unit) == "ratio":
        return ""
    return canonical_attr


def is_plausible_headcount(value: Any, unit: Any = None) -> bool:
    """A headcount is a bare whole number under a sane ceiling.

    Rejects money, percentages, scaled magnitudes, and share/parcel counts that
    the extractor mislabelled as employee_count.
    """
    blob = f"{value or ''} {unit or ''}"
    if _HEADCOUNT_NOISE_RE.search(blob):
        return False
    match = _AMOUNT_RE.search(blob.replace(",", ""))
    if match is None:
        return False
    try:
        number = float(match.group())
    except ValueError:
        return False
    return 0 <= number <= 5_000_000 and number == int(number)


def value_dimension(value: Any, unit: Any = None) -> str:
    """Classify a value as money / ratio / count / scaled / plain."""
    blob = f"{value or ''} {unit or ''}"
    if "%" in blob:
        return "ratio"
    if canonicalize_currency_code(value, unit):
        return "money"
    if COUNT_UNIT_RE.search(blob):
        return "count"
    if SCALE_RE.search(blob):
        return "scaled"
    return "plain"


def dimensions_compatible(
    left_value: Any,
    left_unit: Any = None,
    right_value: Any = None,
    right_unit: Any = None,
) -> bool:
    """True when two values are of comparable kinds."""
    left = value_dimension(left_value, left_unit)
    right = value_dimension(right_value, right_unit)
    return right in DIMENSION_COMPATIBILITY.get(left, {left})


def normalized_value_key(value: Any, unit: Any = None) -> str:
    """Identity for numeric/text values, ignoring hedges, currency symbols, and spacing."""
    text = HEDGE_RE.sub(" ", f"{value or ''} {unit or ''}").replace(",", "")
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
