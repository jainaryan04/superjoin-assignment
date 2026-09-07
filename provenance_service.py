"""Fact relationship engine. Evidence stays as source text; links live in fact_relationships."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any, Iterable

from pathlib import Path

from database import (
    RELATIONSHIP_TYPES,
    format_fact_statement,
    get_fact,
    insert_relationship,
    list_all_relationships,
    list_facts_for_document,
    relationship_exists,
    search_facts,
)
from fact_extractor import LLMClient, _parse_json_object
from relationship_rules import (
    allowed_relationship_types,
    fact_canonical_attribute,
    filter_relationship_candidates,
    is_hierarchical_child,
    is_revenue_child_of_total,
    is_sibling,
    periods_compatible,
    revenue_category,
    same_entity,
    same_scope,
    semantically_same_fact,
)

LINK_SYSTEM_PROMPT = """You classify relationships BETWEEN facts.

Only use a relationship_type from each candidate's allowed_types.
If none apply, use NONE.

Evidence is source text. Do not treat another fact as evidence.

Return ONLY valid JSON:
{"links": [{"candidate_id": "...", "relationship_type": "PART_OF|CONTRADICTS|CORROBORATES|RECONCILES|NONE", "direction": "candidate_to_parent", "confidence": 0.0, "reasoning": "short"}]}

Rules:
- PART_OF: candidate is a hierarchical component of the parent (child -> parent). Never link siblings.
- CONTRADICTS: same entity, same canonical_attribute, same scope, compatible period, different values.
- Never CONTRADICTS between different attributes (total_revenue vs product_revenue is not a contradiction).
- Never PART_OF between siblings (product_revenue vs services_revenue).
- Never create a relationship from a fact to itself (A -> A) for any type.
"""

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_UNIT_RE = re.compile(
    r"\b(crore|cr|lakh|lac|million|billion|thousand|percent|pct|%|usd|inr|rs|₹)\b",
    re.IGNORECASE,
)


CANDIDATE_LOG_PATH = Path("data") / "relationship_candidates.json"
_candidate_log: list[dict[str, Any]] = []


def link_fact_relationships(
    source_document: str | None = None,
    *,
    llm: LLMClient | None = None,
    db_path: str | None = None,
    nearby_page_window: int = 5,
) -> dict[str, int]:
    """Discover connections between nearby facts. Does not write fact_evidence rows."""
    global _candidate_log
    _candidate_log = []
    facts = list_facts_for_document(source_document, db_path=db_path)
    added = 0
    touched = 0
    rejected = 0
    raw_generated = 0
    self_removed = 0
    duplicate_removed = 0
    seen: set[tuple[str, str, str]] = set()

    for parent in facts:
        nearby = _nearby_facts(parent, facts, nearby_page_window)
        plausible, filter_rejects = filter_relationship_candidates(parent, nearby)
        for item in filter_rejects:
            print(
                f"Rejected relationship {item['source_fact']} -> {item['target_fact']} "
                f"[{item['proposed_relationship']}]: {item['reason']}"
            )
        _candidate_log.extend(filter_rejects)
        relations = _heuristic_links(parent, nearby, facts)
        if llm is not None:
            relations.extend(_llm_links(llm, parent, plausible))
        raw_generated += len(relations)
        wrote = 0
        for relation in _dedupe_relations(relations):
            source_id = str(relation.get("source_fact_id") or "")
            target_id = str(relation.get("target_fact_id") or "")
            rel_type = str(relation.get("relationship_type") or "").strip().upper().replace(" ", "_")
            source = _fact_by_id(facts, source_id)
            target = _fact_by_id(facts, target_id)
            key = (source_id, target_id, rel_type)
            row = {
                "source_fact": format_fact_statement(source or {}),
                "target_fact": format_fact_statement(target or {}),
                "candidate": format_fact_statement(source or {}),
                "proposed_relationship": rel_type,
                "accepted": False,
                "reason": "",
                "source_fact_id": source_id,
                "target_fact_id": target_id,
                "source_canonical_attribute": fact_canonical_attribute(source),
                "target_canonical_attribute": fact_canonical_attribute(target),
                "source_raw_attribute": (source or {}).get("raw_attribute"),
                "target_raw_attribute": (target or {}).get("raw_attribute"),
            }
            if source_id == target_id:
                self_removed += 1
                rejected += 1
                row["reason"] = "Self-links are not allowed."
                print(
                    f"Rejected relationship {row['source_fact']} -> {row['target_fact']} "
                    f"[{rel_type}]: {row['reason']}"
                )
                _candidate_log.append(row)
                continue
            check = validate_relationship(source, target, rel_type, cohort=facts)
            row["reason"] = check["reason"] if not check["ok"] else (relation.get("reasoning") or check["reason"])
            row["accepted"] = check["ok"]
            if not check["ok"]:
                rejected += 1
                print(
                    f"Rejected relationship {row['source_fact']} -> {row['target_fact']} "
                    f"[{rel_type}]: {check['reason']}"
                )
                _candidate_log.append(row)
                continue
            already = key in seen or relationship_exists(source_id, target_id, rel_type, db_path=db_path)
            if insert_relationship(
                {
                    "source_fact_id": source_id,
                    "target_fact_id": target_id,
                    "relationship_type": rel_type,
                    "confidence": relation.get("confidence") or 0.7,
                    "reasoning": relation.get("reasoning") or "",
                },
                db_path=db_path,
            ):
                wrote += 1
                seen.add(key)
                row["accepted"] = True
                _candidate_log.append(row)
            else:
                if already:
                    duplicate_removed += 1
                    row["accepted"] = True
                    row["reason"] = "Merged into existing relationship."
                else:
                    rejected += 1
                    row["accepted"] = False
                    row["reason"] = "Duplicate or invalid insert."
                seen.add(key)
                _candidate_log.append(row)
        if wrote:
            touched += 1
            added += wrote

    _save_candidate_log()
    stored = len(list_all_relationships(db_path=db_path))
    return {
        "facts_touched": touched,
        "relationships_added": added,
        "relationships_rejected": rejected,
        "raw_relationships_generated": raw_generated,
        "duplicate_relationships_removed": duplicate_removed,
        "self_relationships_removed": self_removed,
        "final_relationships_stored": stored,
    }


def evidence_tree_text(fact: dict[str, Any]) -> str:
    """ASCII tree of evidence snippets only (not related facts)."""
    statement = format_fact_statement(fact)
    evidence = list(fact.get("evidence") or [])
    if not evidence:
        return statement

    lines = [statement, ""]
    last_index = len(evidence) - 1
    for index, item in enumerate(evidence):
        last = index == last_index
        branch = "└──" if last else "├──"
        hang = "    " if last else "│   "
        evidence_type = item.get("evidence_type") or "DIRECT"
        snippet = " ".join(str(item.get("evidence_text") or "").split())
        page = item.get("page_number")
        lines.append(f"{branch} {evidence_type}")
        lines.append(f"{hang}{snippet}")
        lines.append(f"{hang}Page {page}")
        if not last:
            lines.append("│")
    return "\n".join(lines)


def _nearby_facts(
    parent: dict[str, Any],
    facts: Iterable[dict[str, Any]],
    page_window: int,
) -> list[dict[str, Any]]:
    parent_page = int(parent.get("page_number") or 0)
    parent_doc = parent.get("source_document")
    nearby: list[dict[str, Any]] = []
    for fact in facts:
        if fact["id"] == parent["id"] or str(fact.get("id") or "") == str(parent.get("id") or ""):
            continue
        if parent_doc and fact.get("source_document") != parent_doc:
            continue
        fact_page = int(fact.get("page_number") or 0)
        if page_window >= 0 and abs(fact_page - parent_page) > page_window:
            continue
        nearby.append(fact)
    return nearby


def _heuristic_links(
    parent: dict[str, Any],
    nearby: list[dict[str, Any]],
    cohort: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Link hierarchical / revenue children to a parent. Numeric sums are optional evidence.

    Never inserts or updates facts. Children are never treated as parents.
    """
    if revenue_category(parent) in {
        "product_revenue",
        "services_revenue",
        "subscription_revenue",
        "licensing_revenue",
    }:
        return []
    parent_amount, parent_unit = parse_amount(parent.get("value"), parent.get("unit"))
    children = [fact for fact in nearby if is_hierarchical_child(fact, parent)]
    if not children:
        return []

    summing_ids: set[str] = set()
    scored: list[tuple[dict[str, Any], float]] = []
    if parent_amount is not None:
        for candidate in children:
            amount, unit = parse_amount(candidate.get("value"), candidate.get("unit"))
            if amount is None or not units_compatible(parent_unit, unit):
                continue
            scored.append((candidate, amount))
        for size in range(min(6, len(scored)), 1, -1):
            for combo in itertools.combinations(scored, size):
                total = sum(item[1] for item in combo)
                if amounts_match(total, parent_amount):
                    summing_ids = {item[0]["id"] for item in combo}
                    break
            if summing_ids:
                break

    links: list[dict[str, Any]] = []
    for candidate in children:
        amount, _unit = parse_amount(candidate.get("value"), candidate.get("unit"))
        if summing_ids and candidate["id"] in summing_ids:
            reason = f"{amount} is a numeric component of {parent_amount}"
        elif is_revenue_child_of_total(candidate, parent):
            reason = "Revenue hierarchy: child category of total revenue."
        elif is_hierarchical_child(candidate, parent):
            reason = "Parent-child attribute hierarchy."
        else:
            continue
        check = validate_relationship(candidate, parent, "PART_OF", cohort=cohort)
        if not check["ok"]:
            print(
                f"Rejected relationship {format_fact_statement(candidate)} -> "
                f"{format_fact_statement(parent)} [PART_OF]: {check['reason']}"
            )
            _candidate_log.append(
                {
                    "source_fact": format_fact_statement(candidate),
                    "target_fact": format_fact_statement(parent),
                    "candidate": format_fact_statement(candidate),
                    "proposed_relationship": "PART_OF",
                    "accepted": False,
                    "reason": check["reason"],
                    "source_fact_id": candidate.get("id"),
                    "target_fact_id": parent.get("id"),
                }
            )
            continue
        links.append(
            {
                "source_fact_id": candidate["id"],
                "target_fact_id": parent["id"],
                "candidate": candidate,
                "relationship_type": "PART_OF",
                "confidence": 0.92 if candidate["id"] in summing_ids else 0.8,
                "reasoning": reason,
            }
        )
    return links


def _llm_links(
    llm: LLMClient,
    parent: dict[str, Any],
    plausible: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not plausible:
        return []
    payload = {
        "parent": _fact_brief(parent),
        "candidates": [
            {
                **_fact_brief(item["candidate"]),
                "allowed_types": item["allowed_types"],
            }
            for item in plausible
        ],
    }
    user_prompt = (
        "Classify each candidate relative to the parent fact. "
        "Use only allowed_types for that candidate.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    raw = llm.complete(LINK_SYSTEM_PROMPT, user_prompt)
    parsed = _parse_json_object(raw)
    items = parsed.get("links", [])
    if not isinstance(items, list):
        return []

    allowed_by_id = {
        item["candidate"]["id"]: set(item["allowed_types"]) for item in plausible
    }
    by_id = {item["candidate"]["id"]: item["candidate"] for item in plausible}
    links: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate = by_id.get(str(item.get("candidate_id") or ""))
        if candidate is None:
            continue
        rel_type = str(item.get("relationship_type") or "NONE").strip().upper().replace(" ", "_")
        if rel_type == "SUPPORTING":
            rel_type = "SUPPORTS"
        if rel_type not in RELATIONSHIP_TYPES:
            continue
        if rel_type not in allowed_by_id.get(candidate["id"], set()):
            continue
        if rel_type == "PART_OF":
            source_id, target_id = candidate["id"], parent["id"]
            if is_hierarchical_child(parent, candidate):
                source_id, target_id = parent["id"], candidate["id"]
        else:
            source_id, target_id = candidate["id"], parent["id"]
        if str(source_id) == str(target_id):
            continue
        try:
            confidence = float(item.get("confidence") or 0.7)
        except (TypeError, ValueError):
            confidence = 0.7
        links.append(
            {
                "source_fact_id": source_id,
                "target_fact_id": target_id,
                "candidate": candidate,
                "relationship_type": rel_type,
                "confidence": max(0.0, min(1.0, confidence)),
                "reasoning": str(item.get("reasoning") or ""),
            }
        )
    return links


def _fact_by_id(facts: list[dict[str, Any]], fact_id: str) -> dict[str, Any] | None:
    for fact in facts:
        if fact.get("id") == fact_id:
            return fact
    return None


def _fact_brief(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": fact["id"],
        "id": fact["id"],
        "statement": format_fact_statement(fact),
        "entity": fact.get("entity"),
        "attribute": fact.get("canonical_attribute") or fact.get("attribute"),
        "raw_attribute": fact.get("raw_attribute"),
        "canonical_attribute": fact.get("canonical_attribute"),
        "value": fact.get("value"),
        "unit": fact.get("unit"),
        "period": fact.get("period"),
        "page_number": fact.get("page_number"),
        "evidence_text": fact.get("evidence_text"),
    }


def validate_relationship(
    source: dict[str, Any] | None,
    target: dict[str, Any] | None,
    relationship_type: str,
    *,
    cohort: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rel_type = str(relationship_type or "").strip().upper().replace(" ", "_")
    if source is None or target is None:
        return {"ok": False, "result": "FAILED", "reason": "Missing source or target fact."}
    source_id = str(source.get("id") or source.get("source_fact_id") or "")
    target_id = str(target.get("id") or target.get("target_fact_id") or "")
    if source_id and target_id and source_id == target_id:
        return {"ok": False, "result": "FAILED", "reason": "Self-links are not allowed."}
    if semantically_same_fact(source, target):
        return {
            "ok": False,
            "result": "FAILED",
            "reason": "A fact cannot relate to itself.",
        }
    if rel_type == "CONTRADICTS":
        if not same_entity(source, target):
            return {"ok": False, "result": "FAILED", "reason": "Entities differ."}
        if not periods_compatible(source, target):
            return {"ok": False, "result": "FAILED", "reason": "Periods are incompatible."}
        if fact_canonical_attribute(source) != fact_canonical_attribute(target):
            return {
                "ok": False,
                "result": "FAILED",
                "reason": "Different attributes cannot contradict.",
            }
        if not same_scope(source, target):
            return {
                "ok": False,
                "result": "FAILED",
                "reason": "Different scope cannot contradict.",
            }
        if _values_equivalent(source, target):
            return {
                "ok": False,
                "result": "FAILED",
                "reason": "Contradiction requires different values.",
            }
        return {"ok": True, "result": "PASSED", "reason": "Same attribute, scope, period, different values."}

    if rel_type == "PART_OF":
        if is_sibling(source, target):
            return {
                "ok": False,
                "result": "FAILED",
                "reason": "Sibling facts cannot be PART_OF each other.",
            }
        if fact_canonical_attribute(source) == fact_canonical_attribute(target):
            return {
                "ok": False,
                "result": "FAILED",
                "reason": "A fact cannot be PART_OF itself or the same canonical attribute.",
            }
        if revenue_category(target) in {
            "product_revenue",
            "services_revenue",
            "subscription_revenue",
            "licensing_revenue",
        }:
            return {
                "ok": False,
                "result": "FAILED",
                "reason": "Children may never become parents.",
            }
        if is_hierarchical_child(source, target) or is_revenue_child_of_total(source, target):
            return {
                "ok": True,
                "result": "PASSED",
                "reason": "Child category PART_OF parent total.",
            }
        return {
            "ok": False,
            "result": "FAILED",
            "reason": "PART_OF requires a parent-child hierarchy (e.g. product_revenue -> total_revenue).",
        }

    if rel_type in {"CORROBORATES", "RECONCILES"}:
        if fact_canonical_attribute(source) != fact_canonical_attribute(target):
            return {
                "ok": False,
                "result": "FAILED",
                "reason": "Same canonical attribute required.",
            }
        return {"ok": True, "result": "PASSED", "reason": "Same canonical attribute."}

    if rel_type in RELATIONSHIP_TYPES:
        allowed = allowed_relationship_types(source, target)
        if rel_type not in allowed and rel_type not in allowed_relationship_types(target, source):
            return {
                "ok": False,
                "result": "FAILED",
                "reason": "Candidate pair is not eligible for this relationship type.",
            }
        return {"ok": True, "result": "PASSED", "reason": "Pair passed candidate filters."}

    return {"ok": False, "result": "FAILED", "reason": f"Unknown relationship type {rel_type}."}


def _values_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_n, left_u = parse_amount(left.get("value"), left.get("unit"))
    right_n, right_u = parse_amount(right.get("value"), right.get("unit"))
    if left_n is not None and right_n is not None and units_compatible(left_u, right_u):
        return amounts_match(left_n, right_n)
    return normalize_value(left.get("value")) == normalize_value(right.get("value"))


def normalize_value(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def rebuild_relationships(
    source_document: str | None = None,
    *,
    llm: LLMClient | None = None,
    db_path: str | None = None,
) -> dict[str, int]:
    from database import cleanup_relationships, delete_all_relationships

    cleaned = cleanup_relationships(db_path)
    deleted = delete_all_relationships(db_path)
    result = link_fact_relationships(source_document, llm=llm, db_path=db_path)
    result["relationships_deleted"] = deleted
    result["self_relationships_removed"] = (
        cleaned.get("self_relationships_removed", 0) + result.get("self_relationships_removed", 0)
    )
    result["duplicate_relationships_removed"] = (
        cleaned.get("duplicate_relationships_removed", 0)
        + result.get("duplicate_relationships_removed", 0)
    )
    return result


def delete_invalid_relationships(db_path: str | None = None) -> int:
    from database import delete_relationships_by_id

    facts = list_facts_for_document(None, db_path=db_path)
    invalid_ids: list[str] = []
    for item in list_all_relationships(db_path=db_path):
        check = validate_relationship(
            item.get("source_fact_record"),
            item.get("target_fact_record"),
            item.get("relationship_type") or "",
            cohort=facts,
        )
        if not check["ok"] and item.get("id"):
            invalid_ids.append(str(item["id"]))
    return delete_relationships_by_id(invalid_ids, db_path=db_path)


def annotate_relationship_validations(
    records: list[dict[str, Any]],
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    facts = list_facts_for_document(None, db_path=db_path)
    annotated: list[dict[str, Any]] = []
    for item in records:
        check = validate_relationship(
            item.get("source_fact_record"),
            item.get("target_fact_record"),
            item.get("relationship_type") or "",
            cohort=facts,
        )
        row = dict(item)
        row["validation_result"] = check["result"]
        row["validation_reason"] = check["reason"]
        annotated.append(row)
    return annotated


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for relation in relations:
        key = (
            str(relation["source_fact_id"]),
            str(relation["target_fact_id"]),
            str(relation["relationship_type"] or "").strip().upper().replace(" ", "_"),
        )
        current = best.get(key)
        if current is None:
            best[key] = dict(relation)
            continue
        merged = dict(current)
        if float(relation.get("confidence") or 0) > float(current.get("confidence") or 0):
            merged = dict(relation)
            merged["reasoning"] = _combine_reasons(relation.get("reasoning"), current.get("reasoning"))
        else:
            merged["reasoning"] = _combine_reasons(current.get("reasoning"), relation.get("reasoning"))
            merged["confidence"] = max(
                float(current.get("confidence") or 0),
                float(relation.get("confidence") or 0),
            )
        best[key] = merged
    return list(best.values())


def _combine_reasons(*parts: Any) -> str:
    seen: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen)


def parse_amount(value: Any, unit: Any = None) -> tuple[float | None, str]:
    text = f"{value or ''} {unit or ''}"
    match = _NUMBER_RE.search(text.replace(",", ""))
    if not match:
        return None, _unit_token(text)
    return float(match.group()), _unit_token(text)


def units_compatible(left: str, right: str) -> bool:
    if not left or not right:
        return True
    return left == right


def amounts_match(left: float, right: float, *, rel: float = 0.02, abs_tol: float = 0.51) -> bool:
    return abs(left - right) <= max(abs_tol, rel * max(abs(left), abs(right), 1.0))


def _unit_token(text: str) -> str:
    match = _UNIT_RE.search(text or "")
    if not match:
        return ""
    token = match.group(1).lower()
    aliases = {
        "cr": "crore",
        "lac": "lakh",
        "pct": "%",
        "percent": "%",
        "rs": "inr",
        "₹": "inr",
    }
    return aliases.get(token, token)


def load_candidate_debug_log() -> list[dict[str, Any]]:
    if CANDIDATE_LOG_PATH.exists():
        try:
            return json.loads(CANDIDATE_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return list(_candidate_log)


def _save_candidate_log() -> None:
    CANDIDATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_LOG_PATH.write_text(
        json.dumps(_candidate_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def debug_rows(facts: list[dict[str, Any]] | None = None, db_path: str | None = None) -> list[dict[str, Any]]:
    rows = facts if facts is not None else search_facts(db_path=db_path)
    debug: list[dict[str, Any]] = []
    for row in rows:
        detail = get_fact(row["id"], db_path=db_path) or row
        evidence = detail.get("evidence") or []
        relationships = detail.get("relationships") or []
        debug.append(
            {
                "id": detail.get("id"),
                "fact": format_fact_statement(detail),
                "evidence_count": detail.get("evidence_count") or len(evidence),
                "relationship_count": detail.get("relationship_count") or len(relationships),
                "evidence_records": evidence,
                "relationship_records": relationships,
            }
        )
    return debug
