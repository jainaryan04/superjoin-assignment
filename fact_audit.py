"""Corpus-level semantic audit of the fact graph.

Produces the same metric set before and after a migration so drift is visible.

Usage:
    python fact_audit.py --db data/facts.db
    python fact_audit.py --db data/facts.db --json out.json
    python fact_audit.py --diff before.json after.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from canonicalization_service import dimensions_compatible, value_dimension

# Attribute fragments that must never resolve to a headcount metric.
NON_HEADCOUNT_RE = re.compile(
    r"(salary|benefit|compensation|option|share|percentage|percent|cost|"
    r"contribution|expense|revenue|payable|insurance|esic|healthcare|session|"
    r"turnover|holder|start_date|date)",
    re.IGNORECASE,
)
# Units a headcount can never legitimately carry.
NON_HEADCOUNT_UNITS = {
    "million", "billion", "crore", "lakh", "%", "inr", "usd", "eur", "gbp",
    "shares", "equity shares", "options",
}
DUP_CURRENCY_RE = re.compile(
    r"\b(INR|USD|EUR|GBP)\b.*(\b(INR|USD|EUR|GBP)\b|[₹$€£])"
)
RATIO_RE = re.compile(r"%")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
    except sqlite3.Error:
        return 0


def audit(db_path: str | Path) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        facts = [dict(r) for r in conn.execute("SELECT * FROM facts").fetchall()]
        rel_rows = conn.execute(
            """
            SELECT r.relationship_type,
                   sf.canonical_attribute AS sa, sf.canonical_value AS scv, sf.unit AS su,
                   tf.canonical_attribute AS ta, tf.canonical_value AS tcv, tf.unit AS tu
            FROM fact_relationships r
            JOIN facts sf ON sf.id = r.source_fact_id
            JOIN facts tf ON tf.id = r.target_fact_id
            """
        ).fetchall()

        by_type = Counter(r["relationship_type"] for r in rel_rows)

        employee = [f for f in facts if f.get("canonical_attribute") == "employee_count"]
        employee_invalid = [
            f
            for f in employee
            if NON_HEADCOUNT_RE.search(str(f.get("raw_attribute") or ""))
            or str(f.get("unit") or "").strip().lower() in NON_HEADCOUNT_UNITS
            or RATIO_RE.search(str(f.get("canonical_value") or ""))
        ]

        # Revenue *level* attributes only -- percentage_of_revenue etc. are ratios
        # and must never carry a monetary value, but they are not revenue levels.
        revenue_levels = {
            "total_revenue", "product_revenue", "services_revenue",
            "subscription_revenue", "licensing_revenue", "revenue",
        }
        revenue = [f for f in facts if str(f.get("canonical_attribute") or "") in revenue_levels]
        revenue_ratio = [
            f
            for f in revenue
            if value_dimension(f.get("canonical_value"), f.get("unit")) == "ratio"
        ]

        currency_anomalies = [
            f for f in facts if DUP_CURRENCY_RE.search(str(f.get("canonical_value") or ""))
        ]

        # Relationships the dimension gate should have blocked: a CONTRADICTS or
        # CORROBORATES between values of genuinely incomparable kinds
        # (e.g. money vs percentage).
        dim_mismatch = [
            r
            for r in rel_rows
            if r["relationship_type"] in {"CONTRADICTS", "CORROBORATES"}
            and not dimensions_compatible(r["scv"], r["su"], r["tcv"], r["tu"])
        ]
        dim_strictly_different = [
            r
            for r in rel_rows
            if r["relationship_type"] in {"CONTRADICTS", "CORROBORATES"}
            and value_dimension(r["scv"], r["su"]) != value_dimension(r["tcv"], r["tu"])
        ]

        mixed_dimension_attrs: dict[str, list[str]] = {}
        per_attr: dict[str, set[str]] = {}
        for f in facts:
            attr = str(f.get("canonical_attribute") or "")
            per_attr.setdefault(attr, set()).add(
                value_dimension(f.get("canonical_value"), f.get("unit"))
            )
        for attr, dims in per_attr.items():
            if "ratio" in dims and ({"money", "scaled"} & dims):
                mixed_dimension_attrs[attr] = sorted(dims)

        return {
            "facts": len(facts),
            "evidence": _table_count(conn, "fact_evidence"),
            "relationships": len(rel_rows),
            "clusters": _table_count(conn, "fact_clusters"),
            "embeddings": _table_count(conn, "fact_embeddings"),
            "relationships_by_type": dict(sorted(by_type.items())),
            "contradictions": by_type.get("CONTRADICTS", 0),
            "employee_count_facts": len(employee),
            "employee_count_invalid": len(employee_invalid),
            "revenue_facts": len(revenue),
            "revenue_facts_with_ratio_value": len(revenue_ratio),
            "currency_anomalies": len(currency_anomalies),
            "dimensional_mismatch_relationships": len(dim_mismatch),
            "dim_strictly_different_relationships": len(dim_strictly_different),
            "mixed_dimension_attributes": len(mixed_dimension_attrs),
            "mixed_dimension_attribute_names": sorted(mixed_dimension_attrs),
            "distinct_canonical_attributes": len(per_attr),
            "facts_missing_evidence": _facts_missing_evidence(conn),
            "facts_missing_embedding": _facts_missing_embedding(conn),
        }
    finally:
        conn.close()


def _facts_missing_evidence(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM facts f
        WHERE NOT EXISTS (SELECT 1 FROM fact_evidence e WHERE e.fact_id = f.id)
        """
    ).fetchone()
    return int(row["n"] if row else 0)


def _facts_missing_embedding(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM facts f
            WHERE NOT EXISTS (SELECT 1 FROM fact_embeddings e WHERE e.fact_id = f.id)
            """
        ).fetchone()
    except sqlite3.Error:
        return 0
    return int(row["n"] if row else 0)


ORDER = [
    "facts",
    "evidence",
    "relationships",
    "clusters",
    "embeddings",
    "contradictions",
    "employee_count_facts",
    "employee_count_invalid",
    "revenue_facts",
    "revenue_facts_with_ratio_value",
    "currency_anomalies",
    "dimensional_mismatch_relationships",
    "dim_strictly_different_relationships",
    "mixed_dimension_attributes",
    "distinct_canonical_attributes",
    "facts_missing_evidence",
    "facts_missing_embedding",
]


def render(report: dict[str, Any], title: str = "AUDIT") -> str:
    lines = [title, "=" * len(title)]
    for key in ORDER:
        lines.append(f"{key:38} {report.get(key, 0)}")
    lines.append(f"{'relationships_by_type':38} {report.get('relationships_by_type')}")
    return "\n".join(lines)


def render_diff(before: dict[str, Any], after: dict[str, Any]) -> str:
    lines = [
        f"{'metric':38} {'before':>10} {'after':>10} {'delta':>10}",
        "-" * 72,
    ]
    for key in ORDER:
        b, a = before.get(key, 0), after.get(key, 0)
        delta = a - b
        lines.append(f"{key:38} {b:>10} {a:>10} {delta:>+10}")
    lines.append("")
    lines.append(f"before types: {before.get('relationships_by_type')}")
    lines.append(f"after  types: {after.get('relationships_by_type')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fact-graph semantic quality.")
    parser.add_argument("--db", type=Path, default=Path("data/facts.db"))
    parser.add_argument("--json", type=Path, help="Write the report as JSON")
    parser.add_argument("--title", default="AUDIT")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.diff:
        before = json.loads(args.diff[0].read_text())
        after = json.loads(args.diff[1].read_text())
        print(render_diff(before, after))
        return

    report = audit(args.db)
    print(render(report, args.title))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
