"""SQLite persistence for facts and provenance evidence."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

DB_PATH = Path("data") / "facts.db"

EVIDENCE_TYPES = ("DIRECT", "SUPPORTING")
RELATIONSHIP_TYPES = (
    "PART_OF",
    "SUPPORTS",
    "CORROBORATES",
    "CONTRADICTS",
    "RECONCILES",
    "TEMPORAL_SUCCESSOR",
)
TYPE_RANK = {name: index for index, name in enumerate(EVIDENCE_TYPES)}

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    identity_key TEXT NOT NULL UNIQUE,
    entity TEXT NOT NULL,
    attribute TEXT NOT NULL,
    raw_attribute TEXT,
    canonical_attribute TEXT,
    canonical_entity TEXT,
    canonical_value TEXT,
    original_entity TEXT,
    original_attribute TEXT,
    original_value TEXT,
    original_period TEXT,
    value TEXT NOT NULL,
    unit TEXT,
    period TEXT,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fact_evidence (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL,
    source_document TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    evidence_text TEXT NOT NULL,
    evidence_type TEXT NOT NULL DEFAULT 'DIRECT',
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (fact_id) REFERENCES facts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fact_relationships (
    id TEXT PRIMARY KEY,
    source_fact_id TEXT NOT NULL,
    target_fact_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_fact_id) REFERENCES facts(id) ON DELETE CASCADE,
    FOREIGN KEY (target_fact_id) REFERENCES facts(id) ON DELETE CASCADE
);
"""

INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity);
CREATE INDEX IF NOT EXISTS idx_facts_attribute ON facts(attribute);
CREATE INDEX IF NOT EXISTS idx_facts_canonical_attribute ON facts(canonical_attribute);
CREATE INDEX IF NOT EXISTS idx_facts_identity ON facts(identity_key);
CREATE INDEX IF NOT EXISTS idx_evidence_fact ON fact_evidence(fact_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON fact_evidence(source_document);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON fact_evidence(evidence_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_unique
    ON fact_evidence(fact_id, source_document, page_number, evidence_text, evidence_type);
CREATE INDEX IF NOT EXISTS idx_rel_source ON fact_relationships(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON fact_relationships(target_fact_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_unique
    ON fact_relationships(source_fact_id, target_fact_id, relationship_type);
"""

SCHEMA = TABLE_SCHEMA + INDEX_SCHEMA
UNIQUE_EVIDENCE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_unique
    ON fact_evidence(fact_id, source_document, page_number, evidence_text, evidence_type);
"""


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    with get_connection(db_path) as conn:
        _migrate_legacy_facts(conn)
        existing = _table_names(conn)
        needs_relationship_backfill = "fact_relationships" not in existing
        conn.executescript(TABLE_SCHEMA)
        migrate_fact_evidence(conn)
        migrate_canonical_columns(conn)
        if needs_relationship_backfill:
            _strip_cross_fact_evidence(conn)
        deduplicate_equivalent_facts(conn)
        _cleanup_relationship_rows(conn)
        conn.executescript(INDEX_SCHEMA)
        conn.commit()


def migrate_fact_evidence(conn: sqlite3.Connection | None = None, db_path: Path | str | None = None) -> None:
    """Add evidence_type and refresh uniqueness for snippet provenance."""
    owns_connection = conn is None
    if owns_connection:
        conn = get_connection(db_path)
    assert conn is not None
    try:
        tables = _table_names(conn)
        if "fact_evidence" not in tables:
            conn.executescript(TABLE_SCHEMA)
            conn.executescript(INDEX_SCHEMA)
            if owns_connection:
                conn.commit()
            return

        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(fact_evidence)").fetchall()
        }
        if "evidence_type" not in columns:
            conn.execute(
                "ALTER TABLE fact_evidence ADD COLUMN evidence_type TEXT NOT NULL DEFAULT 'DIRECT'"
            )
            conn.execute(
                "UPDATE fact_evidence SET evidence_type = 'DIRECT' "
                "WHERE evidence_type IS NULL OR trim(evidence_type) = ''"
            )

        conn.execute("DROP INDEX IF EXISTS idx_evidence_unique")
        conn.executescript(UNIQUE_EVIDENCE_INDEX)
        if owns_connection:
            conn.commit()
    finally:
        if owns_connection:
            conn.close()


CANONICAL_FACT_COLUMNS = {
    "raw_attribute": "TEXT",
    "canonical_attribute": "TEXT",
    "canonical_entity": "TEXT",
    "canonical_value": "TEXT",
    "original_entity": "TEXT",
    "original_attribute": "TEXT",
    "original_value": "TEXT",
    "original_period": "TEXT",
}


def migrate_canonical_columns(
    conn: sqlite3.Connection | None = None, db_path: Path | str | None = None
) -> None:
    """Add canonical fields and backfill existing facts. Does not rebuild relationships."""
    owns_connection = conn is None
    if owns_connection:
        conn = get_connection(db_path)
    assert conn is not None
    try:
        if "facts" not in _table_names(conn):
            conn.executescript(TABLE_SCHEMA)
            conn.executescript(INDEX_SCHEMA)
            if owns_connection:
                conn.commit()
            return
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
        for name, typedef in CANONICAL_FACT_COLUMNS.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {typedef}")
        reprocess_fact_canonicalization(conn)
        if owns_connection:
            conn.commit()
    finally:
        if owns_connection:
            conn.close()


def reprocess_fact_canonicalization(conn: sqlite3.Connection) -> int:
    """Repair canonical fields from the immutable original snapshot.

    Does not merge facts, does not read evidence, and does not change values
    except to restore original_value when it was already captured.
    """
    from canonicalization_service import canonicalize_fact

    rows = conn.execute("SELECT * FROM facts").fetchall()
    updated = 0
    for row in rows:
        fact = dict(row)
        original_entity = fact.get("original_entity") or fact.get("entity")
        original_attribute = (
            fact.get("original_attribute")
            or fact.get("raw_attribute")
            or fact.get("attribute")
        )
        original_value = fact.get("original_value") or fact.get("value")
        original_period = (
            fact["original_period"]
            if fact.get("original_period") not in (None, "")
            else fact.get("period")
        )
        prepared = canonicalize_fact(
            {
                "entity": original_entity,
                "attribute": original_attribute,
                "raw_attribute": original_attribute,
                "original_entity": original_entity,
                "original_attribute": original_attribute,
                "original_value": original_value,
                "original_period": original_period,
                "value": original_value,
                "unit": fact.get("unit"),
                "period": original_period,
            }
        )
        key = identity_key(
            prepared.get("canonical_entity") or prepared["entity"],
            prepared.get("canonical_attribute") or prepared["attribute"],
            original_value,
            prepared.get("unit"),
            prepared.get("period"),
        )
        collision = conn.execute(
            "SELECT id FROM facts WHERE identity_key = ? AND id != ?",
            (key, fact["id"]),
        ).fetchone()
        if collision:
            key = f"{key}|{fact['id']}"
        conn.execute(
            """
            UPDATE facts SET
                identity_key = ?,
                attribute = ?,
                raw_attribute = ?,
                canonical_attribute = ?,
                canonical_entity = ?,
                canonical_value = ?,
                original_entity = ?,
                original_attribute = ?,
                original_value = ?,
                original_period = ?,
                value = ?,
                unit = ?,
                period = ?
            WHERE id = ?
            """,
            (
                key,
                prepared["canonical_attribute"],
                original_attribute,
                prepared["canonical_attribute"],
                prepared.get("canonical_entity"),
                prepared.get("canonical_value") or original_value,
                original_entity,
                original_attribute,
                original_value,
                original_period,
                original_value,
                prepared.get("unit"),
                prepared.get("period"),
                fact["id"],
            ),
        )
        updated += 1
    return updated


def _merge_fact_into(conn: sqlite3.Connection, source_id: str, target_id: str) -> None:
    """Move evidence and relationships from source onto target, then delete source."""
    if source_id == target_id:
        return
    incoming = conn.execute("SELECT * FROM facts WHERE id = ?", (source_id,)).fetchone()
    keeper = conn.execute("SELECT * FROM facts WHERE id = ?", (target_id,)).fetchone()
    if incoming is None or keeper is None:
        return

    evidence_rows = conn.execute(
        "SELECT * FROM fact_evidence WHERE fact_id = ?",
        (source_id,),
    ).fetchall()
    for item in evidence_rows:
        try:
            conn.execute(
                "UPDATE fact_evidence SET fact_id = ? WHERE id = ?",
                (target_id, item["id"]),
            )
        except sqlite3.IntegrityError:
            conn.execute("DELETE FROM fact_evidence WHERE id = ?", (item["id"],))

    conn.execute(
        """
        DELETE FROM fact_relationships
        WHERE (source_fact_id = ? AND target_fact_id = ?)
           OR (source_fact_id = ? AND target_fact_id = ?)
           OR (source_fact_id = target_fact_id AND source_fact_id IN (?, ?))
        """,
        (source_id, target_id, target_id, source_id, source_id, target_id),
    )
    rels = conn.execute(
        """
        SELECT * FROM fact_relationships
        WHERE source_fact_id = ? OR target_fact_id = ?
        """,
        (source_id, source_id),
    ).fetchall()
    for rel in rels:
        new_source = target_id if rel["source_fact_id"] == source_id else rel["source_fact_id"]
        new_target = target_id if rel["target_fact_id"] == source_id else rel["target_fact_id"]
        if new_source == new_target:
            conn.execute("DELETE FROM fact_relationships WHERE id = ?", (rel["id"],))
            continue
        existing = conn.execute(
            """
            SELECT id, confidence, reasoning FROM fact_relationships
            WHERE source_fact_id = ? AND target_fact_id = ? AND relationship_type = ?
            """,
            (new_source, new_target, rel["relationship_type"]),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE fact_relationships
                SET confidence = ?, reasoning = ?
                WHERE id = ?
                """,
                (
                    max(float(existing["confidence"] or 0), float(rel["confidence"] or 0)),
                    _merge_reasoning(existing["reasoning"], rel["reasoning"]),
                    existing["id"],
                ),
            )
            conn.execute("DELETE FROM fact_relationships WHERE id = ?", (rel["id"],))
            continue
        conn.execute(
            """
            UPDATE fact_relationships
            SET source_fact_id = ?, target_fact_id = ?
            WHERE id = ?
            """,
            (new_source, new_target, rel["id"]),
        )

    from canonicalization_service import more_specific_period

    period = more_specific_period(keeper["period"], incoming["period"])
    confidence = max(float(keeper["confidence"] or 0), float(incoming["confidence"] or 0))
    entity = keeper["canonical_entity"] or keeper["entity"]
    attribute = keeper["canonical_attribute"] or keeper["attribute"]
    value = keeper["canonical_value"] or keeper["value"]
    key = identity_key(entity, attribute, value, keeper["unit"], period)
    collision = conn.execute(
        "SELECT id FROM facts WHERE identity_key = ? AND id != ?",
        (key, target_id),
    ).fetchone()
    if collision:
        key = f"{key}|{target_id}"
    conn.execute(
        """
        UPDATE facts SET period = ?, confidence = ?, identity_key = ?
        WHERE id = ?
        """,
        (period, confidence, key, target_id),
    )
    conn.execute("DELETE FROM facts WHERE id = ?", (source_id,))
    _refresh_fact_confidence(conn, target_id)


def _merge_reasoning(*parts: Any) -> str | None:
    seen: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen) if seen else None


def _fact_equivalence_key(fact: dict[str, Any]) -> tuple[str, str, str]:
    from canonicalization_service import normalized_value_key

    entity = _norm(str(fact.get("canonical_entity") or fact.get("entity") or ""))
    attribute = _norm(str(fact.get("canonical_attribute") or fact.get("attribute") or ""))
    value = normalized_value_key(fact.get("canonical_value") or fact.get("value"), fact.get("unit"))
    return entity, attribute, value


def _find_equivalent_fact(
    conn: sqlite3.Connection, fact: dict[str, Any], *, exclude_id: str | None = None
) -> sqlite3.Row | None:
    from canonicalization_service import periods_mergeable

    target_key = _fact_equivalence_key(fact)
    period = fact.get("period")
    rows = conn.execute("SELECT * FROM facts").fetchall()
    for row in rows:
        if exclude_id and row["id"] == exclude_id:
            continue
        if _fact_equivalence_key(dict(row)) != target_key:
            continue
        if periods_mergeable(period, row["period"]):
            return row
    return None


def deduplicate_equivalent_facts(
    conn: sqlite3.Connection | None = None, db_path: Path | str | None = None
) -> dict[str, int]:
    """Merge facts that share entity, canonical attribute, and normalized value."""
    owns_connection = conn is None
    if owns_connection:
        conn = get_connection(db_path)
    assert conn is not None
    merged = 0
    try:
        if "facts" not in _table_names(conn):
            return {"facts_merged": 0, "facts_remaining": 0}
        rows = [dict(row) for row in conn.execute("SELECT * FROM facts").fetchall()]
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for fact in rows:
            groups.setdefault(_fact_equivalence_key(fact), []).append(fact)
        for group in groups.values():
            if len(group) < 2:
                continue
            nonempty_periods = {
                _norm(str(item.get("period") or ""))
                for item in group
                if str(item.get("period") or "").strip()
            }
            if len(nonempty_periods) > 1:
                by_period: dict[str, list[dict[str, Any]]] = {}
                for item in group:
                    period_key = _norm(str(item.get("period") or ""))
                    if not period_key:
                        continue
                    by_period.setdefault(period_key, []).append(item)
                for cluster in by_period.values():
                    merged += _merge_fact_cluster(conn, cluster)
                continue
            merged += _merge_fact_cluster(conn, group)
        remaining = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
        if owns_connection:
            conn.commit()
        return {"facts_merged": merged, "facts_remaining": int(remaining or 0)}
    finally:
        if owns_connection:
            conn.close()


def _merge_fact_cluster(conn: sqlite3.Connection, group: list[dict[str, Any]]) -> int:
    if len(group) < 2:
        return 0
    ranked = sorted(
        group,
        key=lambda item: (
            0 if str(item.get("period") or "").strip() else 1,
            str(item.get("id") or ""),
        ),
    )
    keeper = ranked[0]
    merged = 0
    for item in ranked[1:]:
        _merge_fact_into(conn, item["id"], keeper["id"])
        merged += 1
    return merged


def migrate_fact_relationships(
    conn: sqlite3.Connection | None = None, db_path: Path | str | None = None
) -> None:
    """Ensure fact_relationships exists and strip legacy cross-fact evidence rows."""
    owns_connection = conn is None
    if owns_connection:
        conn = get_connection(db_path)
    assert conn is not None
    try:
        existing = _table_names(conn)
        first_create = "fact_relationships" not in existing
        conn.executescript(TABLE_SCHEMA)
        if first_create:
            _strip_cross_fact_evidence(conn)
        _cleanup_relationship_rows(conn)
        conn.executescript(INDEX_SCHEMA)
        if owns_connection:
            conn.commit()
    finally:
        if owns_connection:
            conn.close()


def _strip_cross_fact_evidence(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        DELETE FROM fact_evidence
        WHERE upper(IFNULL(evidence_type, 'DIRECT')) NOT IN ('DIRECT')
        """
    )
    for row in conn.execute("SELECT id FROM facts").fetchall():
        _refresh_fact_confidence(conn, row["id"])


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def insert_facts(facts: list[dict[str, Any]], db_path: Path | str | None = None) -> int:
    """Upsert canonical facts and attach one or more evidence snippets.

    Matching canonical entity/attribute/value/unit/period rows are merged so later
    documents can corroborate the same fact. Relationships are added later by
    provenance_service.link_fact_relationships().
    """
    if not facts:
        return 0

    from canonicalization_service import canonicalize_fact

    touched: set[str] = set()
    with get_connection(db_path) as conn:
        for fact in facts:
            evidence_items = _evidence_from_fact(fact)
            if not evidence_items:
                continue
            prepared = dict(fact)
            original_entity = str(prepared.get("original_entity") or prepared.get("entity") or "").strip()
            original_attribute = str(
                prepared.get("original_attribute") or prepared.get("attribute") or ""
            ).strip()
            original_value = str(prepared.get("original_value") or prepared.get("value") or "").strip()
            original_period = prepared.get("original_period")
            if original_period in (None, ""):
                original_period = prepared.get("period")
            prepared["original_entity"] = original_entity
            prepared["original_attribute"] = original_attribute
            prepared["original_value"] = original_value
            prepared["original_period"] = original_period
            prepared = canonicalize_fact(prepared)
            prepared["original_entity"] = original_entity
            prepared["original_attribute"] = original_attribute
            prepared["original_value"] = original_value
            prepared["original_period"] = original_period
            prepared["value"] = original_value
            fact_id = _upsert_fact(conn, prepared)
            for item in evidence_items:
                _insert_evidence(conn, fact_id, item)
            _refresh_fact_confidence(conn, fact_id)
            touched.add(fact_id)
        conn.commit()
    return len(touched)


def insert_evidence(item: dict[str, Any], db_path: Path | str | None = None) -> bool:
    fact_id = str(item.get("fact_id") or "")
    if not fact_id:
        return False
    with get_connection(db_path) as conn:
        inserted = _insert_evidence(conn, fact_id, item)
        if inserted:
            _refresh_fact_confidence(conn, fact_id)
            conn.commit()
        return inserted


def insert_relationship(item: dict[str, Any], db_path: Path | str | None = None) -> bool:
    source_id = str(item.get("source_fact_id") or "")
    target_id = str(item.get("target_fact_id") or "")
    if not source_id or not target_id or source_id == target_id:
        return False
    with get_connection(db_path) as conn:
        inserted = _insert_relationship(conn, item)
        conn.commit()
        return inserted


def relationship_exists(
    source_fact_id: str,
    target_fact_id: str,
    relationship_type: str,
    db_path: Path | str | None = None,
) -> bool:
    with get_connection(db_path) as conn:
        return _relationship_exists(
            conn,
            str(source_fact_id),
            str(target_fact_id),
            _normalize_relationship_type(relationship_type),
        )


def _cleanup_relationship_rows(conn: sqlite3.Connection) -> tuple[int, int]:
    if "fact_relationships" not in _table_names(conn):
        return 0, 0
    self_cur = conn.execute(
        "DELETE FROM fact_relationships WHERE source_fact_id = target_fact_id"
    )
    dup_cur = conn.execute(
        """
        DELETE FROM fact_relationships
        WHERE id NOT IN (
            SELECT id FROM (
                SELECT MIN(id) AS id
                FROM fact_relationships
                GROUP BY source_fact_id, target_fact_id, relationship_type
            )
        )
        """
    )
    return int(self_cur.rowcount or 0), int(dup_cur.rowcount or 0)


def cleanup_relationships(db_path: Path | str | None = None) -> dict[str, int]:
    """Delete self-links and duplicates, then restore the unique constraint."""
    with get_connection(db_path) as conn:
        self_removed, duplicate_removed = _cleanup_relationship_rows(conn)
        conn.execute("DROP INDEX IF EXISTS idx_rel_unique")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_unique
            ON fact_relationships(source_fact_id, target_fact_id, relationship_type)
            """
        )
        remaining = conn.execute("SELECT COUNT(*) AS n FROM fact_relationships").fetchone()["n"]
        conn.commit()
    return {
        "self_relationships_removed": self_removed,
        "duplicate_relationships_removed": duplicate_removed,
        "final_relationships_stored": int(remaining or 0),
    }


def delete_all_relationships(db_path: Path | str | None = None) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM fact_relationships")
        conn.commit()
        return int(cur.rowcount or 0)


def delete_relationships_by_id(ids: list[str], db_path: Path | str | None = None) -> int:
    if not ids:
        return 0
    with get_connection(db_path) as conn:
        conn.executemany(
            "DELETE FROM fact_relationships WHERE id = ?",
            [(rel_id,) for rel_id in ids],
        )
        conn.commit()
        return len(ids)


def list_relationships(
    fact_id: str, db_path: Path | str | None = None
) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        return _list_relationships(conn, fact_id)


def list_all_relationships(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                r.id,
                r.source_fact_id,
                r.target_fact_id,
                r.relationship_type,
                r.confidence,
                r.reasoning,
                sf.entity AS source_entity,
                sf.attribute AS source_attribute,
                sf.raw_attribute AS source_raw_attribute,
                sf.canonical_attribute AS source_canonical_attribute,
                sf.value AS source_value,
                sf.unit AS source_unit,
                sf.period AS source_period,
                tf.entity AS target_entity,
                tf.attribute AS target_attribute,
                tf.raw_attribute AS target_raw_attribute,
                tf.canonical_attribute AS target_canonical_attribute,
                tf.value AS target_value,
                tf.unit AS target_unit,
                tf.period AS target_period
            FROM fact_relationships r
            JOIN facts sf ON sf.id = r.source_fact_id
            JOIN facts tf ON tf.id = r.target_fact_id
            ORDER BY r.relationship_type, sf.attribute, tf.attribute
            """
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["source_fact"] = format_fact_statement(
                {
                    "attribute": item["source_attribute"],
                    "value": item["source_value"],
                    "unit": item["source_unit"],
                    "period": item["source_period"],
                }
            )
            item["target_fact"] = format_fact_statement(
                {
                    "attribute": item["target_attribute"],
                    "value": item["target_value"],
                    "unit": item["target_unit"],
                    "period": item["target_period"],
                }
            )
            item["source_fact_record"] = {
                "id": item["source_fact_id"],
                "entity": item.get("source_entity"),
                "attribute": item["source_attribute"],
                "raw_attribute": item.get("source_raw_attribute"),
                "canonical_attribute": item.get("source_canonical_attribute"),
                "value": item["source_value"],
                "unit": item["source_unit"],
                "period": item["source_period"],
            }
            item["target_fact_record"] = {
                "id": item["target_fact_id"],
                "entity": item.get("target_entity"),
                "attribute": item["target_attribute"],
                "raw_attribute": item.get("target_raw_attribute"),
                "canonical_attribute": item.get("target_canonical_attribute"),
                "value": item["target_value"],
                "unit": item["target_unit"],
                "period": item["target_period"],
            }
            item["source_statement"] = item["source_fact"]
            item["target_statement"] = item["target_fact"]
            _decorate_relationship(item)
            results.append(item)
        return results


def find_self_links(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, source_fact_id, target_fact_id, relationship_type
            FROM fact_relationships
            WHERE source_fact_id = target_fact_id
            """
        ).fetchall()
        return [dict(row) for row in rows]


def find_duplicate_relationships(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Groups with more than one row for the same source, target, and type."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                source_fact_id,
                target_fact_id,
                relationship_type,
                COUNT(*) AS duplicate_count,
                GROUP_CONCAT(id) AS relationship_ids
            FROM fact_relationships
            GROUP BY source_fact_id, target_fact_id, relationship_type
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        return [dict(row) for row in rows]


def delete_duplicate_relationships(db_path: Path | str | None = None) -> int:
    """Keep the highest-confidence row in each duplicate group."""
    duplicates = find_duplicate_relationships(db_path)
    if not duplicates:
        return 0
    removed = 0
    with get_connection(db_path) as conn:
        for group in duplicates:
            rows = conn.execute(
                """
                SELECT id FROM fact_relationships
                WHERE source_fact_id = ? AND target_fact_id = ? AND relationship_type = ?
                ORDER BY confidence DESC, id ASC
                """,
                (group["source_fact_id"], group["target_fact_id"], group["relationship_type"]),
            ).fetchall()
            for extra in rows[1:]:
                conn.execute("DELETE FROM fact_relationships WHERE id = ?", (extra["id"],))
                removed += 1
        conn.commit()
    return removed


def validate_relationships(db_path: Path | str | None = None) -> dict[str, Any]:
    self_links = find_self_links(db_path)
    duplicates = find_duplicate_relationships(db_path)
    count_mismatches = validate_provenance_counts(db_path)
    return {
        "ok": not self_links and not duplicates and not count_mismatches,
        "self_links": self_links,
        "duplicates": duplicates,
        "count_mismatches": count_mismatches,
    }


def validate_provenance_counts(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return facts whose stored UI counts do not match table row counts."""
    mismatches: list[dict[str, Any]] = []
    for fact in search_facts(db_path=db_path):
        with get_connection(db_path) as conn:
            evidence_n = conn.execute(
                "SELECT COUNT(*) AS n FROM fact_evidence WHERE fact_id = ?",
                (fact["id"],),
            ).fetchone()["n"]
            relationship_n = conn.execute(
                """
                SELECT COUNT(*) AS n FROM fact_relationships
                WHERE source_fact_id = ? OR target_fact_id = ?
                """,
                (fact["id"], fact["id"]),
            ).fetchone()["n"]
        if evidence_n != fact["evidence_count"] or relationship_n != fact["relationship_count"]:
            mismatches.append(
                {
                    "fact_id": fact["id"],
                    "evidence_count": fact["evidence_count"],
                    "evidence_count_actual": evidence_n,
                    "relationship_count": fact["relationship_count"],
                    "relationship_count_actual": relationship_n,
                }
            )
    return mismatches


def search_facts(
    query: str = "",
    source_document: str | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if query.strip():
        like = f"%{query.strip()}%"
        clauses.append(
            """
            (
                f.entity LIKE ? OR
                f.attribute LIKE ? OR
                IFNULL(f.raw_attribute, '') LIKE ? OR
                IFNULL(f.canonical_attribute, '') LIKE ? OR
                f.value LIKE ? OR
                IFNULL(f.unit, '') LIKE ? OR
                IFNULL(f.period, '') LIKE ? OR
                EXISTS (
                    SELECT 1 FROM fact_evidence e_search
                    WHERE e_search.fact_id = f.id
                      AND (
                          e_search.source_document LIKE ?
                          OR e_search.evidence_text LIKE ?
                      )
                )
            )
            """
        )
        params.extend([like] * 9)

    if source_document:
        clauses.append(
            """
            EXISTS (
                SELECT 1 FROM fact_evidence e_doc
                WHERE e_doc.fact_id = f.id AND e_doc.source_document = ?
            )
            """
        )
        params.append(source_document)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT
            f.id,
            f.entity,
            f.attribute,
            f.raw_attribute,
            f.canonical_attribute,
            f.canonical_entity,
            f.canonical_value,
            f.original_entity,
            f.original_attribute,
            f.original_value,
            f.original_period,
            f.value,
            f.unit,
            f.period,
            f.confidence,
            (
                SELECT COUNT(*) FROM fact_evidence e_count
                WHERE e_count.fact_id = f.id
            ) AS evidence_count,
            (
                SELECT COUNT(*) FROM fact_relationships r_count
                WHERE r_count.source_fact_id = f.id OR r_count.target_fact_id = f.id
            ) AS relationship_count,
            (
                SELECT COUNT(DISTINCT e_src.source_document)
                FROM fact_evidence e_src
                WHERE e_src.fact_id = f.id
            ) AS document_count,
            (
                SELECT GROUP_CONCAT(DISTINCT e_type.evidence_type)
                FROM fact_evidence e_type
                WHERE e_type.fact_id = f.id
            ) AS evidence_types,
            primary_e.source_document,
            primary_e.page_number,
            primary_e.evidence_text,
            primary_e.evidence_type
        FROM facts f
        LEFT JOIN fact_evidence primary_e
            ON primary_e.id = (
                SELECT e.id
                FROM fact_evidence e
                WHERE e.fact_id = f.id
                ORDER BY
                    CASE e.evidence_type
                        WHEN 'DIRECT' THEN 0
                        WHEN 'SUPPORTING' THEN 1
                        WHEN 'DERIVED_FROM' THEN 2
                        WHEN 'EXPLANATORY' THEN 3
                        ELSE 4
                    END,
                    e.confidence DESC,
                    e.page_number ASC,
                    e.id ASC
                LIMIT 1
            )
        {where}
        ORDER BY f.entity, f.attribute, f.value
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def get_fact(fact_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                f.id,
                f.entity,
                f.attribute,
                f.raw_attribute,
                f.canonical_attribute,
                f.canonical_entity,
                f.canonical_value,
                f.original_entity,
                f.original_attribute,
                f.original_value,
                f.original_period,
                f.value,
                f.unit,
                f.period,
                f.confidence,
                (
                    SELECT COUNT(*) FROM fact_evidence e_count
                    WHERE e_count.fact_id = f.id
                ) AS evidence_count,
                (
                    SELECT COUNT(*) FROM fact_relationships r_count
                    WHERE r_count.source_fact_id = f.id OR r_count.target_fact_id = f.id
                ) AS relationship_count,
                (
                    SELECT COUNT(DISTINCT e_src.source_document)
                    FROM fact_evidence e_src
                    WHERE e_src.fact_id = f.id
                ) AS document_count,
                (
                    SELECT GROUP_CONCAT(DISTINCT e_type.evidence_type)
                    FROM fact_evidence e_type
                    WHERE e_type.fact_id = f.id
                ) AS evidence_types
            FROM facts f
            WHERE f.id = ?
            """,
            (fact_id,),
        ).fetchone()
        if row is None:
            return None
        fact = dict(row)
        fact["evidence"] = _list_evidence(conn, fact_id)
        fact["relationships"] = _list_relationships(conn, fact_id)
        return fact


def list_facts_for_document(
    source_document: str | None = None,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Facts plus a representative page/document for nearby-evidence search."""
    params: list[Any] = []
    where = ""
    if source_document:
        where = """
            WHERE EXISTS (
                SELECT 1 FROM fact_evidence e_doc
                WHERE e_doc.fact_id = f.id AND e_doc.source_document = ?
            )
        """
        params.append(source_document)

    sql = f"""
        SELECT
            f.id,
            f.entity,
            f.attribute,
            f.raw_attribute,
            f.canonical_attribute,
            f.canonical_entity,
            f.canonical_value,
            f.original_entity,
            f.original_attribute,
            f.original_value,
            f.original_period,
            f.value,
            f.unit,
            f.period,
            f.confidence,
            COALESCE(direct_e.source_document, any_e.source_document) AS source_document,
            COALESCE(direct_e.page_number, any_e.page_number) AS page_number,
            COALESCE(direct_e.evidence_text, any_e.evidence_text) AS evidence_text
        FROM facts f
        LEFT JOIN fact_evidence direct_e
            ON direct_e.id = (
                SELECT e.id FROM fact_evidence e
                WHERE e.fact_id = f.id AND e.evidence_type = 'DIRECT'
                ORDER BY e.page_number ASC, e.id ASC
                LIMIT 1
            )
        LEFT JOIN fact_evidence any_e
            ON any_e.id = (
                SELECT e.id FROM fact_evidence e
                WHERE e.fact_id = f.id
                ORDER BY e.page_number ASC, e.id ASC
                LIMIT 1
            )
        {where}
        ORDER BY page_number, f.entity, f.attribute
    """
    with get_connection(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def list_source_documents(db_path: Path | str | None = None) -> list[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT source_document
            FROM fact_evidence
            ORDER BY source_document
            """
        ).fetchall()
        return [row["source_document"] for row in rows]


def delete_facts_for_document(source_document: str, db_path: Path | str | None = None) -> None:
    with get_connection(db_path) as conn:
        fact_ids = [
            row["fact_id"]
            for row in conn.execute(
                "SELECT DISTINCT fact_id FROM fact_evidence WHERE source_document = ?",
                (source_document,),
            ).fetchall()
        ]
        conn.execute(
            "DELETE FROM fact_evidence WHERE source_document = ?",
            (source_document,),
        )
        for fact_id in fact_ids:
            remaining = conn.execute(
                "SELECT COUNT(*) AS n FROM fact_evidence WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()["n"]
            if remaining == 0:
                conn.execute(
                    """
                    DELETE FROM fact_relationships
                    WHERE source_fact_id = ? OR target_fact_id = ?
                    """,
                    (fact_id, fact_id),
                )
                conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            else:
                _refresh_fact_confidence(conn, fact_id)
        conn.commit()


def clear_all_facts(db_path: Path | str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM fact_relationships")
        conn.execute("DELETE FROM fact_evidence")
        conn.execute("DELETE FROM facts")
        conn.commit()


def identity_key(entity: str, attribute: str, value: str, unit: str | None, period: str | None) -> str:
    return "|".join(
        [
            _norm(entity),
            _norm(attribute),
            _norm(value),
            _norm(unit or ""),
            _norm(period or ""),
        ]
    )


def provenance_dot(fact: dict[str, Any]) -> str:
    """Graphviz tree: fact → evidence_type → snippet."""
    lines = [
        "digraph Provenance {",
        "  rankdir=LR;",
        '  node [fontname="Helvetica"];',
        "  graph [bgcolor=transparent];",
    ]
    fact_label = _dot_label(format_fact_statement(fact))
    lines.append(f'  fact [label="{fact_label}", shape=box, style=filled, fillcolor="#D6EAF8"];')

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in fact.get("evidence") or []:
        grouped.setdefault(item.get("evidence_type") or "DIRECT", []).append(item)

    colors = {
        "DIRECT": "#D5F5E3",
        "SUPPORTING": "#FCF3CF",
        "DERIVED_FROM": "#D6EAF8",
        "EXPLANATORY": "#F5EEF8",
    }
    for type_index, evidence_type in enumerate(EVIDENCE_TYPES):
        items = grouped.get(evidence_type) or []
        if not items:
            continue
        type_id = f"type_{type_index}"
        lines.append(
            f'  {type_id} [label="{evidence_type}", shape=hexagon, style=filled, '
            f'fillcolor="{colors.get(evidence_type, "#FDEBD0")}"];'
        )
        lines.append(f"  fact -> {type_id};")
        for snippet_index, snippet in enumerate(items):
            ev_id = f"ev_{type_index}_{snippet_index}"
            label = _dot_label(
                f"{snippet['evidence_text']}\\nPage {snippet['page_number']}",
                limit=90,
            )
            lines.append(
                f'  {ev_id} [label="{label}", shape=note, style=filled, fillcolor="#FDEBD0"];'
            )
            lines.append(f"  {type_id} -> {ev_id};")

    lines.append("}")
    return "\n".join(lines)


def format_fact_statement(fact: dict[str, Any]) -> str:
    value = str(fact.get("value") or "").strip()
    unit = str(fact.get("unit") or "").strip()
    if unit and unit.lower() not in value.lower():
        value = f"{value} {unit}".strip()
    attribute = str(
        fact.get("canonical_attribute") or fact.get("attribute") or ""
    ).strip()
    return f"{attribute} = {value}".strip(" =")


def _upsert_fact(conn: sqlite3.Connection, fact: dict[str, Any]) -> str:
    key = identity_key(
        fact.get("canonical_entity") or fact["entity"],
        fact.get("canonical_attribute") or fact["attribute"],
        fact.get("canonical_value") or fact["value"],
        fact.get("unit"),
        fact.get("period"),
    )
    existing = conn.execute(
        "SELECT id FROM facts WHERE identity_key = ?",
        (key,),
    ).fetchone()
    if existing:
        return existing["id"]

    equivalent = _find_equivalent_fact(conn, fact)
    if equivalent:
        from canonicalization_service import more_specific_period

        period = more_specific_period(equivalent["period"], fact.get("period"))
        confidence = max(float(equivalent["confidence"] or 0), float(fact.get("confidence") or 0))
        entity = equivalent["canonical_entity"] or equivalent["entity"]
        attribute = equivalent["canonical_attribute"] or equivalent["attribute"]
        value = equivalent["canonical_value"] or equivalent["value"]
        merged_key = identity_key(entity, attribute, value, equivalent["unit"], period)
        collision = conn.execute(
            "SELECT id FROM facts WHERE identity_key = ? AND id != ?",
            (merged_key, equivalent["id"]),
        ).fetchone()
        if collision:
            merged_key = f"{merged_key}|{equivalent['id']}"
        conn.execute(
            """
            UPDATE facts SET period = ?, confidence = ?, identity_key = ?
            WHERE id = ?
            """,
            (period, confidence, merged_key, equivalent["id"]),
        )
        return equivalent["id"]

    fact_id = str(fact.get("id") or uuid.uuid4())
    canonical_attribute = fact.get("canonical_attribute") or fact["attribute"]
    conn.execute(
        """
        INSERT INTO facts (
            id, identity_key, entity, attribute, raw_attribute,
            canonical_attribute, canonical_entity, canonical_value,
            original_entity, original_attribute, original_value, original_period,
            value, unit, period, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fact_id,
            key,
            fact["entity"],
            canonical_attribute,
            fact.get("raw_attribute") or fact.get("original_attribute") or fact["attribute"],
            canonical_attribute,
            fact.get("canonical_entity") or fact["entity"],
            fact.get("canonical_value") or fact["value"],
            fact.get("original_entity") or fact["entity"],
            fact.get("original_attribute") or fact.get("raw_attribute") or fact["attribute"],
            fact.get("original_value") or fact["value"],
            fact.get("original_period") if fact.get("original_period") not in (None, "") else fact.get("period"),
            fact.get("original_value") or fact["value"],
            fact.get("unit"),
            fact.get("period"),
            float(fact.get("confidence") or 0.5),
        ),
    )
    return fact_id


def _insert_evidence(conn: sqlite3.Connection, fact_id: str, item: dict[str, Any]) -> bool:
    evidence_type = _normalize_evidence_type(item.get("evidence_type"))
    try:
        conn.execute(
            """
            INSERT INTO fact_evidence (
                id, fact_id, source_document, page_number, evidence_text, evidence_type, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(item.get("id") or uuid.uuid4()),
                fact_id,
                item["source_document"],
                int(item["page_number"]),
                item["evidence_text"],
                evidence_type,
                float(item.get("confidence") or 0.5),
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _relationship_exists(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    rel_type: str | None,
) -> bool:
    if not source_id or not target_id or not rel_type:
        return False
    row = conn.execute(
        """
        SELECT 1 FROM fact_relationships
        WHERE source_fact_id = ? AND target_fact_id = ? AND relationship_type = ?
        LIMIT 1
        """,
        (source_id, target_id, rel_type),
    ).fetchone()
    return row is not None


def _insert_relationship(conn: sqlite3.Connection, item: dict[str, Any]) -> bool:
    rel_type = _normalize_relationship_type(item.get("relationship_type"))
    if rel_type is None:
        return False
    source_id = str(item["source_fact_id"])
    target_id = str(item["target_fact_id"])
    if source_id == target_id:
        return False
    existing = conn.execute(
        """
        SELECT id, confidence, reasoning FROM fact_relationships
        WHERE source_fact_id = ? AND target_fact_id = ? AND relationship_type = ?
        """,
        (source_id, target_id, rel_type),
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE fact_relationships
            SET confidence = ?, reasoning = ?
            WHERE id = ?
            """,
            (
                max(float(existing["confidence"] or 0), float(item.get("confidence") or 0)),
                _merge_reasoning(existing["reasoning"], item.get("reasoning")),
                existing["id"],
            ),
        )
        return False
    try:
        conn.execute(
            """
            INSERT INTO fact_relationships (
                id, source_fact_id, target_fact_id, relationship_type, confidence, reasoning
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(item.get("id") or uuid.uuid4()),
                source_id,
                target_id,
                rel_type,
                float(item.get("confidence") or 0.7),
                (str(item.get("reasoning") or "").strip() or None),
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def _list_relationships(conn: sqlite3.Connection, fact_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.source_fact_id,
            r.target_fact_id,
            r.relationship_type,
            r.confidence,
            r.reasoning,
            sf.entity AS source_entity,
            sf.attribute AS source_attribute,
            sf.raw_attribute AS source_raw_attribute,
            sf.canonical_attribute AS source_canonical_attribute,
            sf.value AS source_value,
            sf.unit AS source_unit,
            sf.period AS source_period,
            tf.entity AS target_entity,
            tf.attribute AS target_attribute,
            tf.raw_attribute AS target_raw_attribute,
            tf.canonical_attribute AS target_canonical_attribute,
            tf.value AS target_value,
            tf.unit AS target_unit,
            tf.period AS target_period
        FROM fact_relationships r
        JOIN facts sf ON sf.id = r.source_fact_id
        JOIN facts tf ON tf.id = r.target_fact_id
        WHERE r.source_fact_id = ? OR r.target_fact_id = ?
        ORDER BY r.relationship_type, r.confidence DESC
        """,
        (fact_id, fact_id),
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["source_statement"] = format_fact_statement(
            {
                "attribute": item["source_attribute"],
                "value": item["source_value"],
                "unit": item["source_unit"],
                "period": item["source_period"],
            }
        )
        item["target_statement"] = format_fact_statement(
            {
                "attribute": item["target_attribute"],
                "value": item["target_value"],
                "unit": item["target_unit"],
                "period": item["target_period"],
            }
        )
        item["direction"] = "outgoing" if item["source_fact_id"] == fact_id else "incoming"
        other_id = item["target_fact_id"] if item["direction"] == "outgoing" else item["source_fact_id"]
        item["related_fact_id"] = other_id
        item["related_statement"] = (
            item["target_statement"] if item["direction"] == "outgoing" else item["source_statement"]
        )
        _decorate_relationship(item)
        results.append(item)
    return results


def _decorate_relationship(item: dict[str, Any]) -> dict[str, Any]:
    rel_type = str(item.get("relationship_type") or "")
    item["arrow"] = f"{item.get('source_statement')} -> {item.get('target_statement')}"
    item["direction_label"] = {
        "PART_OF": "Child -> Parent",
        "SUPPORTS": "Supporter -> Supported",
        "CORROBORATES": "Source -> Target",
        "CONTRADICTS": "Source -> Target",
        "RECONCILES": "Source -> Target",
        "TEMPORAL_SUCCESSOR": "Earlier -> Later",
    }.get(rel_type, "Source -> Target")
    return item


def _refresh_fact_confidence(conn: sqlite3.Connection, fact_id: str) -> None:
    rows = conn.execute(
        "SELECT confidence FROM fact_evidence WHERE fact_id = ?",
        (fact_id,),
    ).fetchall()
    if not rows:
        return
    combined = 1.0
    for row in rows:
        combined *= 1.0 - max(0.0, min(1.0, float(row["confidence"])))
    conn.execute(
        "UPDATE facts SET confidence = ? WHERE id = ?",
        (1.0 - combined, fact_id),
    )


def _list_evidence(conn: sqlite3.Connection, fact_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            id,
            source_document,
            page_number,
            evidence_text,
            evidence_type,
            confidence
        FROM fact_evidence
        WHERE fact_id = ?
        ORDER BY
            CASE evidence_type
                WHEN 'DIRECT' THEN 0
                WHEN 'SUPPORTING' THEN 1
                WHEN 'DERIVED_FROM' THEN 2
                WHEN 'EXPLANATORY' THEN 3
                ELSE 4
            END,
            page_number,
            confidence DESC
        """,
        (fact_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _evidence_from_fact(fact: dict[str, Any]) -> list[dict[str, Any]]:
    items = fact.get("evidence")
    if isinstance(items, list) and items:
        cleaned: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("evidence_text") or "").strip()
            source = str(item.get("source_document") or "").strip()
            page = item.get("page_number")
            if not text or not source or page is None:
                continue
            cleaned.append(
                {
                    "source_document": source,
                    "page_number": int(page),
                    "evidence_text": text,
                    "evidence_type": _normalize_evidence_type(item.get("evidence_type")),
                    "confidence": float(item.get("confidence") or fact.get("confidence") or 0.5),
                }
            )
        return cleaned

    text = str(fact.get("evidence_text") or "").strip()
    source = str(fact.get("source_document") or "").strip()
    page = fact.get("page_number")
    if not text or not source or page is None:
        return []
    return [
        {
            "source_document": source,
            "page_number": int(page),
            "evidence_text": text,
            "evidence_type": _normalize_evidence_type(fact.get("evidence_type")),
            "confidence": float(fact.get("confidence") or 0.5),
        }
    ]


def _normalize_evidence_type(value: Any) -> str:
    text = str(value or "DIRECT").strip().upper().replace(" ", "_")
    if text not in EVIDENCE_TYPES:
        return "DIRECT"
    return text


def _normalize_relationship_type(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace(" ", "_")
    aliases = {
        "SUPPORTING": "SUPPORTS",
        "SUPPORT": "SUPPORTS",
        "PART": "PART_OF",
        "COMPONENT": "PART_OF",
        "COMPOSED_OF": "PART_OF",
        "CORROBORATION": "CORROBORATES",
        "CONTRADICTION": "CONTRADICTS",
        "RECONCILIATION": "RECONCILES",
        "TEMPORAL": "TEMPORAL_SUCCESSOR",
        "SUCCESSOR": "TEMPORAL_SUCCESSOR",
        "DERIVED_FROM": "PART_OF",
        "EXPLANATORY": "SUPPORTS",
    }
    text = aliases.get(text, text)
    if text not in RELATIONSHIP_TYPES:
        return None
    return text


def _migrate_legacy_facts(conn: sqlite3.Connection) -> None:
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "facts" not in tables:
        return
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(facts)").fetchall()
    }
    if "identity_key" in columns or "evidence_text" not in columns:
        return

    legacy_rows = [dict(row) for row in conn.execute("SELECT * FROM facts").fetchall()]
    conn.execute("ALTER TABLE facts RENAME TO facts_legacy")
    conn.executescript(TABLE_SCHEMA)

    for record in legacy_rows:
        record["evidence_type"] = "DIRECT"
        fact_id = _upsert_fact(conn, record)
        _insert_evidence(conn, fact_id, record)
        _refresh_fact_confidence(conn, fact_id)

    conn.execute("DROP TABLE facts_legacy")
    conn.commit()


def _dot_label(text: str, limit: int = 64) -> str:
    cleaned = " ".join(str(text).replace("\\", "\\\\").replace('"', '\\"').split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return cleaned


def _norm(value: str) -> str:
    return " ".join(value.strip().lower().split())
