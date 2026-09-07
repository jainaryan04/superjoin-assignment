from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("data") / "facts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    entity TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    unit TEXT,
    period TEXT,
    confidence REAL NOT NULL,
    source_document TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    evidence_text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity);
CREATE INDEX IF NOT EXISTS idx_facts_attribute ON facts(attribute);
CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source_document);
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
        conn.executescript(SCHEMA)
        conn.commit()


def insert_facts(facts: list[dict[str, Any]], db_path: Path | str | None = None) -> int:
    if not facts:
        return 0
    columns = (
        "id",
        "entity",
        "attribute",
        "value",
        "unit",
        "period",
        "confidence",
        "source_document",
        "page_number",
        "evidence_text",
    )
    rows = [tuple(fact[col] for col in columns) for fact in facts]
    with get_connection(db_path) as conn:
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO facts ({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})
            """,
            rows,
        )
        conn.commit()
        return conn.total_changes


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
                entity LIKE ? OR
                attribute LIKE ? OR
                value LIKE ? OR
                unit LIKE ? OR
                period LIKE ? OR
                source_document LIKE ? OR
                evidence_text LIKE ?
            )
            """
        )
        params.extend([like] * 7)

    if source_document:
        clauses.append("source_document = ?")
        params.append(source_document)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT
            id, entity, attribute, value, unit, period, confidence,
            source_document, page_number, evidence_text
        FROM facts
        {where}
        ORDER BY source_document, page_number, entity, attribute
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def list_source_documents(db_path: Path | str | None = None) -> list[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT source_document FROM facts ORDER BY source_document"
        ).fetchall()
        return [row["source_document"] for row in rows]


def delete_facts_for_document(source_document: str, db_path: Path | str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM facts WHERE source_document = ?", (source_document,))
        conn.commit()


def clear_all_facts(db_path: Path | str | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM facts")
        conn.commit()
