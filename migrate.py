"""Apply Fact Knowledge Layer database migrations.

Usage:
    python migrate.py
    python migrate.py --db data/facts.db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from database import (
    DB_PATH,
    cleanup_relationships,
    deduplicate_equivalent_facts,
    init_db,
    migrate_canonical_columns,
    migrate_embeddings,
    migrate_fact_evidence,
    migrate_fact_relationships,
    search_facts,
    list_all_relationships,
)
from provenance_service import rebuild_relationships
from semantic_search_service import index_all_facts


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate facts.db for provenance tracking.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to SQLite database")
    args = parser.parse_args()
    init_db(args.db)
    migrate_fact_evidence(db_path=args.db)
    migrate_fact_relationships(db_path=args.db)
    migrate_canonical_columns(db_path=args.db)
    migrate_embeddings(db_path=args.db)
    deduped = deduplicate_equivalent_facts(db_path=args.db)
    cleaned = cleanup_relationships(db_path=args.db)
    rebuilt = rebuild_relationships(db_path=str(args.db))
    indexed = index_all_facts(args.db)
    facts = search_facts(db_path=args.db)
    rels = list_all_relationships(db_path=args.db)
    print(f"Migrated {args.db}")
    print(
        f"Fact deduplication: merged {deduped.get('facts_merged', 0)}, "
        f"remaining {deduped.get('facts_remaining', len(facts))}"
    )
    print(
        f"Cleanup: self-links removed {cleaned.get('self_relationships_removed', 0)}, "
        f"duplicates removed {cleaned.get('duplicate_relationships_removed', 0)}"
    )
    print(
        f"Raw relationships generated: {rebuilt.get('raw_relationships_generated', 0)}\n"
        f"Duplicate relationships removed: {rebuilt.get('duplicate_relationships_removed', 0)}\n"
        f"Self relationships removed: {rebuilt.get('self_relationships_removed', 0)}\n"
        f"Final facts stored: {len(facts)}\n"
        f"Final relationships stored: {len(rels)}\n"
        f"Embeddings indexed: {indexed}"
    )


if __name__ == "__main__":
    main()
