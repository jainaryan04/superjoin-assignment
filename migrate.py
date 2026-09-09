"""Canonicalization / relationship migration for the Fact Knowledge Layer.

Pipeline (each stage timed, nothing deleted silently):

    snapshot DB
    -> schema migrations
    -> reprocess_fact_canonicalization
    -> deduplicate_equivalent_facts
    -> rebuild relationships
    -> rebuild clusters
    -> rebuild embeddings / FAISS index

Usage:
    python migrate.py
    python migrate.py --db data/facts.db --json reports/migration.json
    python migrate.py --from-snapshot snapshots/facts.pre-remediation.db --db /tmp/run1.db
    python migrate.py --no-snapshot        # only when the caller already holds one
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from database import (
    DB_PATH,
    cleanup_relationships,
    deduplicate_equivalent_facts,
    delete_all_relationships,
    get_connection,
    init_db,
    list_all_relationships,
    migrate_canonical_columns,
    migrate_embeddings,
    migrate_fact_evidence,
    migrate_fact_relationships,
    rebuild_fact_clusters,
    reprocess_fact_canonicalization,
    search_facts,
)
from provenance_service import link_fact_relationships
from semantic_search_service import index_all_facts

STAGES = [
    "snapshot",
    "schema_migrations",
    "canonicalization",
    "deduplication",
    "relationship_rebuild",
    "cluster_rebuild",
    "embedding_rebuild",
]


class Timings:
    """Per-stage wall-clock timings."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = round(time.perf_counter() - started, 3)

    def render(self) -> str:
        if not self.seconds:
            return "(no stages timed)"
        width = max(max(len(k) for k in self.seconds), len("stage"))
        total = sum(self.seconds.values())
        lines = [f"{'stage':<{width}}  {'seconds':>9}", "-" * (width + 11)]
        ordered = [n for n in STAGES if n in self.seconds]
        ordered += [n for n in self.seconds if n not in STAGES]
        for name in ordered:
            lines.append(f"{name:<{width}}  {self.seconds[name]:>9.3f}")
        lines.append("-" * (width + 11))
        lines.append(f"{'TOTAL':<{width}}  {total:>9.3f}")
        return "\n".join(lines)


def snapshot_database(db_path: Path, snapshot_dir: Path) -> Path | None:
    """Copy the database aside before mutating it. Never deletes anything."""
    if not db_path.exists():
        return None
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = snapshot_dir / f"{db_path.stem}.{stamp}.db"
    shutil.copy2(db_path, target)
    return target


def run_migration(
    db_path: Path,
    *,
    snapshot_dir: Path = Path("snapshots"),
    take_snapshot: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    timings = Timings()
    result: dict[str, Any] = {"db": str(db_path)}

    with timings.stage("snapshot"):
        snapshot = snapshot_database(db_path, snapshot_dir) if take_snapshot else None
    result["snapshot"] = str(snapshot) if snapshot else None

    with timings.stage("schema_migrations"):
        init_db(db_path)
        migrate_fact_evidence(db_path=db_path)
        migrate_fact_relationships(db_path=db_path)
        migrate_canonical_columns(db_path=db_path)
        migrate_embeddings(db_path=db_path)

    with timings.stage("canonicalization"):
        with get_connection(db_path) as conn:
            recanonicalized = reprocess_fact_canonicalization(conn)
            conn.commit()
    result["facts_recanonicalized"] = recanonicalized

    with timings.stage("deduplication"):
        deduped = deduplicate_equivalent_facts(db_path=db_path)
    result["facts_merged"] = deduped.get("facts_merged", 0)
    result["facts_remaining"] = deduped.get("facts_remaining", 0)

    with timings.stage("relationship_rebuild"):
        cleaned = cleanup_relationships(db_path)
        deleted = delete_all_relationships(db_path)
        linked = link_fact_relationships(db_path=str(db_path))
    result["relationships_deleted"] = deleted
    result["relationships_added"] = linked.get("relationships_added", 0)
    result["relationships_rejected"] = linked.get("relationships_rejected", 0)
    result["raw_relationships_generated"] = linked.get("raw_relationships_generated", 0)
    result["self_relationships_removed"] = (
        cleaned.get("self_relationships_removed", 0)
        + linked.get("self_relationships_removed", 0)
    )
    result["duplicate_relationships_removed"] = (
        cleaned.get("duplicate_relationships_removed", 0)
        + linked.get("duplicate_relationships_removed", 0)
    )

    with timings.stage("cluster_rebuild"):
        clusters = rebuild_fact_clusters(db_path)
    result["clusters"] = clusters

    with timings.stage("embedding_rebuild"):
        indexed = index_all_facts(db_path)
    result["embeddings_indexed"] = indexed

    result["facts"] = len(search_facts(db_path=db_path))
    result["relationships"] = len(list_all_relationships(db_path=db_path))
    result["timings_seconds"] = timings.seconds

    if verbose:
        print(f"Migrated {db_path}")
        if snapshot:
            print(f"Snapshot written to {snapshot}")
        print(
            f"Canonicalization: {recanonicalized} fact(s) reprocessed\n"
            f"Deduplication: merged {result['facts_merged']}, "
            f"remaining {result['facts_remaining']}\n"
            f"Relationships: deleted {deleted}, added {result['relationships_added']}, "
            f"rejected {result['relationships_rejected']}\n"
            f"Self relationships removed: {result['self_relationships_removed']}\n"
            f"Duplicate relationships removed: {result['duplicate_relationships_removed']}\n"
            f"Final facts stored: {result['facts']}\n"
            f"Final relationships stored: {result['relationships']}\n"
            f"Clusters: {clusters}\n"
            f"Embeddings indexed: {indexed}"
        )
        print()
        print(timings.render())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate facts.db canonicalization, relationships, clusters and embeddings."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to SQLite database")
    parser.add_argument(
        "--from-snapshot", type=Path, help="Copy this snapshot over --db before migrating"
    )
    parser.add_argument("--snapshot-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--no-snapshot", action="store_true", help="Skip taking a snapshot")
    parser.add_argument("--json", type=Path, help="Write the migration result as JSON")
    args = parser.parse_args()

    if args.from_snapshot:
        args.db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.from_snapshot, args.db)
        print(f"Restored {args.from_snapshot} -> {args.db}")

    result = run_migration(
        args.db,
        snapshot_dir=args.snapshot_dir,
        take_snapshot=not args.no_snapshot,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
