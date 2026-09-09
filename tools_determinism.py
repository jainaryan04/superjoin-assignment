"""Run the full migration N times from one snapshot and compare the outputs.

    python tools_determinism.py --snapshot snapshots/facts.pre-remediation.db --runs 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from migrate import run_migration


def fingerprint(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        facts = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
        clusters = conn.execute("SELECT COUNT(*) AS n FROM fact_clusters").fetchone()["n"]
        rels = conn.execute("SELECT COUNT(*) AS n FROM fact_relationships").fetchone()["n"]

        # Identity by stable fact ids (the snapshot is byte-identical every run).
        id_rows = sorted(
            f"{r['relationship_type']}|{r['pair_low']}|{r['pair_high']}"
            for r in conn.execute(
                "SELECT relationship_type, pair_low, pair_high FROM fact_relationships"
            ).fetchall()
        )
        # Identity by semantics, independent of row ids.
        sem_rows = sorted(
            "|".join(
                str(x or "")
                for x in (
                    r["relationship_type"],
                    r["sa"], r["scv"], r["sp"],
                    r["ta"], r["tcv"], r["tp"],
                )
            )
            for r in conn.execute(
                """
                SELECT r.relationship_type,
                       sf.canonical_attribute sa, sf.canonical_value scv, sf.period sp,
                       tf.canonical_attribute ta, tf.canonical_value tcv, tf.period tp
                FROM fact_relationships r
                JOIN facts sf ON sf.id = r.source_fact_id
                JOIN facts tf ON tf.id = r.target_fact_id
                """
            ).fetchall()
        )
        cluster_rows = sorted(
            f"{r['canonical_attribute']}|{r['canonical_value']}|{r['period'] or ''}|{r['document_count']}"
            for r in conn.execute(
                "SELECT canonical_attribute, canonical_value, period, document_count FROM fact_clusters"
            ).fetchall()
        )
        return {
            "facts": facts,
            "clusters": clusters,
            "relationships": rels,
            "relationship_id_hash": hashlib.sha256("\n".join(id_rows).encode()).hexdigest(),
            "relationship_semantic_hash": hashlib.sha256("\n".join(sem_rows).encode()).hexdigest(),
            "cluster_hash": hashlib.sha256("\n".join(cluster_rows).encode()).hexdigest(),
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify migration determinism.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    prints: list[dict[str, Any]] = []
    timings: list[dict[str, float]] = []
    workdir = Path(tempfile.mkdtemp(prefix="determinism-"))
    for i in range(1, args.runs + 1):
        db = workdir / f"run{i}.db"
        shutil.copy2(args.snapshot, db)
        result = run_migration(db, take_snapshot=False, verbose=False)
        timings.append(result["timings_seconds"])
        fp = fingerprint(db)
        prints.append(fp)
        print(
            f"run {i}: facts={fp['facts']} clusters={fp['clusters']} "
            f"relationships={fp['relationships']} "
            f"rel_id={fp['relationship_id_hash'][:12]} "
            f"rel_sem={fp['relationship_semantic_hash'][:12]} "
            f"clusters_hash={fp['cluster_hash'][:12]}"
        )

    baseline = prints[0]
    identical = all(fp == baseline for fp in prints[1:])
    print()
    print("DETERMINISTIC" if identical else "NON-DETERMINISTIC")
    if not identical:
        for i, fp in enumerate(prints[1:], start=2):
            for key in baseline:
                if fp[key] != baseline[key]:
                    print(f"  run{i} differs on {key}: {baseline[key]} != {fp[key]}")

    if timings:
        stages = list(timings[0])
        print()
        print(f"{'stage':<22} {'min':>8} {'mean':>8} {'max':>8}")
        for stage in stages:
            vals = [t[stage] for t in timings if stage in t]
            print(f"{stage:<22} {min(vals):>8.3f} {sum(vals)/len(vals):>8.3f} {max(vals):>8.3f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"deterministic": identical, "fingerprints": prints, "timings": timings},
                indent=2,
            )
        )
    raise SystemExit(0 if identical else 1)


if __name__ == "__main__":
    main()
