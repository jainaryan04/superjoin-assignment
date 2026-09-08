"""Canonical fact merging: equivalent facts collapse to one row with all evidence."""

from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from canonicalization_service import normalized_value_key
from database import (
    get_fact,
    init_db,
    insert_facts,
    list_all_relationships,
    list_fact_clusters,
    search_facts,
)
from provenance_service import link_fact_relationships


def _fact(**kwargs) -> dict:
    return {
        "unit": None,
        "confidence": 0.9,
        "evidence_type": "DIRECT",
        "page_number": kwargs.get("page", 1),
        "evidence_text": kwargs.get("evidence")
        or f"{kwargs.get('attribute')} {kwargs.get('value')}",
        **kwargs,
    }


class FactMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(tempfile.mkdtemp()) / "facts.db"
        init_db(self.db_path)

    def _totals(self) -> list[dict]:
        return [
            row
            for row in search_facts(db_path=self.db_path)
            if row["canonical_attribute"] == "total_revenue"
        ]

    def test_currency_symbol_variants_merge(self) -> None:
        insert_facts(
            [_fact(entity="Company", attribute="Revenue", value="₹120 crore",
                   period="FY2024", source_document="3.1.pdf", page=1)],
            self.db_path,
        )
        insert_facts(
            [_fact(entity="Company", attribute="Annual Revenue", value="INR 120 crore",
                   period="FY2024", source_document="3.2.pdf", page=2)],
            self.db_path,
        )

        totals = self._totals()
        self.assertEqual(len(totals), 1)
        fact = get_fact(totals[0]["id"], self.db_path)
        self.assertEqual(
            {(e["source_document"], e["page_number"]) for e in fact["evidence"]},
            {("3.1.pdf", 1), ("3.2.pdf", 2)},
        )

    def test_hedged_value_merges_when_normalized_key_matches(self) -> None:
        insert_facts(
            [_fact(entity="Company", attribute="Employee Count", value="520",
                   period=None, source_document="a.pdf")],
            self.db_path,
        )
        insert_facts(
            [_fact(entity="Company", attribute="Headcount", value="Approximately 520",
                   period=None, source_document="b.pdf")],
            self.db_path,
        )

        rows = [
            row
            for row in search_facts(db_path=self.db_path)
            if row["canonical_attribute"] == "employee_count"
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            normalized_value_key(rows[0]["canonical_value"] or rows[0]["value"], rows[0]["unit"]),
            normalized_value_key("520"),
        )
        fact = get_fact(rows[0]["id"], self.db_path)
        self.assertEqual({e["source_document"] for e in fact["evidence"]}, {"a.pdf", "b.pdf"})

        clusters = [
            c for c in list_fact_clusters(self.db_path)
            if c["canonical_attribute"] == "employee_count"
        ]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]["supporting_fact_ids"]), 1)
        self.assertEqual(clusters[0]["document_count"], 2)

    def test_evidence_from_multiple_pdfs_is_preserved(self) -> None:
        for name, page, text in [
            ("3.1.pdf", 1, "FY2024 revenue was ₹120 crore."),
            ("3.2.pdf", 2, "Revenue: INR 120 crore (FY2024)."),
            ("3.3.pdf", 7, "Total revenue 120 crore INR."),
        ]:
            insert_facts(
                [_fact(entity="Company", attribute="Revenue", value="₹120 crore",
                       period="FY2024", source_document=name, page=page, evidence=text)],
                self.db_path,
            )

        totals = self._totals()
        self.assertEqual(len(totals), 1)
        fact = get_fact(totals[0]["id"], self.db_path)
        by_doc = {e["source_document"]: e for e in fact["evidence"]}
        self.assertEqual(set(by_doc), {"3.1.pdf", "3.2.pdf", "3.3.pdf"})
        self.assertEqual(by_doc["3.3.pdf"]["page_number"], 7)
        self.assertEqual(by_doc["3.2.pdf"]["evidence_text"], "Revenue: INR 120 crore (FY2024).")
        self.assertGreaterEqual(fact["evidence_count"], 3)

    def test_contradictory_values_stay_separate(self) -> None:
        insert_facts(
            [_fact(entity="Company", attribute="Revenue", value="₹120 crore",
                   period="FY2024", source_document="a.pdf")],
            self.db_path,
        )
        insert_facts(
            [_fact(entity="Company", attribute="Revenue", value="₹110 crore",
                   period="FY2024", source_document="b.pdf")],
            self.db_path,
        )

        totals = self._totals()
        self.assertEqual(len(totals), 2)
        self.assertEqual(
            {normalized_value_key(r["canonical_value"] or r["value"], r["unit"]) for r in totals},
            {"120|crore", "110|crore"},
        )

        link_fact_relationships(db_path=self.db_path)
        types = {r["relationship_type"] for r in list_all_relationships(self.db_path)}
        self.assertIn("CONTRADICTS", types)

    def test_supporting_documents_reflect_all_evidence(self) -> None:
        insert_facts(
            [_fact(entity="Company", attribute="Revenue", value="₹120 crore",
                   period="FY2024", source_document="A.pdf", page=1)],
            self.db_path,
        )
        insert_facts(
            [_fact(entity="Company", attribute="Annual Revenue", value="INR 120 crore",
                   period="FY2024", source_document="B.pdf", page=2)],
            self.db_path,
        )
        totals = self._totals()
        self.assertEqual(len(totals), 1)
        self.assertEqual(totals[0]["supporting_documents"], ["A.pdf", "B.pdf"])
        self.assertEqual(
            get_fact(totals[0]["id"], self.db_path)["supporting_documents"],
            ["A.pdf", "B.pdf"],
        )


_REVENUE_DOCS = {
    "A.pdf": [
        ("Revenue", "₹120 crore", 1, "FY2024 revenue was ₹120 crore."),
        ("Product Revenue", "₹90 crore", 1, "Product revenue ₹90 crore."),
        ("Services Revenue", "₹30 crore", 1, "Services revenue ₹30 crore."),
    ],
    "B.pdf": [
        ("Annual Revenue", "INR 120 crore", 2, "Annual revenue INR 120 crore."),
        ("Software Products", "INR 90 crore", 2, "Software products INR 90 crore."),
        ("Consulting Services", "INR 30 crore", 2, "Consulting services INR 30 crore."),
    ],
}


def _build_and_link(db_path: Path, doc_order: list[str]) -> Counter:
    init_db(db_path)
    for name in doc_order:
        insert_facts(
            [
                _fact(entity="Company", attribute=attr, value=value, period="FY2024",
                      source_document=name, page=page, evidence=text)
                for (attr, value, page, text) in _REVENUE_DOCS[name]
            ],
            db_path,
        )
    link_fact_relationships(db_path=db_path)
    return Counter(r["relationship_type"] for r in list_all_relationships(db_path))


class RelationshipDeterminismTests(unittest.TestCase):
    def test_relationship_counts_stable_across_merge_order(self) -> None:
        observed: list[Counter] = []
        for _ in range(12):
            for order in (["A.pdf", "B.pdf"], ["B.pdf", "A.pdf"]):
                db_path = Path(tempfile.mkdtemp()) / "facts.db"
                observed.append(_build_and_link(db_path, order))

        baseline = observed[0]
        for counts in observed[1:]:
            self.assertEqual(counts, baseline, (baseline, counts))

        # PART_OF must survive the merge regardless of which row won.
        self.assertGreaterEqual(baseline["PART_OF"], 2)
        self.assertGreaterEqual(baseline["COMPUTED_SUPPORT"], 1)
        self.assertEqual(baseline.get("CORROBORATES", 0), 0)

    def test_merged_totals_keep_part_of_from_every_child(self) -> None:
        db_path = Path(tempfile.mkdtemp()) / "facts.db"
        _build_and_link(db_path, ["B.pdf", "A.pdf"])
        rels = list_all_relationships(db_path)
        part_of_targets = {
            r["target_canonical_attribute"]
            for r in rels
            if r["relationship_type"] == "PART_OF"
        }
        self.assertEqual(part_of_targets, {"total_revenue"})
        part_of_sources = {
            r["source_canonical_attribute"]
            for r in rels
            if r["relationship_type"] == "PART_OF"
        }
        self.assertEqual(part_of_sources, {"product_revenue", "services_revenue"})


if __name__ == "__main__":
    unittest.main()
