"""Relationship quality: uniqueness, fuzzy corroboration, evidence, clusters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from database import (
    get_fact,
    init_db,
    insert_facts,
    insert_relationship,
    list_all_relationships,
    list_fact_clusters,
    rebuild_fact_clusters,
    search_facts,
)
from provenance_service import link_fact_relationships
from relationship_rules import values_approximately_equivalent


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


class RelationshipQualityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(tempfile.mkdtemp()) / "facts.db"
        init_db(self.db_path)

    def test_fuzzy_numeric_corroboration(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Company",
                    attribute="Employee Count",
                    value="500",
                    period=None,
                    source_document="a.pdf",
                ),
                _fact(
                    entity="Company",
                    attribute="Headcount",
                    value="Approximately 520",
                    period=None,
                    source_document="b.pdf",
                ),
            ],
            self.db_path,
        )
        rows = search_facts(db_path=self.db_path)
        self.assertTrue(values_approximately_equivalent(rows[0], rows[1]))
        link_fact_relationships(db_path=self.db_path)
        types = {row["relationship_type"] for row in list_all_relationships(self.db_path)}
        self.assertIn("CORROBORATES", types)
        self.assertNotIn("RECONCILES", types)
        reasoning = next(
            row["reasoning"]
            for row in list_all_relationships(self.db_path)
            if row["relationship_type"] == "CORROBORATES"
        )
        self.assertIn("5%", reasoning)

    def test_ceo_without_evidence_is_unresolved(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Company",
                    attribute="CEO",
                    value="Rohit Sharma",
                    period="FY2024",
                    source_document="a.pdf",
                    evidence="The CEO is Rohit Sharma.",
                ),
                _fact(
                    entity="Company",
                    attribute="CEO",
                    value="Priya Mehta",
                    period="FY2025",
                    source_document="b.pdf",
                    evidence="Priya Mehta is CEO.",
                ),
            ],
            self.db_path,
        )
        link_fact_relationships(db_path=self.db_path)
        types = {row["relationship_type"] for row in list_all_relationships(self.db_path)}
        self.assertIn("POTENTIAL_CONTRADICTION", types)
        self.assertNotIn("RECONCILES", types)

    def test_ceo_with_resignation_evidence_reconciles(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Company",
                    attribute="CEO",
                    value="Rohit Sharma",
                    period="FY2024",
                    source_document="a.pdf",
                    evidence="The CEO is Rohit Sharma.",
                ),
                _fact(
                    entity="Company",
                    attribute="CEO",
                    value="Priya Mehta",
                    period="FY2025",
                    source_document="b.pdf",
                    evidence="Priya Mehta appointed CEO after Rohit Sharma resigned in July 2025.",
                ),
            ],
            self.db_path,
        )
        link_fact_relationships(db_path=self.db_path)
        rels = [row for row in list_all_relationships(self.db_path) if row["relationship_type"] == "RECONCILES"]
        self.assertEqual(len(rels), 1)
        self.assertIn("#", rels[0]["reasoning"])
        self.assertTrue(rels[0]["supporting_fact_ids"])

    def test_computed_support_and_part_of(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Revenue",
                    attribute="Revenue",
                    value="₹120 crore",
                    period="FY2024",
                    source_document="a.pdf",
                ),
                _fact(
                    entity="Revenue",
                    attribute="Product Revenue",
                    value="₹90 crore",
                    period="FY2024",
                    source_document="a.pdf",
                ),
                _fact(
                    entity="Revenue",
                    attribute="Services Revenue",
                    value="₹30 crore",
                    period="FY2024",
                    source_document="a.pdf",
                ),
            ],
            self.db_path,
        )
        link_fact_relationships(db_path=self.db_path)
        types = {row["relationship_type"] for row in list_all_relationships(self.db_path)}
        self.assertIn("PART_OF", types)
        self.assertIn("COMPUTED_SUPPORT", types)
        computed = next(
            row for row in list_all_relationships(self.db_path) if row["relationship_type"] == "COMPUTED_SUPPORT"
        )
        self.assertEqual(computed["reasoning"].replace(" ", ""), "90+30=120")
        self.assertEqual(len(computed["computed_components"]), 2)

    def test_no_duplicate_or_reverse_pairs(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Revenue",
                    attribute="Revenue",
                    value="₹120 crore",
                    period="FY2024",
                    source_document="survey.pdf",
                ),
                _fact(
                    entity="Revenue",
                    attribute="Annual Revenue",
                    value="INR 130 crore",
                    period="FY2024",
                    source_document="rbi.pdf",
                ),
            ],
            self.db_path,
        )
        # Distinct values -> two separate facts (a merge would leave only one row).
        facts = search_facts(db_path=self.db_path)
        self.assertEqual(len(facts), 2)
        left, right = facts[0], facts[1]
        payload = {
            "source_fact_id": left["id"],
            "target_fact_id": right["id"],
            "relationship_type": "CORROBORATES",
            "confidence": 0.9,
            "reasoning": "Same canonical attribute and normalized value.",
            "supporting_fact_ids": [left["id"], right["id"]],
        }
        self.assertTrue(insert_relationship(payload, self.db_path))
        self.assertFalse(insert_relationship(payload, self.db_path))
        reverse = dict(payload)
        reverse["source_fact_id"] = right["id"]
        reverse["target_fact_id"] = left["id"]
        self.assertFalse(insert_relationship(reverse, self.db_path))
        rels = [row for row in list_all_relationships(self.db_path) if row["relationship_type"] == "CORROBORATES"]
        self.assertEqual(len(rels), 1)

    def test_corroboration_clusters_count_documents(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Revenue",
                    attribute="Revenue",
                    value="₹120 crore",
                    period="FY2024",
                    source_document="Survey.pdf",
                ),
                _fact(
                    entity="Revenue",
                    attribute="Annual Revenue",
                    value="INR 120 crore",
                    period="FY2024",
                    source_document="RBI.pdf",
                ),
                _fact(
                    entity="Revenue",
                    attribute="Revenue",
                    value="120 INR crore",
                    period="FY2024",
                    source_document="IMF.pdf",
                ),
            ],
            self.db_path,
        )
        rebuild_fact_clusters(self.db_path)
        clusters = [
            row
            for row in list_fact_clusters(self.db_path)
            if row["canonical_attribute"] == "total_revenue"
        ]
        self.assertTrue(clusters)
        self.assertGreaterEqual(max(row["document_count"] for row in clusters), 2)

    def test_hedged_and_plain_value_merge_into_one_fact(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Company",
                    attribute="Employee Count",
                    value="520",
                    period=None,
                    source_document="a.pdf",
                ),
                _fact(
                    entity="Company",
                    attribute="Headcount",
                    value="Approximately 520",
                    period=None,
                    source_document="b.pdf",
                ),
            ],
            self.db_path,
        )
        rows = [
            row
            for row in search_facts(db_path=self.db_path)
            if row["canonical_attribute"] == "employee_count"
        ]
        self.assertEqual(len(rows), 1)
        fact = get_fact(rows[0]["id"], self.db_path)
        self.assertEqual({e["source_document"] for e in fact["evidence"]}, {"a.pdf", "b.pdf"})

        rebuild_fact_clusters(self.db_path)
        clusters = [
            row
            for row in list_fact_clusters(self.db_path)
            if row["canonical_attribute"] == "employee_count"
        ]
        self.assertEqual(len(clusters), 1, clusters)
        self.assertEqual(len(clusters[0]["supporting_fact_ids"]), 1)
        self.assertEqual(clusters[0]["document_count"], 2)

    def test_different_companies_ceos_do_not_contradict(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Acme",
                    attribute="CEO",
                    value="Rohit Sharma",
                    period="FY2024",
                    source_document="acme.pdf",
                    evidence="Acme CEO is Rohit Sharma.",
                ),
                _fact(
                    entity="Globex",
                    attribute="CEO",
                    value="Priya Mehta",
                    period="FY2024",
                    source_document="globex.pdf",
                    evidence="Globex CEO is Priya Mehta.",
                ),
            ],
            self.db_path,
        )
        link_fact_relationships(db_path=self.db_path)
        types = {row["relationship_type"] for row in list_all_relationships(self.db_path)}
        self.assertNotIn("CONTRADICTS", types)
        self.assertNotIn("POTENTIAL_CONTRADICTION", types)
        self.assertNotIn("CORROBORATES", types)
        self.assertNotIn("RECONCILES", types)

    def test_different_companies_employee_counts_do_not_contradict(self) -> None:
        insert_facts(
            [
                _fact(
                    entity="Acme",
                    attribute="Employee Count",
                    value="500",
                    period=None,
                    source_document="acme.pdf",
                ),
                _fact(
                    entity="Globex",
                    attribute="Headcount",
                    value="800",
                    period=None,
                    source_document="globex.pdf",
                ),
            ],
            self.db_path,
        )
        link_fact_relationships(db_path=self.db_path)
        types = {row["relationship_type"] for row in list_all_relationships(self.db_path)}
        self.assertNotIn("CONTRADICTS", types)
        self.assertNotIn("POTENTIAL_CONTRADICTION", types)
        self.assertNotIn("UNRESOLVED_DIFFERENCE", types)
        self.assertNotIn("CORROBORATES", types)


if __name__ == "__main__":
    unittest.main()
