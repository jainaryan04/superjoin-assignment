"""Cross-document CORROBORATES / CONTRADICTS / RECONCILES."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from canonicalization_service import normalized_value_key
from database import get_fact, init_db, insert_facts, list_all_relationships, search_facts
from provenance_service import link_fact_relationships


def _fact(
    *,
    entity: str,
    attribute: str,
    value: str,
    period: str | None,
    document: str,
    page: int,
    evidence: str | None = None,
) -> dict:
    return {
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "unit": None,
        "period": period,
        "confidence": 0.9,
        "source_document": document,
        "page_number": page,
        "evidence_text": evidence or f"{attribute} {value} {period or ''}".strip(),
        "evidence_type": "DIRECT",
    }


class CrossDocumentReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(tempfile.mkdtemp()) / "facts.db"
        init_db(self.db_path)
        insert_facts(
            [
                _fact(
                    entity="Company",
                    attribute="Revenue",
                    value="₹120 crore",
                    period="FY2024",
                    document="report.pdf",
                    page=1,
                ),
                _fact(
                    entity="Company",
                    attribute="Product Revenue",
                    value="₹90 crore",
                    period="FY2024",
                    document="report.pdf",
                    page=1,
                ),
                _fact(
                    entity="Company",
                    attribute="Services Revenue",
                    value="₹30 crore",
                    period="FY2024",
                    document="report.pdf",
                    page=1,
                ),
                _fact(
                    entity="Company",
                    attribute="Employee Count",
                    value="500",
                    period=None,
                    document="report.pdf",
                    page=2,
                ),
                _fact(
                    entity="Company",
                    attribute="CEO",
                    value="Rohit Sharma",
                    period="FY2024",
                    document="report.pdf",
                    page=2,
                ),
            ],
            self.db_path,
        )
        insert_facts(
            [
                _fact(
                    entity="Company",
                    attribute="Annual Revenue",
                    value="INR 120 crore",
                    period="FY2024",
                    document="update.pdf",
                    page=1,
                ),
                _fact(
                    entity="Company",
                    attribute="Software Products",
                    value="₹90 crore",
                    period="FY2024",
                    document="update.pdf",
                    page=1,
                ),
                _fact(
                    entity="Company",
                    attribute="Consulting Services",
                    value="₹30 crore",
                    period="FY2024",
                    document="update.pdf",
                    page=1,
                ),
                _fact(
                    entity="Company",
                    attribute="Revenue",
                    value="₹110 crore",
                    period="FY2024",
                    document="update.pdf",
                    page=1,
                    evidence="FY2024 Revenue = ₹110 crore",
                ),
                _fact(
                    entity="Company",
                    attribute="Revenue",
                    value="₹35 crore",
                    period="Q1FY2025",
                    document="update.pdf",
                    page=2,
                ),
                _fact(
                    entity="Company",
                    attribute="Headcount",
                    value="520",
                    period=None,
                    document="update.pdf",
                    page=2,
                ),
                _fact(
                    entity="Company",
                    attribute="CEO",
                    value="Priya Mehta",
                    period="2025-04-01",
                    document="update.pdf",
                    page=3,
                    evidence="Priya Mehta appointed CEO after Rohit Sharma resigned, dated 1 April 2025",
                ),
            ],
            self.db_path,
        )
        self.result = link_fact_relationships(db_path=self.db_path)
        self.rels = list_all_relationships(self.db_path)

    def _pairs(self, rel_type: str) -> set[tuple[str, str, str]]:
        found: set[tuple[str, str, str]] = set()
        for row in self.rels:
            if row["relationship_type"] != rel_type:
                continue
            left = (
                row.get("source_canonical_attribute") or row.get("source_attribute"),
                str(row.get("source_value") or ""),
            )
            right = (
                row.get("target_canonical_attribute") or row.get("target_attribute"),
                str(row.get("target_value") or ""),
            )
            found.add((rel_type, *tuple(sorted((f"{left[0]}={left[1]}", f"{right[0]}={right[1]}")))))
        return found

    def test_equivalents_merge_instead_of_corroborating(self) -> None:
        rows = search_facts(db_path=self.db_path)
        fy24_120 = [
            r
            for r in rows
            if r["canonical_attribute"] == "total_revenue"
            and r["period"] == "FY2024"
            and normalized_value_key(r["canonical_value"] or r["value"], r["unit"]) == "120|crore"
        ]
        self.assertEqual(len(fy24_120), 1)
        fact = get_fact(fy24_120[0]["id"], self.db_path)
        self.assertEqual(
            {e["source_document"] for e in fact["evidence"]},
            {"report.pdf", "update.pdf"},
        )
        # merged equivalents no longer appear as CORROBORATES edges
        pairs = self._pairs("CORROBORATES")
        self.assertNotIn(
            ("CORROBORATES", "total_revenue=INR 120 crore", "total_revenue=₹120 crore"), pairs
        )
        # CORROBORATES only survives where facts could not be merged (approx values)
        for row in self.rels:
            if row["relationship_type"] == "CORROBORATES":
                self.assertEqual(row["source_canonical_attribute"], "employee_count")

    def test_expected_contradiction(self) -> None:
        pairs = self._pairs("CONTRADICTS")
        self.assertIn(("CONTRADICTS", "total_revenue=₹110 crore", "total_revenue=₹120 crore"), pairs)
        for row in self.rels:
            if row["relationship_type"] != "CONTRADICTS":
                continue
            attrs = {
                row.get("source_canonical_attribute"),
                row.get("target_canonical_attribute"),
            }
            self.assertEqual(len(attrs), 1)

    def test_expected_reconciliations(self) -> None:
        pairs = self._pairs("RECONCILES")
        self.assertNotIn(("RECONCILES", "employee_count=500", "employee_count=520"), pairs)
        self.assertIn(("RECONCILES", "holder=Priya Mehta", "holder=Rohit Sharma"), pairs)
        for row in self.rels:
            if row["relationship_type"] != "RECONCILES":
                continue
            self.assertTrue(str(row.get("reasoning") or "").strip())
            self.assertTrue(row.get("supporting_fact_ids"))

    def test_employees_corroborate_within_tolerance(self) -> None:
        pairs = self._pairs("CORROBORATES")
        self.assertIn(("CORROBORATES", "employee_count=500", "employee_count=520"), pairs)

    def test_computed_support_for_revenue_sum(self) -> None:
        computed = [row for row in self.rels if row["relationship_type"] == "COMPUTED_SUPPORT"]
        self.assertTrue(computed)
        self.assertTrue(any("90" in str(row.get("reasoning")) and "30" in str(row.get("reasoning")) for row in computed))
        self.assertTrue(any(len(row.get("computed_components") or []) >= 2 for row in computed))

    def test_merges_cross_document_equivalents(self) -> None:
        rows = search_facts(db_path=self.db_path)
        totals = [row for row in rows if row["canonical_attribute"] == "total_revenue"]
        # One row per distinct (period, normalized value): the two "120 crore FY2024"
        # totals from report.pdf and update.pdf collapse into a single fact.
        distinct = {
            (r["period"], normalized_value_key(r["canonical_value"] or r["value"], r["unit"]))
            for r in totals
        }
        self.assertEqual(len(totals), len(distinct))
        merged = [
            r
            for r in totals
            if r["period"] == "FY2024"
            and normalized_value_key(r["canonical_value"] or r["value"], r["unit"]) == "120|crore"
        ]
        self.assertEqual(len(merged), 1)
        fact = get_fact(merged[0]["id"], self.db_path)
        self.assertEqual(
            {e["source_document"] for e in fact["evidence"]}, {"report.pdf", "update.pdf"}
        )
        # distinct values / periods are still kept apart
        self.assertTrue(
            any(
                normalized_value_key(r["canonical_value"] or r["value"], r["unit"]) == "110|crore"
                for r in totals
            )
        )
        self.assertGreaterEqual(
            len([row for row in rows if row["canonical_attribute"] == "holder"]), 2
        )

    def test_part_of_still_created(self) -> None:
        part_of = [row for row in self.rels if row["relationship_type"] == "PART_OF"]
        self.assertGreaterEqual(len(part_of), 2)


if __name__ == "__main__":
    unittest.main()
