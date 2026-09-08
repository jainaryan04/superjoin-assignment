"""Canonicalization and revenue hierarchy tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from canonicalization_service import canonicalize_fact
from database import (
    deduplicate_equivalent_facts,
    find_self_links,
    get_connection,
    get_fact,
    init_db,
    insert_facts,
    insert_relationship,
    list_all_relationships,
    migrate_canonical_columns,
    search_facts,
)
from fact_integrity import list_fact_integrity
from provenance_service import link_fact_relationships, rebuild_relationships, validate_relationship


PAGE_TEXT = (
    "Revenue FY2024 = ₹120 crore\n"
    "Product Revenue = ₹90 crore\n"
    "Services Revenue = ₹30 crore"
)


def _part_of_count(fact: dict) -> int:
    """PART_OF links touching a fact. COMPUTED_SUPPORT is a separate, expected type."""
    return len([r for r in fact.get("relationships") or [] if r["relationship_type"] == "PART_OF"])


def _sample_facts() -> list[dict]:
    return [
        {
            "entity": "Revenue",
            "attribute": "fiscal_year",
            "value": "₹120 crore",
            "unit": None,
            "period": "FY2024",
            "confidence": 0.9,
            "source_document": "demo.pdf",
            "page_number": 1,
            "evidence_text": PAGE_TEXT,
            "evidence_type": "DIRECT",
        },
        {
            "entity": "Revenue",
            "attribute": "Product Revenue",
            "value": "₹90 crore",
            "unit": None,
            "period": "FY2024",
            "confidence": 0.88,
            "source_document": "demo.pdf",
            "page_number": 1,
            "evidence_text": PAGE_TEXT,
            "evidence_type": "DIRECT",
        },
        {
            "entity": "Revenue",
            "attribute": "Services Revenue",
            "value": "₹30 crore",
            "unit": None,
            "period": "FY2024",
            "confidence": 0.87,
            "source_document": "demo.pdf",
            "page_number": 1,
            "evidence_text": PAGE_TEXT,
            "evidence_type": "DIRECT",
        },
    ]


class CanonicalizationTests(unittest.TestCase):
    def test_fiscal_year_is_period_not_attribute(self) -> None:
        fact = canonicalize_fact(
            {
                "entity": "Revenue",
                "attribute": "fiscal_year",
                "value": "₹120 crore",
                "period": "FY2024",
                "evidence_text": "Revenue FY2024 = ₹120 crore",
            }
        )
        self.assertEqual(fact["raw_attribute"], "fiscal_year")
        self.assertEqual(fact["canonical_attribute"], "total_revenue")
        self.assertEqual(fact["period"], "FY2024")
        self.assertEqual(fact["canonical_value"], "120 INR crore")
        self.assertEqual(fact["value"], "₹120 crore")
        self.assertEqual(fact["unit"], "crore")

    def test_normalizes_currency_dates_and_addresses(self) -> None:
        money = canonicalize_fact(
            {
                "entity": "Acme",
                "attribute": "total_revenue",
                "value": "Rs. 1,200 million",
                "period": "FY2024",
            }
        )
        date = canonicalize_fact(
            {
                "entity": "Acme",
                "attribute": "incorporation_date",
                "value": "15 March 2024",
            }
        )
        address = canonicalize_fact(
            {
                "entity": "Acme",
                "attribute": "registered_address",
                "value": "221B Baker St., London",
            }
        )
        self.assertEqual(money["canonical_value"], "1200 INR million")
        self.assertEqual(money["value"], "Rs. 1,200 million")
        self.assertEqual(date["canonical_value"], "2024-03-15")
        self.assertEqual(date["value"], "15 March 2024")
        self.assertEqual(address["canonical_value"], "221b baker street, london")
        self.assertEqual(address["value"], "221B Baker St., London")

    def test_employee_and_ceo_entity_resolution(self) -> None:
        headcount = canonicalize_fact(
            {"entity": "Company", "attribute": "Headcount", "value": "500"}
        )
        employees = canonicalize_fact(
            {"entity": "Employees", "attribute": "Employee Count", "value": "520"}
        )
        ceo = canonicalize_fact(
            {"entity": "Company", "attribute": "CEO", "value": "Rohit Sharma"}
        )
        incoming = canonicalize_fact(
            {
                "entity": "Priya Mehta | position=CEO",
                "attribute": "role",
                "value": "CEO",
            }
        )
        approx = canonicalize_fact(
            {"entity": "Company", "attribute": "Employee Count", "value": "Approximately 500"}
        )
        software = canonicalize_fact(
            {
                "entity": "Company",
                "attribute": "Software Products",
                "value": "₹90 crore",
                "period": "FY2024",
            }
        )
        self.assertEqual(headcount["canonical_attribute"], "employee_count")
        self.assertEqual(employees["canonical_attribute"], "employee_count")
        self.assertEqual(ceo["canonical_entity"], "CEO")
        self.assertEqual(ceo["canonical_attribute"], "holder")
        self.assertEqual(ceo["value"], "Rohit Sharma")
        self.assertEqual(incoming["canonical_attribute"], "holder")
        self.assertEqual(incoming["value"], "Priya Mehta")
        from canonicalization_service import normalized_value_key

        self.assertEqual(
            normalized_value_key(approx["value"], approx.get("unit")),
            normalized_value_key("500"),
        )
        self.assertEqual(software["canonical_attribute"], "product_revenue")

    def test_page_context_does_not_create_synthetic_services_total(self) -> None:
        fact = canonicalize_fact(
            {
                "entity": "Revenue",
                "attribute": "fiscal_year",
                "value": "₹120 crore",
                "period": "FY2024",
                "evidence_text": PAGE_TEXT,
            }
        )
        self.assertEqual(fact["canonical_attribute"], "total_revenue")
        self.assertEqual(fact["value"], "₹120 crore")
        self.assertNotEqual(fact["canonical_attribute"], "services_revenue")

    def test_named_revenue_lines(self) -> None:
        product = canonicalize_fact(
            {
                "entity": "Revenue",
                "attribute": "Product Revenue",
                "value": "₹90 crore",
                "period": "FY2024",
            }
        )
        services = canonicalize_fact(
            {
                "entity": "Revenue",
                "attribute": "Services Revenue",
                "value": "₹30 crore",
                "period": "FY2024",
            }
        )
        total = canonicalize_fact(
            {
                "entity": "Revenue",
                "attribute": "Revenue FY2024",
                "value": "₹120 crore",
                "period": "FY2024",
            }
        )
        self.assertEqual(product["canonical_attribute"], "product_revenue")
        self.assertEqual(services["canonical_attribute"], "services_revenue")
        self.assertEqual(total["canonical_attribute"], "total_revenue")
        self.assertEqual(total["period"], "FY2024")


class RevenueHierarchyDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(tempfile.mkdtemp()) / "facts.db"
        init_db(self.db_path)
        insert_facts(_sample_facts(), self.db_path)

    def test_canonical_facts_and_part_of_links(self) -> None:
        rows = search_facts(db_path=self.db_path)
        by_attr = {row["canonical_attribute"]: row for row in rows}
        self.assertEqual(set(by_attr), {"total_revenue", "product_revenue", "services_revenue"})
        self.assertEqual(by_attr["total_revenue"]["raw_attribute"], "fiscal_year")
        self.assertEqual(by_attr["product_revenue"]["raw_attribute"], "Product Revenue")
        self.assertEqual(by_attr["services_revenue"]["raw_attribute"], "Services Revenue")

        result = link_fact_relationships("demo.pdf", db_path=self.db_path)
        rels = list_all_relationships(self.db_path)
        part_of = [row for row in rels if row["relationship_type"] == "PART_OF"]
        contradicts = [row for row in rels if row["relationship_type"] == "CONTRADICTS"]
        self.assertEqual(len(part_of), 2, part_of)
        self.assertEqual(result["relationships_added"], len(rels))
        self.assertLessEqual(
            {row["relationship_type"] for row in rels}, {"PART_OF", "COMPUTED_SUPPORT"}
        )
        self.assertEqual(contradicts, [])

        pairs = {
            (row["source_canonical_attribute"], row["target_canonical_attribute"])
            for row in part_of
        }
        self.assertEqual(
            pairs,
            {
                ("product_revenue", "total_revenue"),
                ("services_revenue", "total_revenue"),
            },
        )
        sibling_pairs = {
            (row["source_canonical_attribute"], row["target_canonical_attribute"])
            for row in part_of
            if {row["source_canonical_attribute"], row["target_canonical_attribute"]}
            == {"product_revenue", "services_revenue"}
        }
        self.assertEqual(sibling_pairs, set())

        total = get_fact(by_attr["total_revenue"]["id"], self.db_path)
        product = get_fact(by_attr["product_revenue"]["id"], self.db_path)
        services = get_fact(by_attr["services_revenue"]["id"], self.db_path)
        self.assertEqual(_part_of_count(total), 2)
        self.assertEqual(_part_of_count(product), 1)
        self.assertEqual(_part_of_count(services), 1)
        self.assertEqual(total["evidence_count"], 1)
        self.assertEqual(total["evidence"][0]["evidence_text"], PAGE_TEXT)

    def test_exactly_three_immutable_facts_and_two_part_of_links(self) -> None:
        before = {
            (row["canonical_attribute"], row["value"], row["original_value"])
            for row in search_facts(db_path=self.db_path)
        }
        self.assertEqual(len(search_facts(db_path=self.db_path)), 3)
        self.assertEqual(
            before,
            {
                ("total_revenue", "₹120 crore", "₹120 crore"),
                ("product_revenue", "₹90 crore", "₹90 crore"),
                ("services_revenue", "₹30 crore", "₹30 crore"),
            },
        )
        self.assertNotIn(("services_revenue", "₹120 crore", "₹120 crore"), before)
        self.assertNotIn(("product_revenue", "₹120 crore", "₹120 crore"), before)

        link_fact_relationships("demo.pdf", db_path=self.db_path)
        after = {
            (row["canonical_attribute"], row["value"], row["original_value"])
            for row in search_facts(db_path=self.db_path)
        }
        self.assertEqual(before, after)
        self.assertEqual(len(search_facts(db_path=self.db_path)), 3)

        rels = list_all_relationships(self.db_path)
        part_of = [row for row in rels if row["relationship_type"] == "PART_OF"]
        self.assertEqual(len(part_of), 2)
        self.assertLessEqual(
            {row["relationship_type"] for row in rels}, {"PART_OF", "COMPUTED_SUPPORT"}
        )
        pairs = {
            (row["source_canonical_attribute"], row["target_canonical_attribute"])
            for row in part_of
        }
        self.assertEqual(
            pairs,
            {("product_revenue", "total_revenue"), ("services_revenue", "total_revenue")},
        )
        for row in list_fact_integrity(str(self.db_path)):
            self.assertEqual(row["integrity_status"], "PASS", row)

        by_attr = {row["canonical_attribute"]: row for row in search_facts(db_path=self.db_path)}
        product = by_attr["product_revenue"]
        services = by_attr["services_revenue"]
        total = by_attr["total_revenue"]
        sibling = validate_relationship(services, product, "PART_OF", cohort=[product, services, total])
        self.assertFalse(sibling["ok"])
        self_link = validate_relationship(services, services, "PART_OF", cohort=[services])
        self.assertFalse(self_link["ok"])
        child_parent = validate_relationship(product, services, "PART_OF", cohort=[product, services, total])
        self.assertFalse(child_parent["ok"])

    def test_no_duplicate_or_self_relationships(self) -> None:
        first = link_fact_relationships("demo.pdf", db_path=self.db_path)
        second = link_fact_relationships("demo.pdf", db_path=self.db_path)
        rels = list_all_relationships(self.db_path)
        part_of = [row for row in rels if row["relationship_type"] == "PART_OF"]
        self.assertEqual(len(part_of), 2)
        self.assertEqual(first["final_relationships_stored"], len(rels))
        self.assertEqual(second["final_relationships_stored"], len(rels))
        self.assertGreaterEqual(second["duplicate_relationships_removed"], 1)
        self.assertEqual(second["self_relationships_removed"], 0)
        types = {row["relationship_type"] for row in rels}
        self.assertLessEqual(types, {"PART_OF", "COMPUTED_SUPPORT"})
        self.assertNotIn("CORROBORATES", types)
        keys = {
            (row["source_fact_id"], row["target_fact_id"], row["relationship_type"])
            for row in part_of
        }
        self.assertEqual(len(keys), 2)
        self.assertEqual(find_self_links(self.db_path), [])

        by_attr = {row["canonical_attribute"]: row for row in search_facts(db_path=self.db_path)}
        total = by_attr["total_revenue"]
        self.assertFalse(
            insert_relationship(
                {
                    "source_fact_id": total["id"],
                    "target_fact_id": total["id"],
                    "relationship_type": "CORROBORATES",
                    "confidence": 0.9,
                    "reasoning": "self corroboration",
                },
                self.db_path,
            )
        )
        product = by_attr["product_revenue"]
        self.assertFalse(
            insert_relationship(
                {
                    "source_fact_id": product["id"],
                    "target_fact_id": total["id"],
                    "relationship_type": "PART_OF",
                    "confidence": 0.9,
                    "reasoning": "duplicate",
                },
                self.db_path,
            )
        )
        self.assertEqual(
            len([r for r in list_all_relationships(self.db_path) if r["relationship_type"] == "PART_OF"]),
            2,
        )
        for rel_type in ("CORROBORATES", "CONTRADICTS", "PART_OF", "RECONCILES", "SUPPORTS"):
            check = validate_relationship(total, total, rel_type, cohort=[total])
            self.assertFalse(check["ok"], rel_type)

    def test_cleanup_removes_self_and_duplicate_rows(self) -> None:
        link_fact_relationships("demo.pdf", db_path=self.db_path)
        by_attr = {row["canonical_attribute"]: row for row in search_facts(db_path=self.db_path)}
        total = by_attr["total_revenue"]
        product = by_attr["product_revenue"]
        with get_connection(self.db_path) as conn:
            conn.execute("DROP INDEX IF EXISTS idx_rel_unique")
            conn.execute(
                """
                INSERT INTO fact_relationships (
                    id, source_fact_id, target_fact_id, relationship_type, confidence, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("self-1", total["id"], total["id"], "CORROBORATES", 0.5, "self"),
            )
            conn.execute(
                """
                INSERT INTO fact_relationships (
                    id, source_fact_id, target_fact_id, relationship_type, confidence, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("dup-1", product["id"], total["id"], "PART_OF", 0.4, "dup"),
            )
            conn.commit()
        rebuilt = rebuild_relationships(db_path=self.db_path)
        rels = list_all_relationships(self.db_path)
        part_of = [row for row in rels if row["relationship_type"] == "PART_OF"]
        self.assertEqual(rebuilt["final_relationships_stored"], len(rels))
        self.assertEqual(len(part_of), 2)
        self.assertLessEqual(
            {row["relationship_type"] for row in rels}, {"PART_OF", "COMPUTED_SUPPORT"}
        )
        self.assertEqual(find_self_links(self.db_path), [])
        self.assertNotIn("CORROBORATES", {row["relationship_type"] for row in rels})

    def test_merges_periodless_duplicate_total_revenue(self) -> None:
        insert_facts(
            [
                {
                    "entity": "Revenue",
                    "attribute": "fiscal_year",
                    "value": "₹120 crore",
                    "unit": None,
                    "period": None,
                    "confidence": 0.8,
                    "source_document": "demo.pdf",
                    "page_number": 2,
                    "evidence_text": "Total revenue stood at ₹120 crore",
                    "evidence_type": "DIRECT",
                }
            ],
            self.db_path,
        )
        rows = search_facts(db_path=self.db_path)
        self.assertEqual(len(rows), 3)
        by_attr = {row["canonical_attribute"]: row for row in rows}
        total = get_fact(by_attr["total_revenue"]["id"], self.db_path)
        self.assertEqual(total["period"], "FY2024")
        self.assertEqual(total["evidence_count"], 2)
        snippets = {item["evidence_text"] for item in total["evidence"]}
        self.assertIn(PAGE_TEXT, snippets)
        self.assertIn("Total revenue stood at ₹120 crore", snippets)

        result = link_fact_relationships("demo.pdf", db_path=self.db_path)
        rels = list_all_relationships(self.db_path)
        part_of = [row for row in rels if row["relationship_type"] == "PART_OF"]
        self.assertEqual(len(part_of), 2, rels)
        self.assertEqual(result["final_relationships_stored"], len(rels))
        self.assertLessEqual(
            {row["relationship_type"] for row in rels}, {"PART_OF", "COMPUTED_SUPPORT"}
        )
        self.assertEqual(find_self_links(self.db_path), [])
        pairs = {
            (row["source_canonical_attribute"], row["target_canonical_attribute"])
            for row in part_of
        }
        self.assertEqual(
            pairs,
            {("product_revenue", "total_revenue"), ("services_revenue", "total_revenue")},
        )

    def test_deduplicate_existing_periodless_total(self) -> None:
        by_attr = {row["canonical_attribute"]: row for row in search_facts(db_path=self.db_path)}
        keeper = by_attr["total_revenue"]
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO facts (
                    id, identity_key, entity, attribute, value, unit, period, confidence,
                    canonical_attribute, canonical_entity, canonical_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "dup-total",
                    "dup-total-key",
                    "Revenue",
                    "total_revenue",
                    "₹120 crore",
                    None,
                    None,
                    0.7,
                    "total_revenue",
                    "Revenue",
                    "₹120 crore",
                ),
            )
            conn.execute(
                """
                INSERT INTO fact_evidence (
                    id, fact_id, source_document, page_number, evidence_text,
                    evidence_type, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "ev-dup",
                    "dup-total",
                    "demo.pdf",
                    2,
                    "Revenue is ₹120 crore",
                    "DIRECT",
                    0.7,
                ),
            )
            conn.commit()
        stats = deduplicate_equivalent_facts(db_path=self.db_path)
        self.assertEqual(stats["facts_remaining"], 3)
        self.assertGreaterEqual(stats["facts_merged"], 1)
        total = get_fact(keeper["id"], self.db_path)
        self.assertIsNotNone(total)
        self.assertEqual(total["period"], "FY2024")
        self.assertGreaterEqual(total["evidence_count"], 2)

    def test_migration_backfills_existing_rows(self) -> None:
        db_path = Path(tempfile.mkdtemp()) / "legacy.db"
        init_db(db_path)
        fact_id = "legacy-total"
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO facts (
                    id, identity_key, entity, attribute, value, unit, period, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    "old-key",
                    "Revenue",
                    "fiscal_year",
                    "₹120 crore",
                    None,
                    "FY2024",
                    0.9,
                ),
            )
            conn.execute(
                """
                INSERT INTO fact_evidence (
                    id, fact_id, source_document, page_number, evidence_text,
                    evidence_type, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "ev-1",
                    fact_id,
                    "demo.pdf",
                    1,
                    "Revenue FY2024 = ₹120 crore",
                    "DIRECT",
                    0.9,
                ),
            )
            conn.commit()
        migrate_canonical_columns(db_path=db_path)
        fact = get_fact(fact_id, db_path)
        self.assertEqual(fact["raw_attribute"], "fiscal_year")
        self.assertEqual(fact["canonical_attribute"], "total_revenue")
        self.assertEqual(fact["attribute"], "total_revenue")
        self.assertEqual(fact["period"], "FY2024")


if __name__ == "__main__":
    unittest.main()
