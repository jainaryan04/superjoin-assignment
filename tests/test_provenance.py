"""Automated validation: evidence is snippets; relationships are fact links."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from database import (
    find_duplicate_relationships,
    find_self_links,
    get_fact,
    init_db,
    insert_facts,
    insert_relationship,
    list_all_relationships,
    search_facts,
    validate_provenance_counts,
    validate_relationships,
)
from fact_extractor import LLMClient
from provenance_service import (
    evidence_tree_text,
    link_fact_relationships,
    parse_amount,
    validate_relationship,
)
from relationship_rules import is_hierarchical_child, is_sibling


def _by_canonical(rows: list[dict]) -> dict[str, dict]:
    return {row["canonical_attribute"]: row for row in rows}


def _seed_revenue_document(db_path: Path) -> None:
    insert_facts(
        [
            {
                "entity": "Company",
                "attribute": "Revenue FY2024",
                "value": "₹120 crore",
                "unit": None,
                "period": "FY2024",
                "confidence": 0.9,
                "source_document": "demo.pdf",
                "page_number": 1,
                "evidence_text": "Revenue FY2024 = ₹120 crore",
                "evidence_type": "DIRECT",
            },
            {
                "entity": "Company",
                "attribute": "Product Revenue",
                "value": "₹90 crore",
                "unit": None,
                "period": "FY2024",
                "confidence": 0.88,
                "source_document": "demo.pdf",
                "page_number": 2,
                "evidence_text": "Product Revenue = ₹90 crore",
                "evidence_type": "DIRECT",
            },
            {
                "entity": "Company",
                "attribute": "Services Revenue",
                "value": "₹30 crore",
                "unit": None,
                "period": "FY2024",
                "confidence": 0.87,
                "source_document": "demo.pdf",
                "page_number": 2,
                "evidence_text": "Services Revenue = ₹30 crore",
                "evidence_type": "DIRECT",
            },
        ],
        db_path,
    )


class ScriptedLLM(LLMClient):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        start = user_prompt.find("{")
        payload = json.loads(user_prompt[start:]) if start >= 0 else {}
        links = []
        for candidate in payload.get("candidates", []):
            attr = str(
                candidate.get("canonical_attribute")
                or candidate.get("attribute")
                or candidate.get("statement")
                or ""
            ).lower()
            if "product_revenue" in attr or "services_revenue" in attr or "product revenue" in attr or "services revenue" in attr:
                links.append(
                    {
                        "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
                        "relationship_type": "PART_OF",
                        "direction": "candidate_to_parent",
                        "confidence": 0.91,
                        "reasoning": "component of total revenue",
                    }
                )
        return json.dumps({"links": links})


class ProvenanceSeparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(tempfile.mkdtemp()) / "facts.db"
        init_db(self.db_path)

    def test_isolated_fact_has_single_direct_evidence(self) -> None:
        insert_facts(
            [
                {
                    "entity": "Company",
                    "attribute": "CEO",
                    "value": "Ada Lovelace",
                    "confidence": 0.8,
                    "source_document": "bio.pdf",
                    "page_number": 1,
                    "evidence_text": "The CEO is Ada Lovelace.",
                }
            ],
            self.db_path,
        )
        link_fact_relationships("bio.pdf", db_path=self.db_path)
        rows = search_facts(db_path=self.db_path)
        self.assertEqual(rows[0]["evidence_count"], 1)
        self.assertEqual(rows[0]["relationship_count"], 0)

    def test_revenue_parts_are_relationships_not_evidence(self) -> None:
        _seed_revenue_document(self.db_path)
        link_fact_relationships("demo.pdf", db_path=self.db_path)

        by_attr = _by_canonical(search_facts(db_path=self.db_path))
        self.assertEqual(by_attr["total_revenue"]["evidence_count"], 1)
        self.assertEqual(by_attr["product_revenue"]["evidence_count"], 1)
        self.assertEqual(by_attr["services_revenue"]["evidence_count"], 1)

        revenue = get_fact(by_attr["total_revenue"]["id"], self.db_path)
        product = get_fact(by_attr["product_revenue"]["id"], self.db_path)
        services = get_fact(by_attr["services_revenue"]["id"], self.db_path)

        self.assertEqual(revenue["evidence"][0]["evidence_text"], "Revenue FY2024 = ₹120 crore")
        self.assertEqual(product["evidence"][0]["evidence_text"], "Product Revenue = ₹90 crore")
        self.assertEqual(services["evidence"][0]["evidence_text"], "Services Revenue = ₹30 crore")

        pairs = {
            (rel["source_statement"], rel["relationship_type"], rel["target_statement"])
            for rel in revenue["relationships"]
        }
        self.assertIn(
            ("product_revenue = ₹90 crore", "PART_OF", "total_revenue = ₹120 crore"),
            pairs,
        )
        self.assertIn(
            ("services_revenue = ₹30 crore", "PART_OF", "total_revenue = ₹120 crore"),
            pairs,
        )
        self.assertEqual(revenue["relationship_count"], 2)
        self.assertEqual(product["relationship_count"], 1)
        self.assertEqual(services["relationship_count"], 1)

        tree = evidence_tree_text(revenue)
        self.assertIn("DIRECT", tree)
        self.assertIn("Revenue FY2024 = ₹120 crore", tree)
        self.assertNotIn("Product Revenue", tree)

        self.assertEqual(validate_provenance_counts(self.db_path), [])
        self.assertEqual(find_self_links(self.db_path), [])
        self.assertEqual(find_duplicate_relationships(self.db_path), [])
        report = validate_relationships(self.db_path)
        self.assertTrue(report["ok"])

    def test_no_self_links_or_duplicate_part_of(self) -> None:
        _seed_revenue_document(self.db_path)
        link_fact_relationships("demo.pdf", db_path=self.db_path)
        link_fact_relationships("demo.pdf", db_path=self.db_path)

        self.assertEqual(find_self_links(self.db_path), [])
        self.assertEqual(find_duplicate_relationships(self.db_path), [])

        by_attr = _by_canonical(search_facts(db_path=self.db_path))
        revenue = get_fact(by_attr["total_revenue"]["id"], self.db_path)
        product = get_fact(by_attr["product_revenue"]["id"], self.db_path)
        part_of = [
            rel
            for rel in revenue["relationships"]
            if rel["relationship_type"] == "PART_OF"
        ]
        self.assertEqual(len(part_of), 2)
        keys = {
            (rel["source_fact_id"], rel["target_fact_id"], rel["relationship_type"])
            for rel in part_of
        }
        self.assertEqual(len(keys), 2)
        for rel in part_of:
            self.assertNotEqual(rel["source_fact_id"], rel["target_fact_id"])
            self.assertIn("->", rel["arrow"])
            self.assertEqual(rel["direction_label"], "Child -> Parent")

        rejected = insert_relationship(
            {
                "source_fact_id": product["id"],
                "target_fact_id": product["id"],
                "relationship_type": "PART_OF",
                "confidence": 0.5,
                "reasoning": "should be rejected",
            },
            self.db_path,
        )
        self.assertFalse(rejected)
        self.assertEqual(find_self_links(self.db_path), [])

        for row in search_facts(db_path=self.db_path):
            detail = get_fact(row["id"], self.db_path)
            self.assertEqual(row["relationship_count"], len(detail["relationships"]))
            self.assertEqual(row["evidence_count"], len(detail["evidence"]))
        self.assertEqual(validate_provenance_counts(self.db_path), [])

    def test_llm_classifier_writes_relationships_only(self) -> None:
        _seed_revenue_document(self.db_path)
        link_fact_relationships("demo.pdf", llm=ScriptedLLM(), db_path=self.db_path)
        revenue = next(
            row for row in search_facts(db_path=self.db_path) if row["canonical_attribute"] == "total_revenue"
        )
        self.assertEqual(revenue["evidence_count"], 1)
        self.assertGreaterEqual(revenue["relationship_count"], 2)

    def test_sample_dataset_creates_exactly_two_part_of_links(self) -> None:
        _seed_revenue_document(self.db_path)
        result = link_fact_relationships("demo.pdf", db_path=self.db_path)
        rels = list_all_relationships(self.db_path)
        part_of = [row for row in rels if row["relationship_type"] == "PART_OF"]
        self.assertEqual(len(part_of), 2, part_of)
        self.assertEqual(result["relationships_added"], 2)
        types = {row["relationship_type"] for row in rels}
        self.assertEqual(types, {"PART_OF"})
        pairs = {
            (row["source_statement"], row["target_statement"])
            for row in part_of
        }
        self.assertEqual(
            pairs,
            {
                ("product_revenue = ₹90 crore", "total_revenue = ₹120 crore"),
                ("services_revenue = ₹30 crore", "total_revenue = ₹120 crore"),
            },
        )

    def test_rejects_cross_attribute_contradiction_and_sibling_part_of(self) -> None:
        _seed_revenue_document(self.db_path)
        link_fact_relationships("demo.pdf", llm=EvilLLM(), db_path=self.db_path)

        rows = search_facts(db_path=self.db_path)
        by_attr = _by_canonical(rows)
        revenue = get_fact(by_attr["total_revenue"]["id"], self.db_path)
        product = get_fact(by_attr["product_revenue"]["id"], self.db_path)
        services = get_fact(by_attr["services_revenue"]["id"], self.db_path)

        types = {rel["relationship_type"] for rel in list_all_relationships(self.db_path)}
        self.assertNotIn("CONTRADICTS", types)

        pairs = {
            (rel["source_statement"], rel["relationship_type"], rel["target_statement"])
            for rel in list_all_relationships(self.db_path)
        }
        self.assertIn(
            ("product_revenue = ₹90 crore", "PART_OF", "total_revenue = ₹120 crore"),
            pairs,
        )
        self.assertIn(
            ("services_revenue = ₹30 crore", "PART_OF", "total_revenue = ₹120 crore"),
            pairs,
        )
        self.assertNotIn(
            ("services_revenue = ₹30 crore", "PART_OF", "product_revenue = ₹90 crore"),
            pairs,
        )
        self.assertNotIn(
            ("total_revenue = ₹120 crore", "CONTRADICTS", "product_revenue = ₹90 crore"),
            pairs,
        )
        self.assertEqual(revenue["relationship_count"], 2)
        self.assertEqual(product["relationship_count"], 1)
        self.assertEqual(services["relationship_count"], 1)

        self.assertTrue(is_hierarchical_child(product, revenue))
        self.assertTrue(is_sibling(product, services))
        contradiction = validate_relationship(revenue, product, "CONTRADICTS", cohort=rows)
        self.assertFalse(contradiction["ok"])
        self.assertIn("Different attributes", contradiction["reason"])
        sibling = validate_relationship(services, product, "PART_OF", cohort=[revenue, product, services])
        self.assertFalse(sibling["ok"])


class EvilLLM(LLMClient):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        start = user_prompt.find("{")
        payload = json.loads(user_prompt[start:]) if start >= 0 else {}
        parent = payload.get("parent") or {}
        links = []
        for candidate in payload.get("candidates", []):
            statement = candidate.get("statement") or ""
            parent_stmt = parent.get("statement") or ""
            attr = str(
                candidate.get("canonical_attribute")
                or candidate.get("attribute")
                or statement
            ).lower()
            parent_attr = str(
                parent.get("canonical_attribute") or parent.get("attribute") or parent_stmt
            ).lower()
            if "product_revenue" in attr and "total_revenue" in parent_attr:
                links.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "relationship_type": "CONTRADICTS",
                        "confidence": 0.99,
                        "reasoning": "incorrect contradiction",
                    }
                )
            if "services_revenue" in attr and "product_revenue" in parent_attr:
                links.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "relationship_type": "PART_OF",
                        "confidence": 0.99,
                        "reasoning": "incorrect sibling part-of",
                    }
                )
            if "product_revenue" in attr and "services_revenue" in parent_attr:
                links.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "relationship_type": "PART_OF",
                        "confidence": 0.99,
                        "reasoning": "incorrect sibling part-of",
                    }
                )
        return json.dumps({"links": links})


if __name__ == "__main__":
    unittest.main()
