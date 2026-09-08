"""FAISS nearest-neighbor search over canonical fact embeddings."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ["FACT_EMBEDDING_BACKEND"] = "hash"

from database import get_fact_embedding, init_db, insert_facts, search_facts
from semantic_search_service import embedding_text, index_all_facts, similar_facts


def _fact(
    entity: str,
    attribute: str,
    value: str,
    *,
    period: str | None = None,
    page: int = 1,
) -> dict:
    return {
        "entity": entity,
        "attribute": attribute,
        "value": value,
        "unit": None,
        "period": period,
        "confidence": 0.9,
        "source_document": "demo.pdf",
        "page_number": page,
        "evidence_text": f"{entity} {attribute} {value}",
        "evidence_type": "DIRECT",
    }


class SemanticSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["FACT_EMBEDDING_BACKEND"] = "hash"
        self.db_path = Path(tempfile.mkdtemp()) / "facts.db"
        init_db(self.db_path)
        insert_facts(
            [
                _fact("Revenue", "fiscal_year", "₹120 crore", period="FY2024", page=1),
                _fact("Revenue", "Product Revenue", "₹90 crore", period="FY2024", page=1),
                _fact("Revenue", "Services Revenue", "₹30 crore", period="FY2024", page=1),
                _fact("Acme", "registered_address", "221B Baker St., London", page=2),
            ],
            self.db_path,
        )

    def test_embedding_text_uses_canonical_fields(self) -> None:
        rows = {row["canonical_attribute"]: row for row in search_facts(db_path=self.db_path)}
        total = rows["total_revenue"]
        self.assertEqual(
            embedding_text(total),
            "Revenue total_revenue 120 INR crore FY2024",
        )
        stored = get_fact_embedding(total["id"], self.db_path)
        self.assertIsNotNone(stored)
        self.assertEqual(stored["dim"], 384)
        self.assertTrue(stored["embedding"])

    def test_similar_facts_rank_revenue_above_address(self) -> None:
        index_all_facts(self.db_path)
        rows = {row["canonical_attribute"]: row for row in search_facts(db_path=self.db_path)}
        neighbors = similar_facts(rows["total_revenue"]["id"], k=3, db_path=self.db_path)
        self.assertGreaterEqual(len(neighbors), 2)
        neighbor_attrs = [row["canonical_attribute"] for row in neighbors]
        self.assertIn("product_revenue", neighbor_attrs)
        self.assertIn("services_revenue", neighbor_attrs)
        self.assertNotEqual(neighbor_attrs[0], "registered_address")
        for row in neighbors:
            self.assertNotEqual(row["id"], rows["total_revenue"]["id"])
            self.assertGreaterEqual(row["similarity"], -1.0)


if __name__ == "__main__":
    unittest.main()
