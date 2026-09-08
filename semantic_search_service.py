"""Embed canonical facts and retrieve nearest neighbors with FAISS."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from database import (
    get_fact,
    get_fact_embedding,
    list_fact_embeddings,
    next_faiss_id,
    search_facts,
    upsert_fact_embedding,
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_MODEL = "hashing-v1"
EMBED_DIM = 384

_INDEX_CACHE: dict[str, Any] = {}
_ST_MODEL = None


def embedding_text(fact: dict[str, Any]) -> str:
    """Format used for every fact embedding."""
    entity = str(fact.get("canonical_entity") or fact.get("entity") or "").strip()
    attribute = str(
        fact.get("canonical_attribute") or fact.get("attribute") or ""
    ).strip()
    value = str(fact.get("canonical_value") or fact.get("value") or "").strip()
    period = str(fact.get("period") or "").strip()
    return " ".join(part for part in (entity, attribute, value, period) if part)


def embedding_text_hash(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


def embedding_backend() -> str:
    forced = (os.getenv("FACT_EMBEDDING_BACKEND") or "").strip().lower()
    if forced in {"hash", "hashing", HASH_MODEL}:
        return "hash"
    if forced in {"st", "sentence-transformers", "minilm"}:
        return "st"
    try:
        import sentence_transformers  # noqa: F401

        return "st"
    except ImportError:
        return "hash"


def embedding_model_name() -> str:
    if embedding_backend() == "st":
        return os.getenv("FACT_EMBEDDING_MODEL") or DEFAULT_MODEL
    return HASH_MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    if embedding_backend() == "st":
        return _sentence_transformer_embed(texts)
    return _hash_embed(texts)


def index_fact_ids(fact_ids: list[str], db_path: Path | str | None = None) -> int:
    """Encode missing/stale facts and refresh the FAISS index from SQLite blobs."""
    updated = 0
    model = embedding_model_name()
    pending_ids: list[str] = []
    pending_texts: list[str] = []
    next_id = next_faiss_id(db_path)

    for fact_id in fact_ids:
        fact = get_fact(fact_id, db_path)
        if fact is None:
            continue
        text = embedding_text(fact)
        digest = embedding_text_hash(text, model)
        existing = get_fact_embedding(fact_id, db_path)
        if (
            existing
            and existing["text_hash"] == digest
            and existing["model"] == model
            and existing["embedding"]
        ):
            continue
        pending_ids.append(fact_id)
        pending_texts.append(text)

    if pending_texts:
        vectors = embed_texts(pending_texts)
        for offset, fact_id in enumerate(pending_ids):
            fact = get_fact(fact_id, db_path)
            if fact is None:
                continue
            stored = get_fact_embedding(fact_id, db_path)
            faiss_id = int(stored["faiss_id"]) if stored else next_id
            if not stored:
                next_id += 1
            vector = np.asarray(vectors[offset], dtype=np.float32)
            text = embedding_text(fact)
            upsert_fact_embedding(
                fact_id=fact_id,
                faiss_id=faiss_id,
                model=model,
                dim=int(vector.shape[0]),
                text_hash=embedding_text_hash(text, model),
                embedding=_vector_to_blob(vector),
                db_path=db_path,
            )
            updated += 1

    rebuild_faiss_index(db_path)
    return updated


def index_all_facts(db_path: Path | str | None = None) -> int:
    rows = search_facts(db_path=db_path)
    return index_fact_ids([row["id"] for row in rows], db_path)


def similar_facts(
    fact_id: str,
    k: int = 5,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return the k nearest facts to `fact_id` by cosine similarity."""
    index_fact_ids([fact_id], db_path)
    index, id_map = rebuild_faiss_index(db_path)
    source = get_fact_embedding(fact_id, db_path)
    if source is None or index is None or index.ntotal == 0:
        return []

    vector = _blob_to_vector(source["embedding"], int(source["dim"]))
    query = vector.reshape(1, -1)
    neighbor_count = min(max(int(k) + 1, 1), index.ntotal)
    scores, ids = index.search(query, neighbor_count)
    neighbors: list[dict[str, Any]] = []
    for score, faiss_id in zip(scores[0], ids[0]):
        if faiss_id < 0:
            continue
        other_id = id_map.get(int(faiss_id))
        if not other_id or other_id == fact_id:
            continue
        fact = get_fact(other_id, db_path)
        if fact is None:
            continue
        neighbors.append(
            {
                **fact,
                "similarity": float(score),
                "embedding_text": embedding_text(fact),
            }
        )
        if len(neighbors) >= k:
            break
    return neighbors


def rebuild_faiss_index(db_path: Path | str | None = None):
    """Rebuild an in-memory FAISS index from packed float32 blobs in SQLite."""
    import faiss

    rows = [
        row
        for row in list_fact_embeddings(db_path)
        if row.get("embedding") and row.get("model") == embedding_model_name()
    ]
    cache_key = _cache_key(db_path)
    if not rows:
        dim = EMBED_DIM
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        _INDEX_CACHE[cache_key] = (index, {})
        return index, {}

    dim = int(rows[0]["dim"] or EMBED_DIM)
    matrix = np.vstack(
        [_blob_to_vector(row["embedding"], dim) for row in rows]
    ).astype(np.float32)
    faiss.normalize_L2(matrix)
    ids = np.array([int(row["faiss_id"]) for row in rows], dtype=np.int64)
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    index.add_with_ids(matrix, ids)
    id_map = {int(row["faiss_id"]): str(row["fact_id"]) for row in rows}
    _INDEX_CACHE[cache_key] = (index, id_map)
    return index, id_map


def reset_faiss_index(db_path: Path | str | None = None) -> None:
    _INDEX_CACHE.pop(_cache_key(db_path), None)


def _sentence_transformer_embed(texts: list[str]) -> np.ndarray:
    global _ST_MODEL
    from sentence_transformers import SentenceTransformer

    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(embedding_model_name())
    vectors = _ST_MODEL.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def _hash_embed(texts: list[str], dim: int = EMBED_DIM) -> np.ndarray:
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            first = int.from_bytes(digest[:4], "little")
            second = int.from_bytes(digest[4:8], "little")
            matrix[row, first % dim] += 1.0
            matrix[row, second % dim] += 0.5
        norm = float(np.linalg.norm(matrix[row]))
        if norm:
            matrix[row] /= norm
    return matrix


def _vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _blob_to_vector(blob: bytes, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.size != dim:
        padded = np.zeros(dim, dtype=np.float32)
        padded[: min(dim, vector.size)] = vector[:dim]
        vector = padded
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return np.asarray(vector, dtype=np.float32)


def _cache_key(db_path: Path | str | None) -> str:
    return str(Path(db_path) if db_path else "default")
