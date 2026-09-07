from __future__ import annotations

import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from pdf_processor import DocumentChunk

SYSTEM_PROMPT = """You extract structured facts from document text.

Return ONLY valid JSON with this shape:
{"facts": [<fact>, ...]}

Each fact must use this generic schema (no domain-specific fields):
{
  "entity": "the subject the fact is about",
  "attribute": "the property or relation name, in snake_case or short phrase",
  "value": "the extracted value as a string",
  "unit": "unit of measure if present, otherwise null",
  "period": "time period or date if present, otherwise null",
  "confidence": number between 0 and 1,
  "evidence_text": "verbatim quote from the provided text that supports the fact"
}

Rules:
- Extract only facts that are explicitly stated.
- Do not invent attributes such as industry-specific fields unless they appear in the text.
- entity, attribute, value, and evidence_text are required.
- evidence_text MUST be copied from the source text, not paraphrased.
- If nothing extractable exists, return {"facts": []}.
"""


@dataclass
class Fact:
    id: str
    entity: str
    attribute: str
    value: str
    unit: str | None
    period: str | None
    confidence: float
    source_document: str
    page_number: int
    evidence_text: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class LLMClient(ABC):
    """Swap this implementation to change providers without changing extraction logic."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """Works with OpenAI and any OpenAI-compatible API (Groq, Together, local servers)."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        try:
            response = self._client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            message = str(exc).lower()
            if "response_format" not in message and "json_object" not in message:
                raise
            response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response")
        return content


class FactExtractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract_from_chunks(self, chunks: list[DocumentChunk]) -> list[Fact]:
        facts: list[Fact] = []
        seen: set[tuple[str, str, str, str, int]] = set()
        for chunk in chunks:
            for fact in self.extract_from_chunk(chunk):
                key = (
                    fact.entity.lower(),
                    fact.attribute.lower(),
                    fact.value.lower(),
                    fact.source_document,
                    fact.page_number,
                )
                if key in seen:
                    continue
                seen.add(key)
                facts.append(fact)
        return facts

    def extract_from_chunk(self, chunk: DocumentChunk) -> list[Fact]:
        user_prompt = (
            f"Source document: {chunk.source_document}\n"
            f"Page number: {chunk.page_number}\n\n"
            f"Extract facts from this text:\n\n{chunk.text}"
        )
        raw = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        payload = _parse_json_object(raw)
        items = payload.get("facts", [])
        if not isinstance(items, list):
            return []

        facts: list[Fact] = []
        for item in items:
            fact = _to_fact(item, chunk)
            if fact is not None:
                facts.append(fact)
        return facts


def client_from_settings(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> OpenAICompatibleClient:
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not key:
        raise ValueError(
            "Missing API key. Set OPENAI_API_KEY or enter it in the sidebar."
        )
    resolved_model = model or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    resolved_base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    return OpenAICompatibleClient(
        api_key=key,
        model=resolved_model,
        base_url=resolved_base or None,
    )


def _to_fact(item: Any, chunk: DocumentChunk) -> Fact | None:
    if not isinstance(item, dict):
        return None

    entity = _clean_str(item.get("entity"))
    attribute = _clean_str(item.get("attribute"))
    value = _clean_str(item.get("value"))
    evidence = _clean_str(item.get("evidence_text"))
    if not entity or not attribute or not value or not evidence:
        return None

    evidence = _align_evidence(evidence, chunk.text)
    if not evidence:
        return None

    return Fact(
        id=str(uuid.uuid4()),
        entity=entity,
        attribute=attribute,
        value=value,
        unit=_optional_str(item.get("unit")),
        period=_optional_str(item.get("period")),
        confidence=_clamp_confidence(item.get("confidence")),
        source_document=chunk.source_document,
        page_number=chunk.page_number,
        evidence_text=evidence,
    )


def _align_evidence(evidence: str, source_text: str) -> str | None:
    compact_source = _normalize_for_match(source_text)
    compact_evidence = _normalize_for_match(evidence)
    if compact_evidence and compact_evidence in compact_source:
        return evidence.strip()

    snippet = evidence.strip()
    if len(snippet) < 12:
        return None
    # Keep a shortened verbatim window if the model added minor punctuation drift.
    for window in (snippet, snippet[:180], snippet[:80]):
        if _normalize_for_match(window) in compact_source:
            return window.strip()
    return None


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"facts": []}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"facts": []}
    return parsed if isinstance(parsed, dict) else {"facts": []}


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_str(value: Any) -> str | None:
    text = _clean_str(value)
    if not text or text.lower() in {"null", "none", "n/a", "na"}:
        return None
    return text


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, number))


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
