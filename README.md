# Fact Knowledge Layer

**Live app:** https://jainaryan04-superjoin-assignment.streamlit.app/

> The hosted app already has an OpenAI key configured — just upload PDFs and click
> **Extract facts**. Only enter a key in the sidebar if you want to try a
> different model/provider.

Upload PDFs → extract entity–attribute–value facts grounded in a verbatim source
snippet → automatically link facts that **corroborate**, **contradict**, are
**part of** a whole, or can be **reconciled** by context (time, scope, units,
a documented transition).

Nothing is hard‑coded to a filename, schema, or document — the attribute names,
periods, and units are inferred from the text.

---

## Setup and Run

Requires **Python 3.11+** and an OpenAI‑compatible API key.

```bash
git clone <repo-url> && cd superjoin-assignment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# API key: put it in .env, or paste it into the app's sidebar at runtime
echo "OPENAI_API_KEY=sk-..." > .env
# optional:  OPENAI_MODEL=gpt-4o-mini   OPENAI_BASE_URL=<groq/ollama/... endpoint>

streamlit run app.py                                   # opens http://localhost:8501
```

Upload PDFs → **Extract facts**. Sidebar pages show canonical clusters, semantic
"similar facts", and the relationship graph. Demo set: [`files/`](files/) (`3.1`–`3.5.pdf`).

**Deploying to Streamlit Community Cloud:** put the key in **Manage app →
Settings → Secrets** (TOML) as a **top‑level** key — not a platform env var and
not under a `[section]` — then **Reboot** (secret changes don't hot‑reload):

```toml
OPENAI_API_KEY = "sk-..."
```

A `[openai]` / `[llm]` section also works; the app reads both. See
[`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).

## Video Demo

**<https://drive.google.com/drive/u/0/folders/1EIRINDdMnRNVk9wVma2WryjuB2pfsK3q — 3 min>** · walks through the four required cases on `files/3.1`–`3.5.pdf`.

---

## Approach

**Pipeline** (`app.py` orchestrates):
`pdf_processor` (PyMuPDF → page‑local chunks) → `fact_extractor` (one LLM call per
chunk; generic `entity / attribute / value / unit / period / confidence` +
**verbatim** evidence; snippets that don't align back to the page are dropped) →
`canonicalization_service` (period ↔ attribute split, unit/currency/date
normalization, entity resolution, revenue hierarchy) → `database` (SQLite) →
`provenance_service` + `relationship_rules` (relationship discovery) →
`semantic_search_service` (FAISS over canonical embeddings).

**Key decisions**

- **Evidence first.** A fact is a claim *plus* the sentence it came from. Every
  fact links to one or more `fact_evidence` rows (document, page, exact text).
- **Source‑independent identity.** A fact's key is
  `(canonical_entity, canonical_attribute, normalized_value(value,unit), period)` —
  **not** the filename. So "Revenue FY2024: ₹120 crore" and "Annual Revenue: INR
  120 crore" become **one fact with two evidence records**, not two facts joined
  by a weak edge.
- **Dimensional safety.** Values are typed (money / ratio / count / scaled / plain);
  a rupee figure and a percentage are never compared, so they can't produce a
  false contradiction.
- **Relationship types:** `CORROBORATES`, `CONTRADICTS` (same attribute + same
  period + different value), `RECONCILES` (different period / documented CEO
  transition, with the reconciling sentence cited), `PART_OF` + `COMPUTED_SUPPORT`
  (revenue components → total, incl. a `90 + 30 = 120` check),
  `UNRESOLVED_DIFFERENCE` when a difference exists but nothing explains it.
- **Deterministic linking.** Relationship generation reads only the evidence set
  and canonical fields — never merge‑order‑dependent row metadata — so a rebuild
  is reproducible.
- **Incremental.** New PDFs merge into the existing layer; only affected
  embeddings/clusters are recomputed.

**AI tools used:** built with **Claude Code** (Anthropic) as the coding agent.
The app uses an OpenAI‑compatible chat model (`gpt-4o-mini` by default) for fact
extraction and optional relationship classification; embeddings use
`sentence-transformers` when installed, otherwise a deterministic hash fallback.

---

## Limitations and Next Steps

- **No per‑document period inference.** If a line item ("Software Products: INR 90
  crore") doesn't restate the fiscal year, it stays a separate fact from the
  dated version. Next: infer a document's reporting period and apply it.
- **`holder` facts can pick up a spurious period** (e.g. `FY2025`) from
  surrounding text — a person's role shouldn't be fiscal‑year scoped.
- **Undated ratios** (`operating_margin` 15% vs 18%) fall to
  `UNRESOLVED_DIFFERENCE` because the engine can't tell they're different periods.
- **No place aliases** — "Bengaluru" and "Bangalore" aren't linked.
- **Extraction is LLM‑dependent**: fact counts vary slightly per run; scanned
  PDFs need OCR.
- On large corpora, `PART_OF` can over‑link revenue components across fiscal
  years (period‑scoping for the hierarchy is a known gap).

---

## Additional Notes

- Keep the API key out of git — it lives in `.env` (git‑ignored) or the sidebar.
- `snapshots/` and `reports/` hold before/after migration snapshots and audit
  output used while hardening canonicalization.
- Run without an internet model by pointing `OPENAI_BASE_URL` at a local
  Ollama / vLLM endpoint.
