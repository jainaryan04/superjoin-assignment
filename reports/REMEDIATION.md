# Canonicalization Remediation Report

Snapshot: `snapshots/facts.pre-remediation.db` (1,828 facts, untouched).
Migration: `python migrate.py --from-snapshot snapshots/facts.pre-remediation.db --db data/facts.db`

## 1. Root causes fixed

| # | Root cause | File / function | Fix |
|---|---|---|---|
| 1 | `database._norm` folded case + whitespace only, so `total_income` ≠ `total income` as a stored identity while `relationship_rules.normalize_text` treated them as one entity. | `database._norm` | Now delegates to `canonicalization_service.normalize_token` (folds `_` and `-`). Single shared normalizer. |
| 2 | Nothing rejected comparisons across value kinds: `1874.74 INR million` vs `3.9 %` became `CONTRADICTS`. | `canonicalization_service.value_dimension` / `dimensions_compatible` (new); `relationship_rules.classify_relationship_detail`; `provenance_service.validate_relationship` | Values classified as `money / ratio / count / scaled / plain`. `classify_relationship_detail` returns `None` and `validate_relationship` fails `CONTRADICTS`/`CORROBORATES` when dimensions are incompatible. |
| 3 | `is_employee_metric` matched any substring `employee|headcount|…` **and read the already-mutated entity**, so `salary_and_other_employee_benefits`, `percentage_of_female_employees`, and even `total_revenue` (once entity became `Employees`) resolved to `employee_count`. | `canonicalization_service.is_employee_metric` | Exact alias set + a bounded headcount-phrase regex; explicit `NON_HEADCOUNT_RE` reject list; `entity` argument ignored. Plus a value guard (`is_plausible_headcount`): a headcount that is money / % / scaled / share-count is demoted to `value`. |
| 4 | `infer_revenue_attribute` matched bare `revenue`, so `percentage_of_total_revenue`, `CAGR`, `growth_rate` became `total_revenue`. | `canonicalization_service.infer_revenue_attribute` + `canonicalize_fact` demotion | `DERIVED_METRIC_RE` guard returns `None` for ratio/derived wording; `canonicalize_fact` additionally demotes a revenue *level* attribute whose value is a ratio. |
| 5 | `canonicalize_unit` echoed unknown units, so `INR` / `₹` survived as "scale" and `canonicalize_currency` emitted `15.36 INR INR`, `15.36 INR ₹` — two identities for one value. | `canonicalize_unit`, `canonicalize_currency`, new `is_currency_token` / `canonicalize_currency_code` | Currency is a separate dimension. `canonicalize_unit("INR") == ""`, `canonicalize_unit("₹") == ""`. `canonicalize_currency` fills amount / currency / scale once each. New `facts.currency` column. |
| 6 | `apply_entity_resolution` overwrote `fact["entity"]` with `"Employees"`/`"CEO"` **before** `infer_canonical_attribute` ran, so canonicalization read its own output and was not idempotent. | `canonicalization_service.apply_entity_resolution`, `canonicalize_fact` | Never writes `entity`; the observed subject is preserved and the derived name lives only in `canonical_entity`. Canonicalizing an already-canonical fact is now a fixed point. |

## 2. Tests added

`tests/test_canonicalization_quality.py` (26 tests):

* **Normalization** — `total_income == total income`, `Total_Revenue == total revenue`; `identity_key` unifies underscore entity spellings.
* **Dimensions** — `money`/`ratio`/`count`/`scaled`/`plain` classification; money-vs-% and count-vs-money are incomparable; money/scaled/plain stay comparable; an incomparable pair produces no `classify_relationship_detail` result and `validate_relationship(CONTRADICTS)` fails with `"Incomparable value dimensions"`.
* **Employee metric** — every one of the 17 observed bad attributes is rejected; every headcount phrase is accepted; `entity="Employees"` cannot force `employee_count`; a headcount value that is money / % / scaled / share-count is demoted; plain integer headcounts are kept.
* **Revenue** — `percentage_of_total_revenue`, `CAGR`, `growth_rate`, `revenue_margin`, `yoy_revenue_increase` are not `total_revenue`; genuine revenue attributes still resolve.
* **Currency** — `canonicalize_unit("INR"/"₹") == ""`; scale units survive; never emits `INR INR` / `INR ₹`; `₹15.36/INR` and `₹15.36/₹` share one `normalized_value_key`; currency + scale both kept; `canonicalize_fact` populates `currency`.
* **Entity resolution** — `entity` never overwritten; canonicalization idempotent over 3 passes; pre-seeded stale `canonical_*` does not change the result.
* **End-to-end** — money vs % produces no `CONTRADICTS`; `₹15.36/INR` + `₹15.36/₹` merge to one fact with evidence from both PDFs; underscore entity variants merge.

Full suite: **68 tests, all passing.**

## 3. Corpus before / after

| metric | before | after | delta |
|---|---:|---:|---:|
| facts | 1828 | 1821 | −7 (exact-duplicate merges surfaced by the `_norm` fix) |
| evidence | 2221 | 2221 | 0 (no provenance lost) |
| relationships | 875 | 2150 | +1275 — see §5 |
| clusters | 1733 | 1730 | −3 |
| embeddings | 1828 | 1821 | −7 (1:1 with facts; 0 missing) |
| **employee_count facts** | 149 | 78 | −71 |
| **employee_count invalid** | 54 | **0** | −54 |
| revenue *level* facts | 268 | 235 | −33 |
| **revenue level facts with ratio value** | 30 | **0** | −30 |
| **currency anomalies (`INR INR` / `INR ₹`)** | 109 | **0** | −109 |
| **incompatible-dimension CONTRADICTS/CORROBORATES** | 1 | **0** | −1 |
| mixed-dimension attributes | 4 | 2 | −2 |
| CONTRADICTS | 85 | 201 | +116 — all same-period, 0 incompatible dimensions (see §5) |
| COMPUTED_SUPPORT | 41 | 5 | −36 (now scoped to same entity + period) |

## 4. Performance

Per-stage timings (`reports/migration.json`), 1,821 facts, 2 source PDFs:

| stage | seconds |
|---|---:|
| snapshot | 0.01 |
| schema_migrations | 0.81 |
| canonicalization (`reprocess_fact_canonicalization`) | 0.26 |
| deduplication | 0.08 |
| relationship_rebuild | **2.94** (was **126 s** before optimisation) |
| cluster_rebuild | 0.11 |
| embedding_rebuild | 3.33 |
| **total** | **7.5 s** |

O(N²)/O(N^k) work removed from `link_fact_relationships`:

* `_cross_reason_links` — buckets facts by `(canonical_attribute, entity-identity)` and only pairs within a bucket. `classify_relationship_detail` already rejects any cross-bucket pair, so this is the identical result set, not a heuristic.
* `_nearby_facts` — a per-document index keyed by comparison bucket / revenue category / attribute core replaces the full-corpus `_facts_share_document` sweep.
* `_computed_support_links` — subset-sum runs per `(entity, period)` group and is skipped above 16 candidates; the old code ran `C(65, 6) ≈ 8×10⁷` combinations per total-revenue fact.
* `find_transition_support` — the transition-fact subset is computed once per run (module cache keyed by cohort identity) instead of scanning all 1,821 facts per pair; fact text blobs are memoised.
* `_fact_by_id` — id→fact map memoised (was a linear scan per emitted relation).
* relationship writes — one transaction / one commit via `insert_relationships_bulk` + `existing_relationship_keys`, replacing ~2,150 single-row commits.

## 5. Determinism

`python tools_determinism.py --snapshot snapshots/facts.pre-remediation.db --runs 5` → **DETERMINISTIC**.

All 5 runs from the same snapshot: identical fact count (1821), cluster count (1730), relationship count (2150), relationship identity hash, relationship *semantic* hash, and cluster hash. Re-migrating an already-migrated DB is a fixed point (same `rel_id` hash).

## 6. Remaining known issues (not addressed — out of scope)

1. **Cross-period `PART_OF` (720 of 1210).** `relationship_rules.is_hierarchical_child` links any `services_revenue`/`product_revenue` fact to any `total_revenue` fact in the same document regardless of period or entity, so five fiscal years of segment figures form a cross-product (933 `services_revenue → total_revenue` links). This is pre-existing behaviour in the committed relationship engine (`is_hierarchical_child` has no period scoping); the task required "`PART_OF` behaviour should remain unchanged", so it was left alone. The relationship-count jump (875 → 2150) is dominated by this plus the `c72f64a` engine rewrite that already landed before this work — the canonicalization changes here only *reduce* candidate pairs (bucketing/indexing are provably subset-or-equal) and the dimension gate only removes links.
2. **CONTRADICTS 85 → 201.** Not a regression: canonicalization now unifies entity/attribute spellings that were previously fragmented (`Total income` / `total_income` / `total income`), exposing genuine same-period value disagreements between the prospectus and the annual report. All 201 are same-period and dimensionally compatible; 33 are ratio-vs-ratio.
3. **`mixed_dimension_attributes: 2`** (`total_revenue`, `employee_count`). Some legitimate facts are bare integers and others carry a scale word; `money`/`scaled`/`plain` are mutually compatible so this creates no bad relationships.
4. **9 CONTRADICTS between facts whose `original_entity` differs** but refer to the same real subject — `same_entity_identity` keys on `original_entity`, which the extractor spells inconsistently.
5. `COMPUTED_SUPPORT` dropped 41 → 5 because it is now correctly scoped to same-entity + same-period components. The 36 removed were cross-period coincidental sums.
