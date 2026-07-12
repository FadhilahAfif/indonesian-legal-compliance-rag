# Grounded Generation Review

Review date: 2026-07-12

## Rubric

- `1.0`: fully supported or directly answers the case.
- `0.5`: partially supported or relevant but incomplete.
- `0.0`: unsupported, contradicted, or does not answer the case.
- Faithfulness is `N/A` when no substantive legal claim was generated.
- Correct abstentions receive answer relevance `1.0`.

The exact per-case scores are stored in `grounded_review.json` and are applied
by `eval.run_grounded` when calculating the final report.

## Final Metrics

| Metric | Target | Result | Status |
| --- | ---: | ---: | --- |
| Faithfulness | >= 0.90 | 1.0000 (41 reviewed claims) | Pass |
| Citation precision | >= 0.90 | 1.0000 | Pass |
| Abstention accuracy | >= 0.85 | 0.9333 | Pass |
| Valid output rate | - | 1.0000 | Pass |
| Answer relevance | No numeric target | 0.6167 (60 reviewed cases) | Reported limitation |
| Unsupported legal claims | 0 | 0 | Pass |

Answer relevance improved from the historical model baseline of `0.3833`, but
remains the main quality limitation. The extractive fallback does not infer or
synthesize: it can return a faithful but incomplete passage, and four
answerable cases still abstain at retrieval.

## Answer Relevance Scores

- `1.0`: m1-006, m1-012, m1-013, m1-018, m1-019, m1-021, m1-023,
  m1-027, m1-028, m1-030, m1-035, m1-036, m1-037, m1-039, m1-042,
  m1-044, m1-045, and m1-046 through m1-060.
- `0.5`: m1-001, m1-005, m1-008, m1-011, m1-016, m1-020, m1-024,
  m1-025, m1-031, m1-043.
- `0.0`: all remaining cases.

## Safety Audit

Every accepted answer copies its short answer and supporting quote from the
selected retrieved document. Regulation, source, page, article, and chunk ID
are reconstructed from retrieval metadata. No model-generated claim or
citation is accepted, and all 41 substantive answers were fully supported by
their cited quote.

Queries asking for current information (`saat ini`, `terbaru`, or `tahun ini`)
and under-specified personal totals now abstain. This corrected the two
unanswerable false positives for current Bandung UMK and an unspecified total
severance calculation.

## Decision

The grounded-generation milestone passes its documented acceptance criteria.
The deterministic extractive fallback is the production answer renderer for
the initial release. Its moderate answer relevance is retained as an explicit
product limitation rather than hidden behind unsupported generated prose.
