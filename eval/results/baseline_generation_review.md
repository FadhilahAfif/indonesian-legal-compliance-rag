# Baseline Generation Review

Review date: 2026-07-09

## Rubric

- `1.0`: fully supported or directly answers the case.
- `0.5`: partially supported/relevant but incomplete, mixed, or internally inconsistent.
- `0.0`: unsupported, contradicted, or does not answer the case.
- Faithfulness is `N/A` when no substantive legal claim was generated.
- Correct abstentions receive answer relevance `1.0`.

## Scores

Faithfulness: `19.5 / 42 = 0.4643`

- `1.0`: m1-006, m1-007, m1-012, m1-016, m1-022, m1-026, m1-028, m1-031, m1-032, m1-035, m1-036, m1-042, m1-055, m1-059
- `0.5`: m1-001, m1-003, m1-008, m1-015, m1-018, m1-019, m1-020, m1-029, m1-041, m1-043, m1-060
- `0.0`: m1-002, m1-004, m1-005, m1-011, m1-013, m1-014, m1-017, m1-021, m1-023, m1-027, m1-030, m1-033, m1-037, m1-038, m1-039, m1-046, m1-058
- `N/A`: m1-009, m1-010, m1-024, m1-025, m1-034, m1-040, m1-044, m1-045, m1-047, m1-048, m1-049, m1-050, m1-051, m1-052, m1-053, m1-054, m1-056, m1-057

Answer relevance: `23 / 60 = 0.3833`

- `1.0`: m1-007, m1-008, m1-012, m1-028, m1-031, m1-035, m1-041, m1-047, m1-048, m1-049, m1-050, m1-051, m1-052, m1-053, m1-054, m1-056, m1-057
- `0.5`: m1-001, m1-003, m1-006, m1-016, m1-019, m1-020, m1-029, m1-032, m1-036, m1-037, m1-042, m1-043
- All remaining cases: `0.0`

## Main failure patterns

- Seven of 15 unanswerable cases returned status `answer`; two more used abstaining prose but still returned the wrong structured status.
- Relevant evidence was retrieved for 33 of 45 answerable cases, but generation frequently ignored it or reversed its meaning.
- Outputs contain wrong regulation years, prompt repetition, internal contradictions, and unsupported legal conclusions.
- Citation precision is `0.2791`; citations currently mirror top reranked chunks rather than verified support for each claim.
