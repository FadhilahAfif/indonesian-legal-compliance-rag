# Model Comparison Review

## Decision

Reject Qwen2.5-3B-Instruct, the SFT model, and the GRPO model as grounded
generators. Keep the deterministic extractive fallback for the application.
Fine-tuning did not improve any primary quality metric because all three models
failed the output contract on every attempted generation.

## Protocol

- Evaluation date: 2026-07-12.
- Cases: 60 reviewed cases (45 answerable, 15 unanswerable).
- Retrieval: the same committed final-retrieval artifact for every model.
- Retrieval equality check: zero mismatches between base, SFT, and GRPO runs.
- Decoding: deterministic, seed 42, 4-bit loading, maximum 512 new tokens.
- Uploaded archive SHA-256: `0D16D1F61BEE70EC612836FE11A3949E1B8FE45A096CF1DACD3A54D7F04A93BF`.

## Results

| Model | Valid outputs | Citation precision | Abstention accuracy | Mean generation latency | P95 case latency | Peak VRAM | Loaded footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 0/43 | N/E | 0.2167 | 24.81 s | 40.22 s | 2686.71 MB | 1916.96 MB |
| SFT | 0/43 | N/E | 0.2167 | 13.72 s | 40.23 s | 2686.36 MB | 1916.96 MB |
| GRPO | 0/43 | N/E | 0.2167 | 12.15 s | 39.23 s | 2686.36 MB | 1916.96 MB |

All 43 attempted generations from each model failed with `output must start
with a valid STATUS line`. The other 17 cases abstained at retrieval before a
model was called: 13 were correct unanswerable abstentions and 4 were false
abstentions on answerable cases. Two unanswerable cases reached generation and
were invalid, as were 41 answerable cases.

Faithfulness, answer relevance, and citation precision are **not evaluable**,
not zero: no model produced a valid claim or citation. Latency therefore does
not rescue any candidate. Load time is also excluded from the selection
because Hub download and cache state differed between models.

The raw benchmark reported `parameter_count` from quantized parameter storage.
That value is implementation-specific and is not treated as the architectural
parameter count; loaded footprint and peak VRAM are the comparable size
measurements. The runner now labels this field `quantized_parameter_elements`.

## Training Decision

- Do not retrain SFT or GRPO for the initial release.
- Do not use the general Indonesian Alpaca data as evidence of legal-domain
  capability.
- Preserve the historical notebooks and their outputs as learning artifacts.
- If training is reconsidered, first build provenance-backed legal QA data,
  split by regulation or article, and use groundedness, citation correctness,
  abstention, and answer quality rewards. Do not reuse reasoning-length or
  constant language rewards.

## Artifact Retention

The compact `comparison.json` is versioned with this review. Per-model
predictions are omitted because all model answers are null and each file mostly
duplicates the same retrieved context. The uploaded ZIP hash above preserves a
verifiable reference to the complete local artifact.

## Model Card Publication

The benchmark decision, training data, method, limitations, and Indonesian
language metadata were published and verified against the versioned sources:

- SFT: commit `5f1627dc1d3ffda08bbef7a7f51ddfc5dee97f77`.
- GRPO: commit `5e3621a7ed26390afbc5f12d8ab629fc91ea10a2`.

Both remote cards report `language: id`, the Indonesian Alpaca dataset, and the
Apache-2.0 model license. Remote content matches the local model-card sources.
