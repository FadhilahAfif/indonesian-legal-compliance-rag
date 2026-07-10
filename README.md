# Indonesian Legal Compliance RAG

An Indonesian legal compliance assistant that retrieves evidence from regulations, reranks relevant passages, and generates answers with verifiable citations.

> **Status:** active engineering redevelopment. The current repository contains the original training and RAG experiments; evaluation and application code will be added incrementally.

## Scope

The initial knowledge base covers:

- PP Nomor 5 Tahun 2021
- PP Nomor 35 Tahun 2021
- PP Nomor 51 Tahun 2023
- UU Nomor 6 Tahun 2023

The project explores:

- Qwen2.5 3B instruction tuning with LoRA
- GRPO experiments
- Dense retrieval with BGE-M3
- BM25 and hybrid retrieval
- Parent-child chunking
- Cross-encoder reranking
- Grounded answers with document citations

## Repository

```text
notebooks/
├── Fine_tuning_submission_PGABL_Muhammad_Afif_Fadhilah.ipynb
├── GRPO_submission_PGABL_Muhammad_Afif_Fadhilah.ipynb
├── RAG_submission_PGABL_Muhammad_Afif_Fadhilah.ipynb
├── M1_baseline_evaluation.ipynb
├── M2_retrieval_ablation.ipynb
├── retrieval_calibration.ipynb
├── retrieval_validation.ipynb
└── grounded_generation_benchmark.ipynb
docs/
└── data-sources.md
data/
└── eval_cases.jsonl
eval/
├── run_baseline.py
├── run_grounded.py
└── validate_cases.py
scripts/
└── check_environment.py
tests/
├── test_eval_cases.py
└── test_rag.py
src/
└── rag.py
requirements.txt
requirements-training.txt
```

The legal PDF files and generated model artifacts are intentionally not committed. See [Regulatory Data Sources](docs/data-sources.md) for provenance and corpus rules.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| Fine-tuning | LoRA supervised fine-tuning of Qwen2.5 3B |
| GRPO | Reward-based post-training experiment |
| RAG | Hybrid retrieval, reranking, generation, and study case |
| M1 baseline | Deterministic benchmark runner for the reviewed evaluation set |
| M2 retrieval | GPU ablation for BM25, dense, hybrid, and hybrid + reranker |
| Retrieval calibration | GPU sweep for candidate depth, fusion weight, and abstention threshold |
| Retrieval validation | GPU validation of the selected configuration and scope guard |
| Grounded generation | GPU benchmark for structured answers, citations, and abstention |

The notebooks were developed for a GPU-enabled Google Colab environment.

## Models

- [SFT model](https://huggingface.co/Slotherynn/legal-chatbot-qwen-sft)
- [GRPO model](https://huggingface.co/Slotherynn/legal-chatbot-qwen-grpo)

## Environment

- Python 3.12
- Linux
- NVIDIA CUDA-compatible GPU for model inference
- NVIDIA T4-class GPU or better for the submitted training configuration

Create the runtime environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

Install and validate the training environment only when running SFT or GRPO:

```bash
python -m pip install -r requirements-training.txt
python scripts/check_environment.py --training
```

The dependency set uses one compatible LangChain 1.x family. Legacy retrievers are supplied by `langchain-classic`.

## Evaluation Set

M1 currently contains 60 reviewed cases: 45 answerable and 15 unanswerable. Validate their structure and coverage locally:

```bash
python -m eval.validate_cases
python -m unittest tests.test_eval_cases -v
```

The strict gate requires every case to retain its reviewed status:

```bash
python -m eval.validate_cases --require-reviewed
```

Do not use this evaluation set for training.

### Run the retrieval ablation

With the four historical PDFs in `data/raw/`, compare BM25, dense, hybrid,
and reranked hybrid retrieval:

```bash
python -m src.rag --device cuda
```

For Google Colab, open `notebooks/M2_retrieval_ablation.ipynb`, select a T4
GPU or better, and run every cell.

For a CPU-only smoke benchmark that does not load embedding models:

```bash
python -m src.rag --methods bm25
```

The runner writes `eval/results/retrieval_ablation.json` and
`eval/results/retrieval_predictions.jsonl`. HyDE and web fallback are disabled.

Current 60-case result:

| Method | Recall@5 | MRR | Source hit rate | Abstention accuracy | Mean query latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.6667 | 0.4744 | 0.8000 | 0.7500 | 0.0067 s |
| Dense | 0.7556 | **0.6274** | **0.9111** | 0.7500 | 0.0244 s |
| Hybrid | **0.7778** | 0.6056 | 0.8667 | 0.7500 | 0.0410 s |
| Hybrid + reranker | 0.7556 | 0.6259 | **0.9111** | **0.8167** | 0.2431 s |

All methods returned valid metadata and zero duplicate results. Latency covers
per-query retrieval after indexing and model loading. No method has reached the
initial Recall@5 target of `0.85`, so parameter calibration remains pending.

### Run retrieval calibration

Open `notebooks/retrieval_calibration.ipynb` in Google Colab, select a T4 GPU
or better, and run every cell. The default sweep compares five parent/child
chunk and overlap configurations, candidate depths `10`, `20`, and `40`, and
BM25 weights `0.2`, `0.4`, and `0.6`. It then evaluates reranker thresholds
`0.1`, `0.2`, `0.24`, and `0.3` on the best hybrid setup.

The notebook writes `eval/results/retrieval_calibration.json` and
`eval/results/retrieval_calibration_predictions.jsonl`.

Current calibration result:

| Configuration | Candidate depth | BM25 weight | Threshold | Recall@5 | MRR | Source hit rate | Abstention accuracy | Mean query latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best hybrid | 10 | 0.4 | - | **0.8444** | 0.6507 | 0.8889 | 0.7500 | 0.0320 s |
| Best hybrid + reranker | 10 | 0.4 | 0.30 | **0.8444** | **0.6637** | **0.9111** | **0.8667** | 0.1909 s |

The best configuration uses parent chunks `1000/100` and child chunks
`300/30` (size/overlap). Calibration improved exact-page Recall@5 from
`0.8222` to `0.8444`, one hit short of the next attainable score (`39/45 =
0.8667`). The reranked setup meets the source hit target, returns valid
metadata, and has zero duplicate results. The frozen evaluation set was not
changed after reviewing the seven exact-page misses.

For the final retrieval check after code changes, run
`notebooks/retrieval_validation.ipynb`. It evaluates only the selected
configuration, so it does not repeat the full calibration sweep.

Final validation with the scope guard produced:

| Method | Recall@5 | MRR | Source hit rate | Abstention accuracy | Mean query latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid + reranker | 0.8444 | 0.6637 | 0.9111 | **0.9000** | 0.1668 s |

All 285 returned quotes were matched to their original PDF source and page.
The retrieval milestone accepts the one-hit exact-page Recall@5 deviation
because all acceptance criteria pass and further tuning against reviewed
misses would risk fitting the frozen evaluation set.

### Grounded answer contract

`src.rag.generate_grounded_answer` uses deterministic decoding and returns a
structured answer with a short answer, legal basis, application, practical
steps, limitations, citations, and a legal disclaimer. Every legal claim must
reference a precomputed snippet ID. The parser reconstructs source metadata and
the exact short quote from retrieval, so model-generated quote text is never
trusted. Unknown snippet IDs are rejected. Questions and document text remain
untrusted JSON data in the prompt.

Run the local contract and prompt-injection checks with:

```bash
python -m unittest discover -s tests -v
```

Run the 60-case generation benchmark in Google Colab by opening
`notebooks/grounded_generation_benchmark.ipynb`, selecting a T4 GPU, and
running every cell. The notebook checks the environment and evaluation set,
downloads the historical corpus, and runs:

```bash
python -m eval.run_grounded
```

The runner uses the first five cases as a format gate. It continues to all 60
cases only when every attempted generation returns a valid grounded schema;
otherwise it saves the partial diagnostics and stops early.

Download `grounded_report.json` and `grounded_predictions.jsonl` from the final
cell. The report calculates retrieval, citation precision, output validity,
abstention, latency, and VRAM. Faithfulness and answer relevance remain unset
until the predictions receive manual review, so the grounded-generation
milestone remains open.

### Run the M1 baseline in Colab

Open `notebooks/M1_baseline_evaluation.ipynb`, select a T4 GPU or better, and run every cell. The notebook installs the runtime dependencies, downloads the historical four-document corpus, validates the reviewed cases, and runs:

```bash
python -m eval.run_baseline
```

It writes `eval/results/baseline_report.json` and `eval/results/baseline_predictions.jsonl`. The completed review and scoring rubric are stored in `eval/results/baseline_generation_review.md`.

## Roadmap

1. Build a manually reviewed Indonesian legal QA benchmark.
2. Fix retrieval duplication, metadata, and citation handling.
3. Compare BM25, dense, hybrid, reranking, and HyDE with ablation tests.
4. Add grounded generation, abstention, and citation evaluation.
5. Compare the base, SFT, and GRPO models on the same benchmark.
6. Build a minimal Gradio demo and publish reproducible results.

## Current Limitations

- The existing SFT and GRPO dataset is general Indonesian instruction data, not a curated legal dataset.
- The M1 baseline has low generation quality: faithfulness `0.4643`, answer relevance `0.3833`, and citation precision `0.2791`.
- The first grounded-generation run produced `43/43` invalid model outputs. JSON prefill fixed the top-level schema, but `0/3` model-generated quotes matched retrieval; citation by precomputed snippet ID now awaits a five-case GPU gate.
- PP Nomor 5 Tahun 2021 is no longer in force, and PP Nomor 51 Tahun 2023 has since been amended; the four-document corpus is a historical evaluation scope, not a statement of current law.
- The current notebooks are experiments and are not a production legal service.

## Disclaimer

The system is under active development. Its output is not legal advice and must be verified against official regulations or a qualified legal professional.

## License

Code is released under the [MIT License](LICENSE). Third-party models, datasets, and regulations retain their respective licenses and terms.
