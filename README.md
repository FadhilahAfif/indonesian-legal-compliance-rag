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
└── M1_baseline_evaluation.ipynb
docs/
└── data-sources.md
data/
└── eval_cases.jsonl
eval/
├── run_baseline.py
└── validate_cases.py
scripts/
└── check_environment.py
tests/
└── test_eval_cases.py
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

### Run the M1 baseline in Colab

Open `notebooks/M1_baseline_evaluation.ipynb`, select a T4 GPU or better, and run every cell. The notebook installs the runtime dependencies, downloads the historical four-document corpus, validates the reviewed cases, and runs:

```bash
python -m eval.run_baseline
```

It writes `eval/results/baseline_report.json` and `eval/results/baseline_predictions.jsonl`. Generation faithfulness and answer relevance remain unset until the predictions are reviewed; the runner does not invent proxy scores for them.

## Roadmap

1. Build a manually reviewed Indonesian legal QA benchmark.
2. Fix retrieval duplication, metadata, and citation handling.
3. Compare BM25, dense, hybrid, reranking, and HyDE with ablation tests.
4. Add grounded generation, abstention, and citation evaluation.
5. Compare the base, SFT, and GRPO models on the same benchmark.
6. Build a minimal Gradio demo and publish reproducible results.

## Current Limitations

- The existing SFT and GRPO dataset is general Indonesian instruction data, not a curated legal dataset.
- The M1 benchmark is still awaiting a baseline run.
- PP Nomor 5 Tahun 2021 is no longer in force, and PP Nomor 51 Tahun 2023 has since been amended; the four-document corpus is a historical evaluation scope, not a statement of current law.
- The current notebooks are experiments and are not a production legal service.

## Disclaimer

The system is under active development. Its output is not legal advice and must be verified against official regulations or a qualified legal professional.

## License

Code is released under the [MIT License](LICENSE). Third-party models, datasets, and regulations retain their respective licenses and terms.
