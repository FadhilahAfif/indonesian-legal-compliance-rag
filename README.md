# Indonesian Legal Compliance RAG

An Indonesian legal compliance assistant that retrieves evidence from regulations, reranks relevant passages, and generates answers with verifiable citations.

> **Status:** active portfolio redevelopment. The current repository contains the original training and RAG experiments; evaluation and application code will be added incrementally.

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
└── RAG_submission_PGABL_Muhammad_Afif_Fadhilah.ipynb
requirements.txt
```

The legal PDF files and generated model artifacts are intentionally not committed.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| Fine-tuning | LoRA supervised fine-tuning of Qwen2.5 3B |
| GRPO | Reward-based post-training experiment |
| RAG | Hybrid retrieval, reranking, generation, and study case |

The notebooks were developed for a GPU-enabled Google Colab environment.

## Models

- [SFT model](https://huggingface.co/Slotherynn/legal-chatbot-qwen-sft)
- [GRPO model](https://huggingface.co/Slotherynn/legal-chatbot-qwen-grpo)

## Original Notebook Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The requirements currently describe the submitted notebook environment and will be normalized during repository redevelopment. CUDA-compatible hardware is required for the original training and inference notebooks.

## Roadmap

1. Build a manually reviewed Indonesian legal QA benchmark.
2. Fix retrieval duplication, metadata, and citation handling.
3. Compare BM25, dense, hybrid, reranking, and HyDE with ablation tests.
4. Add grounded generation, abstention, and citation evaluation.
5. Compare the base, SFT, and GRPO models on the same benchmark.
6. Build a minimal Gradio demo and publish reproducible results.

## Current Limitations

- The existing SFT and GRPO dataset is general Indonesian instruction data, not a curated legal dataset.
- Retrieval and generation do not yet have a reproducible benchmark.
- The current notebooks are experiments and are not a production legal service.

## Disclaimer

This project is for education and portfolio demonstration. Its output is not legal advice and must be verified against official regulations or a qualified legal professional.

## License

Code is released under the [MIT License](LICENSE). Third-party models, datasets, and regulations retain their respective licenses and terms.
