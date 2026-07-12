---
base_model: unsloth/Qwen2.5-3B-Instruct-bnb-4bit
tags:
- text-generation
- transformers
- unsloth
- qwen2
- sft
license: apache-2.0
language:
- id
datasets:
- Ichsan2895/alpaca-gpt4-indonesian
---

# Legal Chatbot Qwen SFT

This is a historical supervised fine-tuning experiment based on
Qwen2.5-3B-Instruct. Despite its repository name, it was trained on a general Indonesian
instruction dataset, not a curated legal QA dataset, and must not be treated as
an Indonesian legal expert.

## Training

- Method: 4-bit QLoRA with Unsloth and TRL.
- Dataset: `Ichsan2895/alpaca-gpt4-indonesian`, licensed CC BY-SA 4.0 and
  derived from `FreedomIntelligence/alpaca-gpt4-indonesian`.
- Split: random 90/10 train/evaluation split with seed 42.
- LoRA: rank 16, alpha 16, applied to attention and MLP projection modules.
- Maximum sequence length: 2048.
- Training: 800 steps, learning rate `2e-4`.
- Historical validation loss: 1.0338. This measures the general instruction
  validation split and is not a legal-quality metric.

## Legal RAG Benchmark

The model was evaluated on 60 reviewed Indonesian legal cases using the same
frozen retrieval results, prompt, deterministic decoding, and validator as the
base and GRPO models.

| Metric | Result |
| --- | ---: |
| Attempted generations | 43 |
| Valid grounded outputs | 0 |
| Citation precision | Not evaluable |
| Faithfulness | Not evaluable |
| Answer relevance | Not evaluable |
| Abstention accuracy | 0.2167 |
| Mean generation latency | 13.72 s |
| Peak VRAM (4-bit load) | 2686.36 MB |
| Loaded footprint | 1916.96 MB |

Every attempted generation failed the required grounded-output contract. The
model is rejected for the production RAG application; fine-tuning showed no
quality improvement over the base model.

## Limitations and Intended Use

- Not for production legal answers or legal advice.
- Not trained on the four regulatory documents used by the benchmark.
- Does not reliably follow the required citation and abstention contract.
- The benchmark corpus is historical and does not represent current law.
- Outputs must be verified against official regulations and qualified counsel.

The project retains this model and its training notebook only as a learning
artifact. The application uses a deterministic extractive fallback instead.
