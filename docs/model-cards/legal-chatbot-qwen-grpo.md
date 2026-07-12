---
base_model: Slotherynn/legal-chatbot-qwen-sft
tags:
- text-generation
- transformers
- unsloth
- qwen2
- grpo
license: apache-2.0
language:
- id
datasets:
- Ichsan2895/alpaca-gpt4-indonesian
---

# Legal Chatbot Qwen GRPO

This is a historical GRPO experiment derived from
`Slotherynn/legal-chatbot-qwen-sft`. Despite its repository name, it was
trained on general Indonesian instructions rather than provenance-backed legal
QA data and must not be treated as an Indonesian legal expert.

## Training

- Method: GRPO with Unsloth and TRL after the historical SFT run.
- Dataset: first 2,000 examples from
  `Ichsan2895/alpaca-gpt4-indonesian`, licensed CC BY-SA 4.0 and derived from
  `FreedomIntelligence/alpaca-gpt4-indonesian`.
- LoRA: rank 16, alpha 16, applied to attention and MLP projection modules.
- Maximum sequence length: 1024.
- Training: 200 steps, learning rate `5e-6`.
- Historical rewards: format, reasoning length, ROUGE-L correctness, and
  language. Training logs show the format and reasoning rewards remained zero
  while the language reward remained constant, so these rewards did not
  provide the intended learning signal.

## Legal RAG Benchmark

The model was evaluated on 60 reviewed Indonesian legal cases using the same
frozen retrieval results, prompt, deterministic decoding, and validator as the
base and SFT models.

| Metric | Result |
| --- | ---: |
| Attempted generations | 43 |
| Valid grounded outputs | 0 |
| Citation precision | Not evaluable |
| Faithfulness | Not evaluable |
| Answer relevance | Not evaluable |
| Abstention accuracy | 0.2167 |
| Mean generation latency | 12.15 s |
| Peak VRAM (4-bit load) | 2686.36 MB |
| Loaded footprint | 1916.96 MB |

Every attempted generation failed the required grounded-output contract. GRPO
did not improve any primary quality metric over either the base or SFT model,
so this model is rejected for the production RAG application.

## Limitations and Intended Use

- Not for production legal answers or legal advice.
- Not trained on the four regulatory documents used by the benchmark.
- Does not reliably follow the required citation and abstention contract.
- The reward design is unsuitable evidence of grounded legal reasoning.
- The benchmark corpus is historical and does not represent current law.

The project retains this model and its training notebook only as a learning
artifact. Any future GRPO run must use legal QA data with clear provenance and
rewards for groundedness, citation correctness, abstention, and answer quality.
