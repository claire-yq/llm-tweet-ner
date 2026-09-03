# High-Demand vs. Low-Demand NER on Small Language Models

Fine-tuning small language models for named entity recognition on [Tweebank-NER](https://huggingface.co/datasets/tner/tweebank_ner). Based on the psychological concept of task demands, we propose a "high-demand" and "low-demand" evaluation method / data format and compare their performance.

## Repository contents

| File | Description |
|---|---|
| `llama.ipynb` | Pipeline for Llama 3.1 8B |
| `mistral.ipynb` | Pipeline for Mistral v0.3 7B |
| `qwen.ipynb` | Pipeline for Qwen 3 4B |
| `f1.py` | Scoring script: computes overall and per-entity-class precision/recall/F1 from model predictions |

Pipelines are adapted from Unsloth notebooks. Each contains code for data formatting, QLoRA fine-tuning, generation, and evaluation.
