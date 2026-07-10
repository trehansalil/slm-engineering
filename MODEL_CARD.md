---
license: apache-2.0
language:
  - en
tags:
  - legal
  - finance
  - causal-lm
  - llama
  - from-scratch
  - small-language-model
library_name: transformers
pipeline_tag: text-generation
model-index:
  - name: slm-125m-base
    results: []
datasets:
  - HFforLegal/case-law
  - PleIAs/SEC
  - HuggingFaceFW/fineweb-edu
---

# SLM-125M-Base: A Legal/Financial Small Language Model

A 125M-parameter decoder-only transformer trained from scratch on a curated legal and financial corpus. Built on the LLaMA architecture with a custom 16K byte-level BPE tokenizer.

This is a **base model** (no instruction tuning or RLHF). It is intended as a foundation for downstream fine-tuning on legal and financial NLP tasks.

## Model Details

| | |
|---|---|
| **Architecture** | LLaMA (decoder-only transformer) |
| **Parameters** | 125.8M (tied embeddings) |
| **Layers / Hidden / Heads** | 12 / 768 / 12 (MHA, head dim 64) |
| **Intermediate (SwiGLU)** | 3,072 |
| **Context length** | 1,024 tokens |
| **Vocabulary** | 16,384 (byte-level BPE) |
| **Precision** | bfloat16 |
| **License** | Apache 2.0 |

## Training Data

The model was trained on a **legal-first data mix** (~40/40/20) totaling **2.04 billion tokens** after cleaning and deduplication:

| Source | Dataset | Domain | Approx. Share |
|--------|---------|--------|--------------|
| US Case Law | [HFforLegal/case-law](https://huggingface.co/datasets/HFforLegal/case-law) (split: us) | Legal | ~40% |
| SEC Filings | [PleIAs/SEC](https://huggingface.co/datasets/PleIAs/SEC) | Financial | ~40% |
| FineWeb-Edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT) | General | ~20% |

### Data Processing

All data underwent a rigorous cleaning and deduplication pipeline:

- **6-step deterministic cleaning:** line filtering, boilerplate stripping, repetition detection, English language filtering, OCR quality gating, minimum length enforcement
- **Exact deduplication** via blake2b hashing on normalized text
- **Near-deduplication** via MinHash LSH (128 permutations, Jaccard threshold 0.7)
- **Contamination stripping:** 13-gram overlap removal against CaseHOLD and LexGLUE evaluation sets to prevent benchmark leakage

## Training Procedure

| | |
|---|---|
| **Hardware** | 8× NVIDIA H100 GPUs (Modal serverless) |
| **Parallelism** | PyTorch DDP (DistributedDataParallel) |
| **Epochs** | 1 |
| **Total steps** | 3,889 |
| **Global batch size** | 524,288 tokens |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.95, weight decay=0.1) |
| **Learning rate** | 6×10⁻⁴ with cosine decay to 6×10⁻⁵ |
| **Warmup** | 200M tokens (~381 steps) |
| **Gradient clipping** | 1.0 |
| **Wall time** | 19 minutes |
| **Throughput** | ~1.96M tokens/sec |

### Training Loss

| Step | Train Loss | Val Loss | Val Perplexity |
|------|-----------|----------|----------------|
| 1,000 | 2.80 | 2.81 | 16.54 |
| 2,000 | 2.55 | 2.53 | 12.56 |
| 3,000 | 2.44 | 2.42 | 11.29 |
| 3,500 | 2.38 | 2.41 | **11.08** |

## Evaluation

| Metric | Value |
|--------|-------|
| **Validation loss** | 2.4054 |
| **Validation perplexity** | 11.08 |

Evaluation was conducted on a held-out 1% split (20.6M tokens) of the training corpus.

## Intended Use

This model is designed as a **base model for fine-tuning** on domain-specific legal and financial tasks, including:

- Legal document classification
- Contract analysis and clause extraction
- SEC filing summarization
- Legal question answering
- Financial sentiment analysis

### Out-of-Scope Use

- This is a small (125M) model and is not suitable as a general-purpose assistant or chatbot without significant fine-tuning
- The model has not been instruction-tuned or aligned — it may generate harmful, biased, or factually incorrect content
- Not suitable for legal advice, financial decisions, or any application requiring factual accuracy without human review

## Limitations

- **Model size:** At 125M parameters, this model has limited capacity compared to larger LLMs. It is best suited for focused domain tasks rather than broad language understanding.
- **Single epoch:** The model was trained for 1 epoch over 2.04B tokens. Additional training epochs or more data may improve performance.
- **Context length:** Limited to 1,024 tokens. Documents longer than this must be chunked.
- **English only:** The model was trained exclusively on English-language text.
- **Temporal cutoff:** Training data reflects documents available as of mid-2026. The model has no knowledge of events after this date.
- **No safety alignment:** The model has no RLHF, constitutional AI, or other safety training.

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Saliltrehan7/slm-125m-base")
tokenizer = AutoTokenizer.from_pretrained("Saliltrehan7/slm-125m-base")

prompt = "The court held that the defendant"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Infrastructure

The entire pipeline — data processing, tokenizer training, model pretraining, and deployment — ran on [Modal](https://modal.com)'s serverless GPU platform. Total compute cost: **$12.36**.

| Phase | Cost |
|-------|------|
| Data pipeline (clean, dedup, tokenize) | $1.57 |
| Pretraining (8× H100, 19 min) | $10.59 |
| Eval + upload | $0.20 |
| **Total** | **$12.36** |

## Citation

```bibtex
@misc{slm125m2026,
  title={SLM-125M-Base: A Legal/Financial Small Language Model},
  author={Salil Trehan},
  year={2026},
  url={https://huggingface.co/Saliltrehan7/slm-125m-base}
}
```
