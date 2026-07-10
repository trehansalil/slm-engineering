# SLM-125M: Legal/Financial Small Language Model — Project Summary

**Date:** 2026-07-10
**Model:** [Saliltrehan7/slm-125m-base](https://huggingface.co/Saliltrehan7/slm-125m-base)
**Platform:** [Modal](https://modal.com) (serverless GPU)

---

## Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | LLaMA (decoder-only transformer) |
| Parameters | 125.8M (tied embeddings) |
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 (MHA, head dim 64) |
| Intermediate (SwiGLU) | 3,072 |
| Context length | 1,024 tokens |
| Vocabulary | 16,384 (byte-level BPE) |
| Precision | bfloat16 |

---

## Data Pipeline

### Sources (Legal-first mix ~40/40/20)

| Source | HuggingFace ID | Domain | Role |
|--------|---------------|--------|------|
| case-law | HFforLegal/case-law (split: us) | Legal | US judicial opinions |
| SEC | PleIAs/SEC | Financial | SEC filings (10-K, 10-Q, 8-K) |
| fineweb-edu | HuggingFaceFW/fineweb-edu (sample-10BT) | General | High-quality web text |

### Cleaning (6-step deterministic pipeline)
1. **Line filtering** — remove lines <40 chars or >30% non-alphanumeric
2. **Boilerplate stripping** — remove headers/footers with high symbol density
3. **Repetition check** — drop docs where top-10 4-grams exceed 50% of content
4. **English detection** — langdetect filter (sampled at 5,000 chars)
5. **OCR gate** — drop docs with >20% non-dictionary words (strict for case-law)
6. **Minimum length** — discard docs under 600 chars post-cleaning

### Deduplication
- **Exact dedup** — blake2b hash on normalized text
- **Near dedup** — MinHash LSH (128 perms, 0.7 Jaccard threshold)
- **Contamination stripping** — 13-gram overlap with CaseHOLD/LexGLUE eval sets (480K eval n-grams removed)

### Tokenization
- **Tokenizer:** 16,384 vocab byte-level BPE trained on corpus
- **Special tokens:** `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`, `<|user|>`, `<|assistant|>`, `<|system|>`
- **Packing:** uint16 packed 1,024-token windows, 99/1 train/val split
- **Final corpus:** 2.04B train tokens (1,991,282 windows), 20.6M val tokens (20,119 windows)

---

## Training

| Parameter | Value |
|-----------|-------|
| GPUs | 8× NVIDIA H100 (Modal) |
| Parallelism | DDP (DistributedDataParallel) via torchrun |
| Epochs | 1 |
| Total steps | 3,889 |
| Tokens seen | 2.04B |
| Global batch size | 524,288 tokens |
| Micro batch size | 32 |
| Grad accumulation | 2 per GPU |
| Optimizer | AdamW (β1=0.9, β2=0.95, wd=0.1) |
| Learning rate | 6e-4 → 6e-5 (cosine decay) |
| Warmup | 200M tokens (~381 steps) |
| Gradient clipping | 1.0 |
| Wall time | **19.0 minutes** |
| Throughput | ~1.96M tok/s |

### Loss Curve

| Step | Train Loss | Val Loss | Val PPL |
|------|-----------|----------|---------|
| 20 | 9.68 | — | — |
| 100 | 6.81 | — | — |
| 500 | 3.48 | — | — |
| 1,000 | 2.80 | 2.81 | 16.54 |
| 2,000 | 2.55 | 2.53 | 12.56 |
| 3,000 | 2.44 | 2.42 | 11.29 |
| 3,500 | 2.38 | **2.41** | **11.08** |
| 3,880 | 2.39 | — | — |

**Final validation perplexity: 11.08** (measured at step 3,500 checkpoint)

---

## Costs

| Component | Resource | Cost |
|-----------|----------|------|
| Data pipeline (download, clean, dedup, tokenize) | CPU | $1.57 |
| Pretraining (19 min) | 8× H100 | $10.31 |
| Pretraining overhead | CPU + Memory | $0.28 |
| Final eval + HF upload | L4 + CPU | $0.20 |
| **Total** | | **$12.36** |

All compute ran on Modal's serverless platform. No persistent infrastructure costs.

---

## Pipeline Phases (Execution Order)

| Phase | Description | Time | Status |
|-------|-------------|------|--------|
| 0 | Smoke test + source measurement | ~2 min | Done |
| 1 | Download + clean (14 shards) | ~8 min | Done |
| 2 | Dedup + decontamination + corpus assembly | ~5 min | Done |
| 3 | Train BPE tokenizer | ~3 min | Done |
| 4 | Tokenize + pack windows | ~4 min | Done |
| 5 | Pretrain (8×H100 DDP) | 19 min | Done |
| 6 | HuggingFace upload | ~1 min | Done |
| — | Final eval | ~1 min | Done |

**Total wall time: ~43 minutes** (including image builds and cold starts)

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | Single source of truth — model, data, training hyperparameters |
| `cleaning.py` | Deterministic 6-step cleaning pipeline |
| `dedup.py` | Hashing and n-gram helpers for deduplication |
| `modal_app.py` | Modal app with all phase functions + pretrain + upload + eval |
| `train_llm.py` | Standalone DDP training script (launched via torchrun) |
| `REPLICATION_GUIDE.md` | Full replication guide with expected outputs |
