# SLM-125M: Legal/Financial Small Language Model

A 125M-parameter decoder-only transformer trained from scratch on legal and financial text, then fine-tuned for grounded Q&A. The entire pipeline — data cleaning, pretraining, SFT, and on-device inference — runs reproducibly for under $15.

**HuggingFace:** [Saliltrehan7/slm-125m-base](https://huggingface.co/Saliltrehan7/slm-125m-base)

## Model

| | |
|---|---|
| Architecture | LLaMA (decoder-only transformer) |
| Parameters | 125.8M (tied embeddings) |
| Layers / Hidden / Heads | 12 / 768 / 12 (MHA, head dim 64) |
| FFN (SwiGLU) | 3,072 |
| Context length | 1,024 tokens |
| Vocabulary | 16,384 (byte-level BPE) |
| Precision | bfloat16 |

## Training Data

Legal-first mix (~40/40/20), 2.04B tokens after cleaning and deduplication:

| Source | Dataset | Domain |
|--------|---------|--------|
| US Case Law | [HFforLegal/case-law](https://huggingface.co/datasets/HFforLegal/case-law) | Legal |
| SEC Filings | [PleIAs/SEC](https://huggingface.co/datasets/PleIAs/SEC) | Financial |
| FineWeb-Edu | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | General |

Processing pipeline: 6-step deterministic cleaning → exact + near deduplication (MinHash LSH) → contamination stripping (13-gram overlap with CaseHOLD/LexGLUE) → BPE tokenization → uint16 packed 1024-token windows.

## Results

### Pretraining (base model)

8x H100 on Modal, 1 epoch, 19 minutes, **$12.36 total**.

| Step | Val Loss | Val Perplexity |
|------|----------|----------------|
| 1,000 | 2.81 | 16.54 |
| 2,000 | 2.53 | 12.56 |
| 3,500 | 2.41 | **11.08** |

### SFT (instruction-tuned)

12K grounded Q&A pairs generated via Azure OpenAI + Gemini Flash, filtered through a 5-stage gauntlet (format, TF-IDF dedup, grounding check, task balance, target cap).

| | Modal (A100) | Local (MPS) |
|---|---|---|
| Epochs | 10 | 8 (best @ 4) |
| Best Val Loss | 6.19 (ppl 485.69) | 4.36 (ppl 78.24) |

### Inference (Apple Silicon)

CoreML conversion with KV-cached single-token decode:

| | Speed |
|---|---|
| Prefill | 49 tok/s |
| Decode | **56 tok/s** |

## Project Structure

```
Makefile                     # All pipeline targets (make help)
config.py                    # Central config — model arch, data mix, hyperparams

pretrain/                    # Phases 1-6: raw data → base model
  pipeline.py                #   Modal orchestrator (clean, dedup, tokenize, pretrain, upload)
  cleaning.py                #   Deterministic 6-step cleaning pipeline
  dedup.py                   #   Hashing and n-gram dedup helpers
  train_ddp.py               #   Multi-GPU DDP training script

sft/                         # Phases 7-8: SFT data generation + fine-tuning
  datagen_azure.py           #   Generate Q&A pairs via Azure OpenAI
  datagen_gemini.py          #   Generate Q&A pairs via Gemini Flash
  finetune_modal.py          #   Full fine-tuning on Modal A100
  finetune_local.py          #   Full fine-tuning on Mac MPS/CPU

inference/                   # Phase 9: serving
  convert_coreml.py          #   Convert to CoreML with KV cache
  chat_pytorch.py            #   PyTorch inference (CPU/MPS)
  chat_coreml.py             #   CoreML inference on Apple Silicon (ANE)

docs/
  MODEL_CARD.md              # HuggingFace model card
  PROJECT_SUMMARY.md         # End-to-end project writeup
  REPLICATION_GUIDE.md       # Step-by-step reproduction instructions
notes/                       # Voice-memo transcripts and project setup notes
```

## Environment Variables

SFT data generation requires API keys set as [Modal secrets](https://modal.com/docs/guide/secrets):

| Variable | Used by | Purpose |
|----------|---------|---------|
| `AZURE_API_KEY` | `datagen_azure.py` | Azure OpenAI endpoint |
| `AZURE_BASE_URL` | `datagen_azure.py` | Azure OpenAI base URL |
| `AZURE_API_VERSION` | `datagen_azure.py` | Azure API version |
| `AZURE_MODEL` | `datagen_azure.py` | Deployment name |
| `GEMINI_API_KEY` | `datagen_gemini.py` | Google Gemini API key |

DDP training uses `LOCAL_RANK` (set automatically by `torchrun`). Local SFT accepts `NUM_EPOCHS` to override the default epoch count.

## Quick Start

```bash
# Install dependencies
pip install torch transformers coremltools numpy

# Chat with PyTorch (downloads model from local_model/sft_local)
python inference/chat_pytorch.py

# Convert to CoreML and chat on Apple Silicon
python inference/convert_coreml.py
python inference/chat_coreml.py
```

## Pipeline Commands

Run `make help` to see all targets:

```
Pretrain pipeline (Modal)
  make smoke               Smoke test data pipeline
  make clean-data          Stream + clean all sources
  make dedup               Dedup + contamination strip
  make tokenizer           Train 16K BPE tokenizer
  make tokenize            Tokenize into 1024-token windows
  make pretrain            Pretrain on 8x H100
  make upload              Upload base model to HuggingFace
  make eval                Final perplexity eval

SFT data generation (Modal)
  make sft-data-azure      Generate Q&A pairs via Azure OpenAI
  make sft-data-gemini     Generate Q&A pairs via Gemini Flash

SFT fine-tuning
  make sft-modal           Fine-tune on Modal A100
  make sft-local           Fine-tune on local Mac (MPS)

Inference
  make chat                PyTorch chat
  make chat-coreml         CoreML chat (KV cached, Apple Silicon)
  make convert-coreml      Convert model to CoreML
```

## Infrastructure

All cloud compute runs on [Modal](https://modal.com)'s serverless platform. Local training and inference run on Apple Silicon (MPS / ANE).

| Phase | Resource | Cost |
|-------|----------|------|
| Data pipeline | CPU (Modal) | $1.57 |
| Pretraining | 8x H100, 19 min | $10.59 |
| Eval + upload | L4 | $0.20 |
| SFT data gen | CPU (Modal) | ~$2.00 |
| SFT training | A100 / local MPS | ~$0.50 |
| **Total** | | **~$15** |

## License

Apache 2.0
