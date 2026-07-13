.PHONY: help smoke clean-data dedup tokenizer tokenize pretrain upload eval \
       sft-data-azure sft-data-gemini sft-tokenize sft-modal sft-local \
       sft-modal-fresh sft-local-fresh \
       chat chat-coreml convert-coreml

help: ## Show all targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Pretrain pipeline (Modal) ──────────────────────────────────────

smoke: ## Smoke test data pipeline (10 docs/source)
	modal run pretrain/pipeline.py::main

clean-data: ## Phase 1: stream + clean all sources
	modal run pretrain/pipeline.py::clean

dedup: ## Phase 2: dedup + contamination strip
	modal run pretrain/pipeline.py::dedup

tokenizer: ## Phase 3: train 16K BPE tokenizer
	modal run pretrain/pipeline.py::tokenizer

tokenize: ## Phase 4: tokenize corpus into 1024-token windows
	modal run pretrain/pipeline.py::tokenize

pretrain: ## Phase 5: pretrain on multi-GPU (8x H100)
	modal run pretrain/pipeline.py::pretrain

upload: ## Phase 6: upload base model to HuggingFace
	modal run pretrain/pipeline.py::upload

eval: ## Run final perplexity eval on GPU
	modal run pretrain/pipeline.py::final_eval

# ── SFT data generation (Modal) ───────────────────────────────────

sft-data-azure: ## Generate SFT pairs via Azure OpenAI
	modal run sft/datagen_azure.py::generate

sft-data-gemini: ## Generate SFT pairs via Gemini Flash
	modal run sft/datagen_gemini.py::generate

# ── SFT fine-tuning ───────────────────────────────────────────────

sft-tokenize: ## Tokenize SFT dataset on Modal
	modal run sft/finetune_modal.py::tokenize_sft

sft-modal: ## SFT on Modal A100 (20 epochs, ckpt every 2)
	modal run sft/finetune_modal.py::finetune $(ARGS)

sft-modal-fresh: ## SFT from scratch (ignore existing checkpoints)
	modal run sft/finetune_modal.py::finetune --fresh $(ARGS)

sft-local: ## SFT on local Mac (MPS)
	python sft/finetune_local.py $(ARGS)

sft-local-fresh: ## SFT from scratch on local Mac (ignore existing checkpoints)
	python sft/finetune_local.py --fresh $(ARGS)

# ── Inference ─────────────────────────────────────────────────────

chat: ## Chat via PyTorch (CPU/MPS)
	python inference/chat_pytorch.py

chat-coreml: ## Chat via CoreML (Apple Silicon, KV cached)
	python inference/chat_coreml.py

convert-coreml: ## Convert best model to CoreML with KV cache
	python inference/convert_coreml.py
