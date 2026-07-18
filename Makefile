.PHONY: help smoke clean-data dedup tokenizer tokenize pretrain upload eval \
       sft-data-azure sft-data-gemini sft-tokenize sft-modal sft-local \
       sft-modal-fresh sft-local-fresh \
       chat chat-remote chat-coreml convert-coreml

help: ## Show all targets
	@echo "Usage: make <target> [ARGS=\"--flag value ...\"]"
	@echo ""
	@awk '\
	/^# Args: none/   { next } \
	/^# Args: same/   { sub(/^# Args: /, ""); args = "      " $$0 "\n"; next } \
	/^# Args:$$/      { next } \
	/^#   --/         { sub(/^#   /, ""); args = args "      " $$0 "\n"; next } \
	/^[a-zA-Z_-]+:.*## / { \
		split($$0, a, ":.*## "); \
		printf "  \033[36m%-20s\033[0m %s\n", a[1], a[2]; \
		if (args != "") printf "%s", args; \
		args = "" \
	}' $(MAKEFILE_LIST)

# ── Pretrain pipeline (Modal) ──────────────────────────────────────

# Usage: make smoke [ARGS="--n-per-source <int>"]
# Args:
#   --n-per-source  (optional, default=10)  Number of documents per source to test
smoke: ## Smoke test data pipeline (10 docs/source)
	modal run pretrain/pipeline.py::main

# Usage: make clean-data [ARGS="--fineweb-shards <int> --only <source>"]
# Args:
#   --fineweb-shards  (optional, default=1)   Number of FineWeb-Edu shards to process
#   --only            (optional, default="")  Process only the named source (empty = all)
clean-data: ## Phase 1: stream + clean all sources
	modal run pretrain/pipeline.py::clean

# Usage: make dedup [ARGS="--compute-sigs <bool>"]
# Args:
#   --compute-sigs  (optional, default=True)  Compute MinHash signatures (skip if already done)
dedup: ## Phase 2: dedup + contamination strip
	modal run pretrain/pipeline.py::dedup

# Usage: make tokenizer
# Args: none
tokenizer: ## Phase 3: train 16K BPE tokenizer
	modal run pretrain/pipeline.py::tokenizer

# Usage: make tokenize
# Args: none
tokenize: ## Phase 4: tokenize corpus into 1024-token windows
	modal run pretrain/pipeline.py::tokenize

# Usage: make pretrain [ARGS="--n-epochs <int>"]
# Args:
#   --n-epochs  (optional, default=3)  Number of pretraining epochs
pretrain: ## Phase 5: pretrain on multi-GPU (8x H100)
	modal run pretrain/pipeline.py::pretrain

# Usage: make upload
# Args: none
upload: ## Phase 6: upload base model to HuggingFace
	modal run pretrain/pipeline.py::upload

# Usage: make eval
# Args: none
eval: ## Run final perplexity eval on GPU
	modal run pretrain/pipeline.py::final_eval

# ── SFT data generation (Modal) ───────────────────────────────────

# Usage: make sft-data-azure
# Args: none
sft-data-azure: ## Generate SFT pairs via Azure OpenAI
	modal run sft/datagen_azure.py::generate

# Usage: make sft-data-gemini
# Args: none
sft-data-gemini: ## Generate SFT pairs via Gemini Flash
	modal run sft/datagen_gemini.py::generate

# ── SFT fine-tuning ───────────────────────────────────────────────

# Usage: make sft-tokenize
# Args: none
sft-tokenize: ## Tokenize SFT dataset on Modal
	modal run sft/finetune_modal.py::tokenize_sft

# Usage: make sft-modal [ARGS="--n-epochs <int> --ckpt-every-epochs <int> --lr <float> ..."]
# Args:
#   --n-epochs          (optional, default=20)    Total epochs to train (cumulative across resumes)
#   --ckpt-every-epochs (optional, default=2)     Save checkpoint every N epochs
#   --fresh             (optional, default=False) Ignore existing checkpoints, start from base model
#   --lr                (optional, default=2e-5)  Learning rate
#   --batch-size        (optional, default=16)    Batch size
#   --grad-accum        (optional, default=2)     Gradient accumulation steps
#   --weight-decay      (optional, default=0.01)  Weight decay
sft-modal: ## SFT on Modal A100 (20 epochs, ckpt every 2)
	modal run sft/finetune_modal.py::finetune $(ARGS)

# Usage: make sft-modal-fresh [ARGS="--n-epochs <int> --lr <float> ..."]
# Args: same as sft-modal (--fresh is pre-set)
sft-modal-fresh: ## SFT from scratch (ignore existing checkpoints)
	modal run sft/finetune_modal.py::finetune --fresh $(ARGS)

# Usage: make download-sft
# Args: none
download-sft: ## Download best SFT model + tokenizer from Modal volume
	modal run remote_sft/download.py

# Usage: make sft-local [ARGS="--model <path> --resume <path> --n-epochs <int> ..."]
# Args:
#   --model             (optional, default=local_model/)  Base model directory
#   --resume            (optional, default=None)          Resume from a specific checkpoint path
#   --fresh             (optional)                        Start from scratch, ignore existing checkpoints
#   --n-epochs          (optional, default=20)            Total epochs to train
#   --batch-size        (optional, default=16)            Batch size
#   --grad-accum        (optional, default=2)             Gradient accumulation steps
#   --lr                (optional, default=2e-5)          Learning rate
#   --min-lr            (optional, default=2e-6)          Minimum learning rate
#   --weight-decay      (optional, default=0.01)          Weight decay
#   --grad-clip         (optional, default=1.0)           Gradient clipping
#   --device            (optional, default=auto)          Device (auto-selects MPS/CUDA/CPU)
#   --log-every         (optional, default=20)            Log every N steps
#   --ckpt-every-epochs (optional, default=2)             Save checkpoint every N epochs
sft-local: ## SFT on local Mac (MPS)
	python sft/finetune_local.py $(ARGS)

# Usage: make sft-local-fresh [ARGS="--n-epochs <int> --lr <float> ..."]
# Args: same as sft-local (--fresh is pre-set)
sft-local-fresh: ## SFT from scratch on local Mac (ignore existing checkpoints)
	python sft/finetune_local.py --fresh $(ARGS)

# ── Inference ─────────────────────────────────────────────────────

# Usage: make chat [ARGS="--prompt <str> --model <path> --max-tokens <int> ..."]
# Args:
#   --model        (optional, default=local_model/)  Model directory
#   --tokenizer    (optional, default=--model)       Tokenizer directory (if different from model)
#   --prompt       (optional, default=None)          Non-interactive single prompt (omit for REPL)
#   --system       (optional, default=built-in)      System prompt
#   --max-tokens   (optional, default=256)           Maximum tokens to generate
#   --temperature  (optional, default=0.7)           Sampling temperature
chat: ## Chat via PyTorch (MPS/CUDA/CPU)
	python inference/chat_pytorch.py $(ARGS)

# Usage: make chat-remote [ARGS="--prompt <str> --max-tokens <int> ..."]
# Args: same as chat (--model and --tokenizer are pre-set)
chat-remote: ## Chat via PyTorch using remote (Modal) SFT model
	python inference/chat_pytorch.py --model remote_sft/ --tokenizer remote_sft/ $(ARGS)

# Usage: make chat-coreml [ARGS="--prompt <str> --model-dir <path> --max-tokens <int> ..."]
# Args:
#   --model-dir    (optional, default=coreml_model/)  CoreML model directory
#   --prompt       (optional, default=None)           Non-interactive single prompt (omit for REPL)
#   --system       (optional, default=built-in)       System prompt
#   --max-tokens   (optional, default=256)            Maximum tokens to generate
#   --temperature  (optional, default=0.7)            Sampling temperature
chat-coreml: ## Chat via CoreML (Apple Silicon, KV cached)
	python inference/chat_coreml.py $(ARGS)

# Usage: make convert-coreml [ARGS="--model <path> --seq-len <int> --output <path>"]
# Args:
#   --model    (optional, default=local_model/)   Source model directory
#   --seq-len  (optional, default=1024)           Sequence length for the CoreML model
#   --output   (optional, default=coreml_model/)  Output directory for CoreML model
convert-coreml: ## Convert best model to CoreML with KV cache
	python inference/convert_coreml.py
