"""Chat with the fine-tuned 125M SLM locally on Mac.

Usage:
    python inference/chat_pytorch.py                          # interactive chat
    python inference/chat_pytorch.py --prompt "What is..."    # single question
    python inference/chat_pytorch.py --model local_model/sft_best  # custom path

Requires: pip install torch transformers
"""

from __future__ import annotations

import argparse
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

DEFAULT_MODEL_DIR = "local_model/sft_best"
SYSTEM_PROMPT = "You are a helpful legal and financial assistant. Answer based only on the provided context."

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_dir: str, tokenizer_dir: str | None = None):
    print(f"Loading model from {model_dir}...")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or model_dir)
    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
    )
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.1f}M params on {device}")
    return model, tokenizer


def build_prompt(tokenizer, system: str, user: str) -> list[int]:
    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    ids = []
    for role, content in [("system", system), ("user", user)]:
        role_id = tokenizer.convert_tokens_to_ids(ROLE_TOKENS[role])
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        ids.extend([role_id] + content_ids + [eos_id])
    assistant_id = tokenizer.convert_tokens_to_ids(ROLE_TOKENS["assistant"])
    ids.append(assistant_id)
    return ids


@torch.no_grad()
def generate(model, tokenizer, prompt_ids: list[int],
             max_new_tokens: int = 256, temperature: float = 0.7,
             top_k: int = 50, top_p: float = 0.9) -> str:
    device = next(model.parameters()).device
    max_ctx = getattr(model.config, "max_position_embeddings", 1024)
    if len(prompt_ids) >= max_ctx:
        print(f"Warning: prompt ({len(prompt_ids)} tokens) fills the context window ({max_ctx}), truncating prompt")
        prompt_ids = prompt_ids[-(max_ctx - 2):]
    if len(prompt_ids) + max_new_tokens > max_ctx:
        max_new_tokens = max(1, max_ctx - len(prompt_ids))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    do_sample = temperature > 0

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        eos_token_id=eos_id,
        pad_token_id=tokenizer.convert_tokens_to_ids("<|pad|>"),
    )

    new_tokens = output[0][len(prompt_ids):]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def interactive(model, tokenizer, system_prompt: str, max_tokens: int,
                temperature: float):
    print("\n--- SLM-125M SFT Chat ---")
    print("Type your question (or 'quit' to exit)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        prompt_ids = build_prompt(tokenizer, system_prompt, user_input)
        print(f"[{len(prompt_ids)} prompt tokens]")

        response = generate(model, tokenizer, prompt_ids,
                            max_new_tokens=max_tokens,
                            temperature=temperature)
        print(f"SLM: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with SLM-125M SFT model")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR,
                        help="Path to model directory")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer directory (defaults to --model)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--system", type=str, default=SYSTEM_PROMPT,
                        help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.tokenizer)

    if args.prompt:
        prompt_ids = build_prompt(tokenizer, args.system, args.prompt)
        response = generate(model, tokenizer, prompt_ids,
                            max_new_tokens=args.max_tokens,
                            temperature=args.temperature)
        print(response)
    else:
        interactive(model, tokenizer, args.system, args.max_tokens,
                    args.temperature)


if __name__ == "__main__":
    main()
