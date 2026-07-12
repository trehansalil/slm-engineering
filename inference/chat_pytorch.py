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


def load_model(model_dir: str):
    print(f"Loading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params/1e6:.1f}M params")
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
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
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
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--system", type=str, default=SYSTEM_PROMPT,
                        help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

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
