"""Chat with the CoreML SLM using KV-cached inference on Apple Silicon.

Each decode step processes a single token with the Neural Engine,
reusing cached key/value states from prior tokens.

Usage:
    python inference/chat_coreml.py                          # interactive chat
    python inference/chat_coreml.py --prompt "What is..."    # single question
"""

from __future__ import annotations

import argparse
import json
import time

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

MODEL_DIR = "local_model/sft_coreml"
MODEL_PATH = f"{MODEL_DIR}/slm-125m-sft.mlpackage"
SYSTEM_PROMPT = "You are a helpful legal and financial assistant. Answer based only on the provided context."

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def load_model(model_dir: str):
    print(f"Loading CoreML model...")
    t0 = time.time()

    with open(f"{model_dir}/model_meta.json") as f:
        meta = json.load(f)

    model = ct.models.MLModel(f"{model_dir}/slm-125m-sft.mlpackage")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    rotary_cos = np.load(f"{model_dir}/rotary_cos.npy")
    rotary_sin = np.load(f"{model_dir}/rotary_sin.npy")
    causal_mask = np.load(f"{model_dir}/causal_mask.npy")

    print(f"Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer, meta, rotary_cos, rotary_sin, causal_mask


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


def generate(model, tokenizer, meta, rotary_cos, rotary_sin, causal_mask,
             prompt_ids: list[int], max_new_tokens: int = 256,
             temperature: float = 0.7, top_k: int = 50) -> tuple[str, float, float]:
    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    num_layers = meta["num_layers"]
    num_heads = meta["num_heads"]
    head_dim = meta["head_dim"]
    max_seq = meta["max_seq_len"]

    k_cache = np.zeros((num_layers, num_heads, max_seq, head_dim), dtype=np.float16)
    v_cache = np.zeros_like(k_cache)

    def step(token_id: int, pos: int):
        nonlocal k_cache, v_cache
        cos = rotary_cos[pos:pos + 1][np.newaxis].astype(np.float16)
        sin = rotary_sin[pos:pos + 1][np.newaxis].astype(np.float16)
        mask = causal_mask[pos:pos + 1][np.newaxis, np.newaxis].astype(np.float16)
        scatter = np.full((1, num_heads, 1, head_dim), pos, dtype=np.int32)

        out = model.predict({
            "input_ids": np.array([[token_id]], dtype=np.int32),
            "cos": cos,
            "sin": sin,
            "attn_mask": mask,
            "scatter_pos": scatter,
            "k_cache": k_cache,
            "v_cache": v_cache,
        })

        k_cache = out["new_k_cache"]
        v_cache = out["new_v_cache"]
        return out["logits"][0, 0, :]

    # Prefill: process prompt tokens
    t_prefill = time.time()
    for pos, token_id in enumerate(prompt_ids):
        logits = step(token_id, pos)
    t_prefill = time.time() - t_prefill

    # Decode: generate new tokens
    t_decode = time.time()
    generated_ids = []
    pos = len(prompt_ids)

    for _ in range(max_new_tokens):
        if pos >= max_seq:
            break

        if temperature > 0:
            scaled = logits / temperature
            if top_k > 0:
                top_idx = np.argpartition(scaled, -top_k)[-top_k:]
                filtered = np.full_like(scaled, -np.inf)
                filtered[top_idx] = scaled[top_idx]
                scaled = filtered
            probs = np.exp(scaled - np.max(scaled))
            probs /= probs.sum()
            next_id = int(np.random.choice(len(probs), p=probs))
        else:
            next_id = int(np.argmax(logits))

        if next_id == eos_id:
            break

        generated_ids.append(next_id)
        logits = step(next_id, pos)
        pos += 1

    t_decode = time.time() - t_decode

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    decode_tps = len(generated_ids) / t_decode if t_decode > 0 else 0
    prefill_tps = len(prompt_ids) / t_prefill if t_prefill > 0 else 0
    return text, prefill_tps, decode_tps


def interactive(model, tokenizer, meta, rotary_cos, rotary_sin, causal_mask,
                system_prompt: str, max_tokens: int, temperature: float):
    print("\n--- SLM-125M CoreML Chat (KV cached) ---")
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

        response, prefill_tps, decode_tps = generate(
            model, tokenizer, meta, rotary_cos, rotary_sin, causal_mask,
            prompt_ids, max_new_tokens=max_tokens, temperature=temperature,
        )
        print(f"SLM: {response}")
        print(f"[prefill: {prefill_tps:.0f} tok/s | decode: {decode_tps:.0f} tok/s]\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with CoreML SLM (KV cached)")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--system", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer, meta, rot_cos, rot_sin, mask = load_model(args.model_dir)

    if args.prompt:
        prompt_ids = build_prompt(tokenizer, args.system, args.prompt)
        response, prefill_tps, decode_tps = generate(
            model, tokenizer, meta, rot_cos, rot_sin, mask,
            prompt_ids, max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(response)
        print(f"[prefill: {prefill_tps:.0f} tok/s | decode: {decode_tps:.0f} tok/s]")
    else:
        interactive(model, tokenizer, meta, rot_cos, rot_sin, mask,
                    args.system, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
