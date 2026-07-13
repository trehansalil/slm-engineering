"""Convert the SFT model to CoreML with KV cache for fast inference.

Builds a single-token decode model with explicit KV cache I/O, bypassing
HuggingFace's dynamic ops. Each inference step processes 1 token and
updates the cache — no recomputation of prior tokens.

Usage:
    python inference/convert_coreml.py
    python inference/convert_coreml.py --model local_model/sft_local
    python inference/convert_coreml.py --seq-len 512
"""

from __future__ import annotations

import argparse
import os
import shutil

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM


DEFAULT_MODEL_DIR = "local_model/sft_local"
OUTPUT_DIR = "local_model/sft_coreml"


class LlamaDecodeStep(nn.Module):
    """Single-token LLaMA decode step with explicit KV cache.

    All position-dependent values (rotary embeddings, causal mask) are
    passed as inputs so the model itself is position-independent and
    traces cleanly for CoreML.
    """

    def __init__(self, hf_model):
        super().__init__()
        m = hf_model.model
        cfg = m.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.scale = self.head_dim ** -0.5

        self.embed_tokens = m.embed_tokens
        self.final_norm = m.norm
        self.lm_head = hf_model.lm_head

        self.input_norms = nn.ModuleList()
        self.q_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()
        self.o_projs = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.gate_projs = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        self.down_projs = nn.ModuleList()

        for layer in m.layers:
            self.input_norms.append(layer.input_layernorm)
            self.q_projs.append(layer.self_attn.q_proj)
            self.k_projs.append(layer.self_attn.k_proj)
            self.v_projs.append(layer.self_attn.v_proj)
            self.o_projs.append(layer.self_attn.o_proj)
            self.post_norms.append(layer.post_attention_layernorm)
            self.gate_projs.append(layer.mlp.gate_proj)
            self.up_projs.append(layer.mlp.up_proj)
            self.down_projs.append(layer.mlp.down_proj)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, input_ids, cos, sin, attn_mask, scatter_pos,
                k_cache, v_cache):
        """
        Args:
            input_ids:    [1, 1] int — single token
            cos:          [1, 1, head_dim] — rotary cos at current position
            sin:          [1, 1, head_dim] — rotary sin at current position
            attn_mask:    [1, 1, 1, max_seq] — causal mask row for current pos
            scatter_pos:  [1, heads, 1, head_dim] int — filled with position idx
            k_cache:      [layers, heads, max_seq, head_dim] — key cache
            v_cache:      [layers, heads, max_seq, head_dim] — value cache
        Returns:
            logits:       [1, 1, vocab]
            new_k_cache:  [layers, heads, max_seq, head_dim]
            new_v_cache:  [layers, heads, max_seq, head_dim]
        """
        h = self.embed_tokens(input_ids)

        cos_4d = cos.unsqueeze(1)   # [1, 1, 1, head_dim]
        sin_4d = sin.unsqueeze(1)

        new_ks = []
        new_vs = []

        for i in range(self.num_layers):
            normed = self.input_norms[i](h)

            q = self.q_projs[i](normed).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_projs[i](normed).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_projs[i](normed).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)

            q = (q * cos_4d) + (self._rotate_half(q) * sin_4d)
            k = (k * cos_4d) + (self._rotate_half(k) * sin_4d)

            ki = k_cache[i].unsqueeze(0).scatter(2, scatter_pos, k)
            vi = v_cache[i].unsqueeze(0).scatter(2, scatter_pos, v)

            attn_w = torch.matmul(q, ki.transpose(-2, -1)) * self.scale + attn_mask
            attn_w = torch.softmax(attn_w, dim=-1)
            attn_out = torch.matmul(attn_w, vi)

            attn_out = self.o_projs[i](
                attn_out.transpose(1, 2).reshape(1, 1, self.hidden_size)
            )
            h = h + attn_out

            normed = self.post_norms[i](h)
            h = h + self.down_projs[i](
                F.silu(self.gate_projs[i](normed)) * self.up_projs[i](normed)
            )

            new_ks.append(ki.squeeze(0))
            new_vs.append(vi.squeeze(0))

        h = self.final_norm(h)
        logits = self.lm_head(h)

        return logits, torch.stack(new_ks), torch.stack(new_vs)


def precompute_rotary(hf_model, max_seq_len: int):
    """Extract rotary embeddings from the HF model."""
    m = hf_model.model
    pos_ids = torch.arange(max_seq_len).unsqueeze(0)
    dummy = torch.zeros(1, max_seq_len, m.config.hidden_size)
    with torch.no_grad():
        cos, sin = m.rotary_emb(dummy, position_ids=pos_ids)
    return cos.squeeze(0).numpy(), sin.squeeze(0).numpy()  # [max_seq, head_dim]


def precompute_causal_mask(max_seq_len: int):
    """Build a causal mask: row i attends to positions 0..i."""
    mask = np.zeros((max_seq_len, max_seq_len), dtype=np.float32)
    for i in range(max_seq_len):
        mask[i, i + 1 :] = -65000.0
    return mask  # [max_seq, max_seq]


def verify_kv_model(hf_model, kv_model, seq_len: int, max_seq_len: int):
    """Verify KV-cached model matches HF model output."""
    hf_model.eval()
    kv_model.eval()

    cos_np, sin_np = precompute_rotary(hf_model, max_seq_len)
    mask_np = precompute_causal_mask(max_seq_len)

    test_ids = torch.randint(0, 100, (1, seq_len))

    with torch.no_grad():
        ref_logits = hf_model(input_ids=test_ids).logits  # [1, seq_len, vocab]

    num_layers = kv_model.num_layers
    num_heads = kv_model.num_heads
    head_dim = kv_model.head_dim

    k_cache = torch.zeros(num_layers, num_heads, max_seq_len, head_dim)
    v_cache = torch.zeros_like(k_cache)

    with torch.no_grad():
        for pos in range(seq_len):
            token = test_ids[:, pos : pos + 1]
            cos = torch.from_numpy(cos_np[pos : pos + 1]).unsqueeze(0)
            sin = torch.from_numpy(sin_np[pos : pos + 1]).unsqueeze(0)
            amask = torch.from_numpy(
                mask_np[pos : pos + 1]
            ).unsqueeze(0).unsqueeze(0)
            scatter = torch.full(
                (1, num_heads, 1, head_dim), pos, dtype=torch.long
            )

            logits, k_cache, v_cache = kv_model(
                token, cos, sin, amask, scatter, k_cache, v_cache
            )

    max_diff = (ref_logits[0, -1, :] - logits[0, 0, :]).abs().max().item()
    top_ref = ref_logits[0, -1, :].argmax().item()
    top_kv = logits[0, 0, :].argmax().item()

    print(f"  Verification (seq_len={seq_len}):")
    print(f"    Max logit diff: {max_diff:.6f}")
    print(f"    Top token — ref: {top_ref}, kv: {top_kv} {'✓' if top_ref == top_kv else '✗'}")
    if max_diff > 0.01:
        raise ValueError(f"KV model output diverges: max_diff={max_diff}")


def convert(model_dir: str, max_seq_len: int, output_dir: str):
    print(f"Loading model from {model_dir} (eager attention)...")
    model = LlamaForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // num_heads
    print(f"Model: {n_params / 1e6:.1f}M params, {num_layers} layers, "
          f"{num_heads} heads, head_dim={head_dim}")

    print("Building KV-cached decode model...")
    kv_model = LlamaDecodeStep(model)
    kv_model.eval()

    print("Verifying correctness...")
    verify_kv_model(model, kv_model, seq_len=16, max_seq_len=max_seq_len)

    print("Tracing...")
    example_inputs = (
        torch.randint(0, 100, (1, 1)),                                  # input_ids
        torch.randn(1, 1, head_dim),                                     # cos
        torch.randn(1, 1, head_dim),                                     # sin
        torch.zeros(1, 1, 1, max_seq_len),                               # attn_mask
        torch.zeros(1, num_heads, 1, head_dim, dtype=torch.long),        # scatter_pos
        torch.zeros(num_layers, num_heads, max_seq_len, head_dim),       # k_cache
        torch.zeros(num_layers, num_heads, max_seq_len, head_dim),       # v_cache
    )
    with torch.no_grad():
        traced = torch.jit.trace(kv_model, example_inputs)

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="cos", shape=(1, 1, head_dim)),
            ct.TensorType(name="sin", shape=(1, 1, head_dim)),
            ct.TensorType(name="attn_mask", shape=(1, 1, 1, max_seq_len)),
            ct.TensorType(name="scatter_pos", shape=(1, num_heads, 1, head_dim),
                          dtype=np.int32),
            ct.TensorType(name="k_cache",
                          shape=(num_layers, num_heads, max_seq_len, head_dim)),
            ct.TensorType(name="v_cache",
                          shape=(num_layers, num_heads, max_seq_len, head_dim)),
        ],
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="new_k_cache"),
            ct.TensorType(name="new_v_cache"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    mlmodel.author = "slm-125m-sft"
    mlmodel.short_description = "125M SLM decode step with KV cache"
    mlmodel.version = "2.0"

    output_path = f"{output_dir}/slm-125m-sft.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    # Save rotary embeddings and causal mask for inference
    print("Saving rotary embeddings and causal mask...")
    cos_np, sin_np = precompute_rotary(model, max_seq_len)
    mask_np = precompute_causal_mask(max_seq_len)
    np.save(f"{output_dir}/rotary_cos.npy", cos_np)
    np.save(f"{output_dir}/rotary_sin.npy", sin_np)
    np.save(f"{output_dir}/causal_mask.npy", mask_np)

    # Save model metadata
    import json
    meta = {
        "max_seq_len": max_seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": cfg.vocab_size,
    }
    with open(f"{output_dir}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Copy tokenizer
    for tf in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = f"{model_dir}/{tf}"
        dst = f"{output_dir}/{tf}"
        try:
            shutil.copy2(src, dst)
        except FileNotFoundError:
            pass

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(output_path)
        for f in fns
    ) / 1e6
    print(f"\nDone! CoreML model: {output_path} ({size_mb:.1f} MB)")
    print(f"  KV cache shape: [{num_layers}, {num_heads}, {max_seq_len}, {head_dim}]")
    print(f"  Cache memory: {num_layers * num_heads * max_seq_len * head_dim * 2 * 2 / 1e6:.1f} MB")
    print(f"\nTo run inference: python inference/chat_coreml.py")


def main():
    parser = argparse.ArgumentParser(description="Convert SLM to CoreML with KV cache")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    convert(args.model, args.seq_len, args.output)


if __name__ == "__main__":
    main()
