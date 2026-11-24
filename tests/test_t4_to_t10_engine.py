"""
Synthetic Unit & Plumbing Tests (T4–T10) for ZAQERIS / QERLIO Native GGUF Inference Engine.

NOTE (v0.3.1 honesty requirement):
All tests in this module construct and execute a tiny deterministic **synthetic / toy 2-layer
Seed-OSS GGUF v3** fixture with planted output strings ("The calculated answer is 42.",
"Analyzing the query step by step.", and a get_weather tool call for Paris). These are unit and
API/plumbing tests only; they do NOT constitute real 36B model execution or Section 2 "tested"
evidence. Real-weight Hermes-4.3-36B tests live in `tests/test_real_hermes_36b_optin.py`
(opt-in via `QERLIO_RUN_REAL_HERMES=1`) and `scripts/verify_hermes_36b_real.py`.
- Synthetic T4: Forward-pass parity on the tiny 2-layer fixture
- Synthetic T5: 64-token greedy parity on the tiny 2-layer fixture and sampler math
- Synthetic T6: KV cache equivalence on the tiny 2-layer fixture
- Synthetic T7: Multi-turn chat and thinking-tag plumbing on the toy GGUF
- Synthetic T8: Toy GGUF tool-call plumbing (<tool_call>...</tool_call>) and schema mapping
- Synthetic T9: Error handling, context bounds, stop sequences, and cancellation cleanup
- Static Guard Audit: C++20 source-string bounds audit and C ASan/UBSan CI coverage check
  (actual sanitizer execution runs in the CI `cpp-sanitizers` job)
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest

from core.inference.dequant import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q6_K,
)
from core.inference.gguf_loader import GGUFLoader
from core.inference.kv_cache import (
    ContextLengthExceededError,
    InsufficientMemoryError,
    KVCache,
)
from core.inference.sampler import Sampler, SamplerConfig
from core.inference.seed_oss_model import (
    SeedOssConfig,
    SeedOssEngine,
    SeedOssForCausalLM,
    UnsupportedArchitectureError,
)
from core.inference.tokenizer import _bytes_to_unicode

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "seed_oss_reference_logits_and_tokens.json"


# ---------------------------------------------------------------------------
# Deterministic GGUF v3 Q4_K / Q6_K / F32 checkpoint builder (pure NumPy + stdlib)
# ---------------------------------------------------------------------------


def _quantize_q4_k_numpy(arr_f32: np.ndarray) -> bytes:
    flat = np.asarray(arr_f32, dtype=np.float32).reshape(-1)
    assert flat.size % 256 == 0
    n_blocks = flat.size // 256
    blocks = flat.reshape(n_blocks, 8, 32)

    out = bytearray(n_blocks * 144)
    for b in range(n_blocks):
        sub = blocks[b]
        sub_mins = np.maximum(0.0, -np.min(sub, axis=1))
        shifted = sub + sub_mins[:, None]
        sub_maxs = np.max(shifted, axis=1)
        sub_scales = sub_maxs / 15.0

        max_scale = float(np.max(sub_scales))
        max_min = float(np.max(sub_mins))

        d = np.float16(max_scale / 63.0 if max_scale > 0 else 0.0)
        dmin = np.float16(max_min / 63.0 if max_min > 0 else 0.0)
        d_f32 = float(d)
        dmin_f32 = float(dmin)

        sc = np.zeros(8, dtype=np.uint8)
        m = np.zeros(8, dtype=np.uint8)
        for j in range(8):
            if d_f32 > 0:
                sc[j] = int(np.clip(np.round(sub_scales[j] / d_f32), 0, 63))
            if dmin_f32 > 0:
                m[j] = int(np.clip(np.round(sub_mins[j] / dmin_f32), 0, 63))

        scales_12 = bytearray(12)
        for j in range(4):
            scales_12[j] = int((sc[j] & 63) | ((sc[j + 4] >> 4) << 6))
            scales_12[j + 4] = int((m[j] & 63) | ((m[j + 4] >> 4) << 6))
            scales_12[j + 8] = int((sc[j + 4] & 0x0F) | ((m[j + 4] & 0x0F) << 4))

        qs_128 = bytearray(128)
        for c in range(4):
            d1 = d_f32 * float(sc[2 * c])
            m1 = dmin_f32 * float(m[2 * c])
            d2 = d_f32 * float(sc[2 * c + 1])
            m2 = dmin_f32 * float(m[2 * c + 1])

            v1 = sub[2 * c]
            v2 = sub[2 * c + 1]
            for l in range(32):
                q_low = int(np.clip(np.round((v1[l] + m1) / d1), 0, 15)) if d1 > 0 else 0
                q_high = int(np.clip(np.round((v2[l] + m2) / d2), 0, 15)) if d2 > 0 else 0
                qs_128[c * 32 + l] = (q_low & 0x0F) | ((q_high & 0x0F) << 4)

        base = b * 144
        out[base : base + 2] = struct.pack("<e", d)
        out[base + 2 : base + 4] = struct.pack("<e", dmin)
        out[base + 4 : base + 16] = scales_12
        out[base + 16 : base + 144] = qs_128

    return bytes(out)


def _quantize_q6_k_numpy(arr_f32: np.ndarray) -> bytes:
    flat = np.asarray(arr_f32, dtype=np.float32).reshape(-1)
    assert flat.size % 256 == 0
    n_blocks = flat.size // 256
    blocks = flat.reshape(n_blocks, 16, 16)

    out = bytearray(n_blocks * 210)
    for b in range(n_blocks):
        sub = blocks[b]
        abs_max = np.max(np.abs(sub), axis=1)
        sub_scales = abs_max / 31.0
        max_sc = float(np.max(sub_scales))
        d = np.float16(max_sc / 127.0 if max_sc > 0 else 0.0)
        d_f32 = float(d)

        sc_i8 = np.zeros(16, dtype=np.int8)
        for j in range(16):
            if d_f32 > 0:
                sc_i8[j] = int(np.clip(np.round(sub_scales[j] / d_f32), 1, 127))
            else:
                sc_i8[j] = 1

        vals_256 = blocks[b].reshape(256)
        ql = bytearray(128)
        qh = bytearray(64)

        for n in range(2):
            sc_half = sc_i8[n * 8 : (n + 1) * 8]
            for l in range(32):
                is_sub = l // 16
                s1 = d_f32 * float(sc_half[is_sub + 0])
                s2 = d_f32 * float(sc_half[is_sub + 2])
                s3 = d_f32 * float(sc_half[is_sub + 4])
                s4 = d_f32 * float(sc_half[is_sub + 6])

                v1 = vals_256[n * 128 + l + 0]
                v2 = vals_256[n * 128 + l + 32]
                v3 = vals_256[n * 128 + l + 64]
                v4 = vals_256[n * 128 + l + 96]

                q1 = int(np.clip(np.round(v1 / s1) + 32, 0, 63)) if s1 != 0 else 32
                q2 = int(np.clip(np.round(v2 / s2) + 32, 0, 63)) if s2 != 0 else 32
                q3 = int(np.clip(np.round(v3 / s3) + 32, 0, 63)) if s3 != 0 else 32
                q4 = int(np.clip(np.round(v4 / s4) + 32, 0, 63)) if s4 != 0 else 32

                ql[n * 64 + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4)
                ql[n * 64 + 32 + l] = (q2 & 0x0F) | ((q4 & 0x0F) << 4)
                qh[n * 32 + l] = (
                    ((q1 >> 4) & 3)
                    | (((q2 >> 4) & 3) << 2)
                    | (((q3 >> 4) & 3) << 4)
                    | (((q4 >> 4) & 3) << 6)
                )

        base = b * 210
        out[base : base + 128] = ql
        out[base + 128 : base + 192] = qh
        out[base + 192 : base + 208] = struct.pack("<16b", *[int(x) for x in sc_i8])
        out[base + 208 : base + 210] = struct.pack("<e", d)

    return bytes(out)


def _build_vocab_and_merges() -> Tuple[List[str], List[int], List[str]]:
    special_entries = [
        ("<seed:bos>", 3),
        ("<seed:pad>", 3),
        ("<seed:eos>", 3),
        ("<seed:think>", 4),
        ("</seed:think>", 4),
        ("<|eot_id|>", 3),
        ("<|start_header_id|>", 3),
        ("<|end_header_id|>", 3),
        ("<think>", 4),
        ("</think>", 4),
        ("<tool_call>", 4),
        ("</tool_call>", 4),
        ("<tools>", 4),
        ("</tools>", 4),
        ("<tool_response>", 4),
        ("</tool_response>", 4),
    ]
    tokens: List[str] = [t for t, _ in special_entries]
    token_types: List[int] = [tt for _, tt in special_entries]

    byte_enc = _bytes_to_unicode()
    for b in range(256):
        tokens.append(byte_enc[b])
        token_types.append(1)

    def _map_ascii(s: str) -> str:
        return "".join(byte_enc[b] for b in s.encode("utf-8"))

    merges: List[str] = []

    def _add_merge_seq(word_ascii: str) -> str:
        mapped = [_map_ascii(ch) for ch in word_ascii]
        cur = mapped[0]
        for nxt in mapped[1:]:
            pair_str = f"{cur} {nxt}"
            merged = cur + nxt
            if merged not in tokens:
                tokens.append(merged)
                token_types.append(1)
                merges.append(pair_str)
            cur = merged
        return cur

    for w in ["system", "user", "assistant", "Hello", "Paris", "weather", "42", "think", "step", "\n\n"]:
        _add_merge_seq(w)

    extra_user_tokens = [
        '{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}',
        "Analyzing the query step by step.",
        "The calculated answer is 42.",
        "The weather in Paris is 21 celsius.",
    ]
    for ut in extra_user_tokens:
        if len(tokens) < 320:
            tokens.append(ut)
            token_types.append(4)

    while len(tokens) < 320:
        idx = len(tokens)
        tokens.append(f"[PAD{idx}]")
        token_types.append(5)

    return tokens, token_types, merges


def _create_deterministic_seed_oss_weights() -> Dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(20260301))

    vocab_size = 320
    hidden_size = 256
    intermediate_size = 512
    num_layers = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    q_mat, _ = np.linalg.qr(rng.standard_normal((hidden_size, hidden_size)).astype(np.float64))
    q_mat2, _ = np.linalg.qr(rng.standard_normal((hidden_size, hidden_size)).astype(np.float64))
    unit_l2 = np.vstack([q_mat.T, q_mat2.T[: vocab_size - hidden_size]]).astype(np.float32)
    unit_l2 = unit_l2 / np.linalg.norm(unit_l2, axis=1, keepdims=True)
    token_embd = (unit_l2 * math.sqrt(hidden_size)).astype(np.float32)

    next_token_map = np.arange(vocab_size, dtype=np.int64)
    cycle_tokens = [t for t in range(16, 276)]
    for idx_i, src_tok in enumerate(cycle_tokens):
        dst_tok = cycle_tokens[(idx_i + 1) % len(cycle_tokens)]
        next_token_map[src_tok] = dst_tok

    for t in range(0, 16):
        next_token_map[t] = 16 + (t * 11) % 240
    for t in range(276, vocab_size):
        next_token_map[t] = 16 + ((t - 260) * 13) % 240

    tokens_list, _, _ = _build_vocab_and_merges()
    byte_enc = _bytes_to_unicode()
    nn_mapped = "".join(byte_enc[b] for b in "\n\n".encode("utf-8"))
    tok_nn = tokens_list.index(nn_mapped)
    tok_tool_json = tokens_list.index(
        '{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}'
    )
    tok_think_body = tokens_list.index("Analyzing the query step by step.")
    tok_answer_body = tokens_list.index("The calculated answer is 42.")

    structured_links = [
        (tok_nn, tok_answer_body),
        (tok_answer_body, 5),
        (318, 8),
        (8, tok_think_body),
        (tok_think_body, 9),
        (9, tok_answer_body),
        (319, 10),
        (10, tok_tool_json),
        (tok_tool_json, 11),
        (11, 5),
    ]
    for src_t, dst_t in structured_links:
        next_token_map[src_t] = dst_t

    output_weight = (
        rng.standard_normal((vocab_size, hidden_size), dtype=np.float32) * 0.002
    ).astype(np.float32)
    inv_h = 1.0 / float(hidden_size)
    for src_t in range(vocab_size):
        dst1 = int(next_token_map[src_t])
        output_weight[dst1] += (2.40 * inv_h) * token_embd[src_t]
        for rank_idx, coeff in enumerate((1.35, 1.00, 0.70, 0.40), start=1):
            runner_up = 16 + ((dst1 - 16 + rank_idx * 47) % 240)
            if runner_up != dst1:
                output_weight[runner_up] += (coeff * inv_h) * token_embd[src_t]

    for src_t, dst_t in structured_links:
        output_weight[dst_t] = (
            rng.standard_normal(hidden_size, dtype=np.float32) * 0.001
        ).astype(np.float32)
    for src_t, dst_t in structured_links:
        output_weight[dst_t] += (3.00 * inv_h) * token_embd[src_t]
    output_weight[8] = (4.20 * inv_h) * token_embd[318]
    output_weight[10] = (4.20 * inv_h) * token_embd[319]

    output_norm = np.ones(hidden_size, dtype=np.float32) + (
        rng.uniform(-0.05, 0.05, size=hidden_size).astype(np.float32)
    )

    layers = []
    for l_idx in range(num_layers):
        scale = 0.04 / (l_idx + 1)
        layer_dict = {
            "attn_norm.weight": np.ones(hidden_size, dtype=np.float32)
            + rng.uniform(-0.05, 0.05, size=hidden_size).astype(np.float32),
            "attn_q.weight": (
                rng.standard_normal((q_dim, hidden_size), dtype=np.float32) * scale
            ).astype(np.float32),
            "attn_q.bias": (rng.standard_normal(q_dim, dtype=np.float32) * 0.01).astype(
                np.float32
            ),
            "attn_k.weight": (
                rng.standard_normal((kv_dim, hidden_size), dtype=np.float32) * scale
            ).astype(np.float32),
            "attn_k.bias": (rng.standard_normal(kv_dim, dtype=np.float32) * 0.01).astype(
                np.float32
            ),
            "attn_v.weight": (
                rng.standard_normal((kv_dim, hidden_size), dtype=np.float32) * scale
            ).astype(np.float32),
            "attn_v.bias": (rng.standard_normal(kv_dim, dtype=np.float32) * 0.01).astype(
                np.float32
            ),
            "attn_output.weight": (
                rng.standard_normal((hidden_size, q_dim), dtype=np.float32) * scale
            ).astype(np.float32),
            "post_attention_norm.weight": np.ones(hidden_size, dtype=np.float32)
            + rng.uniform(-0.05, 0.05, size=hidden_size).astype(np.float32),
            "ffn_gate.weight": (
                rng.standard_normal((intermediate_size, hidden_size), dtype=np.float32) * scale
            ).astype(np.float32),
            "ffn_up.weight": (
                rng.standard_normal((intermediate_size, hidden_size), dtype=np.float32) * scale
            ).astype(np.float32),
            "ffn_down.weight": (
                rng.standard_normal((hidden_size, intermediate_size), dtype=np.float32) * scale
            ).astype(np.float32),
        }
        layers.append(layer_dict)

    l0 = layers[0]
    l0["attn_norm.weight"][:] = 1.0
    l0["attn_q.weight"][31, :] = (24.0 * inv_h) * token_embd[tok_nn]
    l0["attn_q.bias"][31] = -8.0
    l0["attn_k.weight"][31, :] = (
        (24.0 * inv_h) * token_embd[12]
        + (16.0 * inv_h) * token_embd[8]
        + (4.0 * inv_h) * token_embd[0]
    )
    l0["attn_k.bias"][31] = 0.0
    l0["attn_v.weight"][0, :] = (4.0 * inv_h) * token_embd[8]
    l0["attn_v.weight"][1, :] = (4.0 * inv_h) * token_embd[12]
    l0["attn_v.bias"][0] = 0.0
    l0["attn_v.bias"][1] = 0.0
    l0["attn_output.weight"][:, 0] = 2.5 * token_embd[318]
    l0["attn_output.weight"][:, 1] = 2.5 * token_embd[319]

    return {
        "config": {
            "architecture": "seed_oss",
            "name": "SeedOss-36B-Micro-Reference",
            "block_count": num_layers,
            "context_length": 2048,
            "embedding_length": hidden_size,
            "feed_forward_length": intermediate_size,
            "head_count": num_heads,
            "head_count_kv": num_kv_heads,
            "key_length": head_dim,
            "value_length": head_dim,
            "rope_freq_base": 10000000.0,
            "rms_norm_eps": 1e-6,
            "vocab_size": vocab_size,
            "bos_token_id": 0,
            "eos_token_id": 5,
            "pad_token_id": 1,
        },
        "token_embd.weight": token_embd,
        "output_norm.weight": output_norm,
        "output.weight": output_weight,
        "layers": layers,
    }


def write_reference_gguf_v3(
    path: Path,
    model_data: Dict[str, Any],
    architecture_override: str | None = None,
) -> None:
    cfg = model_data["config"]
    arch = architecture_override or cfg["architecture"]
    tokens, token_types, merges = _build_vocab_and_merges()

    def _pack_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kvs: List[Tuple[str, int, bytes]] = []

    def _add_str(k: str, v: str) -> None:
        kvs.append((k, 8, _pack_str(v)))

    def _add_u32(k: str, v: int) -> None:
        kvs.append((k, 4, struct.pack("<I", int(v))))

    def _add_f32(k: str, v: float) -> None:
        kvs.append((k, 6, struct.pack("<f", float(v))))

    def _add_str_array(k: str, arr: Sequence[str]) -> None:
        buf = bytearray(struct.pack("<IQ", 8, len(arr)))
        for item in arr:
            buf.extend(_pack_str(item))
        kvs.append((k, 9, bytes(buf)))

    def _add_i32_array(k: str, arr: Sequence[int]) -> None:
        buf = struct.pack("<IQ", 5, len(arr)) + struct.pack(f"<{len(arr)}i", *arr)
        kvs.append((k, 9, buf))

    _add_str("general.architecture", arch)
    _add_str("general.type", "model")
    _add_str("general.name", cfg["name"])
    _add_u32("general.alignment", 32)
    _add_u32("general.file_type", 15)
    _add_u32("general.quantization_version", 2)

    _add_u32(f"{arch}.block_count", cfg["block_count"])
    _add_u32(f"{arch}.context_length", cfg["context_length"])
    _add_u32(f"{arch}.embedding_length", cfg["embedding_length"])
    _add_u32(f"{arch}.feed_forward_length", cfg["feed_forward_length"])
    _add_u32(f"{arch}.attention.head_count", cfg["head_count"])
    _add_u32(f"{arch}.attention.head_count_kv", cfg["head_count_kv"])
    _add_u32(f"{arch}.attention.key_length", cfg["key_length"])
    _add_u32(f"{arch}.attention.value_length", cfg["value_length"])
    _add_f32(f"{arch}.rope.freq_base", cfg["rope_freq_base"])
    _add_f32(f"{arch}.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])

    _add_str("tokenizer.ggml.model", "gpt2")
    _add_str("tokenizer.ggml.pre", "seed-coder")
    _add_str_array("tokenizer.ggml.tokens", tokens)
    _add_i32_array("tokenizer.ggml.token_type", token_types)
    _add_str_array("tokenizer.ggml.merges", merges)
    _add_u32("tokenizer.ggml.bos_token_id", cfg["bos_token_id"])
    _add_u32("tokenizer.ggml.eos_token_id", cfg["eos_token_id"])
    _add_u32("tokenizer.ggml.padding_token_id", cfg["pad_token_id"])

    tensor_specs: List[Tuple[str, np.ndarray, int]] = [
        ("output.weight", model_data["output.weight"], GGML_TYPE_Q6_K),
        ("output_norm.weight", model_data["output_norm.weight"], GGML_TYPE_F32),
        ("token_embd.weight", model_data["token_embd.weight"], GGML_TYPE_Q4_K),
    ]
    for i, layer in enumerate(model_data["layers"]):
        p = f"blk.{i}."
        tensor_specs.extend(
            [
                (f"{p}attn_k.bias", layer["attn_k.bias"], GGML_TYPE_F32),
                (f"{p}attn_k.weight", layer["attn_k.weight"], GGML_TYPE_Q4_K),
                (f"{p}attn_norm.weight", layer["attn_norm.weight"], GGML_TYPE_F32),
                (f"{p}attn_output.weight", layer["attn_output.weight"], GGML_TYPE_Q4_K),
                (f"{p}attn_q.bias", layer["attn_q.bias"], GGML_TYPE_F32),
                (f"{p}attn_q.weight", layer["attn_q.weight"], GGML_TYPE_Q4_K),
                (f"{p}attn_v.bias", layer["attn_v.bias"], GGML_TYPE_F32),
                (f"{p}attn_v.weight", layer["attn_v.weight"], GGML_TYPE_Q6_K),
                (f"{p}ffn_down.weight", layer["ffn_down.weight"], GGML_TYPE_Q6_K),
                (f"{p}ffn_gate.weight", layer["ffn_gate.weight"], GGML_TYPE_Q4_K),
                (f"{p}ffn_up.weight", layer["ffn_up.weight"], GGML_TYPE_Q4_K),
                (
                    f"{p}post_attention_norm.weight",
                    layer["post_attention_norm.weight"],
                    GGML_TYPE_F32,
                ),
            ]
        )

    alignment = 32
    encoded_tensors: List[Tuple[str, Tuple[int, ...], int, int, bytes]] = []
    current_offset = 0
    for tname, arr, gtype in tensor_specs:
        arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
        gguf_dims = tuple(reversed(arr_f32.shape))
        if gtype == GGML_TYPE_F32:
            raw = arr_f32.tobytes()
        elif gtype == GGML_TYPE_Q4_K:
            raw = _quantize_q4_k_numpy(arr_f32)
        elif gtype == GGML_TYPE_Q6_K:
            raw = _quantize_q6_k_numpy(arr_f32)
        else:
            raise ValueError(f"Unexpected tensor quantization type: {gtype}")

        encoded_tensors.append((tname, gguf_dims, gtype, current_offset, raw))
        padded_len = ((len(raw) + alignment - 1) // alignment) * alignment
        current_offset += padded_len

    header_buf = bytearray()
    header_buf.extend(b"GGUF")
    header_buf.extend(struct.pack("<IQQ", 3, len(encoded_tensors), len(kvs)))

    for k, vtype, vbytes in kvs:
        header_buf.extend(_pack_str(k))
        header_buf.extend(struct.pack("<I", vtype))
        header_buf.extend(vbytes)

    for tname, gguf_dims, gtype, offset, _raw in encoded_tensors:
        header_buf.extend(_pack_str(tname))
        header_buf.extend(struct.pack("<I", len(gguf_dims)))
        for d in gguf_dims:
            header_buf.extend(struct.pack("<Q", int(d)))
        header_buf.extend(struct.pack("<IQ", int(gtype), int(offset)))

    header_end = len(header_buf)
    data_offset = ((header_end + alignment - 1) // alignment) * alignment
    header_buf.extend(b"\x00" * (data_offset - header_end))

    with open(path, "wb") as f:
        f.write(header_buf)
        for _tname, _dims, _gtype, offset, raw in encoded_tensors:
            f.seek(data_offset + offset)
            f.write(raw)
            pad = ((len(raw) + alignment - 1) // alignment) * alignment - len(raw)
            if pad > 0:
                f.write(b"\x00" * pad)


@pytest.fixture(scope="module")
def reference_fixture() -> Dict[str, Any]:
    assert FIXTURE_PATH.exists(), f"Missing reference fixture: {FIXTURE_PATH}"
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference_gguf_path(tmp_path_factory: pytest.TempPathFactory, reference_fixture: Dict[str, Any]) -> Path:
    tmp_dir = tmp_path_factory.mktemp("seed_oss_gguf")
    gguf_file = tmp_dir / "seed_oss_reference.gguf"
    model_data = _create_deterministic_seed_oss_weights()
    write_reference_gguf_v3(gguf_file, model_data)
    sha = hashlib.sha256(gguf_file.read_bytes()).hexdigest()
    assert sha == reference_fixture["generator"]["gguf_sha256"], (
        f"Deterministic GGUF SHA-256 mismatch: expected {reference_fixture['generator']['gguf_sha256']}, got {sha}"
    )
    return gguf_file


# ---------------------------------------------------------------------------
# Strict Rule Verification: Zero Prohibited Third-Party Inference Libraries
# ---------------------------------------------------------------------------


class TestStrictDependencyPolicy:
    """Verify core/inference/ modules use only Python stdlib and NumPy."""

    FORBIDDEN_MODULES = {
        "torch",
        "transformers",
        "llama_cpp",
        "gguf",
        "vllm",
        "onnxruntime",
        "safetensors",
        "tiktoken",
        "tokenizers",
        "sentencepiece",
    }

    def test_no_forbidden_imports_in_core_inference(self) -> None:
        inference_dir = REPO_ROOT / "core" / "inference"
        py_files = sorted(inference_dir.glob("*.py"))
        assert len(py_files) >= 5
        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_mod = alias.name.split(".")[0]
                        assert root_mod not in self.FORBIDDEN_MODULES, (
                            f"Forbidden import {alias.name!r} in {py_file}"
                        )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root_mod = node.module.split(".")[0]
                    assert root_mod not in self.FORBIDDEN_MODULES, (
                        f"Forbidden from-import {node.module!r} in {py_file}"
                    )


# ---------------------------------------------------------------------------
# T4: Single Forward Pass Parity against HuggingFace Reference Fixture
# ---------------------------------------------------------------------------


class TestSyntheticT4ForwardPassParity:
    """
    Synthetic T4: Single forward pass on 5 fixed prompts using the toy 2-layer GGUF fixture.
    (Unit parity check; real-weight parity is verified in tests/test_real_hermes_36b_optin.py.)
    """

    def test_dynamic_hyperparameter_loading_from_gguf(
        self, reference_gguf_path: Path, reference_fixture: Dict[str, Any]
    ) -> None:
        expected_cfg = reference_fixture["model_config"]
        with SeedOssEngine(reference_gguf_path) as engine:
            cfg = engine.config
            assert isinstance(cfg, SeedOssConfig)
            assert cfg.architecture == "seed_oss"
            assert cfg.block_count == expected_cfg["block_count"]
            assert cfg.context_length == expected_cfg["context_length"]
            assert cfg.embedding_length == expected_cfg["embedding_length"]
            assert cfg.feed_forward_length == expected_cfg["feed_forward_length"]
            assert cfg.head_count == expected_cfg["head_count"]
            assert cfg.head_count_kv == expected_cfg["head_count_kv"]
            assert cfg.key_length == expected_cfg["key_length"]
            assert cfg.value_length == expected_cfg["value_length"]
            assert math.isclose(cfg.rope_freq_base, expected_cfg["rope_freq_base"], rel_tol=1e-6)
            assert math.isclose(cfg.rms_norm_eps, expected_cfg["rms_norm_eps"], rel_tol=1e-5)
            assert cfg.vocab_size == expected_cfg["vocab_size"]

    def test_forward_pass_5_prompts_parity(
        self, reference_gguf_path: Path, reference_fixture: Dict[str, Any]
    ) -> None:
        prompts = reference_fixture["prompts"]
        tolerances = reference_fixture["quantization_tolerances"]
        assert len(prompts) == 5

        top1_matches = 0
        with SeedOssEngine(reference_gguf_path) as engine:
            for item in prompts:
                token_ids = engine.tokenizer.encode(
                    item["prompt_text"], add_bos=False, parse_special=True
                )
                assert token_ids == item["prompt_token_ids"], (
                    f"Tokenization mismatch on {item['id']}"
                )

                logits_2d = engine.forward(token_ids)
                assert logits_2d.shape == (len(token_ids), engine.config.vocab_size)
                assert logits_2d.dtype == np.float32
                last_logits = logits_2d[-1]

                # Also verify last_token_only=True produces the exact same 1D vector
                last_only = engine.forward(token_ids, last_token_only=True)
                assert np.allclose(last_logits, last_only, atol=1e-6)

                ref_logits = np.array(item["reference_logits"], dtype=np.float32)
                f32_logits = np.array(item["unquantized_f32_logits"], dtype=np.float32)

                max_diff_ref = float(np.max(np.abs(last_logits - ref_logits)))
                max_diff_f32 = float(np.max(np.abs(last_logits - f32_logits)))

                assert max_diff_ref <= tolerances["gguf_dequant_vs_hf_max_abs_tol"], (
                    f"Prompt {item['id']} max diff vs HF GGUF reference {max_diff_ref} "
                    f"exceeds {tolerances['gguf_dequant_vs_hf_max_abs_tol']}"
                )
                assert max_diff_f32 <= tolerances["unquantized_f32_vs_q4k_q6k_max_abs_tol"], (
                    f"Prompt {item['id']} max diff vs unquantized F32 {max_diff_f32} "
                    f"exceeds {tolerances['unquantized_f32_vs_q4k_q6k_max_abs_tol']}"
                )

                pred_top5 = [int(x) for x in np.argsort(last_logits)[::-1][:5]]
                pred_top1 = pred_top5[0]
                if pred_top1 == item["reference_top1_token_id"]:
                    top1_matches += 1

                overlap_ref = len(set(pred_top5) & set(item["reference_top5_token_ids"]))
                overlap_f32 = len(set(pred_top5) & set(item["unquantized_f32_top5_token_ids"]))
                assert overlap_ref == 5
                assert overlap_f32 >= tolerances["min_top5_overlap"], (
                    f"Prompt {item['id']} top-5 overlap vs unquantized F32 is {overlap_f32}/5 (< 4/5)"
                )

        assert top1_matches == 5, f"Expected 5/5 top-1 matches, got {top1_matches}/5"

    def test_lazy_mmap_vs_preloaded_weights_exact_parity(
        self, reference_gguf_path: Path, reference_fixture: Dict[str, Any]
    ) -> None:
        # Preloaded weights use NumPy matmul, lazy weights use the C GEMM kernels; the two
        # accumulate in different orders, so float32 results agree to rounding error, not bitwise.
        token_ids = reference_fixture["prompts"][0]["prompt_token_ids"]
        with SeedOssEngine(reference_gguf_path, preload_weights=True) as eng_preloaded, SeedOssEngine(
            reference_gguf_path, preload_weights=False
        ) as eng_lazy:
            logits_pre = eng_preloaded.forward(token_ids)
            logits_lazy = eng_lazy.forward(token_ids)
            np.testing.assert_allclose(logits_lazy, logits_pre, atol=1e-5, rtol=1e-4)
            assert np.array_equal(np.argmax(logits_pre, axis=-1), np.argmax(logits_lazy, axis=-1))


# ---------------------------------------------------------------------------
# T5: Greedy 64-Token Generation Parity & Sampler Verification
# ---------------------------------------------------------------------------


class TestSyntheticT5GreedyGenerationAndSampler:
    """
    Synthetic T5: 64-token greedy parity on the toy 2-layer GGUF fixture and unit sampler checks.
    """

    def test_greedy_64_tokens_on_all_5_prompts(
        self, reference_gguf_path: Path, reference_fixture: Dict[str, Any]
    ) -> None:
        prompts = reference_fixture["prompts"]
        greedy_cfg = SamplerConfig(temperature=0.0, top_k=1)

        with SeedOssEngine(reference_gguf_path) as engine:
            for item in prompts:
                gen = engine.generate(
                    item["prompt_token_ids"],
                    sampler_config=greedy_cfg,
                    max_new_tokens=64,
                    stop_token_ids=[],
                )
                generated_ids = gen["token_ids"]
                assert len(generated_ids) == 64
                assert generated_ids == item["reference_greedy_64_tokens"], (
                    f"Greedy 64-token mismatch on {item['id']} vs HF GGUF reference"
                )
                assert generated_ids == item["unquantized_f32_greedy_64_tokens"], (
                    f"Greedy 64-token mismatch on {item['id']} vs HF unquantized F32 reference"
                )
                assert gen["finish_reason"] == "length"

    def test_sampler_temperature_topk_topp_and_repeat_penalty(self) -> None:
        logits = np.array([1.0, 4.0, 3.0, 2.0, -2.0], dtype=np.float32)

        # Greedy decoding via temperature=0.0 or top_k=1
        greedy_sampler = Sampler(SamplerConfig(temperature=0.0, top_k=20))
        assert greedy_sampler.sample(logits) == 1

        topk1_sampler = Sampler(SamplerConfig(temperature=0.8, top_k=1))
        assert topk1_sampler.sample(logits) == 1

        # Repetition penalty reduces positive logit (> 0 divided by penalty) and penalizes negative (< 0 multiplied by penalty)
        penalized = Sampler.apply_repetition_penalty(logits, previous_tokens=[1, 4], penalty=2.0)
        assert math.isclose(penalized[1], 2.0, rel_tol=1e-6)
        assert math.isclose(penalized[4], -4.0, rel_tol=1e-6)

        # Top-k=2 keeps only indices 1 and 2
        s_topk2 = Sampler(SamplerConfig(temperature=1.0, top_k=2, top_p=1.0, seed=42))
        dist_topk2 = s_topk2.sample_distribution(logits)
        assert dist_topk2[0] == 0.0 and dist_topk2[3] == 0.0 and dist_topk2[4] == 0.0
        assert math.isclose(float(np.sum(dist_topk2)), 1.0, rel_tol=1e-6)

        # Top-p=0.50 keeps only top token when top token has probability > 0.50
        s_topp = Sampler(SamplerConfig(temperature=0.5, top_k=0, top_p=0.50, seed=42))
        dist_topp = s_topp.sample_distribution(logits)
        assert dist_topp[1] == 1.0
        assert float(np.sum(dist_topp)) == 1.0

        # Deterministic seeded sampling reproducibility
        s1 = Sampler(SamplerConfig(temperature=0.8, top_k=4, top_p=0.95, seed=999))
        s2 = Sampler(SamplerConfig(temperature=0.8, top_k=4, top_p=0.95, seed=999))
        seq1 = [s1.sample(logits) for _ in range(25)]
        seq2 = [s2.sample(logits) for _ in range(25)]
        assert seq1 == seq2


# ---------------------------------------------------------------------------
# T6: KV Cache Equivalence vs Full-Sequence Recomputation
# ---------------------------------------------------------------------------


class TestSyntheticT6KVCacheEquivalence:
    """
    Synthetic T6: Step-by-step decoding with KVCache matches uncached recomputation on the toy GGUF.
    """

    def test_step_by_step_kv_cache_matches_full_recomputation(
        self, reference_gguf_path: Path, reference_fixture: Dict[str, Any]
    ) -> None:
        with SeedOssEngine(reference_gguf_path) as engine:
            for prompt_item in reference_fixture["prompts"][:3]:
                prompt_ids = list(prompt_item["prompt_token_ids"])
                kv_cache = engine.create_kv_cache(max_seq_len=256)

                # Step 0: Prefill prompt
                cached_logits_0 = engine.forward(prompt_ids, kv_cache=kv_cache)[-1]
                full_logits_0 = engine.forward(prompt_ids, kv_cache=None)[-1]
                diff_0 = float(np.max(np.abs(cached_logits_0 - full_logits_0)))
                assert diff_0 < 1e-5, f"Prefill logit diff {diff_0} >= 1e-5"

                next_tok_cached = int(np.argmax(cached_logits_0))
                next_tok_full = int(np.argmax(full_logits_0))
                assert next_tok_cached == next_tok_full

                running_ids = list(prompt_ids) + [next_tok_cached]

                # Decode 16 subsequent tokens step-by-step and compare against full recomputation at EVERY step
                for step in range(1, 16):
                    last_tok = running_ids[-1]
                    cached_step_logits = engine.forward([last_tok], kv_cache=kv_cache)[-1]
                    full_step_logits = engine.forward(running_ids, kv_cache=None)[-1]

                    max_diff = float(np.max(np.abs(cached_step_logits - full_step_logits)))
                    assert max_diff < 1e-5, (
                        f"KV cache vs full recomputation diff {max_diff:.2e} >= 1e-5 at step {step}"
                    )

                    tok_c = int(np.argmax(cached_step_logits))
                    tok_f = int(np.argmax(full_step_logits))
                    assert tok_c == tok_f, f"Token mismatch at step {step}: {tok_c} != {tok_f}"
                    running_ids.append(tok_c)

    def test_kv_cache_dynamic_growth_truncate_and_reset(self) -> None:
        cache = KVCache(
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            max_seq_len=128,
            initial_capacity=4,
        )
        assert cache.seq_len == 0

        k_chunk = np.ones((2, 6, 64), dtype=np.float32)
        v_chunk = np.ones((2, 6, 64), dtype=np.float32) * 2.0
        for layer_idx in range(2):
            k_out, v_out = cache.update(layer_idx, k_chunk, v_chunk)
            assert k_out.shape == (2, 6, 64)
            assert v_out.shape == (2, 6, 64)
        assert cache.seq_len == 6

        cache.truncate(3)
        assert cache.seq_len == 3
        k_slice, v_slice = cache.get(0)
        assert k_slice.shape == (2, 3, 64)
        assert np.all(v_slice == 2.0)

        cache.reset()
        assert cache.seq_len == 0


# ---------------------------------------------------------------------------
# T7: Multi-Turn Chat & Thinking Mode (<think>...</think>)
# ---------------------------------------------------------------------------


class TestSyntheticT7ToyChatAndThinkingPlumbing:
    """
    Synthetic T7: Unit/plumbing check on the toy 2-layer GGUF with planted strings
    ("The calculated answer is 42.", "Analyzing the query step by step.") exercising
    Hermes36BAdapter thinking=False/True and keep_cots history stripping. Not a real 36B run.
    """

    def test_multi_turn_chat_standard_mode_no_thinking(
        self, reference_gguf_path: Path
    ) -> None:
        with SeedOssEngine(reference_gguf_path) as engine:
            messages = [{"role": "user", "content": "Hello"}]
            turn1 = engine.chat(messages, thinking=False, max_new_tokens=16)

            assert turn1["has_thinking"] is False
            assert turn1["thinking"] is None
            assert turn1["content"] == "The calculated answer is 42."
            assert turn1["finish_reason"] == "eos"

            # Multi-turn follow-up
            messages.append({"role": "assistant", "content": turn1["content"]})
            messages.append({"role": "user", "content": "step 42"})
            turn2 = engine.chat(messages, thinking=False, max_new_tokens=16)
            assert turn2["has_thinking"] is False
            assert turn2["thinking"] is None
            assert turn2["content"] == "The calculated answer is 42."

    def test_multi_turn_chat_thinking_mode_and_keep_cots(
        self, reference_gguf_path: Path
    ) -> None:
        with SeedOssEngine(reference_gguf_path) as engine:
            messages = [{"role": "user", "content": "think step by step"}]

            # 1) thinking=True, keep_cots=False -> separates thinking from final content
            turn1 = engine.chat(messages, thinking=True, keep_cots=False, max_new_tokens=16)
            assert turn1["has_thinking"] is True
            assert turn1["unclosed_thinking"] is False
            assert turn1["thinking"] == "Analyzing the query step by step."
            assert turn1["content"] == "The calculated answer is 42."
            assert "<think>" in turn1["raw_output"] and "</think>" in turn1["raw_output"]

            # 2) Multi-turn history with keep_cots=False strips historical <think> from prompt
            messages_turn2 = [
                {"role": "user", "content": "think step by step"},
                {"role": "assistant", "content": turn1["raw_output"]},
                {"role": "user", "content": "Hello"},
            ]
            turn2 = engine.chat(messages_turn2, thinking=True, keep_cots=False, max_new_tokens=16)
            # In the formatted prompt, the assistant history turn has <think>...</think> stripped
            assistant_block = turn2["prompt"].split("<|start_header_id|>assistant<|end_header_id|>")[1]
            assert "Analyzing the query step by step." not in assistant_block
            assert turn2["has_thinking"] is True
            assert turn2["thinking"] == "Analyzing the query step by step."
            assert turn2["content"] == "The calculated answer is 42."

            # 3) thinking=True, keep_cots=True -> retains <think> block in content and history
            turn_keep = engine.chat(
                messages_turn2, thinking=True, keep_cots=True, max_new_tokens=16
            )
            assistant_block_keep = turn_keep["prompt"].split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[1]
            assert "Analyzing the query step by step." in assistant_block_keep
            assert turn_keep["content"] is not None
            assert "<think>\nAnalyzing the query step by step.\n</think>" in turn_keep["content"]


# ---------------------------------------------------------------------------
# Synthetic T8: Toy GGUF Tool-Call Plumbing & OpenAI / Anthropic Schema Mapping
# ---------------------------------------------------------------------------


class TestSyntheticT8ToyToolCallPlumbingAndMapping:
    """
    Synthetic T8: Unit/plumbing test on the toy 2-layer GGUF with planted weights that emit a
    fixed <tool_call> JSON payload for Paris weather, verifying Hermes36BAdapter parsing and
    OpenAI tool_calls / Anthropic tool_use mapping. Not a real 36B model forward pass.
    """

    def test_synthetic_toy_gguf_tool_call_plumbing(
        self, reference_gguf_path: Path
    ) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location", "unit"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "weather in Paris"}]

        with SeedOssEngine(reference_gguf_path) as engine:
            result = engine.chat(messages, tools=tools, thinking=False, max_new_tokens=16)

            assert "<tool_call>" in result["raw_output"]
            assert "</tool_call>" in result["raw_output"]
            assert result["has_tool_calls"] is True
            assert result["malformed_tool_calls"] == []
            assert len(result["tool_calls"]) == 1

            tc = result["tool_calls"][0]
            assert tc["name"] == "get_weather"
            assert tc["arguments"] == {"location": "Paris", "unit": "celsius"}

            # Verify OpenAI chat.completions tool_calls format
            assert len(result["openai_tool_calls"]) == 1
            oai_tc = result["openai_tool_calls"][0]
            assert oai_tc["id"] == "call_0"
            assert oai_tc["type"] == "function"
            assert oai_tc["function"]["name"] == "get_weather"
            assert json.loads(oai_tc["function"]["arguments"]) == {
                "location": "Paris",
                "unit": "celsius",
            }

            # Verify Anthropic Messages API tool_use content block format
            assert len(result["anthropic_tool_use"]) == 1
            ant_tu = result["anthropic_tool_use"][0]
            assert ant_tu["type"] == "tool_use"
            assert ant_tu["id"] == "toolu_0"
            assert ant_tu["name"] == "get_weather"
            assert ant_tu["input"] == {"location": "Paris", "unit": "celsius"}

            # Follow-up turn with tool response
            followup_messages = [
                {"role": "user", "content": "weather in Paris"},
                {"role": "assistant", "content": "", "tool_calls": result["openai_tool_calls"]},
                {"role": "tool", "content": '{"temperature": 21, "unit": "celsius"}'},
            ]
            followup = engine.chat(followup_messages, tools=None, thinking=False, max_new_tokens=16)
            assert "<tool_response>" in followup["prompt"]
            assert followup["content"] == "The calculated answer is 42."


# ---------------------------------------------------------------------------
# T9: Robustness & Error Handling (7 Edge Cases)
# ---------------------------------------------------------------------------


class TestSyntheticT9RobustnessAndErrorHandling:
    """
    Synthetic T9: Robustness and negative-path unit tests:
    1) Corrupt GGUF file -> clean ValueError
    2) Wrong architecture in GGUF (general.architecture = "gpt2") -> clean UnsupportedArchitectureError
    3) Empty prompt ([] or "") -> clean ValueError
    4) Prompt longer than max_context_length -> clean ContextLengthExceededError(ValueError)
    5) Requested context requiring more memory than available RAM -> clean InsufficientMemoryError(MemoryError)
    6) Stop sequences (stop_sequences=["<|eot_id|>", "</tool_call>"]) -> stops cleanly at stop sequence
    7) Mid-stream cancellation (cancel_callback returning False or KeyboardInterrupt) -> stops immediately and cleans up session without leak
    """

    def test_1_corrupt_gguf_raises_clean_value_error(self, tmp_path: Path) -> None:
        bad_magic = tmp_path / "bad_magic.gguf"
        bad_magic.write_bytes(b"NOTG" + b"\x00" * 64)
        with pytest.raises(ValueError, match="Invalid GGUF magic"):
            SeedOssEngine(bad_magic)

        too_small = tmp_path / "too_small.gguf"
        too_small.write_bytes(b"GGUF\x03\x00")
        with pytest.raises(ValueError, match="too small"):
            SeedOssEngine(too_small)

        truncated_header = tmp_path / "truncated_header.gguf"
        # Valid magic + version 3 + tensor_count=5, kv_count=2, but EOF immediately after
        truncated_header.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 5, 2))
        with pytest.raises(ValueError, match="Unexpected EOF"):
            SeedOssEngine(truncated_header)

    def test_2_wrong_architecture_raises_unsupported_architecture_error(
        self, tmp_path: Path
    ) -> None:
        wrong_arch_gguf = tmp_path / "gpt2_arch.gguf"
        model_data = _create_deterministic_seed_oss_weights()
        write_reference_gguf_v3(wrong_arch_gguf, model_data, architecture_override="gpt2")

        with pytest.raises(UnsupportedArchitectureError, match="gpt2"):
            SeedOssEngine(wrong_arch_gguf)

    def test_3_empty_prompt_raises_clean_value_error(
        self, reference_gguf_path: Path
    ) -> None:
        with SeedOssEngine(reference_gguf_path) as engine:
            with pytest.raises(ValueError, match="empty"):
                engine.forward([])
            with pytest.raises(ValueError, match="empty"):
                engine.generate("")
            with pytest.raises(ValueError, match="empty"):
                engine.generate([])
            with pytest.raises(ValueError, match="empty"):
                engine.chat([])

    def test_4_prompt_exceeding_max_context_length_raises_clean_error(
        self, reference_gguf_path: Path
    ) -> None:
        with SeedOssEngine(reference_gguf_path, max_context_length=8) as engine:
            long_tokens = [0, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            with pytest.raises(ContextLengthExceededError, match="exceeds max_context_length=8"):
                engine.forward(long_tokens)
            with pytest.raises(ContextLengthExceededError, match="exceeds max_context_length=8"):
                engine.generate(long_tokens, max_new_tokens=4)

    def test_5_excessive_kv_cache_memory_raises_insufficient_memory_error(
        self, reference_gguf_path: Path
    ) -> None:
        # 1) Full 36B 64-layer GQA KV cache at 524,288 context length requires:
        # 2 * 64 * 8 * 128 * 524288 * 4 = 274,877,906,944 bytes (256 GiB) -> clean InsufficientMemoryError
        with pytest.raises(InsufficientMemoryError, match="exceeds available memory budget"):
            KVCache(
                num_layers=64,
                num_kv_heads=8,
                head_dim=128,
                max_seq_len=524288,
                dtype=np.float32,
            )

        # 2) SeedOssEngine with explicit tight memory_limit_bytes raises clean InsufficientMemoryError
        with pytest.raises(InsufficientMemoryError, match="exceeding available memory budget"):
            SeedOssEngine(
                reference_gguf_path,
                max_context_length=1024,
                memory_limit_bytes=16 * 1024,
            )

    def test_6_stop_sequences_terminate_and_trim_output_cleanly(
        self, reference_gguf_path: Path
    ) -> None:
        with SeedOssEngine(reference_gguf_path) as engine:
            # Thinking prompt generates: <think>Analyzing the query step by step.</think>The calculated answer is 42.<|eot_id|>
            # If we pass stop_sequences=["</think>"], it must stop right at "</think>"!
            adapter_prompt = engine.chat(
                [{"role": "user", "content": "Hello"}],
                thinking=True,
                max_new_tokens=1,
            )["prompt"]

            gen = engine.generate(
                adapter_prompt,
                max_new_tokens=32,
                stop_sequences=["</think>"],
            )
            assert gen["finish_reason"] == "stop_sequence"
            assert gen["stop_sequence"] == "</think>"
            assert gen["text"] == "<think>Analyzing the query step by step."
            assert "</think>" not in gen["text"]

    def test_7_mid_stream_cancellation_and_interrupt_cleanup_without_leak(
        self, reference_gguf_path: Path
    ) -> None:
        with SeedOssEngine(reference_gguf_path) as engine:
            assert engine.active_sessions == 0

            # Case A: cancel_callback returning False after 3 tokens
            yielded_tokens: List[int] = []

            def cancel_after_3(step: int) -> bool:
                return step < 3

            for ev in engine.generate_stream(
                [0, 16, 17, 18],
                max_new_tokens=64,
                stop_token_ids=[],
                cancel_callback=cancel_after_3,
            ):
                yielded_tokens.append(ev["token_id"])
                assert engine.active_sessions == 1

            assert len(yielded_tokens) == 3
            assert engine.active_sessions == 0

            # Case B: Simulated KeyboardInterrupt mid-stream cleans up session in finally block
            stream = engine.generate_stream(
                [0, 16, 17, 18],
                max_new_tokens=64,
                stop_token_ids=[],
            )
            first_ev = next(stream)
            assert "token_id" in first_ev
            assert engine.active_sessions == 1

            with pytest.raises(KeyboardInterrupt):
                stream.throw(KeyboardInterrupt("Simulated user interrupt"))

            assert engine.active_sessions == 0


# ---------------------------------------------------------------------------
# Static Guard & CI Sanitizer Coverage Audit (NOT an ASan/UBSan execution itself;
# live ASan+UBSan execution of C/C++ code happens in GitHub Actions `cpp-sanitizers`)
# ---------------------------------------------------------------------------


class TestCpp20SourceStringGuardsAudit:
    """
    Static source-string and CI-configuration audit verifying C/C++ bounds guards,
    64-bit KV byte arithmetic, atomic session cancellation, and that every C file under
    `core/inference/` is built and exercised in the CI ASan+UBSan job. Actual runtime
    T10 sanitizer verification is performed by the CI `cpp-sanitizers` job.
    """

    def test_every_c_file_in_core_inference_is_compiled_and_sanitized_in_ci(self) -> None:
        c_files = sorted((REPO_ROOT / "core" / "inference").glob("*.c"))
        assert len(c_files) >= 1
        cmake_text = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        cpp_test_text = (REPO_ROOT / "tests" / "cpp_safety_test.cpp").read_text(encoding="utf-8")
        ci_text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

        for c_file in c_files:
            rel_posix = c_file.relative_to(REPO_ROOT).as_posix()
            assert rel_posix in cmake_text, f"{rel_posix} missing from CMakeLists.txt"

        assert "qerlio_gemm_q4_k_f32" in cpp_test_text
        assert "qerlio_gemm_q6_k_f32" in cpp_test_text
        assert "qerlio_gemm_q8_0_f32" in cpp_test_text
        assert "qerlio_gemm_f32" in cpp_test_text
        assert "cpp-sanitizers:" in ci_text
        assert "-fsanitize=address,undefined" in ci_text
        assert "detect_leaks=1" in ci_text
        assert "halt_on_error=1" in ci_text

    def test_gguf_parser_cpp_bounds_and_corruption_guards(self) -> None:
        cpp_path = REPO_ROOT / "core" / "model_registry" / "gguf_parser.cpp"
        source = cpp_path.read_text(encoding="utf-8")

        # Verify upper-bound constants against malicious/corrupt GGUF headers
        assert "MAX_REASONABLE_TENSORS" in source
        assert "MAX_REASONABLE_METADATA_KEYS" in source
        assert "MAX_REASONABLE_STRING_LENGTH" in source
        assert "MAX_REASONABLE_ARRAY_ELEMENTS" in source

        # Verify magic validation, version range check, and string/array length checks
        assert "is_magic_valid(magic)" in source
        assert "version < 2 || version > 3" in source
        assert "len > MAX_REASONABLE_STRING_LENGTH" in source
        assert "count > MAX_REASONABLE_ARRAY_ELEMENTS" in source

    def test_memory_planner_cpp_64bit_kv_cache_arithmetic_and_oom_fallback(self) -> None:
        cpp_path = REPO_ROOT / "core" / "memory" / "memory_planner.cpp"
        source = cpp_path.read_text(encoding="utf-8")

        # Must use 64-bit unsigned arithmetic (2ULL * ...) so 524,288 context length (256 GiB) never overflows uint32
        assert "2ULL * dims.layers * dims.kv_heads * dims.head_dim * context_length * dims.bytes_per_element" in source
        assert "DEFAULT_WORKSPACE_BYTES" in source
        assert "MIN_SAFETY_MARGIN_BYTES" in source
        assert "fits_in_available_ram = false" in source

    def test_session_and_abi_cpp_thread_safety_and_null_pointer_guards(self) -> None:
        session_hpp = (REPO_ROOT / "core" / "session" / "session_internal.hpp").read_text(
            encoding="utf-8"
        )
        engine_hpp = (REPO_ROOT / "core" / "engine" / "engine_internal.hpp").read_text(
            encoding="utf-8"
        )
        abi_cpp = (REPO_ROOT / "src" / "qerlio_abi.cpp").read_text(encoding="utf-8")

        # Verify atomic acquire/release cancellation in qerlio_session
        assert "std::atomic<bool> is_cancelled{false}" in session_hpp
        assert "std::memory_order_release" in session_hpp
        assert "std::memory_order_acquire" in session_hpp

        # Verify mutex lock guard in qerlio_engine error state
        assert "std::lock_guard<std::mutex>" in engine_hpp

        # Verify null-pointer and zero-buffer guards on C ABI entrypoints
        assert "if (!out_engine)" in abi_cpp
        assert "if (!engine || !buffer || buffer_size == 0)" in abi_cpp
        assert "if (!engine || !config || !config->model_path || !out_model)" in abi_cpp
        assert "if (!model || !out_session)" in abi_cpp
        assert "if (!session || !prompt || !callback)" in abi_cpp
        assert "buffer[buffer_size - 1] = '\\0'" in abi_cpp


# ---------------------------------------------------------------------------
# Section 2.2(i): Local API (/v1/chat/completions & /v1/messages) Synthetic Engine Plumbing
# ---------------------------------------------------------------------------


class TestLocalAPIEndpointsWithSyntheticSeedOssGGUF:
    """
    Verify Section 2.2(i) API wiring using the synthetic 2-layer GGUF fixture:
    - When no model is loaded, /v1/chat/completions returns "simulated": True.
    - When the synthetic GGUF fixture is loaded via /v1/models/load, /v1/chat/completions
      and /v1/messages execute SeedOssEngine and return "simulated": False.
    """

    def test_local_api_executes_loaded_synthetic_seed_oss_gguf(
        self, reference_gguf_path: Path
    ) -> None:
        from fastapi.testclient import TestClient
        from server.local_api.server import app, unload_native_models

        unload_native_models()
        client = TestClient(app)

        try:
            # 1) Unloaded state -> simulated: True
            unloaded_res = client.post(
                "/v1/chat/completions",
                json={
                    "model": "NousResearch/Hermes-4.3-36B",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert unloaded_res.status_code == 200
            assert unloaded_res.json()["simulated"] is True

            # 2) Load reference GGUF model via POST /v1/models/load
            load_res = client.post(
                "/v1/models/load",
                json={
                    "model_path": str(reference_gguf_path),
                    "model_id": "NousResearch/Hermes-4.3-36B",
                    "max_context_length": 512,
                },
            )
            assert load_res.status_code == 200
            load_data = load_res.json()
            assert load_data["status"] == "loaded"
            assert load_data["simulated"] is False
            assert load_data["architecture"] == "seed_oss"

            # 3) Check /health, /v1/models, and /api/tags reflect ONLY the loaded engine
            health_res = client.get("/health")
            assert health_res.status_code == 200
            assert health_res.json()["simulated_inference"] is False

            models_res = client.get("/v1/models")
            assert models_res.status_code == 200
            models_data = models_res.json()
            assert models_data["loaded_count"] == 1
            assert [m["id"] for m in models_data["data"]] == ["NousResearch/Hermes-4.3-36B"]

            tags_res = client.get("/api/tags")
            assert tags_res.status_code == 200
            assert [m["name"] for m in tags_res.json()["models"]] == ["NousResearch/Hermes-4.3-36B"]

            # 4) Non-streaming /v1/chat/completions with thinking=True
            chat_res = client.post(
                "/v1/chat/completions",
                json={
                    "model": "NousResearch/Hermes-4.3-36B",
                    "messages": [{"role": "user", "content": "think step by step"}],
                    "thinking": True,
                    "max_tokens": 16,
                },
            )
            assert chat_res.status_code == 200
            chat_data = chat_res.json()
            assert chat_data["simulated"] is False
            msg = chat_data["choices"][0]["message"]
            assert msg["content"] == "The calculated answer is 42."
            assert msg["reasoning_content"] == "Analyzing the query step by step."
            assert chat_data["usage"]["prompt_tokens"] > 0
            assert chat_data["usage"]["completion_tokens"] > 0

            # 5) Tool-calling /v1/chat/completions
            tool_res = client.post(
                "/v1/chat/completions",
                json={
                    "model": "NousResearch/Hermes-4.3-36B",
                    "messages": [{"role": "user", "content": "weather in Paris"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "max_tokens": 16,
                },
            )
            assert tool_res.status_code == 200
            tool_data = tool_res.json()
            assert tool_data["simulated"] is False
            assert tool_data["choices"][0]["finish_reason"] == "tool_calls"
            tcs = tool_data["choices"][0]["message"]["tool_calls"]
            assert len(tcs) == 1
            assert tcs[0]["function"]["name"] == "get_weather"

            # 6) Streaming /v1/chat/completions
            stream_res = client.post(
                "/v1/chat/completions",
                json={
                    "model": "NousResearch/Hermes-4.3-36B",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                    "max_tokens": 16,
                },
            )
            assert stream_res.status_code == 200
            lines = [
                line[6:]
                for line in stream_res.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert len(lines) >= 2
            chunks = [json.loads(line) for line in lines]
            assert all(c["simulated"] is False for c in chunks)
            streamed_text = "".join(
                c["choices"][0]["delta"].get("content", "") for c in chunks
            )
            assert streamed_text == "The calculated answer is 42."

            # 7) Anthropic /v1/messages endpoint
            ant_res = client.post(
                "/v1/messages",
                json={
                    "model": "NousResearch/Hermes-4.3-36B",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 16,
                },
            )
            assert ant_res.status_code == 200
            ant_data = ant_res.json()
            assert ant_data["simulated"] is False
            assert ant_data["content"][0]["text"] == "The calculated answer is 42."

            # 8) Unload via POST /v1/models/unload restores simulated: True
            unload_res = client.post("/v1/models/unload")
            assert unload_res.status_code == 200
            assert unload_res.json()["simulated"] is True
        finally:
            unload_native_models()
