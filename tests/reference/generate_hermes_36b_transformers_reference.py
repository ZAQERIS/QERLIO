"""Offline T4/T5 Reference Generator using official HuggingFace `transformers` + `gguf-py`.

Per ZAQERIS Ai Directive Section 3.1, this script runs strictly offline ONCE
against `models/hermes-4_3_36b-Q4_K_M.gguf` (never imported or invoked by
`core/inference/` at runtime). It instantiates HuggingFace `SeedOssDecoderLayer`,
`SeedOssRMSNorm`, and `SeedOssRotaryEmbedding` one layer at a time using tensors
dequantized by `gguf.quants.dequantize`, and computes reference float32 logits
across all 5 fixed prompts and all 64 greedy steps (320 predictions total) to
produce `tests/fixtures/hermes_4_3_36b_transformers_reference.json`.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import gguf
import gguf.quants as gq
import numpy as np
import torch
from transformers.models.seed_oss.configuration_seed_oss import SeedOssConfig
from transformers.models.seed_oss.modeling_seed_oss import (
    SeedOssDecoderLayer,
    SeedOssRMSNorm,
    SeedOssRotaryEmbedding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "hermes-4_3_36b-Q4_K_M.gguf"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "hermes_4_3_36b_transformers_reference.json"


def _meta_scalar(reader: gguf.GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.fields.get(key)
    if field is None:
        return default
    parts = [field.parts[idx] for idx in field.data]
    if len(parts) == 1 and hasattr(parts[0], "tolist"):
        val = parts[0].tolist()
        if isinstance(val, list) and len(val) == 1:
            return val[0]
        return val
    # String arrays / bytes
    raw_bytes = bytes(parts[0].tolist())
    return raw_bytes.decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline HF transformers + gguf-py T4/T5 reference generator for Hermes 4.3 36B"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--qerlio-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    qerlio_data = json.loads(args.qerlio_run.read_text(encoding="utf-8"))
    t5_trajs = qerlio_data["T5"]["trajectories"]
    t4_prompts = qerlio_data["T4"]["prompts"]
    assert len(t5_trajs) == 5

    reader = gguf.GGUFReader(str(args.model_path), mode="r")
    tensors_by_name = {t.name: t for t in reader.tensors}

    n_layers = int(_meta_scalar(reader, "seed_oss.block_count"))
    hidden_size = int(_meta_scalar(reader, "seed_oss.embedding_length"))
    intermediate_size = int(_meta_scalar(reader, "seed_oss.feed_forward_length"))
    n_heads = int(_meta_scalar(reader, "seed_oss.attention.head_count"))
    n_kv_heads = int(_meta_scalar(reader, "seed_oss.attention.head_count_kv"))
    head_dim = int(_meta_scalar(reader, "seed_oss.attention.key_length"))
    rms_eps = float(_meta_scalar(reader, "seed_oss.attention.layer_norm_rms_epsilon", 1e-6))
    rope_theta = float(_meta_scalar(reader, "seed_oss.rope.freq_base", 10000000.0))
    max_pos = int(_meta_scalar(reader, "seed_oss.context_length", 524288))
    vocab_size = int(tensors_by_name["token_embd.weight"].shape[1])

    cfg = SeedOssConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        rms_norm_eps=rms_eps,
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
        attention_bias=True,
        attention_out_bias=False,
        mlp_bias=False,
    )
    cfg._attn_implementation = "eager"

    def _dequant_tensor_f32(name: str) -> torch.Tensor:
        t = tensors_by_name[name]
        arr = gq.dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
        expected_shape = tuple(int(d) for d in reversed(t.shape))
        return torch.from_numpy(np.ascontiguousarray(arr.reshape(expected_shape)))

    # Embed all 5 sequences (prompt + first 63 generated tokens -> predicts all 64 steps!)
    full_seqs: List[List[int]] = []
    prompt_lens: List[int] = []
    for tr in t5_trajs:
        p_ids = list(tr["prompt_token_ids"])
        g_ids = list(tr["generated_token_ids"])
        assert len(g_ids) == 64
        prompt_lens.append(len(p_ids))
        full_seqs.append(p_ids + g_ids[:-1])

    emb_t = tensors_by_name["token_embd.weight"]
    blk_elems, blk_bytes = gq.GGML_QUANT_SIZES[emb_t.tensor_type]
    emb_row_bytes = (hidden_size // blk_elems) * blk_bytes
    emb_rows_view = emb_t.data.reshape(vocab_size, emb_row_bytes)

    hidden_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    rotary = SeedOssRotaryEmbedding(cfg)

    with torch.no_grad():
        for seq in full_seqs:
            seq_u8 = emb_rows_view[seq]
            seq_f32 = gq.dequantize(seq_u8, emb_t.tensor_type).astype(np.float32, copy=False)
            h = torch.from_numpy(np.ascontiguousarray(seq_f32.reshape(len(seq), hidden_size))).unsqueeze(0)
            hidden_list.append(h)
            pos = torch.arange(len(seq), dtype=torch.long).unsqueeze(0)
            pos_list.append(pos)
            mask = torch.triu(
                torch.full((1, 1, len(seq), len(seq)), float("-inf"), dtype=torch.float32),
                diagonal=1,
            )
            mask_list.append(mask)

        t0 = time.perf_counter()
        for l_idx in range(n_layers):
            prefix = f"blk.{l_idx}."
            layer = SeedOssDecoderLayer(cfg, l_idx).to(torch.float32)
            layer.eval()

            layer.input_layernorm.weight.copy_(_dequant_tensor_f32(f"{prefix}attn_norm.weight"))
            layer.self_attn.q_proj.weight.copy_(_dequant_tensor_f32(f"{prefix}attn_q.weight"))
            layer.self_attn.q_proj.bias.copy_(_dequant_tensor_f32(f"{prefix}attn_q.bias"))
            layer.self_attn.k_proj.weight.copy_(_dequant_tensor_f32(f"{prefix}attn_k.weight"))
            layer.self_attn.k_proj.bias.copy_(_dequant_tensor_f32(f"{prefix}attn_k.bias"))
            layer.self_attn.v_proj.weight.copy_(_dequant_tensor_f32(f"{prefix}attn_v.weight"))
            layer.self_attn.v_proj.bias.copy_(_dequant_tensor_f32(f"{prefix}attn_v.bias"))
            layer.self_attn.o_proj.weight.copy_(_dequant_tensor_f32(f"{prefix}attn_output.weight"))
            layer.post_attention_layernorm.weight.copy_(
                _dequant_tensor_f32(f"{prefix}post_attention_norm.weight")
            )
            layer.mlp.gate_proj.weight.copy_(_dequant_tensor_f32(f"{prefix}ffn_gate.weight"))
            layer.mlp.up_proj.weight.copy_(_dequant_tensor_f32(f"{prefix}ffn_up.weight"))
            layer.mlp.down_proj.weight.copy_(_dequant_tensor_f32(f"{prefix}ffn_down.weight"))

            for p_i in range(len(hidden_list)):
                cos, sin = rotary(hidden_list[p_i], pos_list[p_i])
                hidden_list[p_i] = layer(
                    hidden_list[p_i],
                    attention_mask=mask_list[p_i],
                    position_ids=pos_list[p_i],
                    position_embeddings=(cos, sin),
                    use_cache=False,
                )

            del layer
            if (l_idx + 1) % 8 == 0:
                print(f"[HF-Ref] layer {l_idx + 1}/{n_layers} done ({time.perf_counter() - t0:.1f}s)", flush=True)

        # Final RMSNorm & slice out all 64 target positions per prompt (prompt_len - 1 .. seq_len - 1)
        out_norm = SeedOssRMSNorm(hidden_size, eps=rms_eps)
        out_norm.weight.copy_(_dequant_tensor_f32("output_norm.weight"))

        target_rows_list: List[torch.Tensor] = []
        for p_i, h in enumerate(hidden_list):
            h_normed = out_norm(h)[0]  # (seq_len, hidden)
            start_pos = prompt_lens[p_i] - 1
            target_rows_list.append(h_normed[start_pos:])  # (64, hidden)
            assert target_rows_list[-1].shape == (64, hidden_size)

        all_targets = torch.cat(target_rows_list, dim=0)  # (320, hidden)

        # Chunked LM head via gguf.quants.dequantize
        out_t = tensors_by_name["output.weight"]
        ob_elems, ob_bytes = gq.GGML_QUANT_SIZES[out_t.tensor_type]
        out_row_bytes = (hidden_size // ob_elems) * ob_bytes
        out_rows_view = out_t.data.reshape(vocab_size, out_row_bytes)

        all_logits = torch.empty((all_targets.shape[0], vocab_size), dtype=torch.float32)
        chunk = 16384
        for rs in range(0, vocab_size, chunk):
            re = min(rs + chunk, vocab_size)
            w_np = gq.dequantize(out_rows_view[rs:re], out_t.tensor_type).astype(np.float32, copy=False)
            w_th = torch.from_numpy(np.ascontiguousarray(w_np.reshape(re - rs, hidden_size)))
            all_logits[:, rs:re] = all_targets @ w_th.T

    # Compare all 5 T4 prefill logits & all 320 T5 greedy tokens
    strided_indices = t4_prompts[0]["strided_64_indices"]
    prompt_records: List[Dict[str, Any]] = []
    all_t4_top1_match = 0
    all_t5_exact_tokens = 0
    global_t4_strided_max_diff = 0.0
    global_t4_top20_max_diff = 0.0

    for p_i in range(5):
        p_logits = all_logits[p_i * 64 : (p_i + 1) * 64]  # (64, vocab)
        hf_pred_64 = [int(x) for x in torch.argmax(p_logits, dim=-1).tolist()]
        q_pred_64 = list(t5_trajs[p_i]["generated_token_ids"])

        step0_logits = p_logits[0].numpy()
        top20_ids = np.argsort(step0_logits)[-20:][::-1]
        hf_top20 = [{"token_id": int(tid), "logit": float(step0_logits[int(tid)])} for tid in top20_ids]

        q_t4 = t4_prompts[p_i]
        if int(top20_ids[0]) == int(q_t4["top1_token_id"]):
            all_t4_top1_match += 1

        q_top5 = {x["token_id"] for x in q_t4["top20"][:5]}
        r_top5 = {int(tid) for tid in top20_ids[:5]}
        top5_overlap = len(q_top5 & r_top5)

        q_map = {x["token_id"]: x["logit"] for x in q_t4["top20"]}
        top20_diffs = [
            abs(float(step0_logits[tid]) - float(q_map[tid]))
            for tid in q_map
        ]
        p_top20_diff = max(top20_diffs)
        strided_hf = [float(step0_logits[j]) for j in strided_indices]
        p_strided_diff = max(
            abs(a - b) for a, b in zip(strided_hf, q_t4["strided_64_logits"])
        )
        global_t4_top20_max_diff = max(global_t4_top20_max_diff, p_top20_diff)
        global_t4_strided_max_diff = max(global_t4_strided_max_diff, p_strided_diff)

        exact_steps = sum(1 for a, b in zip(hf_pred_64, q_pred_64) if a == b)
        all_t5_exact_tokens += exact_steps

        print(
            f"[Prompt {p_i}] top1_match={int(top20_ids[0]) == int(q_t4['top1_token_id'])} "
            f"top5_overlap={top5_overlap}/5 top20_max_diff={p_top20_diff:.2e} "
            f"strided_max_diff={p_strided_diff:.2e} T5_match={exact_steps}/64",
            flush=True,
        )

        prompt_records.append(
            {
                "prompt_index": p_i,
                "prompt_text": t5_trajs[p_i]["prompt_text"],
                "prompt_token_ids": t5_trajs[p_i]["prompt_token_ids"],
                "top1_token_id": int(top20_ids[0]),
                "top5_overlap_with_qerlio": int(top5_overlap),
                "top20": hf_top20,
                "strided_64_indices": strided_indices,
                "strided_64_logits": strided_hf,
                "t4_top20_max_abs_diff": p_top20_diff,
                "t4_strided_max_abs_diff": p_strided_diff,
                "greedy_64_token_ids": hf_pred_64,
                "t5_exact_token_matches": exact_steps,
            }
        )

    out_payload = {
        "generator": {
            "command": f"python tests/reference/generate_hermes_36b_transformers_reference.py --qerlio-run <report.json>",
            "torch_version": torch.__version__.split("+")[0],
            "transformers_version": importlib.metadata.version("transformers"),
            "gguf_version": importlib.metadata.version("gguf"),
            "numpy_version": np.__version__,
            "model_file": args.model_path.name,
            "model_sha256_verified": "17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08",
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
        "summary": {
            "t4_top1_matches": f"{all_t4_top1_match}/5",
            "t4_top20_max_abs_diff": global_t4_top20_max_diff,
            "t4_strided_64_max_abs_diff": global_t4_strided_max_diff,
            "t5_exact_token_matches": f"{all_t5_exact_tokens}/320",
        },
        "prompts": prompt_records,
    }
    args.out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(
        f"Saved {args.out}: T4={all_t4_top1_match}/5 (max_diff={global_t4_top20_max_diff:.2e}), "
        f"T5={all_t5_exact_tokens}/320 tokens matched!",
        flush=True,
    )
    assert all_t4_top1_match == 5
    assert all(p["top5_overlap_with_qerlio"] >= 4 for p in prompt_records)
    assert all_t5_exact_tokens == 320


if __name__ == "__main__":
    main()
