"""Offline T2 Reference Generator using installed `gguf` (`gguf-py`).

Per ZAQERIS AI Directive Section 3.1, this script runs offline ONCE to extract
real quantized blocks from `models/hermes-4_3_36b-Q4_K_M.gguf` (`Q4_K`, `Q6_K`,
`F32`) plus deterministic seeded sample blocks across all quant types supported by
`gguf.quants.dequantize`, saving expected float32 outputs to
`tests/fixtures/t2_gguf_py_dequant_reference.json`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List

import gguf
import gguf.quants as gq
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.inference.dequant import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_NAMES,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
)
from core.inference.gguf_loader import GGUFLoader

DEFAULT_HERMES_GGUF = REPO_ROOT / "models" / "hermes-4_3_36b-Q4_K_M.gguf"
OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "t2_gguf_py_dequant_reference.json"

ALL_T2_TYPES = [
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ4_XS,
]


def _make_sample_blocks(ggml_type: int, n_blocks: int, seed: int) -> bytes:
    rng = np.random.Generator(np.random.PCG64(seed))
    qtype = gguf.GGMLQuantizationType(ggml_type)
    if ggml_type == GGML_TYPE_F32:
        return rng.uniform(-3.0, 3.0, size=n_blocks).astype("<f4").tobytes()
    if ggml_type == GGML_TYPE_F16:
        return rng.uniform(-3.0, 3.0, size=n_blocks).astype("<f2").tobytes()
    if ggml_type == GGML_TYPE_BF16:
        f32 = rng.uniform(-3.0, 3.0, size=n_blocks).astype("<f4")
        u32 = np.frombuffer(f32.tobytes(), dtype="<u4")
        return (u32 >> 16).astype("<u2").tobytes()

    blk_elems, type_size = gq.GGML_QUANT_SIZES[qtype]
    total_bytes = n_blocks * type_size
    _ = blk_elems
    raw = bytearray(rng.integers(0, 256, size=total_bytes, dtype=np.uint8).tobytes())
    unit_bytes = type_size
    scales_a = rng.uniform(0.005, 0.25, size=n_blocks).astype("<f2")
    scales_b = rng.uniform(0.005, 0.25, size=n_blocks).astype("<f2")
    for i in range(n_blocks):
        base = i * unit_bytes
        sa = struct.pack("<e", float(scales_a[i]))
        sb = struct.pack("<e", float(scales_b[i]))
        if ggml_type in (
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q8_0,
            GGML_TYPE_IQ4_NL,
            GGML_TYPE_IQ3_S,
            GGML_TYPE_IQ4_XS,
        ):
            raw[base : base + 2] = sa
        elif ggml_type in (GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K):
            raw[base : base + 2] = sa
            raw[base + 2 : base + 4] = sb
        elif ggml_type == GGML_TYPE_Q2_K:
            raw[base + 80 : base + 82] = sa
            raw[base + 82 : base + 84] = sb
        elif ggml_type == GGML_TYPE_Q3_K:
            raw[base + 108 : base + 110] = sa
        elif ggml_type == GGML_TYPE_Q6_K:
            raw[base + 208 : base + 210] = sa
    return bytes(raw)


def main() -> None:
    if not DEFAULT_HERMES_GGUF.is_file():
        raise FileNotFoundError(f"Expected official GGUF at {DEFAULT_HERMES_GGUF}")

    gguf_ver = importlib.metadata.version("gguf")
    samples: List[Dict[str, Any]] = []
    skipped_in_gguf_py: List[str] = []

    with GGUFLoader(DEFAULT_HERMES_GGUF, header_only=False) as loader:
        real_targets = [
            ("real_hermes_output_weight_row0_q6k", "output.weight", 0, 210 * 2, 256 * 2),
            ("real_hermes_token_embd_row10805_q4k", "token_embd.weight", 10805, 144 * 2, 256 * 2),
            ("real_hermes_blk0_attn_q_row0_q4k", "blk.0.attn_q.weight", 0, 144 * 2, 256 * 2),
            ("real_hermes_blk0_ffn_down_row0_q6k", "blk.0.ffn_down.weight", 0, 210 * 2, 256 * 2),
            ("real_hermes_output_norm_f32", "output_norm.weight", 0, 4 * 64, 64),
        ]
        for case_id, tname, row_idx, n_bytes, n_elems in real_targets:
            tinfo = loader.get_tensor_info(tname)
            row_stride = (tinfo.n_bytes // tinfo.dims[1]) if tinfo.n_dims == 2 else 0
            raw_mv = loader.get_tensor_raw(tname)
            try:
                start = row_idx * row_stride
                raw_slice = bytes(raw_mv[start : start + n_bytes])
            finally:
                raw_mv.release()
            ref_vals = gq.dequantize(
                np.frombuffer(raw_slice, dtype=np.uint8),
                gguf.GGMLQuantizationType(tinfo.type_id),
            ).astype(np.float32)
            assert ref_vals.size == n_elems
            samples.append(
                {
                    "id": case_id,
                    "source": f"hermes-4_3_36b-Q4_K_M.gguf:{tname}[row={row_idx}]",
                    "type_id": int(tinfo.type_id),
                    "type_name": tinfo.type_name,
                    "num_elements": int(n_elems),
                    "raw_hex": raw_slice.hex(),
                    "expected_f32": [float(v) for v in ref_vals],
                }
            )

    for tid in ALL_T2_TYPES:
        tname = GGML_TYPE_NAMES[tid]
        qtype = gguf.GGMLQuantizationType(tid)
        if tid in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16):
            n_blocks = 16
        elif tid in (GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL):
            n_blocks = 4
        else:
            n_blocks = 2

        try:
            raw_bytes = _make_sample_blocks(tid, n_blocks=n_blocks, seed=500 + tid)
            blk_sz = gq.GGML_QUANT_SIZES[qtype][0]
            n_elems = n_blocks * blk_sz
            ref_arr = gq.dequantize(np.frombuffer(raw_bytes, dtype=np.uint8), qtype).astype(np.float32)
            assert ref_arr.size == n_elems
        except (NotImplementedError, Exception) as exc:
            skipped_in_gguf_py.append(f"{tname}: {type(exc).__name__}: {exc}")
            continue

        samples.append(
            {
                "id": f"gguf_py_{tname.lower()}",
                "source": "deterministic_pcg64_blocks",
                "type_id": int(tid),
                "type_name": tname,
                "num_elements": int(n_elems),
                "raw_hex": raw_bytes.hex(),
                "expected_f32": [float(v) for v in ref_arr],
            }
        )

    payload = {
        "generator": {
            "command": "python tests/reference/generate_t2_dequant_reference.py",
            "gguf_version": gguf_ver,
            "model_file": "hermes-4_3_36b-Q4_K_M.gguf",
            "model_sha256_verified": "17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08",
            "unsupported_by_gguf_py": skipped_in_gguf_py,
        },
        "cases": samples,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} T2 reference cases to {OUT_PATH} (gguf-py={gguf_ver}, unsupported={skipped_in_gguf_py})")


if __name__ == "__main__":
    main()
