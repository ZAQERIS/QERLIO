"""Offline T3 Reference Generator using HuggingFace `tokenizers` (`NousResearch/Hermes-4.3-36B`).

Per ZAQERIS AI Directive Section 3.1, this script runs offline (or with cached
Hub files) ONCE to verify and regenerate `tests/fixtures/hermes_tokenizer_reference.json`
covering 64 multilingual, code, math, emoji, whitespace, and special-token strings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import tokenizers
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "hermes_tokenizer_reference.json"


def main() -> None:
    tok_json_path = hf_hub_download("NousResearch/Hermes-4.3-36B", "tokenizer.json")
    hf_tok = tokenizers.Tokenizer.from_file(tok_json_path)

    existing = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    sparse_tokens: Dict[str, Any] = existing["gguf_fallback_tables"]["sparse_tokens"]
    ordered_merges: List[str] = existing["gguf_fallback_tables"]["ordered_merges"]

    for case in existing["cases"]:
        enc = hf_tok.encode(case["text"], add_special_tokens=False)
        case["token_ids"] = list(enc.ids)

    # Ensure fallback sparse_tokens / ordered_merges remain sufficient for headless CI builds
    sys.path.insert(0, str(REPO_ROOT))
    from core.inference.gguf_loader import GGUFLoader
    from core.inference.tokenizer import GGUFTokenizer

    gguf_path = REPO_ROOT / "models" / "hermes-4_3_36b-Q4_K_M.gguf"
    if gguf_path.is_file():
        with GGUFLoader(gguf_path, header_only=True) as loader:
            full_tok = GGUFTokenizer.from_gguf_metadata(loader.metadata)
            # Expand sparse_tokens and ordered_merges with any tokens/merges needed by updated cases
            vocab_list = loader.metadata["tokenizer.ggml.tokens"]
            ttype_list = loader.metadata["tokenizer.ggml.token_type"]
            merges_list = loader.metadata["tokenizer.ggml.merges"]

            # Trace exact BPE merge sequence for every new token/word so headless CI fallback has 100% of merges
            existing_merges_set = set(ordered_merges)
            # Also seed intermediate sub-pieces for every final token in case["token_ids"]
            merge_index = {m: i for i, m in enumerate(merges_list)}
            tok_to_id = {tok: i for i, tok in enumerate(vocab_list)}

            def _record_piece_ancestry(tok_str: str) -> None:
                tid = tok_to_id.get(tok_str)
                if tid is not None:
                    sparse_tokens[str(tid)] = [tok_str, int(ttype_list[tid])]
                if len(tok_str) <= 1:
                    return
                # Find highest-priority split among valid merges forming tok_str
                best_split = None
                best_rank = len(merges_list) + 1
                for cut in range(1, len(tok_str)):
                    cand_m = f"{tok_str[:cut]} {tok_str[cut:]}"
                    rk = merge_index.get(cand_m)
                    if rk is not None and rk < best_rank:
                        best_rank = rk
                        best_split = (cand_m, tok_str[:cut], tok_str[cut:])
                if best_split is not None:
                    cand_m, left_p, right_p = best_split
                    existing_merges_set.add(cand_m)
                    _record_piece_ancestry(left_p)
                    _record_piece_ancestry(right_p)

            for case in existing["cases"]:
                for tid in case["token_ids"]:
                    _record_piece_ancestry(vocab_list[tid])

            ordered_merges = [m for m in merges_list if m in existing_merges_set]

            # Verify full Hermes GGUFTokenizer AND reconstructed headless fallback tokenizer match 64/64!
            for case in existing["cases"]:
                assert full_tok.encode(case["text"]) == case["token_ids"], (
                    f"Mismatch against full Hermes GGUFTokenizer on {case['id']}"
                )
            existing["gguf_fallback_tables"]["ordered_merges"] = ordered_merges

            # Test reconstructed fallback metadata directly
            import tempfile
            from tests.test_t3_tokenizer import _build_gguf_prefix_bytes_from_fixture
            fb_bytes = _build_gguf_prefix_bytes_from_fixture(existing)
            with tempfile.TemporaryDirectory() as td:
                fb_path = Path(td) / "fb.gguf"
                fb_path.write_bytes(fb_bytes)
                fb_tok = GGUFTokenizer.from_gguf_file(str(fb_path))
            for case in existing["cases"]:
                assert fb_tok.encode(case["text"]) == case["token_ids"], (
                    f"Fallback mismatch on {case['id']}"
                )

    existing["tokenizers_version"] = tokenizers.__version__
    existing["generator_command"] = "python tests/reference/generate_t3_tokenizer_reference.py"
    FIXTURE_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {FIXTURE_PATH} cleanly (cases={len(existing['cases'])}, tokenizers={tokenizers.__version__})")


if __name__ == "__main__":
    main()
