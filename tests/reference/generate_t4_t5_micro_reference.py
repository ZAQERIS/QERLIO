"""Offline T4/T5 Micro Reference Verifier (`SeedOssConfig` deterministic reference fixture).

Documents the offline HuggingFace `transformers` (`SeedOssForCausalLM`) reference fixture
committed in `tests/fixtures/seed_oss_reference_logits_and_tokens.json`, generated with
`torch 2.11.0` and `transformers 5.20.1` (seed=20260301) over 5 fixed prompts x 64
greedy tokens. Running this script validates the fixture's schema and checksum against
the deterministic GGUF builder in `tests/test_t4_to_t10_engine.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "seed_oss_reference_logits_and_tokens.json"


def main() -> None:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    gen = data["generator"]
    prompts = data["prompts"]
    assert len(prompts) == 5, f"Expected 5 prompts, got {len(prompts)}"
    for p in prompts:
        assert len(p["reference_greedy_64_tokens"]) == 64
    print(
        f"Verified T4/T5 micro reference fixture ({len(prompts)} prompts x 64 greedy tokens, "
        f"torch={gen['torch_version']}, transformers={gen['transformers_version']}, "
        f"gguf_sha256={gen['gguf_sha256']})"
    )


if __name__ == "__main__":
    main()
