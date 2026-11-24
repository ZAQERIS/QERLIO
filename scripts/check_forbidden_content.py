#!/usr/bin/env python3
"""C14 Forbidden-Content, Honesty & Repository Integrity Scanner."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_TERMS_CI = (
    "zaq" + "ion",
    "zaq" + "ion_dialectic",
    "officially " + "supports",
    "official " + "support",
    "bla" + "zing",
    "state-of-" + "the-art",
    "state of " + "the art",
    "k-quants (" + "sota)",
    "memory " + "blowout",
    "production-" + "ready",
    "production-" + "grade",
    "fully " + "supported",
    "v" + "ram",
    "gpu-" + "hour",
    "24 " + "threads",
    "ra" + "him",
    "h1" + "00",
    "ge" + "force",
    "ry" + "zen",
    "zero" + "gpu",
    "+" + "cu" + "128",
    "nlo" + "hmann",
    "json_" + "license",
    "proprietary / " + "zaqeris",
    "ready for github " + "initialization",
    "approved for upload " + "to github",
    "c:\\" + "users",
    "c:/" + "users",
    "/users" + "/",
)

FORBIDDEN_EXACT_CASE = (
    "ZAQERIS " + "Ai",
    "zaqeris/" + "zaqeris-7b",
    "zaqeris/" + "zqn-12b",
    "zaqeris/" + "zaqeris-axiom",
)

FORBIDDEN_CI_RE = re.compile(
    "|".join(re.escape(term) for term in FORBIDDEN_TERMS_CI),
    re.IGNORECASE,
)

FORBIDDEN_INFERENCE_IMPORTS = {
    "torch",
    "transformers",
    "tokenizers",
    "tiktoken",
    "gguf",
    "llama_cpp",
    "vllm",
    "sglang",
    "onnxruntime",
    "psutil",
    "jax",
    "ctransformers",
    "safetensors",
    "sentencepiece",
}


def _iter_repo_files(root: Path) -> List[Path]:
    if (root / ".git").exists():
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        untracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        rels = sorted(set(tracked + untracked))
        return [root / r for r in rels]

    skip_dirs = {".git", "build", "models", "weights", "__pycache__", ".pytest_cache"}
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and not any(part in skip_dirs for part in p.parts)
    ]


def _has_bad_tokenizer_wording(low_line: str) -> bool:
    if ("zaqeris-" + "7b") not in low_line and ("zaqeris_" + "7b") not in low_line:
        return False
    allowed = "only the tokenizer is reused from " + "qwen2"
    if allowed in low_line:
        return False
    stripped = (
        low_line.replace("qwen35", "")
        .replace("qwen3.8", "")
        .replace("qwen2.5", "")
        .replace("qwen2-", "")
        .replace("qwen1.5", "")
    )
    return ("qw" + "en") in stripped


def check_text_block(label: str, text: str) -> List[str]:
    hits: List[str] = []
    allowed_guard = '"zero' + 'gpu_active" not in'
    for idx, line in enumerate(text.splitlines(), 1):
        if FORBIDDEN_CI_RE.search(line):
            if label == "tests/test_hermes_36b.py" and allowed_guard in line:
                continue
            hits.append(f"FORBIDDEN HIT {label}:{idx}: {line.strip()[:140]}")
        for exact in FORBIDDEN_EXACT_CASE:
            if exact in line:
                hits.append(f"CASE HIT {label}:{idx} ({exact}): {line.strip()[:140]}")
        if _has_bad_tokenizer_wording(line.lower()):
            hits.append(f"TOKENIZER WORDING HIT {label}:{idx}: {line.strip()[:140]}")
    return hits


def run_scan(
    root: Path = REPO_ROOT,
    extra_texts: Optional[Dict[str, str]] = None,
) -> List[str]:
    violations: List[str] = []

    # 1. Zero runtime external ML/inference imports under core/inference/
    for py_path in sorted((root / "core" / "inference").glob("*.py")):
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            )
            for mod in mods:
                if mod.split(".")[0] in FORBIDDEN_INFERENCE_IMPORTS:
                    violations.append(
                        f"AST core/inference forbidden import: {py_path.relative_to(root).as_posix()} -> {mod}"
                    )

    # 2. Zero tracked weight files or build outputs
    if (root / ".git").exists():
        tracked_weights = subprocess.run(
            ["git", "-C", str(root), "ls-files", "models", "*.gguf", "build/*"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if tracked_weights:
            violations.append(f"Tracked weight/build files: {tracked_weights}")

    # 3. Scan repo text files
    for path in _iter_repo_files(root):
        if not path.is_file() or path.suffix.lower() in {".bin", ".png", ".jpg", ".jpeg", ".webp", ".gguf"}:
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        violations.extend(check_text_block(rel, text))

    # 4. Scan optional extra texts (release notes, correction notice, etc.)
    if extra_texts:
        for label, content in extra_texts.items():
            violations.extend(check_text_block(label, content))

    # 5. Pure ASCII .gitignore check (C8)
    gi_bytes = (root / ".gitignore").read_bytes()
    if not all(b < 128 for b in gi_bytes):
        violations.append(".gitignore contains non-ASCII bytes")

    # 6. Check C9 required tokenizer wording on public surfaces
    required_phrase = "only the tokenizer is reused from " + "Qwen2"
    for rel in (
        "README.md",
        "registry/model_architectures.json",
        "scripts/build_architectures_registry.py",
        "docs/backend-matrix/SUPPORT_MATRIX.md",
    ):
        if required_phrase not in (root / rel).read_text(encoding="utf-8"):
            violations.append(f"Missing required C9 wording '{required_phrase}' in {rel}")

    # 7. Honest registry counts check (A5)
    reg = json.loads((root / "registry" / "model_architectures.json").read_text(encoding="utf-8"))
    tested = [a["id"] for a in reg["architectures"] if a["support_status"] == "tested"]
    if (
        reg.get("total_count"),
        reg.get("tested_count"),
        reg.get("experimental_count"),
        reg.get("runtime_version"),
    ) != (184, 1, 183, "0.3.1") or tested != ["seed_36b"]:
        violations.append(
            f"Invalid registry metadata: total={reg.get('total_count')} tested={reg.get('tested_count')} "
            f"exp={reg.get('experimental_count')} ver={reg.get('runtime_version')} tested_ids={tested}"
        )

    # 8. Version string consistency check (A7)
    version_checks = (
        ("CMakeLists.txt", "project(qerlio VERSION 0.3.1"),
        ("pyproject.toml", 'version = "0.3.1"'),
        ("src/qerlio_abi.cpp", 'QERLIO_VERSION_STRING = "0.3.1"'),
        ("server/local_api/server.py", 'version="0.3.1"'),
        ("server/local_api/dashboard.html", "QERLIO RUNTIME v0.3.1"),
        ("server/hf_space_gateway/app.py", 'version="0.3.1"'),
        ("core/remote/hf_remote_client.py", "QERLIO-Runtime/0.3.1"),
        ("tools/benchmark/benchmark_suite.py", '"qerlio_version": "0.3.1"'),
        ("CHANGELOG.md", "## [0.3.1] -"),
    )
    for rel, needle in version_checks:
        if needle not in (root / rel).read_text(encoding="utf-8"):
            violations.append(f"Missing 0.3.1 version string '{needle}' in {rel}")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Run C14 forbidden-content and integrity scan.")
    parser.add_argument(
        "--text-file",
        action="append",
        default=[],
        help="Optional extra text file(s) (e.g. draft release notes) to include in the scan.",
    )
    args = parser.parse_args()

    extra: Dict[str, str] = {}
    for tf in args.text_file:
        p = Path(tf)
        extra[p.name] = p.read_text(encoding="utf-8")

    violations = run_scan(extra_texts=extra or None)
    print(f"forbidden_content_violations={len(violations)}")
    for item in violations:
        print(f"  - {item}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
