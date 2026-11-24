#!/usr/bin/env python3
"""Verify that all relative in-repo Markdown links resolve to existing files or directories."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"(?:\[(?:[^\[\]]|\[[^\]]*\]\([^)]*\))*\])\(([^)#\s]+)(?:#[^)]*)?\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def collect_markdown_files(root: Path = REPO_ROOT) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.rglob("*.md")):
        if ".git" in p.parts or "build" in p.parts or ".pytest_cache" in p.parts:
            continue
        files.append(p)
    return files


def find_broken_links(root: Path = REPO_ROOT) -> List[Tuple[str, str]]:
    broken: List[Tuple[str, str]] = []
    for md_path in collect_markdown_files(root):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        stripped = FENCE_RE.sub("", text)
        for match in LINK_RE.finditer(stripped):
            target = match.group(1).strip()
            if not target or target.startswith(("http://", "https://", "mailto:", "file://", "#")):
                continue
            resolved = (md_path.parent / target).resolve()
            if not resolved.exists():
                broken.append((md_path.relative_to(root).as_posix(), target))
    return broken


def main() -> int:
    broken = find_broken_links()
    print(f"checked_markdown_files={len(collect_markdown_files())} broken_links={len(broken)}")
    for src, target in broken:
        print(f"BROKEN: {src} -> {target}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
