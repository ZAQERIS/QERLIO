import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_forbidden_content import run_scan
from scripts.check_markdown_links import find_broken_links
from server.hf_space_gateway.app import app as space_gateway_app
from server.local_api.server import app as local_app, unload_native_models


def test_zero_broken_markdown_links() -> None:
    broken = find_broken_links(REPO_ROOT)
    assert broken == [], f"Broken Markdown links detected: {broken}"


def test_forbidden_content_and_integrity_scan_clean() -> None:
    violations = run_scan(REPO_ROOT)
    assert violations == [], f"C14/integrity violations detected: {violations}"


def test_gitignore_is_pure_ascii() -> None:
    raw = (REPO_ROOT / ".gitignore").read_bytes()
    assert all(b < 128 for b in raw), ".gitignore must contain ASCII bytes only"


def test_version_strings_consistent_0_3_1() -> None:
    expected = {
        "CMakeLists.txt": "project(qerlio VERSION 0.3.1",
        "pyproject.toml": 'version = "0.3.1"',
        "src/qerlio_abi.cpp": 'QERLIO_VERSION_STRING = "0.3.1"',
        "server/local_api/server.py": 'version="0.3.1"',
        "server/local_api/dashboard.html": "QERLIO RUNTIME v0.3.1",
        "server/hf_space_gateway/app.py": 'version="0.3.1"',
        "core/remote/hf_remote_client.py": "QERLIO-Runtime/0.3.1",
        "tools/benchmark/benchmark_suite.py": '"qerlio_version": "0.3.1"',
        "CHANGELOG.md": "## [0.3.1] -",
    }
    for rel, needle in expected.items():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert needle in text, f"Missing '{needle}' in {rel}"


def test_unloaded_apis_return_empty_model_lists() -> None:
    unload_native_models()
    local_client = TestClient(local_app)

    r_models = local_client.get("/v1/models")
    assert r_models.status_code == 200
    assert r_models.json()["loaded_count"] == 0
    assert r_models.json()["data"] == []

    r_tags = local_client.get("/api/tags")
    assert r_tags.status_code == 200
    assert r_tags.json()["loaded_count"] == 0
    assert r_tags.json()["models"] == []

    gw_client = TestClient(space_gateway_app)
    r_gw_models = gw_client.get("/v1/models")
    assert r_gw_models.status_code == 200
    assert r_gw_models.json()["backend_loaded"] is False
    assert r_gw_models.json()["loaded_count"] == 0
    assert r_gw_models.json()["data"] == []
