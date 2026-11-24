# QERLIO Runtime — Third-Party Dependency Inventory & Licensing Audit

**Policy:** Every external dependency used at runtime, in optional tooling, or offline for reference-fixture generation is tracked accurately in this file with its upstream source, version constraint, and SPDX license identifier. The QERLIO repository itself is licensed under `Apache-2.0` (see [`LICENSE`](../LICENSE)). No vendored C/C++ third-party headers or separate `third_party/licenses/` files are shipped in this repository.

---

## 1. Direct Runtime Dependencies (`core/inference/`)

QERLIO's native inference engine (`core/inference/`) is implemented from scratch using only the Python standard library, NumPy, and standard C11/C++20 (zero runtime dependency on `llama.cpp`, `ggml`, PyTorch, Transformers, or other external inference engines).

| Dependency Name | Upstream Repository / Source | Version Constraint | SPDX License Identifier | Usage in QERLIO | Compliance Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NumPy** | `https://numpy.org` | `>= 1.24` | `BSD-3-Clause` | Python array/numerical kernels and dequantization fallback (`core/inference/`) | Sole third-party runtime Python dependency of the native inference engine |

---

## 2. Optional Gateway, CLI & Test Dependencies (`pyproject.toml` Extras `[server]` / `[dev]`)

These standard open-source packages are used only by the optional local HTTP gateway (`server/local_api/`), CLI helpers (`tools/qerlio_cli.py`), or automated test runner (`tests/`). They are never imported by `core/inference/`.

| Package Name | Upstream Source | Version Constraint | SPDX License Identifier | Usage in QERLIO |
| :--- | :--- | :--- | :--- | :--- |
| **FastAPI** | `https://fastapi.tiangolo.com` | `>= 0.100` | `MIT` | Optional local HTTP gateway (`server/local_api/`, `server/hf_space_gateway/`) |
| **Pydantic** | `https://docs.pydantic.dev` | `>= 2.0` | `MIT` | Request/response schemas and manifest validation |
| **Uvicorn** | `https://www.uvicorn.org` | `>= 0.23` | `BSD-3-Clause` | Optional ASGI server for local HTTP gateway |
| **psutil** | `https://github.com/giampaolo/psutil` | `>= 5.9` | `BSD-3-Clause` | Optional opt-in host diagnostic reporting (`QERLIO_EXPOSE_HOST_METRICS=1`, `qerlio doctor`) |
| **pytest** | `https://docs.pytest.org` | `>= 7.4` | `MIT` | Automated verification suite (`tests/`) |
| **httpx** | `https://www.python-httpx.org` | `>= 0.24` | `BSD-3-Clause` | FastAPI `TestClient` transport in API tests (`tests/`) |

---

## 3. Isolated Optional Compression Adapter Interface (`formats/zaqeris_compression`)

`formats/zaqeris_compression/zaqeris_m3_adapter.py` is an isolated optional Python interface within this Apache-2.0 repository. It imports an optional external module only inside a `try ... except ImportError` guard and cleanly reports `{"status": "UNAVAILABLE"}` when absent. No external compression engine, binary wheel, or non-open package is vendored, bundled, or required by QERLIO.

---

## 4. Offline Reference Tools (Dev/Fixture Generation Only — Never Runtime Dependencies)

External ML libraries and GGUF inspectors are strictly forbidden as runtime dependencies of `core/inference/`. They are used offline only inside [`tests/reference/`](../tests/reference/) to regenerate static reference verification fixtures.

| Tool / Library | Generic Version | SPDX License Identifier | Offline Usage |
| :--- | :--- | :--- | :--- |
| **gguf (`gguf-py`)** | `0.19.0` | `MIT` | Offline GGUF metadata, tensor inspection & T2 dequantization reference fixture generation (`tests/reference/`) |
| **tokenizers** | `0.23.3` | `Apache-2.0` | Offline T3 BPE tokenizer reference fixture generation (`tests/reference/`) |
| **transformers** | `5.20.1` | `Apache-2.0` | Offline T4/T5 `SeedOssForCausalLM` / `SeedOssDecoderLayer` reference verification (`tests/reference/`); never imported by `core/inference/` |
| **torch** | `>= 2.0` (2.11 used for static reference fixtures) | `BSD-3-Clause` | Offline tensor execution for `transformers` reference verification (`tests/reference/`); never imported by `core/inference/` |
| **safetensors** | `>= 0.4` | `Apache-2.0` | Offline conversion & reference inspection only |
