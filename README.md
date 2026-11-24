# QERLIO Runtime

[![CI](https://github.com/ZAQERIS/QERLIO/actions/workflows/ci.yml/badge.svg)](https://github.com/ZAQERIS/QERLIO/actions)
[![Release](https://img.shields.io/github/v/release/ZAQERIS/QERLIO?color=blue)](https://github.com/ZAQERIS/QERLIO/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](sdk/python/)
[![Tests](https://img.shields.io/badge/tests-pytest-success.svg)](tests/)
[![Architectures](https://img.shields.io/badge/architectures-1%20tested%20%7C%20183%20experimental-blueviolet.svg)](registry/model_architectures.json)
[![Universal Gateway](https://img.shields.io/badge/gateway-OpenAI%20%7C%20Anthropic%20%7C%20Ollama-purple.svg)](server/local_api/)
[![Agent Mode](https://img.shields.io/badge/agents-ReAct%20%7C%20Tri--Agent%20Sandbox-emerald.svg)](core/agent/)

> **Your AI. Your Device. Your Rules.**  
> *Developed by ZAQERIS AI*

QERLIO Runtime is a local-first AI execution platform and tooling suite providing a from-scratch GGUF v2/v3 native inference engine (Python standard library, NumPy, and a from-scratch C11 SIMD quantized GEMM library), a C++20 runtime core, chat-template and tool-call parsers, and a multi-protocol local API gateway.

---

## What's New in v0.3.1

* **Strict `tested` Registry Criteria (F1):** Applied the six-part verification standard (`T-A` through `T-F`, defined below) across [`registry/model_architectures.json`](registry/model_architectures.json) (`184` total: **`1` tested**, **`183` experimental**). `seed_36b` (`SeedOssForCausalLM` on official `NousResearch/Hermes-4.3-36B-GGUF`) remains `tested` with committed prompt/token/output logs in [`tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json`](tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json); `qwen35` and the three ZAQERIS catalog entries (`zaqeris_axiom`, `zqn_12b`, `zaqeris_7b`) were recategorized to `experimental` until end-to-end `>= 64`-token runs on official weights are recorded and committed.
* **Official Apache-2.0 `LICENSE` File (F2):** Added the full Apache License, Version 2.0 text at [`LICENSE`](LICENSE).
* **Honest Test Labeling & Opt-In Real-Weight Pytest (F4):** Relabeled [`tests/test_t4_to_t10_engine.py`](tests/test_t4_to_t10_engine.py) and [`tests/test_qwen35_engine.py`](tests/test_qwen35_engine.py) explicitly as synthetic/toy-GGUF unit and plumbing tests (with no silent fallback to toy weights), and added opt-in [`tests/test_real_hermes_36b_optin.py`](tests/test_real_hermes_36b_optin.py) gated on `QERLIO_RUN_REAL_HERMES=1` and the official `models/hermes-4_3_36b-Q4_K_M.gguf` checkpoint (`skip` when absent in default CI).
* **ASan/UBSan Coverage & Static Doc Audit (F3 & F5):** Expanded [`tests/cpp_safety_test.cpp`](tests/cpp_safety_test.cpp) to exercise single-row and multi-row paths of every kernel in [`core/inference/native_gemm.c`](core/inference/native_gemm.c) under GitHub Actions `C++20 AddressSanitizer & UndefinedBehaviorSanitizer` (`detect_leaks=1:halt_on_error=1`), and updated or marked stale historical docs.
* **Full Repository Cleanup (C1–C14):** Fixed broken relative test links in [`docs/models/HERMES_4_3_36B.md`](docs/models/HERMES_4_3_36B.md) with [`scripts/check_markdown_links.py`](scripts/check_markdown_links.py); updated `/v1/models` and `/api/tags` in [`server/local_api/server.py`](server/local_api/server.py) and [`server/hf_space_gateway/app.py`](server/hf_space_gateway/app.py) to return empty lists when unloaded and list only actively loaded models; labeled formula estimates in [`core/compression/quant_matrix.py`](core/compression/quant_matrix.py) and `/v1/telemetry/hud` explicitly as theoretical (`measured=false`) with host memory omitted by default; aligned [`third_party/DEPENDENCY_INVENTORY.md`](third_party/DEPENDENCY_INVENTORY.md) and security/risk docs with the repository tree; added [`pyproject.toml`](pyproject.toml), [`requirements.txt`](requirements.txt), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and PR/issue templates; updated [`CONTRIBUTING.md`](CONTRIBUTING.md); converted [`.gitignore`](.gitignore) to 100% ASCII; standardized `ZAQERIS AI` / `ZAQERIS/ZAQERIS-7B` naming (`trained from scratch; only the tokenizer is reused from Qwen2`); documented `v0.2.2` tag-only history; and reinforced archive markings throughout Phase 0/1 documents.

---

## Installation & Build

* **Supported Python:** Python `3.10+`.
* **Core Inference Engine Dependencies (`core/inference/`):** Python standard library + `numpy>=1.24` (optional C11 `native_gemm` library accelerates `seed_oss` CPU GEMM when built/loaded; NumPy fallback is built in).

1. **Install QERLIO (core runtime only, or with gateway + test dependencies):**
   ```bash
   # Core runtime (numpy only)
   pip install -e .

   # Core runtime + local HTTP gateway/CLI + pytest suite (mirrors requirements.txt)
   pip install -e ".[dev]"
   ```
   See [`third_party/DEPENDENCY_INVENTORY.md`](third_party/DEPENDENCY_INVENTORY.md) and [`pyproject.toml`](pyproject.toml).

2. **Build Optional C11 `native_gemm` Shared Library & C++20 Core (CMake):**
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ctest --test-dir build -C Release --output-on-failure
   ```

3. **Run the Test Suite:**
   ```bash
   python -m pytest tests/ -v
   ```

---

## Native Engine Capabilities (v0.3.0 Baseline + v0.3.1 Corrections)

* **From-Scratch GGUF v2/v3 Loader & Streaming (`core/inference/gguf_loader.py`):** Binary parser and memory-mapped tensor reader with magic/version/bounds validation and row-level dequantized lookup (`get_tensor_rows_f32`).
* **16 GGML Dequantization Kernels (`core/inference/dequant.py`):** Vectorized NumPy + scalar reference decoders matching within `< 1e-6` absolute error for `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `IQ3_S`, and `IQ4_XS`. Verified against offline `gguf-py 0.19.0` reference slices in [`tests/fixtures/t2_gguf_py_dequant_reference.json`](tests/fixtures/t2_gguf_py_dequant_reference.json) via `python -m pytest tests/test_t1_t2_gguf_and_dequant.py -v`.
* **Native Byte-Level BPE Tokenizer (`core/inference/tokenizer.py`):** Pure-Python GGUF tokenizer supporting NFC normalization, `seed-coder` / `qwen35` / `gpt2` pre-tokenization, special/control token splitting, and exact round-trip encode/decode. Verified against offline `tokenizers 0.23.3` on 64 cases in [`tests/fixtures/hermes_tokenizer_reference.json`](tests/fixtures/hermes_tokenizer_reference.json) via `python -m pytest tests/test_t3_tokenizer.py -v`.
* **`SeedOssForCausalLM` Native Engine (`core/inference/seed_oss_model.py`):** Executes ByteDance Seed-OSS / `NousResearch/Hermes-4.3-36B` checkpoints (`seed_oss` architecture: RMSNorm, GQA with QKV biases, half-split RoPE, `post_attention_norm`, SwiGLU MLP) using only the Python standard library, NumPy, and `core/inference/native_gemm.c`. Verified on the official `NousResearch/Hermes-4.3-36B-GGUF` file `models/hermes-4_3_36b-Q4_K_M.gguf` (`21,762,145,216` bytes, SHA-256 `17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08`; all 64 layers, `hidden=5120`, `ffn=27648`, `heads=80`, `kv_heads=8`, `head_dim=128`, `vocab=155136`). Recorded in [`tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json`](tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json) and reproduced with `python scripts/verify_hermes_36b_real.py --model-path models/hermes-4_3_36b-Q4_K_M.gguf --reference-fixture tests/fixtures/hermes_4_3_36b_transformers_reference.json`:
  * **T1:** 27 metadata KVs and 771 tensor descriptors match [`tests/fixtures/hermes_4_3_36b_q4_k_m_metadata.json`](tests/fixtures/hermes_4_3_36b_q4_k_m_metadata.json).
  * **T2:** 21 offline `gguf-py` dequantization cases, `max_abs_err=0.00e+00`.
  * **T3:** 64/64 tokenizer cases match `tokenizers 0.23.3` with exact round-trip.
  * **T4:** 5 fixed prompts, batched prefill in `47.87s`; top-1 identical on 5/5 vs HuggingFace `transformers 5.20.1`, top-5 overlap 5/5, max top-20 logit diff `2.10e-05` ([`tests/fixtures/hermes_4_3_36b_transformers_reference.json`](tests/fixtures/hermes_4_3_36b_transformers_reference.json)).
  * **T5:** 5 prompts × 64 greedy tokens = `320/320` tokens identical to the `transformers` reference (elapsed `2362.87s`).
  * **T6:** cached vs uncached decode `max_abs_diff=7.58e-05`, same top-1 (`id=1371`).
  * **T7:** two-turn chat (`'The capital of France is Paris.'`, then `'The Seine River flows through Paris.'`) and `thinking=True` closed `<think>…</think>` followed by an answer containing `4` (completed deterministically in two segments from a saved 160-token greedy prefix; elapsed `10682.64s`); `keep_cots` False/True history handling checked.
  * **T8:** model-generated `<tool_call>` for `get_weather` parsed into OpenAI `tool_calls` and Anthropic `tool_use`.
  * **T9:** context-overflow error, KV-memory-budget error, stop-sequence trimming (`" Paris"`), and mid-stream cancellation (0 active sessions afterwards).

  Offline reference generators live strictly under [`tests/reference/`](tests/reference/) (`generate_hermes_36b_transformers_reference.py`, `generate_t2_dequant_reference.py`, `generate_t3_tokenizer_reference.py`, `generate_t4_t5_micro_reference.py`) and are never runtime dependencies. Not measured: peak RSS memory, interactive tok/s outside the recorded runs above.
* **Native C11 SIMD Quantized GEMM & Sequential Streaming (`include/qerlio/native_gemm.h`, `core/inference/native_gemm.c`, `core/inference/native_gemm.py`):** From-scratch AVX2+FMA+F16C kernels with a portable scalar fallback for `Q4_K`, `Q6_K`, `Q8_0`, and `F32`, double-buffered sequential tensor streaming (`SequentialTensorStreamer`), and batched `SeedOssEngine.forward_batch()`. A single 64-layer forward pass on `'The capital of France is'` (top-1 `' Paris'`, `id=10805`, `logit=16.4356`) took `419.43s` on the NumPy-only path and `50.75s` with the C GEMM path (measured with `SeedOssEngine("models/hermes-4_3_36b-Q4_K_M.gguf", preload_weights=False).forward(...)`; synthetic unit parity runs in `python -m pytest tests/test_native_gemm.py tests/test_t4_to_t10_engine.py -v`).
* **`Qwen35ForCausalLM` Hybrid Engine (`core/inference/qwen35_model.py`, `experimental`):** Implements Gated DeltaNet recurrent linear-attention + Gated Full-Attention layers with `Qwen35HybridCache` and ChatML formatting. An earlier local check on an unofficial quant (`models/Qwen3.8-27B-UD-Q4_K_M.gguf`, all 64 trunk layers, prompt `'The capital of France is'`) predicted `' Paris'` (`logit=17.5078`, `prefill=610.40s`, 4 greedy tokens `' Paris.\nThe'`); because that file was an unofficial UD quant and generated `< 64` tokens, `qwen35` is classified `experimental` in v0.3.1. CI tests (`python -m pytest tests/test_qwen35_engine.py -v`) exercise synthetic 4-layer weights only.
* **KV Cache, Sampler & Streaming (`core/inference/kv_cache.py`, `core/inference/sampler.py`):** Position-aware GQA `KVCache` with memory budget checking (`InsufficientMemoryError`, `ContextLengthExceededError`), greedy/temperature/top-k/top-p/repetition-penalty sampling, stop sequences, and mid-stream cancellation.
* **Live Gateway Execution (`server/local_api/server.py`, `tools/qerlio_cli.py`):** `/v1/models/load`, `/v1/models/unload`, `/v1/chat/completions`, and `/v1/messages` execute the loaded native engine with `"simulated": false` when a GGUF model is loaded, and return explicit `"simulated": true` stubs only when no model is loaded. CLI `qerlio hermes chat --model-path <model.gguf>` invokes `load_gguf_engine`.

### Known Limitations (v0.3.1)
* Inference executes on CPU only; CUDA/Metal/Vulkan GPU acceleration is not implemented.
* `seed_oss` uses the C11 quantized GEMM (`Q4_K`, `Q6_K`, `Q8_0`, `F32`) when the shared library can be built or loaded and falls back to NumPy otherwise; `qwen35` still uses the NumPy path.
* 36B CPU inference is non-interactive: the recorded 320-token batched generation took `2362.87s`, and the `thinking=True` run took `10682.64s` (completed in two deterministic segments from the saved 160-token prefix). Peak RSS memory was `not measured`.
* Only `seed_oss` (`tested`) and `qwen35` (`experimental`, synthetic CI + partial unofficial run) decoder graphs exist in `core/inference/`; vision encoders, Mixture-of-Experts (`qwen35moe`, `zaqeris_axiom` weights), and all other 182 catalogued architectures have no local weight execution.

---

## Historical Release Notes (Archive — Superseded Where Noted)

* **v0.3.0 (2026-11-22):** Initial merge introducing the native GGUF v2/v3 engine (`core/inference/`), C11 SIMD GEMM, `Hermes-4.3-36B` verification, and `qwen35`. *(Audit correction noted in v0.3.1: v0.3.0 listed `5 tested / 179 experimental`, counted unofficial `qwen35` and parser-only ZAQERIS entries as `tested`, mislabeled synthetic toy tests as real-model tests, and lacked a root `LICENSE` file; corrected in v0.3.1.)*
* **v0.2.2 (2026-11-20, git tag only — no standalone GitHub Release):** Added `Hermes36BAdapter`, tag parsing, GGUF header KV inspection, `qerlio hermes download`, and stub-honesty fixes. *(Historical archive note: the v0.2.2 release listed `183` entries (`3 tested / 180 experimental`) and noted that QERLIO "does not yet execute `SeedOssForCausalLM` tensor weights locally"; that limitation was superseded by `core/inference/seed_oss_model.py` in v0.3.0+.)*
* **v0.2.1 (2026-11-20):** Registry cleanup (`182` historical entries), AXIOM metadata correction (~3.16 bpp), placeholder telemetry removal, and path cleanup.
* **v0.2.0 (2026-11-19):** Initial architecture catalog, sandboxed ReAct agent mode, local multi-protocol API gateway stub, and developer HUD.

---

## Model Architectures: 1 Tested, 183 Experimental

QERLIO keeps a registry of 184 model architectures in [`registry/model_architectures.json`](registry/model_architectures.json), generated by [`scripts/build_architectures_registry.py`](scripts/build_architectures_registry.py). Each entry carries `support_status` using **only** this definition:

### Definition of `tested` vs `experimental`
An architecture may be marked **`tested`** only if **all six** criteria (`T-A` through `T-F`) are true, documented, and verifiable:
* **`T-A` Official weights:** An official weight file or official quantization from the model's official publisher repository (unofficial UD re-quants or third-party re-uploads do not qualify).
* **`T-B` End-to-end execution:** QERLIO loads the official file, tokenizes a documented prompt, runs a forward pass, and generates tokens (header parsing, chat-template/tool-tag unit tests, or synthetic toy GGUFs do not qualify).
* **`T-C` Minimum generation length:** At least `64` newly generated tokens under greedy decoding (or an explicitly documented equivalent protocol) from a documented prompt.
* **`T-D` Documented prompt and output:** Exact prompts/token IDs and generated outputs/token IDs committed in the repository (`tests/fixtures/` or `docs/`).
* **`T-E` Reproducible command:** Exact command recorded alongside the result.
* **`T-F` Verified SHA-256 and byte size:** Exact SHA-256 checksum and byte length recorded next to the result.

Any architecture missing any of `T-A`–`T-F` is **`experimental`**.

* **`tested` (1):**
  * **`seed_36b` (`NousResearch/Hermes-4.3-36B-GGUF` / `hermes-4_3_36b-Q4_K_M.gguf`):** Official repo `NousResearch/Hermes-4.3-36B-GGUF`, size `21,762,145,216` bytes, SHA-256 `17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08`; `5 x 64 = 320` greedy tokens matched HuggingFace `transformers 5.20.1` (`320/320`) in [`tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json`](tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json); reproduce with `python scripts/verify_hermes_36b_real.py --model-path models/hermes-4_3_36b-Q4_K_M.gguf --reference-fixture tests/fixtures/hermes_4_3_36b_transformers_reference.json` or `QERLIO_RUN_REAL_HERMES=1 python -m pytest tests/test_real_hermes_36b_optin.py -v`.
* **`experimental` (183):** Catalog metadata, prompt/tool adapters, header inspection, synthetic unit tests, or incomplete/unofficial runs (including `qwen35`, `zaqeris_axiom`, `zqn_12b`, and `zaqeris_7b`).

| Architectural Family | Tested | Experimental | Representative Architectures |
| :--- | :---: | :---: | :--- |
| **ZAQERIS Native** | 0 | 3 | `zaqeris_axiom`, `zqn_12b`, `zaqeris_7b` |
| **Dense Transformer** | 0 | 51 | Llama 1/2/3/3.1/3.2/3.3, Qwen 1/1.5/2/2.5, Mistral 7B, Gemma 1/2, Phi 1/2/3/4, Yi, Falcon, OLMo |
| **Mixture of Experts** | 0 | 19 | Mixtral 8x7B/8x22B, DeepSeek-V2/V3, Grok-1, DBRX, Snowflake Arctic, Qwen2-MoE, OLMoE |
| **Reasoning & Dialectic** | 1 | 16 | `seed_36b` (ByteDance Seed-OSS-36B / Hermes 4.3 36B — **tested**), DeepSeek-R1 / R1-Zero / Distill, QwQ-32B-Preview, Marco-o1, Sky-T1, NuminaMath |
| **Code Specialist** | 0 | 22 | Qwen2.5-Coder, Codestral, StarCoder 1/2, DeepSeek-Coder-V2, CodeGeeX4 |
| **State Space & Hybrid** | 0 | 15 | `qwen35` (experimental: synthetic CI + prior 4-token unofficial-quant check), Mamba 1/2, Jamba, Zamba 1/2, RWKV 4/5/6/7, RecurrentGemma, HGRN |
| **Vision-Language** | 0 | 25 | LLaVA 1.5/NeXT/OneVision, Qwen2-VL, PaliGemma 1/2, Pixtral 12B, InternVL2 |
| **Embedding & Reranker** | 0 | 15 | ModernBERT, BGE-M3 / Reranker, E5, nomic-embed-text, ColBERTv2 |
| **Speech & Audio** | 0 | 10 | Whisper large-v3 / turbo, Moonshine, SeamlessM4T v1/v2, wav2vec 2.0 |
| **Edge & Micro** | 0 | 7 | SmolLM 1/2, MobileLLM, OpenELM, H2O-Danube 1/2/3 |

**ZAQERIS models**

| ID | Model | Details |
| :--- | :--- | :--- |
| `zaqeris_axiom` | [AXIOM](https://huggingface.co/ZAQERIS/ZAQERIS_AXIOM) | Sparse MoE trained from scratch: 79.43B total / 12.39B active parameters per token (16 experts per layer, top-2 routing); published as ZAQERIS-M3, ~3.16 bits per parameter (`experimental` in registry pending Section 2 real-weight verification). |
| `zqn_12b` | [ZQN-12B](https://huggingface.co/ZAQERIS/ZQN-12B) | 12B model for reasoning and agentic tool use (`experimental` in registry pending Section 2 real-weight verification). |
| `zaqeris_7b` | [ZAQERIS-7B](https://huggingface.co/ZAQERIS/ZAQERIS-7B) | 7B model for function calling and tool use, trained from scratch (only the tokenizer is reused from Qwen2; `experimental` in registry pending Section 2 real-weight verification). |

Inspect architectures via CLI:
```bash
python tools/qerlio_cli.py architectures count
python tools/qerlio_cli.py architectures list --status tested
python tools/qerlio_cli.py architectures list --family "ZAQERIS Native"
python tools/qerlio_cli.py architectures info seed_36b
```

---

## Agent Mode & Tool Execution

QERLIO features an integrated, sandboxed tool execution engine allowing models to execute computations, search files, and run code safely:

```bash
# Execute safe calculator
python tools/qerlio_cli.py agent execute --tool calculator --args '{"expression": "(12 * 15) + 30"}'

# Execute sandboxed python code
python tools/qerlio_cli.py agent execute --tool python_eval --args '{"code": "result = sum(range(101))"}'

# Run offline agent simulation (tool-parsing loop)
python tools/qerlio_cli.py agent run --prompt "Verify theorem steps"
```

---

## Universal Multi-Protocol API Gateway & Diagnostic HUD

Launch the local server and developer dashboard:
```bash
python tools/qerlio_cli.py server --port 8787
```

* **Live Developer HUD:** Open `http://localhost:8787` in any browser to inspect runtime telemetry (unmeasured throughput, TTFT, and memory fields read `"no live measurement"` until measured in an active session), the architecture catalog with its support tiers, and the Agent Studio.
* **OpenAI Protocol:** `POST http://localhost:8787/v1/chat/completions` (SSE Streaming, Function & Tool Calling; requires `POST /v1/models/load` for live native inference, otherwise returns `"simulated": true`)
* **Anthropic Claude Protocol:** `POST http://localhost:8787/v1/messages` (with `tool_use`)
* **Ollama Protocol:** `POST http://localhost:8787/api/generate` & `POST http://localhost:8787/api/chat`
* **Architectures Endpoints:** `GET http://localhost:8787/v1/architectures` (optional `?status=tested|experimental`) & `/v1/architectures/{id}`
* **Agent Endpoints:** `POST http://localhost:8787/v1/agents/execute` & `POST http://localhost:8787/v1/agents/run`

---

## Automated Verification Suite

Run the default CI / offline test suite (uses synthetic toy GGUFs and committed offline reference fixtures; skips [`tests/test_real_hermes_36b_optin.py`](tests/test_real_hermes_36b_optin.py) cleanly when `QERLIO_RUN_REAL_HERMES=1` or the 21.76 GB GGUF is absent):
```bash
python -m pytest tests/ -v
```

Run the opt-in real-weight Hermes-4.3-36B verification test locally (requires official `models/hermes-4_3_36b-Q4_K_M.gguf`):
```bash
QERLIO_RUN_REAL_HERMES=1 python -m pytest tests/test_real_hermes_36b_optin.py -v
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
