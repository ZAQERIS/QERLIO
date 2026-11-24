# Changelog

All notable changes to the QERLIO Runtime project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and internal engineering directive `ZAQ-ENG-DIR-2026-11-20`.

---

## [0.3.1] - 2026-11-24

### Changed
- **Honest `tested` Registry Criteria (`registry/model_architectures.json`, `scripts/build_architectures_registry.py`, `tests/test_architectures_registry.py`):** Recategorized `qwen35` (prior `610.40s` 4-token run used an unofficial UD quant `Qwen3.8-27B-UD-Q4_K_M.gguf`) and ZAQERIS catalog entries (`zaqeris_axiom` / `ZAQERIS/ZAQERIS_AXIOM`, `zqn_12b` / `ZAQERIS/ZQN-12B`, and `zaqeris_7b` / `ZAQERIS/ZAQERIS-7B` — trained from scratch; only the tokenizer is reused from Qwen2 — which lack `>= 64`-token official-weight logs) to `experimental`. Only `seed_36b` (`NousResearch/Hermes-4.3-36B-GGUF`, SHA-256 `17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08`) remains `tested` (`1` tested, `183` experimental, `184` total).
- **Relabeled Synthetic Tests & Added Opt-In Real-Weight Test:** Explicitly labeled [`tests/test_t4_to_t10_engine.py`](tests/test_t4_to_t10_engine.py) and [`tests/test_qwen35_engine.py`](tests/test_qwen35_engine.py) as synthetic toy-GGUF unit tests, removed the silent toy fallback from `test_qwen35_engine.py`, and added opt-in [`tests/test_real_hermes_36b_optin.py`](tests/test_real_hermes_36b_optin.py) (`QERLIO_RUN_REAL_HERMES=1`).
- **API, Telemetry & Dependency Cleanup (C1–C14):** Fixed broken relative links in `docs/models/HERMES_4_3_36B.md` and added `scripts/check_markdown_links.py`; updated `/v1/models` and `/api/tags` in `server/local_api/server.py` and `server/hf_space_gateway/app.py` to list loaded models only (empty when unloaded); labeled `quant_matrix.py` / `/v1/telemetry/hud` formula estimates explicitly as theoretical and omitted host RAM/CPU figures by default; aligned `third_party/DEPENDENCY_INVENTORY.md`, `THREAT_MODEL.md`, and `RISK_REGISTER.md` with the actual tree; standardized `ZAQERIS AI` and `ZAQERIS/ZAQERIS-7B` naming; made `.gitignore` pure ASCII; documented tag `v0.2.2` as tag-only; and added `scripts/check_forbidden_content.py` + `tests/test_repo_cleanliness.py`.
- **Sanitizer Coverage & Stale Doc Fixes:** Expanded multi-row C GEMM kernel exercise (`m=5`) in `tests/cpp_safety_test.cpp` under CI ASan+UBSan (`detect_leaks=1:halt_on_error=1`), added a static CI-coverage test for all `core/inference/*.c` files, updated stale architecture/ADR/matrix docs, clarified inline historical archive annotations throughout Phase 0/1 documents, corrected the `[0.3.0]` heading date to `2026-11-22`, and removed the nonexistent `develop` branch trigger from `.github/workflows/ci.yml`.
- **Version Bump:** Updated runtime version strings to `0.3.1`.

### Added
- **Root `LICENSE`, Packaging & Community Files:** Added the full official Apache License, Version 2.0 text at [`LICENSE`](LICENSE), added [`pyproject.toml`](pyproject.toml), [`requirements.txt`](requirements.txt), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md), and [`.github/ISSUE_TEMPLATE/bug_report.yml`](.github/ISSUE_TEMPLATE/bug_report.yml), and updated [`CONTRIBUTING.md`](CONTRIBUTING.md) with sanitizer, tested-definition, and PR-gate requirements.
- **Committed Real-Weight Record:** Added [`tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json`](tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json) storing exact T1–T9 prompts, token IDs, and outputs for `NousResearch/Hermes-4.3-36B-GGUF`.

### Known Limitations
- CPU execution only; CUDA/Metal/Vulkan GPU acceleration is not implemented.
- `qwen35` runs on the NumPy path and remains `experimental`; vision encoders, MoE weight execution, and all other 182 catalogued architectures remain unexecuted locally.
- 36B CPU latency is non-interactive (`2362.87s` for 320 batched tokens; `10682.64s` for the `thinking=True` run completed in two segments from a saved 160-token greedy prefix). Peak memory is `not measured`.

---

## [0.3.0] - 2026-11-22

> *(Historical v0.3.0 entry — published prior to management review; see `[0.3.1]` above for audit corrections to the `tested` count, synthetic test labels, and missing `LICENSE`.)*

### Added
- **From-Scratch GGUF v2/v3 Loader (`core/inference/gguf_loader.py`):** Pure Python stdlib + NumPy GGUF binary parser, metadata KV reader, alignment calculator, zero-copy memory-mapped tensor access (`mmap`), row-slicing (`get_tensor_rows_f32`), and corrupt/truncated file rejection.
- **16 GGML Dequantization Kernels (`core/inference/dequant.py`):** Vectorized NumPy kernels and pure-Python scalar reference decoders verified within `< 1e-6` absolute error across `F32`, `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `IQ3_S`, and `IQ4_XS`.
- **Native Byte-Level BPE Tokenizer (`core/inference/tokenizer.py`):** Pure-Python tokenizer reading vocab, merges, and special/control tokens directly from GGUF metadata with NFC normalization, `seed-coder` / `qwen35` / `gpt2` pre-tokenization, and exact encode/decode round-trip.
- **`SeedOssForCausalLM` Native Inference Engine (`core/inference/seed_oss_model.py`):** From-scratch implementation of Hermes 4.3 36B (`seed_oss` architecture: RMSNorm, half-split RoPE, GQA with QKV biases, `post_attention_norm`, SwiGLU MLP, dynamic GGUF hyperparameter loading, streaming generation, and `Hermes36BAdapter` integration).
- **`Qwen35ForCausalLM` Hybrid Engine (`core/inference/qwen35_model.py`):** Native implementation of `qwen35` hybrid Gated DeltaNet linear-attention layers + Gated Full-Attention layers with `Qwen35HybridCache`, partial half-split RoPE, and ChatML formatting; verified on full 64-layer `models/Qwen3.8-27B-UD-Q4_K_M.gguf` (`'The capital of France is'` -> `' Paris.\nThe'`).
- **Position-Aware `KVCache` & `Sampler` (`core/inference/kv_cache.py`, `core/inference/sampler.py`):** Dynamic KV cache with upfront memory budget validation (`InsufficientMemoryError`, `ContextLengthExceededError`), plus greedy, temperature, top-k, top-p, repetition penalty, stop sequences, and mid-stream cancellation.
- **Live Local API & CLI Model Execution (`server/local_api/server.py`, `tools/qerlio_cli.py`):** Added `POST /v1/models/load`, `POST /v1/models/unload`, unified `load_gguf_engine`, live non-simulated `/v1/chat/completions` and `/v1/messages` inference (`"simulated": false`), and `qerlio hermes chat --model-path`.
- **C11 SIMD Quantized GEMM & Streaming (`core/inference/native_gemm.c`, `core/inference/native_gemm.py`, `include/qerlio/native_gemm.h`):** AVX2+FMA+F16C kernels with scalar fallback for `Q4_K`, `Q6_K`, `Q8_0`, `F32`; `SequentialTensorStreamer` with double-buffered prefetch; batched `SeedOssEngine.forward_batch()`; kernel bounds tests in `tests/cpp_safety_test.cpp` (ASan/UBSan job also builds the C source).
- **Offline Reference Suite (`tests/reference/`, `tests/fixtures/`):** generators and committed fixtures for T2 (`gguf-py 0.19.0`), T3 (`tokenizers 0.23.3`), and T4/T5 on the real 36B weights (`transformers 5.20.1`); none are runtime dependencies.
- **Real-Weight Verification (`scripts/verify_hermes_36b_real.py`, `docs/models/HERMES_4_3_36B.md`):** T1–T9 run on `hermes-4_3_36b-Q4_K_M.gguf` with recorded results (T4 5/5, T5 320/320, T6 `7.58e-05`).
- **Registry & CI Updates (Historical v0.3.0 — superseded in v0.3.1):** Tagged `seed_36b` and `qwen35` as `"tested"` (184 total: 5 tested, 179 experimental — corrected to `1 tested / 183 experimental` in v0.3.1), and added C++20 AddressSanitizer/UndefinedBehaviorSanitizer testing in CMake/CI (`tests/cpp_safety_test.cpp`).

---

## [0.2.2] - 2026-11-20

> *(Release note: `v0.2.2` exists as git tag `v0.2.2` only and was not published as a standalone GitHub Release.)*

### Added
- **Hermes 4.3 36B (`Seed-OSS`) Chat Template & Tag Parser (`core/models/hermes_adapter.py`):** Prompt formatter and output tag parser for [`NousResearch/Hermes-4.3-36B`](https://huggingface.co/NousResearch/Hermes-4.3-36B) and [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF) (`ByteDance Seed-OSS-36B-Base` / `SeedOssForCausalLM`, `model_type: seed_oss`, 524,288 context length) using the `Llama-3-Chat` template (`<seed:bos>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`).
- **Thinking & Tool-Call Parsing Features:**
  - `thinking` defaults to `False` (`--thinking` flag on CLI enables the reasoning system prompt).
  - `keep_cots` support (`False` by default) to strip or preserve `<think>...</think>` blocks in multi-turn history and parsed output.
  - Edge-case handling for unclosed `<think>` tags, unclosed `<tool_call>` tags, and malformed JSON inside `<tool_call>`.
  - Conversion helpers `to_openai_tool_calls` and `to_anthropic_tool_use` for translating parsed `<tool_call>` blocks into OpenAI and Anthropic protocol schemas.
- **GGUF Metadata KV Parsing:** `parse_gguf_raw` (`tools/qerlio_cli.py`) and `GGUFParser::parse_header` (`core/model_registry/gguf_parser.cpp`) now parse the GGUF v2/v3 metadata key-value section (`general.architecture`, `general.name`, `<arch>.context_length`, `general.file_type` / quantization) and compute real SHA-256 checksums in C++.
- **CLI Subcommands (`qerlio hermes`):**
  - `qerlio hermes chat`: Format prompts locally (`--thinking`, `--keep-cots`) or query an external OpenAI-compatible endpoint.
  - `qerlio hermes test`: Offline template-formatting and tag-parsing self-test (no model inference).
  - `qerlio hermes download`: Download official GGUF files from `NousResearch/Hermes-4.3-36B-GGUF`.
- **Architecture Catalog Entry (`seed_36b`):** Added `seed_36b` (`ByteDance Seed-OSS-36B (Hermes 4.3 36B)`) to `registry/model_architectures.json` under `Reasoning & Dialectic` with `support_status: "experimental"` (183 historical entries: 3 tested, 180 experimental). *(Historical v0.2.2 status: QERLIO did not execute `SeedOssForCausalLM` tensor weights locally until v0.3.0.)*
- **Documentation (`docs/models/HERMES_4_3_36B.md`):** Reference guide covering official `Seed-OSS` metadata, template formatting, tag parsing, and GGUF downloading.
- **Remote Gateway Template Stub (`server/hf_space_gateway/`):** FastAPI/Gradio HTTP gateway scaffold with optional secret-token verification and a 7-day trial expiration check.

### Fixed
- Removed model-impersonating responses and invented tool calls/usage counts from `server/local_api/server.py` and `server/hf_space_gateway/app.py`; stub endpoints now return `"simulated": true` and `"usage": null`.
- Removed unmeasured placeholder performance and memory metrics from `sdk/python/qerlio/runtime.py`, `src/qerlio_abi.cpp`, `core/compression/quant_matrix.py`, and `core/compression/universal_quant.cpp`.

---

## [0.2.1] - 2026-11-20

### Fixed
- **Accurate Architecture Registry:** Filtered catalog down to 182 real, publicly verifiable open-weight architectures across 10 families. Removed 19 duplicate, private, or unreleased entries.
- **Support Tiers:** Introduced explicit `support_status` field (`tested` vs `experimental`). Marked only models with real code paths in the repository as `tested` (`zaqeris_axiom`, `zqn_12b`, `zaqeris_7b`).
- **Corrected ZAQERIS Model Metadata:** AXIOM documented as a sparse MoE (79.43B total / 12.39B active) with ZAQERIS-M3 packed format at ~3.16 bits per parameter (corrected from 3.12 bpp).
- **Removed Fake Telemetry:** Replaced hardcoded throughput, TTFT, and memory numbers in HUD and `/v1/telemetry/hud` with `null` and a clear `no live measurement` label.
- **Repository Cleanliness:** Removed mock UI screenshots from git tracking and configured `capture_hud.py` to save captures outside the repository (`~/.qerlio/hud_captures`). Replaced absolute machine paths with relative links.

---

## [0.2.0] - 2026-11-19

### Added
- **Architecture Catalog:** Registry of model architecture metadata covering 10 architectural families.
- **Built-in Agent Mode:** Sandboxed ReAct loop with native tool calling support for `ZAQERIS/ZAQERIS-7B` (trained from scratch; only the tokenizer is reused from Qwen2), `ZAQERIS/ZQN-12B`, and OpenAI function calling formats.
- **Safe Sandbox Tools:** Calculator, Sandboxed Python Evaluator, Workspace File Search, and Synthetic Web Search.
- **Universal Multi-API Gateway:** Local API server supporting OpenAI, Anthropic, and Ollama protocol compatibility.
- **Developer HUD:** Web-based architecture explorer and agent studio interface.
