# QERLIO Runtime — Platform, Backend, and Architecture Support Matrix

**Version:** 0.3.1  
**Rule:** Any platform, architecture, or backend that has not been verified under the Section 2 `tested` definition (`T-A` through `T-F`) is classified as `EXPERIMENTAL`, `PLANNED`, or `NOT IMPLEMENTED`.

---

## 1. Execution Backend Matrix (v0.3.1)

| Backend Pathway | Scope | Current Status | Notes / Limitations |
| :--- | :--- | :--- | :--- |
| **Native CPU Engine (`core/inference/`)** | Python stdlib + NumPy + C11 SIMD GEMM (`native_gemm.c`) | `IMPLEMENTED` | `seed_oss` uses C11 GEMM (`Q4_K`, `Q6_K`, `Q8_0`, `F32`) when built/loaded, otherwise NumPy; `qwen35` uses NumPy |
| **C++20 Core + Flat C ABI** | Header parsing, SHA-256, KV planner, session cancellation | `IMPLEMENTED` | Verified in CI on Linux/Windows/macOS + ASan/UBSan |
| **CUDA / Vulkan / Metal GPU** | GPU kernel offloading | `NOT IMPLEMENTED` | No GPU execution backend exists in v0.3.1 |
| **Android / NPU (QNN / NeuroPilot)** | Mobile JNI & NPU backends | `PLANNED` | Not implemented or hardware-verified |

---

## 2. Model Architecture Support Matrix (v0.3.1 Section 2 Standard)

| Architecture ID | Representative Checkpoint | Execution Code in Repo | Support Tier | Evidence / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **`seed_36b`** | `NousResearch/Hermes-4.3-36B-GGUF` (`hermes-4_3_36b-Q4_K_M.gguf`) | `core/inference/seed_oss_model.py` | **`TESTED`** | Meets Section 2 `T-A`–`T-F`: SHA-256 `17823599...`, 320/320 greedy tokens vs `transformers 5.20.1` (`tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json`) |
| **`qwen35`** | `Qwen/Qwen3.8-27B` | `core/inference/qwen35_model.py` | `EXPERIMENTAL` | Synthetic CI tests only + earlier 4-token run on an unofficial UD quant (`< 64` tokens, invalid for `tested`) |
| **`zaqeris_axiom`**, **`zqn_12b`**, **`zaqeris_7b`** | `ZAQERIS/ZAQERIS_AXIOM`, `ZAQERIS/ZQN-12B`, `ZAQERIS/ZAQERIS-7B` | Prompt/tool parser & M3 adapter only | `EXPERIMENTAL` | `ZAQERIS-7B` is derived from Qwen/Qwen2.5-Coder-7B-Instruct; no end-to-end `>= 64`-token official-weight run recorded |
| **All other 179 architectures** | Llama, Mistral, Gemma, Phi, MoE, VLM, etc. | Catalog metadata only | `EXPERIMENTAL` | No local weight execution |

---

## 3. Quantization Kernels Implemented (`core/inference/dequant.py` & `native_gemm.c`)

| Quantization Format | Dequant Decoder (`dequant.py`) | C11 SIMD GEMM (`native_gemm.c`) | Status |
| :--- | :---: | :---: | :--- |
| **`Q4_K`, `Q6_K`, `Q8_0`, `F32`** | Yes (`< 1e-6` vs `gguf-py`) | Yes (AVX2+FMA+F16C + scalar fallback) | Used in verified `seed_36b` run & ASan C tests |
| **`F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q5_K`, `IQ4_NL`, `IQ3_S`, `IQ4_XS`** | Yes (`< 1e-6` vs `gguf-py`) | Fallback to dequant + NumPy | Verified against offline `gguf-py 0.19.0` slices |
| **ZAQERIS M3-GPTQ** | Research adapter only (`formats/zaqeris_compression`) | No | `EXPERIMENTAL` |
