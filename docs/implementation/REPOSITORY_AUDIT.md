> **Historical Archive Note (v0.3.1):** Phase 0 environment/discovery snapshot (`2026-11-19`). Note that `gguf`, `transformers`, `tokenizers`, and `torch` are permitted strictly offline under `tests/reference/` and are not runtime dependencies of `core/inference/`, and `ADR-002` (`llama.cpp`) was superseded in v0.3.0 by the from-scratch native engine.

# [HISTORICAL ARCHIVE — 2026-11-19] QERLIO Runtime — Initial Repository & Environment Audit

**Historical Date:** 2026-11-19  
**Auditor:** Principal Systems Architect / QERLIO Core Team  
**Workspace:** QERLIO repository root (initial Phase-0 creation)  
**Phase:** Phase 0 (2026-11-19 Historical Environment Assessment — Superseded by `v0.3.0` / `v0.3.1`)  
**Historical Status:** ARCHIVED PHASE-0 SNAPSHOT

---

## 1. Initial Repository Status (2026-11-19 Archive)

- **Repository Initial State (`2026-11-19`):** Empty workspace before Phase-0 scaffolding.
- **Git Version Control (`2026-11-19`):** Initialized subsequently.
- **Compression Adapter Interface:** `formats/zaqeris_compression/zaqeris_m3_adapter.py` provides an optional Python interface only; no external compression wheel is bundled in the repository.

---

## 2. Host Software Environment

| Subsystem | Specification / Detected Value | Status / Evidence |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 | `VERIFIED` |
| **Host Hardware** | Intentionally not published in this repository | `N/A` |
| **Python Runtime** | Python 3.12.10 (64-bit) | `VERIFIED` (`python --version`) |
| **PyTorch & ML Libs** | PyTorch 2.11.0, `gguf 0.19.0`, `safetensors 0.8.0`, `transformers 5.20.1`, `tokenizers 0.23.3` | `VERIFIED` (active in site-packages) |
| **Node.js Runtime** | Node.js v22.14.0 / npm 10.9.2 | `VERIFIED` |
| **Package Manager** | Windows Package Manager (`winget`) v1.29.380 | `VERIFIED` |
| **Attached Android Devices** | None detected / No ADB daemon in PATH | `NOT AVAILABLE` |
| **Apple Metal / macOS** | Non-Apple Host | `NOT APPLICABLE` |

---

## 3. Historical Toolchain Snapshot (2026-11-19 Archive)

| Tool | Historical Status (`2026-11-19`) | Impact | Remediation Note |
| :--- | :--- | :--- | :--- |
| **CMake** | `VERIFIED` (v4.4.4) | Build configuration ready | Used in CI and local CMake builds |
| **Ninja** | `VERIFIED` (v1.13.2) | Fast build generator ready | Available in toolchain PATH |
| **C++ Compiler** | Host-dependent (`clang` / MSVC / GCC) | Local Windows policy restrictions handled via GitHub Actions CI | Multi-OS CI compiles C11 & C++20 cleanly on Linux/Windows/macOS |
| **Git** | Subsequently installed | Version control active | Standard Git workflows |
| **Android NDK / SDK** | Not configured | Android cross-compilation deferred | Scheduled after desktop core C ABI & native engine work |

---

## 4. Phase-0 Architectural Proposals (`2026-11-19` Archive)

1. **ADR-001 (Core & ABI):** Native Core written in C++20 behind a flat C ABI (`qerlio.h`).
2. **ADR-002 (GGUF Backend — SUPERSEDED in `v0.3.0`):** Initially proposed `llama.cpp` under `backends/gguf_llama_cpp/`. **Never implemented:** `v0.3.0+` builds the native GGUF engine from scratch in `core/inference/`.
3. **ADR-003 (Android Strategy):** JNI wrapper over stable C ABI. NNAPI rejected as primary future path (deprecated in Android 15); focus on CPU (ARM64 NEON) + Vulkan/vendor accelerators.
4. **ADR-004 (Model Integrity):** Zero execution of remote code. Models require SHA-256 validation, strict header bounds checking, and signed/verified manifests.
5. **ADR-005 (Local API Security):** Loopback only (`127.0.0.1`), token-authenticated, strict CORS/CSRF defense.
6. **ADR-006 (Privacy & Telemetry):** Telemetry disabled by default. Network access disabled in inference runtime. No prompt/conversation data in default logs.
7. **ADR-007 (Error Taxonomy & Cancellation):** First-class atomic cancellation tokens and structured error codes.

---

## 5. Phase 0 Audit Conclusion & Next Steps

Phase 0 discovery confirms an active Python environment with `gguf` and `torch`, and zero pre-existing codebase conflicts. The specification architecture is scaffolded with complete directory structure, ADRs, risk register, threat model, and reference C ABI header.
