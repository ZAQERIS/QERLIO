> **Historical Archive Note (v0.3.1):** Phase 1 initial planning snapshot (`2026-11-19`) prior to the v0.3.0 from-scratch native engine (`core/inference/`) which replaced the deferred `llama.cpp` plan in Slice 2.

# [HISTORICAL ARCHIVE — 2026-11-19] QERLIO Runtime — Phase 1 Vertical Slice Execution Plan

**Historical Note:** This Phase-1 execution plan (`2026-11-19`) is archived for historical traceability. Slice 2 (`llama.cpp` integration) was **superseded in `v0.3.0`** by QERLIO's from-scratch native GGUF engine in [`core/inference/`](../../core/inference/).

---

## 1. Historical Scope & Boundaries (2026-11-19 Archive)

- **In Scope for Slice 1 (`2026-11-19`):**
  - Complete memory lifecycle (`create` -> `load` -> `session` -> `cancel` -> `destroy`).
  - Strict GGUF header parsing with bounds validation (preventing buffer overflows and integer wraps).
  - SHA-256 cryptographic verification before loading.
  - Initial scaffolding stream simulation with atomic cancellation.
  - Python-based verification test suite using test harness scripts.
  
- **Historical Slice-2+ Proposals (`2026-11-19` — Superseded):**
  - *(Superseded in `v0.3.0`)* Initial draft considered delegating tensor matrix multiplication kernels to `llama.cpp` in Slice 2. That plan was **never implemented**; QERLIO instead implemented tensor loading, dequantization, C11 SIMD quantized GEMM (`native_gemm.c`), and forward execution from scratch under `core/inference/`.
  - Multi-backend GPU/Vulkan offloading (unimplemented; CPU-only in `v0.3.1`).
  - Android JNI bindings (planned).

---

## 2. Directory Layout for Slice 1

```text
QERLIO/
├── CMakeLists.txt
├── include/
│   └── qerlio/
│       └── qerlio.h
├── src/
│   ├── engine.cpp
│   ├── model.cpp
│   ├── session.cpp
│   └── qerlio_abi.cpp
├── core/
│   ├── model_registry/
│   │   ├── gguf_parser.hpp
│   │   └── gguf_parser.cpp
│   └── memory/
│       └── memory_budget.hpp
├── tools/
│   └── qerlio_cli.py
└── tests/
    ├── test_abi_lifecycle.py
    ├── test_manifest_validation.py
    └── test_malformed_gguf.py
```

---

## 3. Host Environment Path Forward

Since Windows C++ compilers (`cl.exe`, `clang`, `gcc`) and `cmake` are currently missing from PATH, Phase 1 executes two synchronized parallel tracks:
1. **Production C++ Core Codebase:** Written in standards-compliant C++20 with `CMakeLists.txt` so it is ready for compilation on CI runners and local developer machines once compilers are installed.
2. **Operational Python Verification Harness:** Uses the existing local Python 3.12 environment (which already includes `gguf`, `safetensors`, `torch`, `pydantic`, and `pytest`) to immediately execute manifest validation, GGUF structural integrity checks, and conformance testing on real model files without waiting for compiler installation.
