> **Historical Archive Note (v0.3.1):** This Phase-0 document (`2026-11-19`) is **superseded and archived**. The proposed `llama.cpp` integration and `backends/gguf_llama_cpp/` directory were **never created**. In `v0.3.0+`, QERLIO's GGUF v2/v3 inference engine in [`core/inference/`](../../core/inference/) was implemented from scratch using the Python standard library, NumPy, and C11 (`core/inference/native_gemm.c`), with zero dependency on `llama.cpp` or `ggml`.

# ADR-002 [SUPERSEDED — HISTORICAL ARCHIVE]: Initial Phase-0 Proposal for GGUF Backend

## Status (Superseded in v0.3.0 — Never Implemented on Current Main)
Superseded in v0.3.0. Neither `backends/gguf_llama_cpp/`, nor `third_party/licenses/`, nor automated CycloneDX SBOM generation was adopted in this repository. Current runtime inference lives in [`core/inference/`](../../core/inference/), and actual dependency inventory lives in [`third_party/DEPENDENCY_INVENTORY.md`](../../third_party/DEPENDENCY_INVENTORY.md).

## Historical Phase-0 Context (2026-11-19 — Superseded)
At initial Phase-0 scaffolding (`2026-11-19`), writing an inference engine from scratch had been temporarily deferred in favor of designing a C++ ABI boundary (`include/qerlio/qerlio.h`) and registry schema first.

## Historical Phase-0 Proposal (2026-11-19 — Superseded by `core/inference/`)
In that initial `2026-11-19` draft (prior to v0.3.0), Phase 0 considered:
1. **Proposed Reference Backend (Unimplemented):** Phase 0 initially proposed adopting an external `llama.cpp` backend for GGUF execution. *(Superseded in v0.3.0: `core/inference/` implements GGUF parsing, 16 dequantizers, BPE tokenization, C11 SIMD GEMM, and model forward passes from scratch with zero `llama.cpp` dependency.)*
2. **Proposed Pinned Integration (Never Executed):** No `llama.cpp` upstream tag or submodule was ever linked.
3. **Proposed Isolation Directory (Does Not Exist):** `backends/gguf_llama_cpp/` was never created and does not exist in this repository.
4. **Proposed License Directory & SBOM (Never Created):** Neither `third_party/licenses/` nor an automated SBOM exists in the tree; the project's actual `Apache-2.0` license is at root [`LICENSE`](../../LICENSE) and its real dependency list is in [`third_party/DEPENDENCY_INVENTORY.md`](../../third_party/DEPENDENCY_INVENTORY.md).
5. **Retained Principles:** Ownership of the memory planner, C ABI, model hash verification, and local APIs remained with QERLIO and evolved into the current codebase.

## Historical Consequences & Supersession
- The negative consequences noted in Phase 0 (upstream coupling and adapter maintenance churn) were eliminated in `v0.3.0` when QERLIO replaced the unimplemented external backend plan with its self-contained native engine under [`core/inference/`](../../core/inference/).
