> **Historical Archive Note (v0.3.1):** Phase-1 pre-upload snapshot (`2026-11-19`) from the initial scaffolding commit. Superseded by v0.3.0 and the v0.3.1 corrections in [`CHANGELOG.md`](../../CHANGELOG.md) and [`README.md`](../../README.md). Statements below describe only that historical `2026-11-19` scaffolding snapshot and do NOT describe the current repository state.

# [HISTORICAL ARCHIVE — 2026-11-19] QERLIO Runtime — Phase-1 Initial Upload Snapshot

**Historical Date:** 2026-11-19  
**Historical Target:** Initial Phase-1 Scaffolding Snapshot (29 unit/gateway stub tests)  
**Historical Status:** ARCHIVED HISTORICAL SNAPSHOT (2026-11-19 Phase-1 Scaffolding — Superseded by `v0.3.0` / `v0.3.1`)

---

## 1. Source Code Review (2026-11-19 Historical Snapshot)

| Component | Files Inspected (`2026-11-19`) | Historical Findings (`2026-11-19`) | Historical Note |
| :--- | :--- | :--- | :--- |
| **C ABI Boundary** | `include/qerlio/qerlio.h`, `src/qerlio_abi.cpp` | Opaque handle encapsulation; zero exceptions across binary boundary; thread-safe cancellation tokens. | `2026-11-19` snapshot |
| **Model Registry & Parser** | `core/model_registry/gguf_parser.cpp` | Strict magic validation (`0x46554747`), upper bounds enforcement on tensor and metadata counts (`MAX_REASONABLE_TENSORS`). | `2026-11-19` snapshot |
| **Memory Planning** | `core/memory/memory_planner.cpp` | Mathematical pre-flight KV cache estimation; context down-stepping (4096 -> 2048 -> 1024). | `2026-11-19` snapshot |
| **Adaptive Scheduling** | `core/scheduler/adaptive_scheduler.cpp` | Dynamic thread and batch throttling under `WARM`, `HOT`, and `CRITICAL` thermal states. | `2026-11-19` snapshot |
| **Universal Quantization**| `core/compression/universal_quant.cpp`, `quant_matrix.py` | Metadata profiles across Standard GGUF, k-quants, i-quants (sub-3bit), ZAQERIS RVQ, and FP8. | `2026-11-19` snapshot (theoretical sizes) |

---

## 2. Universal API Gateway & HUD Snapshot (2026-11-19 Archive — Stub Mode Prior to v0.3.0)

| Endpoint | Standard Protocol | Historical Test File | Historical Note (`2026-11-19`) |
| :--- | :--- | :--- | :--- |
| `/v1/chat/completions` | OpenAI Chat API | `test_local_api.py` | Stub-mode schema conformance on `2026-11-19` (updated in `v0.3.0`/`v0.3.1`) |
| `/v1/models` | OpenAI Models API | `test_local_api.py` | Reworked in `v0.3.1` to list loaded models only |
| `/v1/messages` | Anthropic Claude Messages API | `test_local_api.py` | Stub-mode schema conformance on `2026-11-19` |
| `/api/tags` | Ollama Tags API | `test_local_api.py` | Reworked in `v0.3.1` to list loaded models only |
| `/api/generate` | Ollama Generate API | `test_local_api.py` | Stub-mode schema conformance on `2026-11-19` |
| `/api/chat` | Ollama Chat API | `test_local_api.py` | Stub-mode schema conformance on `2026-11-19` |
| `/` & `/dashboard` | Developer HUD | `test_quantization_and_hud.py` | Local UI shell check |
| `/v1/health` | System Diagnostics API | `test_local_api.py` | Health schema check |

---

## 3. Repository Configuration Snapshot (2026-11-19 Archive)

- **Network Policy:** Inference runtime operates offline.
- **Local Server Security:** Loopback binding defaults (`127.0.0.1`).
- **2026-11-19 Hygiene Check (Note):** At that Phase-1 snapshot, root `LICENSE`, `pyproject.toml`, and `CODE_OF_CONDUCT.md` had not yet been added and `third_party/DEPENDENCY_INVENTORY.md` contained Phase-0 inaccuracies later resolved in `v0.3.1`.

---

## 4. Historical Automated Test Verdict (2026-11-19 Archive — 29 Phase-1 Tests Only)

```text
======================== 29 passed in 6.06s (historical 2026-11-19 scaffolding) ========================
```

---

## 5. Historical Archive Note (`2026-11-19`)

This archived Phase-1 note records only the initial `2026-11-19` scaffolding state (29 early stub/metadata tests). It is superseded by the `v0.3.0` native inference engine and the `v0.3.1` audit corrections, install files (`pyproject.toml`, `LICENSE`), and release gate rules documented in [`README.md`](../../README.md) and [`CHANGELOG.md`](../../CHANGELOG.md).
