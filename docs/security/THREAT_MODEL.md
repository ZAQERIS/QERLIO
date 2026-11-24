# QERLIO Runtime — Security Architecture & Threat Model

**Standard:** Structured following STRIDE and OWASP Top 10 for LLM Applications  
**Date:** 2026-11-19  
**Status:** Baseline (Phase 0)  

---

## 1. System Assets & Protection Goals

1. **User Data & Conversation History:** Strictly confidential; never transmitted over network without explicit consent; excluded from debug logs by default.
2. **Host Integrity & Operating System Resources:** Protected from unauthorized filesystem modification, shell command execution, or uncontrolled resource exhaustion.
3. **Model Weights & Integrity:** Protected against tampering, unauthorized alteration, or malicious payload embedding.
4. **Local REST Server:** Protected from cross-site request forgery, cross-origin data theft, and DNS rebinding attacks.

---

## 2. Trust Boundaries

```
[ Untrusted External Zone ]
   │  (Remote Model Downloads, Hugging Face URIs, Third-Party Prompts)
───┼──────────────────────────────────────────────────────────────────
   ▼ [ Trust Boundary 1: Ingestion & Verification Gateway ]
   │  • TLS Checksum & SHA-256 Validation
   │  • Path Traversal & Symlink Defense
   │  • Manifest Schema Validation
   ▼
[ QERLIO Local Storage ]
   │  (Verified .gguf files, Local Vector DB, Cached Tokens)
───┼──────────────────────────────────────────────────────────────────
   ▼ [ Trust Boundary 2: In-Memory Runtime Boundary ]
   │  • Untrusted Binary Parser (GGUF Header / Tensor Tables)
   │  • Memory Planner & Buffer Bounds Enforcement
   │  • Sandboxed Execution Engine
   ▼
[ QERLIO Core Engine ] ◄──[ Trust Boundary 3: Local API Boundary ]── (Local HTTP / Sockets)
   │                           • 127.0.0.1 Loopback Only
   │                           • Ephemeral Bearer Token
   │                           • Strict CORS / Host Filtering
   ▼
[ Model Output / Generation Stream ]
   │
───┼──────────────────────────────────────────────────────────────────
   ▼ [ Trust Boundary 4: Tool Calling & Agency Gateway ]
   │  • Strict JSON Schema Validation
   │  • Explicit User Permission Approval
   │  • No Default Arbitrary Shell Execution
   ▼
[ Protected System Resources ]
```

---

## 3. STRIDE Threat Analysis & Mitigations

| Threat Category | Potential Vector | Specific QERLIO Defense |
| :--- | :--- | :--- |
| **Spoofing** | Impersonation of trusted model publisher or local API client | Cryptographic SHA-256 hashes verified on model import; local HTTP server requires a high-entropy random bearer token generated on startup. |
| **Tampering** | Model weights corrupted or tampered on disk | Periodic hash check; rejection of mismatched checksums; read-only memory mappings (`mmap` with read permissions only). |
| **Repudiation** | Audit log evasion during sensitive tool executions | Local append-only audit log for all permission-gated tool calls with timestamp, tool name, and caller identifier. |
| **Information Disclosure** | Memory leaks across sessions; private prompt text logged to disk | Explicit memory zeroization for sensitive buffers; strict rule forbidding prompt and output tokens in default log files. |
| **Denial of Service** | Malformed model causing crash; unbounded prompt exhausting system memory | Fuzz-tested GGUF parser; strict context budgeting; pre-allocation memory checks; cooperative thread cancellation. |
| **Elevation of Privilege** | Model hallucinates or is injected with tool call to execute `cmd.exe` or `rm -rf` | Tool calls pass through an isolated Permission Gateway. No tool is permitted arbitrary OS shell execution. Write and delete operations require explicit user confirmation. |

---

## 4. Supply Chain & Dependency Security

- **No Dynamic Remote Code:** `trust_remote_code=True` behavior from some Python libraries is categorically forbidden in QERLIO.
- **Dependency Policy:** All runtime, optional-gateway, and offline-reference dependencies are documented accurately in `third_party/DEPENDENCY_INVENTORY.md` and `pyproject.toml`; no external C/C++ third-party headers are vendored in the tree, and the inference engine in `core/inference/` is implemented from scratch without `llama.cpp`, `ggml`, or external ML frameworks.
- **SBOM Status:** Dependency tracking is maintained manually in `third_party/DEPENDENCY_INVENTORY.md` and `pyproject.toml`; automated CycloneDX SBOM generation is not currently active in CI.
