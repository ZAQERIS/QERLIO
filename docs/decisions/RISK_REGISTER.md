# QERLIO Runtime — Technical, Legal, and Operational Risk Register

**Owner:** ZAQERIS AI Research Group  
**Document Version:** 1.0  
**Date:** 2026-11-19  

---

## 1. Risk Classification Matrix

| Risk ID | Category | Description | Severity | Likelihood | Mitigation Strategy | Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSK-LEG-01** | Legal / Trademark | Trademark dispute over the name `QERLIO` | High | Medium | The name is treated as a provisional working candidate. Comprehensive WIPO, USPTO, and Egyptian local trademark clearance searches must be completed before commercial marketing. | Legal Counsel / Research Lead |
| **RSK-LEG-02** | Legal / Licensing | Upstream dependency license violation (e.g., GPL infection or missing attribution) | High | Low | Strict SPDX licensing compliance. Maintain `third_party/DEPENDENCY_INVENTORY.md` and `pyproject.toml` (automated CycloneDX/SPDX SBOM generation planned; not yet active in CI). | Release Maintainer |
| **RSK-SEC-01** | Security | Malformed GGUF file exploiting buffer overflow / integer overflow in C++ parser | Critical | Medium | Treat all model files as untrusted binaries. Implement strict bounds checking, fuzzing (AFL++/libFuzzer), and address/undefined-behavior sanitizers. | Security Engineer |
| **RSK-SEC-02** | Security | Prompt injection leading to unauthorized filesystem or tool execution | High | High | Sandboxed execution boundary for local tools. Strict schema validation; no default shell or arbitrary filesystem write access. | Security Engineer |
| **RSK-SEC-03** | Security / Privacy | Local REST API exposure allowing CSRF from malicious web browser tabs | High | Medium | Local loopback binding only (`127.0.0.1`), mandatory ephemeral bearer token authentication, and strict CORS/Host header validation. | Security Engineer |
| **RSK-TEC-01** | Technical / Hardware | Mobile NPU fragmentation and lack of unified APIs | High | High | Avoid relying on deprecated NNAPI. Provide a guaranteed ARM64 CPU baseline, adopt Vulkan for GPU, and add modular vendor-specific adapters (Qualcomm QNN, MediaTek NeuroPilot) only when verified on physical hardware. | Runtime Architect |
| **RSK-TEC-02** | Technical / Stability | Device thermal throttling and battery drain during sustained local inference | High | High | Thermal-aware adaptive scheduler that subscribes to OS thermal events (`PowerManager.addThermalStatusListener`) and throttles batch size or yields threads when thermal limits approach. | Inference Engineer |
| **RSK-TEC-03** | Technical / Memory | Out Of Memory (OOM) crashes due to large KV Cache allocation | Critical | High | Accurate pre-allocation memory planning. Context budgeting: calculate model weights + KV cache + workspace before loading, and reject or downgrade context length if budget is exceeded. | Memory Systems Lead |
| **RSK-OPS-01** | Operational | Over-promising unverified benchmark claims | High | Medium | "Evidence-driven optimization" rule: publish only reproducible results with exact model SHA-256 checksums, prompts/outputs, and verbatim commands; mark unmeasured items `not measured`. | QA / Benchmark Lead |

---

## 2. Review and Escalation Trigger
This risk register is reviewed prior to each major release gate. Any `Critical` or unmitigated `High` risk blocks public stable tagging.
