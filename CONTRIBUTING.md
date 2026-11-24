# Contributing to QERLIO Runtime

We welcome contributions to QERLIO Runtime. All contributions and releases are governed by strict technical, honesty, memory-safety, and PR-gate rules.

**Do not merge or release until the gate is complete.**

---

## 1. Environment, Installation & Build

- **Python:** Python `3.10+` (`numpy>=1.24` for the core inference engine; see [`pyproject.toml`](pyproject.toml)).
- **C/C++ Toolchain:** C11 + C++20 compiler, CMake `3.22+`.

Install editable with gateway and test dependencies:
```bash
pip install -e ".[dev]"
```

Configure and build the C++20 core & C11 `native_gemm` library:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

---

## 2. Running Tests & C11/C++20 Sanitizers (Mandatory)

1. **Default Python Test Suite:**
   ```bash
   python -m pytest tests/ -v
   ```
   - Zero failures allowed.
   - Only `tests/test_real_hermes_36b_optin.py::test_optin_real_hermes_4_3_36b_official_gguf` may skip cleanly in default CI when `QERLIO_RUN_REAL_HERMES!=1` or the official GGUF file is absent.
   - Synthetic toy-GGUF tests MUST remain explicitly labeled as synthetic; never replace a real-weight check with a silent fallback to synthetic fixtures.

2. **Mandatory ASan + UBSan Sanitizer Coverage (`core/inference/*.c` & C++20 Core):**
   ```bash
   cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DQERLIO_BUILD_TESTS=ON \
     -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Wpedantic" \
     -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Wpedantic"
   cmake --build build-asan --target qerlio_cpp_safety_test
   ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
     ctest --test-dir build-asan --output-on-failure
   ```
   - Every `.c` file under `core/inference/` MUST be added to `QERLIO_SOURCES` in [`CMakeLists.txt`](CMakeLists.txt) and actively exercised in [`tests/cpp_safety_test.cpp`](tests/cpp_safety_test.cpp) (not merely compiled).
   - Static source-string audits (such as Pytest T10) are NOT sanitizer runs and MUST NEVER be described as ASan/UBSan results.

3. **Markdown Link & Cleanliness Checks:**
   ```bash
   python scripts/check_markdown_links.py
   python scripts/check_forbidden_content.py
   ```

4. **Engine Purity & ABI Rules:**
   - Modules under `core/inference/` MUST use only the Python standard library, NumPy, and QERLIO's own C11/C++20 code (zero runtime imports of `torch`, `transformers`, `gguf`, `llama_cpp`, `vllm`, `sglang`, `onnxruntime`, `jax`, `psutil`, etc.).
   - Offline reference generators belong strictly under `tests/reference/`.
   - Public C ABI functions in `include/qerlio/qerlio.h` MUST NOT throw C++ exceptions across the binary boundary.

---

## 3. Strict Definition of `tested` & Honesty Rules

An architecture may be marked `support_status: "tested"` in [`registry/model_architectures.json`](registry/model_architectures.json) **ONLY** if ALL six criteria (`T-A` through `T-F`) are satisfied and committed in-repo:
- **`T-A` Official weights:** Official weight file or official publisher quant (never unofficial UD/third-party re-uploads).
- **`T-B` End-to-end execution:** Load weights, tokenize a documented prompt, run a forward pass, and generate tokens.
- **`T-C` Minimum generation length:** At least `64` newly generated tokens under greedy decoding (or documented equivalent protocol).
- **`T-D` Recorded prompt & output:** Exact prompt / token IDs and generated tokens committed in-repo (`tests/fixtures/` or `docs/`).
- **`T-E` Reproducible command:** Verbatim reproducible command recorded next to the result.
- **`T-F` Verified SHA-256 & byte size:** Exact SHA-256 and byte length recorded next to the result.

Additional honesty requirements:
- Every number and non-trivial factual claim MUST cite its reproducible command or committed log/fixture. If unmeasured, write `not measured`.
- Never return unloaded or catalog-only models from `/v1/models` or `/api/tags` as loaded/available inference models, and never label theoretical formulas as live runtime measurements.
- For **`ZAQERIS/ZAQERIS-7B` (`zaqeris_7b`)**, every public description MUST state that it was trained from scratch and that only the tokenizer is reused from Qwen2; NEVER describe it as a Qwen architecture.
- Zero hardware disclosure anywhere public (no GPU/CPU model names, device memory sizes, compute-hour figures, thread counts, or host HUD screenshots), zero personal names or team size, zero legacy/internal codenames outside `zqn_12b`, zero `.gguf` / weight files committed, zero private absolute paths or secrets, and zero unsupported promotional adjectives or slogan claims lacking Section 2 evidence.

---

## 4. Mandatory PR & Release Gate (`G1`–`G11`)

Every PR into `main` and every release MUST satisfy all gate items below before merge or tagging. **Do not merge or release until the gate is complete.**

- **G1:** CI green on ALL jobs (`Python Conformance & Integrity Tests`, `C++20 Core Build` on `ubuntu-latest`, `windows-latest`, `macos-15-intel`, and `C++20 AddressSanitizer & UndefinedBehaviorSanitizer`).
- **G2:** `python -m pytest tests/ -v` passes with zero failures (only opt-in real-weight tests may skip when the env flag or weight file is absent).
- **G3:** ASan+UBSan job green with zero findings, covering every `.c` file under `core/inference/`.
- **G4:** Written approval on the PR from a second reviewer (another team member, NOT the author; self-approval does not count).
- **G5:** Complete `G1`–`G11` checklist included in the PR body, every box ticked with evidence.
- **G6:** Claim-audit table in the PR backing every number and claim with a reproducible command or committed fixture (`not measured` where unmeasured).
- **G7:** Release notes list wins AND honest limitations with zero unsupported marketing terms.
- **G8:** Correct semver tag and matching `CHANGELOG.md` entry.
- **G9:** No force-push and no history rewrite on `main`, PR branches under review, or existing tags.
- **G10:** Forbidden-content scan (`scripts/check_forbidden_content.py`) and Markdown link check (`scripts/check_markdown_links.py`) pass with `0` violations and `0` broken links.
- **G11:** Release-specific deliverables and any prior-release correction notes included and verified before publishing.
