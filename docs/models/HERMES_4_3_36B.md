# NousResearch/Hermes-4.3-36B — Native Engine, Template & GGUF Reference

> **Support Tier in Registry:** `tested` (`seed_36b` — verified under Section 2 `T-A`–`T-F`)  
> **Scope in QERLIO v0.3.1:** From-scratch native GGUF v3 inference engine (`core/inference/seed_oss_model.py`, `native_gemm.c`, `native_gemm.py`, `gguf_loader.py`, `dequant.py`, `tokenizer.py`, `kv_cache.py`, `sampler.py`) using only the Python standard library, NumPy, and C11, integrated with `Hermes36BAdapter` (`Llama-3-Chat` / `<seed:bos>`, `<think>` / `<tool_call>` parsing) and the local API gateway (`/v1/models/load`, `/v1/chat/completions`, `/v1/messages`).

---

## 1. Official Model Metadata

`NousResearch/Hermes-4.3-36B` (model card title: **Hermes 4.3 - Seed 36B**) is an open-weight hybrid-mode reasoning and tool-use model published by **Nous Research**.

* **Base Model:** **ByteDance Seed-OSS-36B-Base** (not Meta Llama).
* **Transformers Architecture Class:** `SeedOssForCausalLM` (`model_type: seed_oss`).
* **Parameters:** 36B.
* **Context Length:** **524,288 tokens** (`max_position_embeddings: 524288`).
* **Chat Template:** `Llama-3-Chat` format with Seed BOS token:
  - BOS Token: `<seed:bos>`
  - Role Header: `<|start_header_id|>{role}<|end_header_id|>\n\n`
  - Turn Terminator: `<|eot_id|>`
* **Official Repositories:**
  - BF16 Safetensors: [`NousResearch/Hermes-4.3-36B`](https://huggingface.co/NousResearch/Hermes-4.3-36B)
  - GGUF Checkpoints: [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF) (`hermes-4_3_36b-Q3_K_M.gguf`, `hermes-4_3_36b-Q4_K_M.gguf`, `hermes-4_3_36b-Q5_K_M.gguf`, `hermes-4_3_36b-Q6_K.gguf`, `hermes-4_3_36b-Q8_0.gguf`, `hermes-4_3_36b.gguf`)

---

## 2. Recommended Sampling & Parser Configuration (From Model Card)

* **Temperature:** `0.6`
* **Top-P:** `0.95`
* **Top-K:** `20`
* **add_generation_prompt:** `True`
* **vLLM Tool Parser:** `--enable-auto-tool-choice --tool-call-parser hermes`
* **SGLang Tool Parser:** `--tool-call-parser qwen25`

---

## 3. Prompt Formatter & Tag Parser (`core/models/hermes_adapter.py`)

### A. Prompt Formatting & `thinking` / `keep_cots` Flags
* **`thinking=False` (Default):** Uses the standard system prompt (`"You are Hermes, created by Nous Research."`) unless a custom system message is provided.
* **`thinking=True`:** Prepends the official deep-thinking system prompt instructing the model to enclose its internal deliberation inside `<think> </think>` tags prior to answering.
* **`keep_cots=False` (Default):** Strips `<think>...</think>` blocks from historical `assistant` turns when formatting multi-turn prompts and removes `<think>...</think>` from `parsed["final_content"]`.
* **`keep_cots=True`:** Preserves `<think>...</think>` blocks (and `reasoning_content` fields) in historical `assistant` turns and retains the `<think>` block in `parsed["final_content"]`.

### B. Tag Parsing & Edge-Case Handling (`parse_generation`)
* **Closed & Unclosed `<think>` Blocks:** Extracts the reasoning text into `parsed["thinking"]`. If `<think>` is opened without a closing `</think>`, sets `parsed["unclosed_thinking"] = True` and strips the unclosed block from `final_content` (when `keep_cots=False`).
* **Structured `<tool_call>` Parsing:** Parses valid `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` blocks into `parsed["tool_calls"]`. Unclosed `<tool_call>` tags or invalid JSON payloads are recorded in `parsed["malformed_tool_calls"]` without raising unhandled exceptions.
* **Protocol Conversions:**
  - `parsed["openai_tool_calls"]` / `Hermes36BAdapter.to_openai_tool_calls(...)`: formats calls as OpenAI `[{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json-str>"}}]`.
  - `parsed["anthropic_tool_use"]` / `Hermes36BAdapter.to_anthropic_tool_use(...)`: formats calls as Anthropic `[{"type": "tool_use", "id": ..., "name": ..., "input": {...}}]`.

---

## 4. CLI Usage (Formatting, Parser Self-Test & GGUF Download)

### A. Run Local Inference with the Native Engine
Load a local GGUF checkpoint directly via CLI or via the local API server (`POST /v1/models/load`):
```bash
python tools/qerlio_cli.py hermes chat --model-path models/hermes-4_3_36b-Q4_K_M.gguf --prompt "Explain dialectic synthesis." --thinking
```

### B. Format a Chat Prompt Locally
By default, `thinking` is disabled (`False`):
```bash
python tools/qerlio_cli.py hermes chat --prompt "Explain dialectic synthesis."
```
To enable the thinking system prompt (`thinking=True`) and optionally keep chain-of-thought blocks (`keep_cots=True`):
```bash
python tools/qerlio_cli.py hermes chat --prompt "Explain dialectic synthesis." --thinking --keep-cots
```

### C. Run Offline Template & Tag-Parser Self-Test
This command tests prompt formatting, `<think>` and `<tool_call>` extraction, and OpenAI/Anthropic format conversion on an offline sample string (it does **not** load or execute model weights):
```bash
python tools/qerlio_cli.py hermes test
```

### D. Download Official GGUF Weights
Download quantized GGUF files from [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF):
```bash
python tools/qerlio_cli.py hermes download --quant Q4_K_M --output-dir models
```

### E. Run Default CI Tests & Opt-In Real-Weight Test
Default synthetic/offline verification suite (skips opt-in real-weight test cleanly when the 21.76 GB file or env flag is absent):
```bash
python -m pytest tests/test_hermes_36b.py tests/test_t1_t2_gguf_and_dequant.py tests/test_t3_tokenizer.py tests/test_t4_to_t10_engine.py -v
```
Opt-in real-weight verification pytest (requires official `models/hermes-4_3_36b-Q4_K_M.gguf`):
```bash
QERLIO_RUN_REAL_HERMES=1 python -m pytest tests/test_real_hermes_36b_optin.py -v
```

---

## 5. Section 2 `tested` Criteria Checklist & Real-Weight T1–T9 Results

| Criterion | Status | Evidence |
| :--- | :--- | :--- |
| `T-A` Official weights | PASS | [`NousResearch/Hermes-4.3-36B-GGUF`](https://huggingface.co/NousResearch/Hermes-4.3-36B-GGUF) / `hermes-4_3_36b-Q4_K_M.gguf` |
| `T-B` End-to-end execution | PASS | `SeedOssEngine` loads `models/hermes-4_3_36b-Q4_K_M.gguf`, tokenizes, runs all 64 layers (`hidden=5120`, `ffn=27648`, `heads=80`, `kv_heads=8`), and generates tokens |
| `T-C` >= 64 greedy tokens | PASS | 5 fixed prompts × 64 greedy tokens (`320` generated tokens total) |
| `T-D` Recorded prompt & output | PASS | Committed in [`tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json`](../../tests/fixtures/hermes_4_3_36b_qerlio_verification_record.json) and reproduced below |
| `T-E` Reproducible command | PASS | `python scripts/verify_hermes_36b_real.py --model-path models/hermes-4_3_36b-Q4_K_M.gguf --reference-fixture tests/fixtures/hermes_4_3_36b_transformers_reference.json` |
| `T-F` Verified SHA-256 & bytes | PASS | Size `21,762,145,216` bytes; SHA-256 `17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08` |

Peak RSS memory during 36B execution: `not measured`.

### Summary Table (T1–T9 on Real 36B Checkpoint)

Command: `python scripts/verify_hermes_36b_real.py --model-path models/hermes-4_3_36b-Q4_K_M.gguf --reference-fixture tests/fixtures/hermes_4_3_36b_transformers_reference.json`

| Test | Result | Recorded Value / Evidence |
| :--- | :--- | :--- |
| T1 GGUF loader | PASS | 27 KVs, 771 tensors match [`tests/fixtures/hermes_4_3_36b_q4_k_m_metadata.json`](../../tests/fixtures/hermes_4_3_36b_q4_k_m_metadata.json) |
| T2 Dequantization | PASS | 21 cases vs `gguf-py 0.19.0`, `max_abs_err=0.00e+00` ([`tests/fixtures/t2_gguf_py_dequant_reference.json`](../../tests/fixtures/t2_gguf_py_dequant_reference.json)) |
| T3 Tokenizer | PASS | 64/64 cases, exact round-trip ([`tests/fixtures/hermes_tokenizer_reference.json`](../../tests/fixtures/hermes_tokenizer_reference.json)) |
| T4 Logits | PASS | top-1 5/5, top-5 overlap 5/5, max top-20 diff `2.10e-05` vs `transformers 5.20.1`; batched prefill `47.87s` |
| T5 Greedy 64 tokens | PASS | `320/320` tokens identical ([`tests/fixtures/hermes_4_3_36b_transformers_reference.json`](../../tests/fixtures/hermes_4_3_36b_transformers_reference.json)); batched generation `2362.87s` |
| T6 KV cache | PASS | `max_abs_diff=7.58e-05`, top-1 `1371` cached vs recompute (`40.27s` recompute) |
| T7 Chat & thinking | PASS | Turn 1 `'The capital of France is Paris.'`; Turn 2 `'The Seine River flows through Paris.'`; `thinking=True` closed `</think>` with answer `**4**` (`10682.64s`, completed in two segments from a saved 160-token greedy prefix); `keep_cots` False/True checked |
| T8 Tool call | PASS | `<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris", "unit": "celsius"}}\n</tool_call>` -> OpenAI `tool_calls` and Anthropic `tool_use` |
| T9 Robustness | PASS | Context overflow error, KV budget error, stop sequence `" Paris"`, cancel callback (`active_sessions=0`) |

### Recorded Prompts & 64-Token Greedy Outputs (T4 & T5)

1. Prompt `'The capital of France is'` (`[0, 1063, 9203, 437, 10341, 506]`, top-1 `' Paris'` id=`10805`, logit=`16.4356`):
   `' Paris. It is located in the north of the country, on the River Seine. Paris is the largest city in France and one of the most important cities in Europe. It is known for its art, fashion, food, and architecture. Some of the most famous landmarks in Paris include the Eiffel Tower, the Louvre'`
2. Prompt `'2 + 2 equals'` (`[0, 145, 1702, 348, 145, 43614]`, top-1 `' '` id=`348`, logit=`20.7355`):
   `' 5?\nI have a question for you.\nWhat is 2 + 2?\nIf you answered 4, you are correct.\nBut what if I told you that 2 + 2 equals 5?\nWould you believe me?\nProbably not.\nBut what if I showed'`
3. Prompt `'def fibonacci(n):\n'` (`[0, 3185, 15797, 123479, 7308, 4283, 326]`, top-1 `'   '` id=`465`, logit=`24.1245`):
   `'    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    else:\n        fib = [0, 1]\n        for i in range(2'`
4. Prompt `'Water boils at 100 degrees'` (`[0, 37209, 154999, 681, 348, 144, 143, 143, 16799]`, top-1 `' Celsius'` id=`107726`, logit=`20.9316`):
   `' Celsius.\nWater boils at 100 degrees Celsius.\nWater boils at 100 degrees Celsius.\nWater boils at 100 degrees Celsius.\nWater boils at 100 degrees Celsius.\nWater boils at 100 degrees Celsius.\nWater boils at 10'`
5. Prompt `'In Python, a dictionary is'` (`[0, 1303, 26657, 139, 387, 33012, 506]`, top-1 `' a'` id=`387`, logit=`21.6034`):
   `' a collection of key-value pairs. Each key is unique and maps to a value. Dictionaries are mutable, meaning you can change their contents after creation. They are also unordered, meaning the items have no defined order.\n\n### Creating a Dictionary\nYou can create a dictionary using curly braces `{}` or the'`

Reference generators: [`tests/reference/generate_hermes_36b_transformers_reference.py`](../../tests/reference/generate_hermes_36b_transformers_reference.py), [`tests/reference/generate_t2_dequant_reference.py`](../../tests/reference/generate_t2_dequant_reference.py), and [`tests/reference/generate_t3_tokenizer_reference.py`](../../tests/reference/generate_t3_tokenizer_reference.py).

---

## 6. Track B Official GGUF SHA-256 & NumPy vs C GEMM Baseline Comparison

* **Artifact:** `NousResearch/Hermes-4.3-36B-GGUF` (`hermes-4_3_36b-Q4_K_M.gguf`)
* **Size:** `21,762,145,216` bytes
* **SHA-256:** `17823599694fa3503ef54bf748d5078c6ce881f4d01616cafa255dc05d215a08`
* **Single 64-layer forward pass (`'The capital of France is'` -> `[0, 1063, 9203, 437, 10341, 506]`, top-1 `' Paris'` id=`10805`, logit=`16.4356`):**
  - NumPy-only path (`v0.3.0` initial baseline): `419.43s`
  - From-scratch C11 SIMD GEMM + `SequentialTensorStreamer`: `50.75s` (measured via `SeedOssEngine("models/hermes-4_3_36b-Q4_K_M.gguf", preload_weights=False).forward(...)`)
