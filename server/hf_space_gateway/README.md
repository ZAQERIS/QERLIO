---
title: QERLIO Remote API Gateway
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
license: apache-2.0
---

# QERLIO Remote API Gateway (Template Stub)

OpenAI-compatible FastAPI/Gradio HTTP gateway scaffold (`POST /v1/chat/completions`, `GET /v1/models`, `GET /healthz`) with optional bearer-token authentication and a configurable 7-day trial expiration check.

## Scope & Honest Status
- **HTTP Gateway Scaffold Only:** This service provides HTTP routing, secret token verification (`QERLIO_GATEWAY_SECRET`), and trial expiration checks (`GATEWAY_LAUNCH_TIMESTAMP`).
- **No Model Weights Attached by Default:** Until a live inference engine is connected, `GET /v1/models` returns an empty loaded list (`"backend_loaded": false`, `"data": []`), `GET /healthz` returns `"loaded_models": []`, and `POST /v1/chat/completions` returns an explicit stub notice (`"simulated": true`, `"usage": null`) without executing or impersonating any model.

## Usage with OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7860/v1",
    api_key="your-gateway-secret"
)

response = client.chat.completions.create(
    model="unloaded-gateway-stub",
    messages=[{"role": "user", "content": "Ping gateway stub."}],
    stream=False
)
print(response.choices[0].message.content)
```
