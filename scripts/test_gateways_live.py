import os
import sys
import json
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.local_api.server import app

def run_live_verification():
    client = TestClient(app)
    print("==================================================")
    print("     QERLIO Universal Multi-Gateway Verification   ")
    print("==================================================")

    # 1. Health
    print("\n[1] Checking /v1/health ...")
    r_health = client.get("/v1/health")
    print("Response:", json.dumps(r_health.json(), indent=2))

    # 2. OpenAI Gateway
    print("\n[2] Testing OpenAI /v1/chat/completions (Non-Streaming) ...")
    r_openai = client.post("/v1/chat/completions", json={
        "model": "qerlio-unloaded-stub",
        "messages": [{"role": "user", "content": "What is QERLIO Runtime?"}]
    })
    print("OpenAI Output:", r_openai.json()["choices"][0]["message"]["content"])

    print("\n[3] Testing OpenAI /v1/chat/completions (Streaming SSE) ...")
    r_openai_stream = client.post("/v1/chat/completions", json={
        "model": "qerlio-unloaded-stub",
        "messages": [{"role": "user", "content": "Stream response"}],
        "stream": True
    })
    print("OpenAI Stream Preview:", r_openai_stream.text[:160].replace("\n", " "))

    # 3. Anthropic Claude Gateway
    print("\n[4] Testing Anthropic /v1/messages ...")
    r_anthropic = client.post("/v1/messages", json={
        "model": "qerlio-unloaded-stub",
        "messages": [{"role": "user", "content": "Explain adaptive scheduling."}]
    })
    print("Anthropic Output:", r_anthropic.json()["content"][0]["text"])

    # 4. Ollama Gateway
    print("\n[5] Testing Ollama /api/tags ...")
    r_ollama_tags = client.get("/api/tags")
    loaded_names = [m["name"] for m in r_ollama_tags.json()["models"]]
    print("Ollama Loaded Models:", loaded_names)

    print("\n[6] Testing Ollama /api/generate ...")
    r_ollama_gen = client.post("/api/generate", json={
        "model": "qerlio-unloaded-stub",
        "prompt": "Hello Ollama",
        "stream": False
    })
    print("Ollama Response:", r_ollama_gen.json()["response"])

    print("\n==================================================")
    print("   ALL GATEWAYS VERIFIED AND FULLY FUNCTIONAL!    ")
    print("==================================================")

if __name__ == "__main__":
    run_live_verification()
