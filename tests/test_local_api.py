import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.local_api.server import app, unload_native_models

client = TestClient(app)


def test_health_endpoints():
    unload_native_models()
    res1 = client.get("/v1/health")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "healthy"
    assert data1["cpu_usage_pct"] is None
    assert data1["ram_available_gb"] is None
    assert data1["loaded_models"] == []
    assert "gateways" in data1
    assert "openai" in data1["gateways"]
    assert "anthropic" in data1["gateways"]
    assert "ollama" in data1["gateways"]

    res2 = client.get("/healthz")
    assert res2.status_code == 200


def test_openai_chat_completions():
    unload_native_models()
    # Non-streaming
    res_non_stream = client.post("/v1/chat/completions", json={
        "model": "qerlio-unloaded-stub",
        "messages": [{"role": "user", "content": "Hello OpenAI gateway"}]
    })
    assert res_non_stream.status_code == 200
    d = res_non_stream.json()
    assert d["object"] == "chat.completion"
    assert d["simulated"] is True
    assert d["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in d

    # Streaming
    res_stream = client.post("/v1/chat/completions", json={
        "model": "qerlio-unloaded-stub",
        "messages": [{"role": "user", "content": "Stream test"}],
        "stream": True
    })
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers["content-type"]
    assert "data: [DONE]" in res_stream.text


def test_openai_models_list():
    unload_native_models()
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert data["loaded_count"] == 0
    assert data["data"] == []


def test_anthropic_messages_endpoint():
    unload_native_models()
    # Non-streaming
    res = client.post("/v1/messages", json={
        "model": "qerlio-unloaded-stub",
        "messages": [{"role": "user", "content": "Hello Claude format"}],
        "max_tokens": 100
    })
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "message"
    assert data["simulated"] is True
    assert data["role"] == "assistant"
    assert len(data["content"]) >= 1
    assert data["content"][0]["type"] == "text"
    assert "stop_reason" in data

    # Streaming
    res_stream = client.post("/v1/messages", json={
        "model": "qerlio-unloaded-stub",
        "messages": [{"role": "user", "content": "Stream Claude"}],
        "stream": True
    })
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers["content-type"]
    assert "event: message_start" in res_stream.text
    assert "event: content_block_delta" in res_stream.text
    assert "event: message_stop" in res_stream.text


def test_ollama_endpoints():
    unload_native_models()
    # /api/tags returns only loaded models (empty when nothing is loaded)
    res_tags = client.get("/api/tags")
    assert res_tags.status_code == 200
    tags_data = res_tags.json()
    assert tags_data["loaded_count"] == 0
    assert tags_data["models"] == []

    # /api/generate
    res_gen = client.post("/api/generate", json={
        "model": "qerlio-unloaded-stub",
        "prompt": "Hello Ollama format",
        "stream": False
    })
    assert res_gen.status_code == 200
    gen_data = res_gen.json()
    assert "response" in gen_data
    assert gen_data["done"] is True
    assert gen_data["simulated"] is True

    # /api/generate streaming (ndjson)
    res_gen_stream = client.post("/api/generate", json={
        "model": "qerlio-unloaded-stub",
        "prompt": "Streaming prompt",
        "stream": True
    })
    assert res_gen_stream.status_code == 200
    assert "application/x-ndjson" in res_gen_stream.headers["content-type"]
    lines = [json.loads(line) for line in res_gen_stream.text.strip().split("\n") if line.strip()]
    assert len(lines) >= 2
    assert lines[-1]["done"] is True

    # /api/chat
    res_chat = client.post("/api/chat", json={
        "model": "qerlio-unloaded-stub",
        "messages": [{"role": "user", "content": "Chat test"}],
        "stream": False
    })
    assert res_chat.status_code == 200
    chat_data = res_chat.json()
    assert chat_data["message"]["role"] == "assistant"
    assert chat_data["done"] is True
    assert chat_data["simulated"] is True
