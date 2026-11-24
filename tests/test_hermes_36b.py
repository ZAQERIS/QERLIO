import os
import sys
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient

from core.models.hermes_adapter import (
    Hermes36BAdapter,
    THINKING_SYSTEM_PROMPT,
    STANDARD_SYSTEM_PROMPT,
    DEFAULT_GENERATION_CONFIG,
)
from core.remote.hf_remote_client import HFRemoteClient
from server.hf_space_gateway.app import app as space_gateway_app
from server.local_api.server import app as local_app


def test_hermes_adapter_base_specs_and_defaults():
    adapter = Hermes36BAdapter()
    assert adapter.thinking is False
    assert adapter.keep_cots is False
    assert adapter.base_model == "ByteDance Seed-OSS-36B-Base"
    assert adapter.architecture_class == "SeedOssForCausalLM"
    assert adapter.model_type == "seed_oss"
    assert adapter.context_length == 524288
    assert adapter.hf_repo == "NousResearch/Hermes-4.3-36B"
    assert adapter.gguf_repo == "NousResearch/Hermes-4.3-36B-GGUF"
    assert adapter.bos_token == "<seed:bos>"
    assert adapter.eos_token == "<|eot_id|>"
    assert DEFAULT_GENERATION_CONFIG["temperature"] == 0.6
    assert DEFAULT_GENERATION_CONFIG["top_p"] == 0.95
    assert DEFAULT_GENERATION_CONFIG["top_k"] == 20


def test_hermes_chat_prompt_standard_default():
    adapter = Hermes36BAdapter()
    messages = [{"role": "user", "content": "What is the capital of Egypt?"}]
    prompt = adapter.format_chat_prompt(messages)
    assert prompt.startswith("<seed:bos>")
    assert STANDARD_SYSTEM_PROMPT in prompt
    assert THINKING_SYSTEM_PROMPT not in prompt
    assert "<|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of Egypt?<|eot_id|>" in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_hermes_chat_prompt_thinking_enabled():
    adapter = Hermes36BAdapter(thinking=True)
    messages = [{"role": "user", "content": "Derive the gradient of cross-entropy loss."}]
    prompt = adapter.format_chat_prompt(messages)
    assert prompt.startswith("<seed:bos>")
    assert THINKING_SYSTEM_PROMPT in prompt
    assert "<think> </think>" in prompt
    assert "Derive the gradient of cross-entropy loss." in prompt


def test_hermes_keep_cots_behavior():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "<think>\nOld internal monologue\n</think>\nFirst answer"},
        {"role": "user", "content": "Follow-up question"},
    ]

    adapter_strip = Hermes36BAdapter(thinking=True, keep_cots=False)
    prompt_strip = adapter_strip.format_chat_prompt(messages)
    assert "Old internal monologue" not in prompt_strip
    assert "First answer" in prompt_strip

    parsed_strip = adapter_strip.parse_generation("<think>\nNew thought\n</think>\nFinal reply")
    assert parsed_strip["thinking"] == "New thought"
    assert parsed_strip["final_content"] == "Final reply"
    assert "<think>" not in parsed_strip["final_content"]

    adapter_keep = Hermes36BAdapter(thinking=True, keep_cots=True)
    prompt_keep = adapter_keep.format_chat_prompt(messages)
    assert "<think>\nOld internal monologue\n</think>" in prompt_keep
    assert "First answer" in prompt_keep

    parsed_keep = adapter_keep.parse_generation("<think>\nNew thought\n</think>\nFinal reply")
    assert parsed_keep["thinking"] == "New thought"
    assert "<think>\nNew thought\n</think>" in parsed_keep["final_content"]
    assert "Final reply" in parsed_keep["final_content"]


def test_hermes_chat_prompt_with_tools():
    adapter = Hermes36BAdapter(thinking=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Fetch current weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }
    ]
    messages = [{"role": "user", "content": "How is the weather in Cairo?"}]
    prompt = adapter.format_chat_prompt(messages, tools=tools)
    assert "<tools>" in prompt
    assert "lookup_weather" in prompt
    assert "</tools>" in prompt
    assert "<tool_call>" in prompt


def test_hermes_parse_generation_with_thinking():
    adapter = Hermes36BAdapter()
    raw = (
        "<think>\nThe user wants to know the capital of Egypt. "
        "Egypt's capital is Cairo.</think>\n"
        "The capital of Egypt is Cairo."
    )
    parsed = adapter.parse_generation(raw)
    assert parsed["has_thinking"] is True
    assert parsed["unclosed_thinking"] is False
    assert "The user wants to know the capital of Egypt." in parsed["thinking"]
    assert parsed["final_content"] == "The capital of Egypt is Cairo."
    assert parsed["has_tool_calls"] is False
    assert parsed["tool_calls"] == []


def test_hermes_parse_unclosed_think_tag():
    adapter = Hermes36BAdapter(keep_cots=False)
    raw = "<think>\nDeliberating step 1 and step 2 before hitting token limit"
    parsed = adapter.parse_generation(raw)
    assert parsed["has_thinking"] is True
    assert parsed["unclosed_thinking"] is True
    assert parsed["thinking"] == "Deliberating step 1 and step 2 before hitting token limit"
    assert parsed["final_content"] is None
    assert parsed["has_tool_calls"] is False


def test_hermes_parse_generation_with_tool_call_and_conversions():
    adapter = Hermes36BAdapter()
    raw = (
        "<think>\nNeed to evaluate 42 * 100.</think>\n"
        '<tool_call>\n{"name": "calculator", "arguments": {"expr": "42 * 100"}}\n</tool_call>'
    )
    parsed = adapter.parse_generation(raw)
    assert parsed["has_thinking"] is True
    assert "Need to evaluate 42 * 100." in parsed["thinking"]
    assert parsed["has_tool_calls"] is True
    assert len(parsed["tool_calls"]) == 1
    assert parsed["tool_calls"][0]["name"] == "calculator"
    assert parsed["tool_calls"][0]["arguments"] == {"expr": "42 * 100"}

    # OpenAI tool_calls conversion
    openai_tc = parsed["openai_tool_calls"]
    assert len(openai_tc) == 1
    assert openai_tc[0]["id"] == "call_0"
    assert openai_tc[0]["type"] == "function"
    assert openai_tc[0]["function"]["name"] == "calculator"
    assert json.loads(openai_tc[0]["function"]["arguments"]) == {"expr": "42 * 100"}

    # Anthropic tool_use conversion
    anthropic_tu = parsed["anthropic_tool_use"]
    assert len(anthropic_tu) == 1
    assert anthropic_tu[0]["type"] == "tool_use"
    assert anthropic_tu[0]["id"] == "toolu_0"
    assert anthropic_tu[0]["name"] == "calculator"
    assert anthropic_tu[0]["input"] == {"expr": "42 * 100"}


def test_hermes_parse_malformed_and_unclosed_tool_calls():
    adapter = Hermes36BAdapter()
    raw = (
        "Checking tools:\n"
        "<tool_call>\n{not valid json at all}\n</tool_call>\n"
        '<tool_call>\n{"arguments": {"missing": "name_field"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "unclosed_tool", "arguments": {"a": 1}}'
    )
    parsed = adapter.parse_generation(raw)
    assert parsed["has_tool_calls"] is False
    assert parsed["tool_calls"] == []
    assert parsed["openai_tool_calls"] == []
    assert parsed["anthropic_tool_use"] == []
    assert len(parsed["malformed_tool_calls"]) == 3
    assert parsed["final_content"] == "Checking tools:"


def test_hermes_parse_generation_plain():
    adapter = Hermes36BAdapter()
    raw = "Direct answer without special tags."
    parsed = adapter.parse_generation(raw)
    assert parsed["has_thinking"] is False
    assert parsed["thinking"] is None
    assert parsed["has_tool_calls"] is False
    assert parsed["final_content"] == "Direct answer without special tags."


def test_hf_remote_client_mock_completion_and_template_prep():
    client = HFRemoteClient(base_url="https://mock-space.hf.space/v1", token="hf_test_123")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer hf_test_123"
    assert client.hermes_adapter.thinking is False

    formatted = client.format_prompt([{"role": "user", "content": "Hello"}], thinking=True)
    assert formatted.startswith("<seed:bos>")
    assert THINKING_SYSTEM_PROMPT in formatted

    mock_resp_data = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<think>\nTesting thinking.\n</think>\n<tool_call>\n{"name": "calc", "arguments": {"x": 2}}\n</tool_call>'
                }
            }
        ]
    }
    mock_bytes = json.dumps(mock_resp_data).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_bytes
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mocked_open:
        res = client.generate(
            model="NousResearch/Hermes-4.3-36B",
            messages=[{"role": "user", "content": "Hello"}],
            thinking=True
        )
        sent_req = mocked_open.call_args[0][0]
        sent_payload = json.loads(sent_req.data.decode("utf-8"))
        assert sent_payload["messages"][0]["role"] == "system"
        assert THINKING_SYSTEM_PROMPT in sent_payload["messages"][0]["content"]
        assert res["thinking"] == "Testing thinking."
        assert len(res["openai_tool_calls"]) == 1
        assert len(res["anthropic_tool_use"]) == 1
        assert res["model"] == "NousResearch/Hermes-4.3-36B"


def test_hf_remote_client_mock_streaming():
    client = HFRemoteClient(base_url="https://mock-space.hf.space/v1")
    sse_lines = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        b'data: {"choices": [{"delta": {"content": " World"}}]}\n',
        b'data: [DONE]\n'
    ]

    mock_resp = MagicMock()
    mock_resp.__iter__.return_value = iter(sse_lines)
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        chunks = list(client.stream_generate(
            model="NousResearch/Hermes-4.3-36B",
            messages=[{"role": "user", "content": "Hi"}]
        ))
        assert chunks == ["Hello", " World"]


def test_space_gateway_honest_stub_no_impersonation():
    test_client = TestClient(space_gateway_app)

    health_res = test_client.get("/healthz")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["status"] == "healthy"
    assert health_data["backend_loaded"] is False
    assert health_data["simulated"] is True
    assert health_data["loaded_models"] == []
    assert "zerogpu_active" not in health_data

    models_res = test_client.get("/v1/models")
    assert models_res.status_code == 200
    models_data = models_res.json()
    assert models_data["backend_loaded"] is False
    assert models_data["loaded_count"] == 0
    assert models_data["data"] == []

    chat_res = test_client.post("/v1/chat/completions", json={
        "model": "NousResearch/Hermes-4.3-36B",
        "messages": [{"role": "user", "content": "Test prompt"}],
        "stream": False
    })
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["simulated"] is True
    assert chat_data["usage"] is None
    content = chat_data["choices"][0]["message"]["content"]
    assert "<think>" not in content
    assert "[Hermes 4.3 36B]" not in content
    assert "[ZAQERIS-7B Native]" not in content

    tools_res = test_client.post("/v1/chat/completions", json={
        "model": "NousResearch/Hermes-4.3-36B",
        "messages": [{"role": "user", "content": "Compute 42 * 2"}],
        "tools": [{"type": "function", "function": {"name": "calculator"}}]
    })
    assert tools_res.status_code == 200
    tools_data = tools_res.json()
    assert tools_data["simulated"] is True
    assert tools_data["usage"] is None
    assert "tool_calls" not in tools_data["choices"][0]["message"]


def test_local_api_does_not_impersonate_hermes_or_zaqeris():
    local_client = TestClient(local_app)
    for model_id in ("NousResearch/Hermes-4.3-36B", "ZAQERIS/ZAQERIS-7B"):
        res = local_client.post("/v1/chat/completions", json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Test prompt"}]
        })
        assert res.status_code == 200
        body = res.json()
        assert body["simulated"] is True
        assert body["usage"] is None
        msg = body["choices"][0]["message"]["content"]
        assert "[Hermes 4.3 36B]" not in msg
        assert "[ZAQERIS-7B Native]" not in msg
        assert "<think>" not in msg


def test_space_gateway_secret_auth_and_expiry():
    test_client = TestClient(space_gateway_app)

    with patch.dict(os.environ, {"QERLIO_GATEWAY_SECRET": "secret_zaq_999"}):
        res_unauth = test_client.get("/v1/models")
        assert res_unauth.status_code == 401

        res_wrong = test_client.post(
            "/v1/chat/completions",
            json={"model": "NousResearch/Hermes-4.3-36B", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong_token"}
        )
        assert res_wrong.status_code == 401

        res_ok = test_client.post(
            "/v1/chat/completions",
            json={"model": "NousResearch/Hermes-4.3-36B", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret_zaq_999"}
        )
        assert res_ok.status_code == 200

    past_timestamp = str(1000000000.0)
    with patch.dict(os.environ, {"GATEWAY_LAUNCH_TIMESTAMP": past_timestamp}):
        res_expired = test_client.get("/v1/models")
        assert res_expired.status_code == 403
        assert "expired" in res_expired.json()["detail"].lower()
