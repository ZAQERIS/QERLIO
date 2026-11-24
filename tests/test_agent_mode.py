import pytest
from core.agent.tools import ToolRegistry
from core.agent.agent_runtime import AgentRuntime
from server.local_api.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_tool_registry_calculator():
    reg = ToolRegistry()
    res = reg.execute("calculator", {"expression": "12 * 12 + 10"})
    assert res["success"] is True
    assert res["result"] == "154"

def test_tool_registry_python_eval():
    reg = ToolRegistry()
    code = "result = [x * 2 for x in range(4)]"
    res = reg.execute("python_eval", {"code": code})
    assert res["success"] is True
    assert res["result"] == "[0, 2, 4, 6]"

def test_tool_registry_unknown_tool():
    reg = ToolRegistry()
    res = reg.execute("unknown_arbitrary_tool", {})
    assert res["success"] is False
    assert "not recognized" in res["error"]

def test_agent_parsing_zqn_dialectical_format():
    runtime = AgentRuntime()
    text = '<think>\nNeed computation\n</think>\n<call name="calculator" args=\'{"expression": "50 * 2"}\'/>'
    calls = runtime.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "calculator"
    assert calls[0]["arguments"] == {"expression": "50 * 2"}

def test_agent_parsing_zaqeris_7b_tool_format():
    runtime = AgentRuntime()
    text = '<tool_call>\n{"name": "file_search", "arguments": {"pattern": "*.py"}}\n</tool_call>'
    calls = runtime.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "file_search"
    assert calls[0]["arguments"]["pattern"] == "*.py"

def test_agent_run_step_execution():
    runtime = AgentRuntime()
    text = '<call name="calculator" args=\'{"expression": "99 + 1"}\'/>'
    step = runtime.run_step(text)
    assert step["has_tool_calls"] is True
    assert len(step["results"]) == 1
    assert step["results"][0]["execution"]["success"] is True
    assert step["results"][0]["execution"]["result"] == "100"
    assert "<tool_response>" in step["next_prompt_suffix"]

def test_api_agents_endpoints():
    res_exec = client.post("/v1/agents/execute", json={
        "tool": "calculator",
        "arguments": {"expression": "2 ** 5"}
    })
    assert res_exec.status_code == 200
    assert res_exec.json()["success"] is True
    assert res_exec.json()["result"] == "32"

    res_run = client.post("/v1/agents/run", json={
        "prompt": "Calculate 100 / 4",
        "model": "ZAQERIS/ZQN-12B"
    })
    assert res_run.status_code == 200
    data = res_run.json()
    assert data["step"]["has_tool_calls"] is True

def test_openai_tool_calling_integration():
    res = client.post("/v1/chat/completions", json={
        "model": "ZAQERIS/ZAQERIS-7B",
        "messages": [{"role": "user", "content": "What is 42 * 2?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
                }
            }
        ]
    })
    assert res.status_code == 200
    data = res.json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert len(data["choices"][0]["message"]["tool_calls"]) == 1

def test_anthropic_tool_calling_integration():
    res = client.post("/v1/messages", json={
        "model": "ZAQERIS/ZAQERIS-7B",
        "messages": [{"role": "user", "content": "Calculate value"}],
        "tools": [{"name": "calculator", "description": "Calculator"}]
    })
    assert res.status_code == 200
    data = res.json()
    assert data["stop_reason"] == "tool_use"
    assert data["content"][0]["type"] == "tool_use"
