import os
import sys
import time
import json
import uuid
import psutil
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Header, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sdk.python.qerlio import QerlioEngine, GenerationConfig, PowerProfile
from core.compression.quant_matrix import QUANT_CATALOG, compute_layer_adaptive_mixed_precision, calculate_sink_aware_kv_cache
from core.model_registry.architecture_catalog import ArchitectureCatalog
from core.agent.agent_runtime import AgentRuntime
from core.agent.tools import ToolRegistry
from core.models.hermes_adapter import Hermes36BAdapter
from core.inference import load_gguf_engine, Qwen35Engine, SeedOssEngine
from core.inference.sampler import SamplerConfig

app = FastAPI(
    title="QERLIO Universal Multi-API Gateway",
    version="0.3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE = QerlioEngine(max_concurrent_sessions=8)
ACTIVE_MODELS: Dict[str, Union[SeedOssEngine, Qwen35Engine]] = {}
SERVER_API_KEY = os.environ.get("QERLIO_API_KEY", "")
ARCH_CATALOG = ArchitectureCatalog()
TOOL_REGISTRY = ToolRegistry()
AGENT_RUNTIME = AgentRuntime(tool_registry=TOOL_REGISTRY)


def load_native_model(
    model_path: str,
    model_id: str = "loaded-local-gguf",
    max_context_length: Optional[int] = None,
    preload_weights: Optional[bool] = None,
) -> Union[SeedOssEngine, Qwen35Engine]:
    """Load a real GGUF v2/v3 model (`seed_oss` or `qwen35`) into ACTIVE_MODELS for live inference."""
    engine = load_gguf_engine(
        model_path,
        max_context_length=max_context_length,
        preload_weights=preload_weights,
    )
    ACTIVE_MODELS[model_id] = engine
    ACTIVE_MODELS["default"] = engine
    return engine


def unload_native_models() -> None:
    """Close and clear all active native model engines."""
    seen = set()
    for eng in list(ACTIVE_MODELS.values()):
        if id(eng) not in seen:
            seen.add(id(eng))
            eng.close()
    ACTIVE_MODELS.clear()


def _get_loaded_engine(model_name: str) -> Optional[SeedOssEngine]:
    if model_name in ACTIVE_MODELS:
        return ACTIVE_MODELS[model_name]
    return ACTIVE_MODELS.get("default")


class OpenAIChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""


class OpenAIChatRequest(BaseModel):
    model: str = "qerlio-unloaded-stub"
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    max_tokens: Optional[int] = 64
    stream: Optional[bool] = False
    thinking: Optional[bool] = False
    keep_cots: Optional[bool] = False
    power_profile: Optional[str] = "balanced"
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    seed: Optional[int] = 1234


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class AnthropicRequest(BaseModel):
    model: str = "qerlio-unloaded-stub"
    messages: List[AnthropicMessage]
    max_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    thinking: Optional[bool] = False
    system: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None


class OllamaChatRequest(BaseModel):
    model: str = "qerlio-unloaded-stub"
    messages: List[OpenAIChatMessage]
    stream: Optional[bool] = False


class OllamaGenerateRequest(BaseModel):
    model: str = "qerlio-unloaded-stub"
    prompt: str
    stream: Optional[bool] = False


class AgentExecuteRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qerlio-unloaded-stub"
    max_iterations: Optional[int] = 3


class ModelLoadRequest(BaseModel):
    model_path: str
    model_id: Optional[str] = "loaded-local-gguf"
    max_context_length: Optional[int] = None
    preload_weights: Optional[bool] = True


def verify_token(authorization: Optional[str] = Header(None)):
    if not SERVER_API_KEY:
        return True
    if not authorization:
        return True
    token = authorization.replace("Bearer ", "").strip()
    if token != SERVER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized"
        )
    return True


@app.get("/health")
@app.get("/v1/health")
@app.get("/healthz")
def get_health():
    expose_host = os.environ.get("QERLIO_EXPOSE_HOST_METRICS") == "1"
    ram = psutil.virtual_memory() if expose_host else None
    loaded_ids = [k for k in ACTIVE_MODELS.keys() if k != "default"]
    return {
        "status": "healthy",
        "runtime": "QERLIO Universal Gateway 0.3.1",
        "cpu_usage_pct": round(psutil.cpu_percent(), 1) if expose_host else None,
        "ram_available_gb": round(ram.available / (1024**3), 2) if ram is not None else None,
        "gateways": ["openai", "anthropic", "ollama"],
        "loaded_models": loaded_ids,
        "live_engine_loaded": len(ACTIVE_MODELS) > 0,
        "simulated_inference": len(ACTIVE_MODELS) == 0,
        "supported_architectures_count": ARCH_CATALOG.total_count,
        "tested_architectures_count": ARCH_CATALOG.tested_count,
        "experimental_architectures_count": ARCH_CATALOG.experimental_count,
        "agent_mode_ready": True
    }


@app.post("/v1/models/load")
def api_load_model(req: ModelLoadRequest, _: bool = Depends(verify_token)):
    try:
        target_id = req.model_id or "loaded-local-gguf"
        eng = load_native_model(
            model_path=req.model_path,
            model_id=target_id,
            max_context_length=req.max_context_length,
            preload_weights=bool(req.preload_weights),
        )
        return {
            "status": "loaded",
            "simulated": False,
            "model_id": target_id,
            "architecture": eng.config.architecture,
            "layers": eng.config.block_count,
            "vocab_size": eng.config.vocab_size,
            "max_context_length": eng.max_context_length,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/v1/models/unload")
def api_unload_models(_: bool = Depends(verify_token)):
    unload_native_models()
    return {"status": "unloaded", "simulated": True}


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>QERLIO Dashboard</h1>"


@app.get("/v1/telemetry/hud")
def get_telemetry_hud():
    expose_host = os.environ.get("QERLIO_EXPOSE_HOST_METRICS") == "1"
    ram = psutil.virtual_memory() if expose_host else None
    return {
        "measurement_status": "no_live_measurement",
        "throughput_tps": None,
        "ttft_ms": None,
        "model_memory_gb": None,
        "ram_total_gb": round(ram.total / (1024**3), 2) if ram is not None else None,
        "ram_available_gb": round(ram.available / (1024**3), 2) if ram is not None else None,
        "thermal_state": None,
        "quant_schemes_count": len(QUANT_CATALOG),
        "supported_architectures_count": ARCH_CATALOG.total_count,
        "tested_architectures_count": ARCH_CATALOG.tested_count,
        "experimental_architectures_count": ARCH_CATALOG.experimental_count,
        "architectures_families": ARCH_CATALOG.list_families(),
        "theoretical_estimates_note": (
            "adaptive_hybrid_7b and sink_aware_16k are theoretical formula estimates, "
            "not live runtime measurements"
        ),
        "adaptive_hybrid_7b": compute_layer_adaptive_mixed_precision(7.0),
        "sink_aware_16k": calculate_sink_aware_kv_cache(context_tokens=16384)
    }


@app.get("/v1/models")
def list_openai_models(_: bool = Depends(verify_token)):
    loaded_ids = [k for k in ACTIVE_MODELS.keys() if k != "default"]
    now = int(time.time())
    data = [
        {
            "id": mid,
            "object": "model",
            "created": now,
            "owned_by": "loaded-local-gguf",
            "status": "loaded",
            "loaded": True,
            "simulated": False,
            "architecture": ACTIVE_MODELS[mid].config.architecture,
        }
        for mid in loaded_ids
    ]
    return {
        "object": "list",
        "loaded_count": len(data),
        "data": data
    }


@app.get("/v1/architectures")
def list_architectures(family: Optional[str] = None, search: Optional[str] = None, status: Optional[str] = None):
    if family:
        items = ARCH_CATALOG.list_by_family(family)
    elif search:
        items = ARCH_CATALOG.search(search)
    else:
        items = ARCH_CATALOG.list_all()
    if status:
        items = [i for i in items if i.get("support_status", "").lower() == status.lower()]
    return {
        "total_count": len(items),
        "global_supported_count": ARCH_CATALOG.total_count,
        "tested_count": ARCH_CATALOG.tested_count,
        "experimental_count": ARCH_CATALOG.experimental_count,
        "families": ARCH_CATALOG.list_families(),
        "data": items
    }


@app.get("/v1/architectures/{arch_id}")
def get_architecture(arch_id: str):
    item = ARCH_CATALOG.get(arch_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Architecture '{arch_id}' not found")
    return item


@app.post("/v1/agents/execute")
def agent_execute_tool(req: AgentExecuteRequest, _: bool = Depends(verify_token)):
    res = TOOL_REGISTRY.execute(req.tool, req.arguments)
    return res


@app.post("/v1/agents/run")
def agent_run_step(req: AgentRunRequest, _: bool = Depends(verify_token)):
    simulated_step = f'<think>\nDeliberating prompt: {req.prompt}\n</think>\n<tool_call>\n{{"name": "calculator", "arguments": {{"expression": "100 / 4"}}}}\n</tool_call>'
    step_res = AGENT_RUNTIME.run_step(simulated_step)
    return {
        "model": req.model,
        "prompt": req.prompt,
        "simulated": True,
        "step": step_res
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAIChatRequest, _: bool = Depends(verify_token)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    live_engine = _get_loaded_engine(req.model)
    if live_engine is not None:
        sampler_cfg = SamplerConfig(
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_p=req.top_p if req.top_p is not None else 0.95,
            top_k=req.top_k if req.top_k is not None else 20,
            seed=req.seed,
        )
        max_new = req.max_tokens if req.max_tokens and req.max_tokens > 0 else 64
        messages_dicts = [{"role": m.role, "content": m.content or ""} for m in req.messages]
        adapter = Hermes36BAdapter(
            thinking=bool(req.thinking),
            keep_cots=bool(req.keep_cots),
        )

        if req.stream:
            if getattr(live_engine.config, "architecture", "") == "qwen35":
                formatted_prompt = Qwen35Engine.format_qwen_chat_prompt(
                    messages_dicts,
                    tools=req.tools,
                    thinking=bool(req.thinking),
                    add_generation_prompt=True,
                )
                stop_seqs = ["<|im_end|>", "<|endoftext|>"]
            else:
                formatted_prompt = adapter.format_chat_prompt(
                    messages_dicts,
                    tools=req.tools,
                    add_generation_prompt=True,
                )
                stop_seqs = ["<|eot_id|>", "<seed:eos>"]

            def live_sse_stream():
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req.model, 'simulated': False, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
                for ev in live_engine.generate_stream(
                    formatted_prompt,
                    sampler_config=sampler_cfg,
                    max_new_tokens=max_new,
                    stop_sequences=stop_seqs,
                ):
                    if ev.get("finish_reason") == "eos":
                        break
                    piece = ev.get("text") or ""
                    if not piece:
                        continue
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "simulated": False,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req.model, 'simulated': False, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(live_sse_stream(), media_type="text/event-stream")

        chat_res = live_engine.chat(
            messages_dicts,
            tools=req.tools,
            thinking=bool(req.thinking),
            keep_cots=bool(req.keep_cots),
            sampler_config=sampler_cfg,
            max_new_tokens=max_new,
        )
        prompt_token_count = (
            len(live_engine.tokenizer.encode(chat_res["prompt"]))
            if live_engine.tokenizer is not None
            else 0
        )
        comp_token_count = len(chat_res["token_ids"])
        usage_dict = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": comp_token_count,
            "total_tokens": prompt_token_count + comp_token_count,
        }

        if chat_res["has_tool_calls"]:
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_ts,
                "model": req.model,
                "simulated": False,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": chat_res["content"],
                            "reasoning_content": chat_res["thinking"],
                            "tool_calls": chat_res["openai_tool_calls"],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": usage_dict,
            }

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "simulated": False,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": chat_res["content"] if chat_res["content"] is not None else chat_res["raw_output"],
                        "reasoning_content": chat_res["thinking"],
                    },
                    "finish_reason": "stop" if chat_res["finish_reason"] in ("eos", "stop_sequence") else "length",
                }
            ],
            "usage": usage_dict,
        }

    # Fallback when NO live model is loaded: explicitly simulated
    if req.tools and len(req.tools) > 0:
        first_tool = req.tools[0]
        tool_name = first_tool.get("function", {}).get("name", "calculator")
        tool_call_payload = {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps({"expression": "42 * 2"})
            }
        }
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "simulated": True,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call_payload]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": None
        }

    if req.stream:
        def sse_stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req.model, 'simulated': True, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
            chunks = ["Hello", " from", " QERLIO", " Universal", " API", " Gateway", "! [Simulated:", " no", " model", " loaded]"]
            for chunk in chunks:
                time.sleep(0.01)
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "simulated": True,
                    "choices": [{"index": 0, "delta": {"content": " " + chunk}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(data)}\n\n"

            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req.model, 'simulated': True, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    content = "Hello from QERLIO Universal API Gateway! [Simulated response: no model is currently loaded into the engine.]"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "simulated": True,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }
        ],
        "usage": None
    }


@app.post("/v1/messages")
def anthropic_messages(req: AnthropicRequest, _: bool = Depends(verify_token)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    msg_id = f"msg_{uuid.uuid4().hex[:16]}"
    live_engine = _get_loaded_engine(req.model)
    if live_engine is not None:
        sampler_cfg = SamplerConfig(
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_p=0.95,
            top_k=20,
        )
        max_new = req.max_tokens if req.max_tokens and req.max_tokens > 0 else 64
        messages_dicts: List[Dict[str, Any]] = []
        if req.system:
            messages_dicts.append({"role": "system", "content": req.system})
        for m in req.messages:
            if isinstance(m.content, str):
                messages_dicts.append({"role": m.role, "content": m.content})
            else:
                joined = "\n".join(str(b.get("text", "")) for b in m.content if isinstance(b, dict))
                messages_dicts.append({"role": m.role, "content": joined})

        chat_res = live_engine.chat(
            messages_dicts,
            tools=req.tools,
            thinking=bool(req.thinking),
            sampler_config=sampler_cfg,
            max_new_tokens=max_new,
        )
        prompt_token_count = (
            len(live_engine.tokenizer.encode(chat_res["prompt"]))
            if live_engine.tokenizer is not None
            else 0
        )
        comp_token_count = len(chat_res["token_ids"])
        content_blocks: List[Dict[str, Any]] = []
        if chat_res["content"]:
            content_blocks.append({"type": "text", "text": chat_res["content"]})
        if chat_res["has_tool_calls"]:
            content_blocks.extend(chat_res["anthropic_tool_use"])
        if not content_blocks:
            content_blocks.append({"type": "text", "text": chat_res["raw_output"]})

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "simulated": False,
            "content": content_blocks,
            "model": req.model,
            "stop_reason": "tool_use" if chat_res["has_tool_calls"] else "end_turn",
            "usage": {"input_tokens": prompt_token_count, "output_tokens": comp_token_count},
        }

    text_out = "QERLIO Anthropic-compatible gateway response. [Simulated: no model is currently loaded into the engine.]"

    if req.tools and len(req.tools) > 0:
        first_tool = req.tools[0]
        tool_name = first_tool.get("name", "calculator")
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "simulated": True,
            "content": [
                {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": {"expression": "25 * 4"}
                }
            ],
            "model": req.model,
            "stop_reason": "tool_use",
            "usage": None
        }

    if req.stream:
        def anthropic_sse():
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'simulated': True, 'content': [], 'model': req.model}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
            for word in text_out.split():
                time.sleep(0.01)
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': ' ' + word}})}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return StreamingResponse(anthropic_sse(), media_type="text/event-stream")

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "simulated": True,
        "content": [{"type": "text", "text": text_out}],
        "model": req.model,
        "stop_reason": "end_turn",
        "usage": None
    }


@app.get("/api/tags")
def ollama_tags():
    loaded_ids = [k for k in ACTIVE_MODELS.keys() if k != "default"]
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    models = [
        {
            "name": mid,
            "model": mid,
            "modified_at": now_iso,
            "size": None,
            "digest": None,
            "status": "loaded",
            "loaded": True,
            "details": {
                "format": "gguf",
                "family": ACTIVE_MODELS[mid].config.architecture,
            },
        }
        for mid in loaded_ids
    ]
    return {
        "models": models,
        "loaded_count": len(models),
    }


@app.post("/api/generate")
def ollama_generate(req: OllamaGenerateRequest):
    if req.stream:
        def ndjson_stream():
            words = ["QERLIO", "Ollama", "gateway", "streaming", "output."]
            for word in words:
                time.sleep(0.01)
                yield json.dumps({"model": req.model, "response": " " + word, "done": False, "simulated": True}) + "\n"
            yield json.dumps({"model": req.model, "response": "", "done": True, "simulated": True, "total_duration": None}) + "\n"

        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")

    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": "Hello from QERLIO Ollama-compatible endpoint. [Simulated: no model is currently loaded.]",
        "done": True,
        "simulated": True,
        "total_duration": None
    }


@app.post("/api/chat")
def ollama_chat(req: OllamaChatRequest):
    if req.stream:
        def ndjson_chat():
            words = ["Hello", "from", "QERLIO", "chat."]
            for word in words:
                time.sleep(0.01)
                yield json.dumps({"model": req.model, "message": {"role": "assistant", "content": " " + word}, "done": False, "simulated": True}) + "\n"
            yield json.dumps({"model": req.model, "done": True, "simulated": True}) + "\n"

        return StreamingResponse(ndjson_chat(), media_type="application/x-ndjson")

    return {
        "model": req.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": "Hello from QERLIO Ollama chat endpoint. [Simulated: no model is currently loaded.]"},
        "done": True,
        "simulated": True
    }


def start_server(host: str = "127.0.0.1", port: int = 8787):
    model_path = os.environ.get("QERLIO_MODEL_PATH")
    if model_path and os.path.isfile(model_path):
        load_native_model(model_path)
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
