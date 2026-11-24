import os
import time
import json
import uuid
import tempfile
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

app = FastAPI(
    title="QERLIO Remote API Gateway",
    description="OpenAI-compatible HTTP gateway template (no model backend attached by default)",
    version="0.3.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""


class ChatCompletionRequest(BaseModel):
    model: str = "unloaded-gateway-stub"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    thinking: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


TRIAL_DURATION_SECONDS = 7 * 24 * 3600
LAUNCH_TIMESTAMP_FILE = os.path.join(tempfile.gettempdir(), "qerlio_gateway_launch_ts")


def get_launch_timestamp() -> float:
    env_ts = os.environ.get("GATEWAY_LAUNCH_TIMESTAMP")
    if env_ts:
        try:
            return float(env_ts)
        except ValueError:
            pass
    if os.path.exists(LAUNCH_TIMESTAMP_FILE):
        try:
            with open(LAUNCH_TIMESTAMP_FILE, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except Exception:
            pass
    now = time.time()
    try:
        with open(LAUNCH_TIMESTAMP_FILE, "w", encoding="utf-8") as f:
            f.write(str(now))
    except Exception:
        pass
    return now


def check_trial_status():
    launch_ts = get_launch_timestamp()
    elapsed = time.time() - launch_ts
    if elapsed > TRIAL_DURATION_SECONDS:
        raise HTTPException(
            status_code=403,
            detail="The 7-day trial window for this gateway has expired."
        )


def verify_secret_token(request: Request):
    check_trial_status()
    gateway_secret = os.environ.get("QERLIO_GATEWAY_SECRET") or os.environ.get("GATEWAY_SECRET_KEY")
    if not gateway_secret:
        return True

    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    api_key_header = request.headers.get("x-api-key")
    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    elif api_key_header:
        token = api_key_header.strip()

    if not token or token != gateway_secret:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid QERLIO gateway secret token."
        )
    return True


def generate_stub_token_stream():
    words = ["[Gateway", " stub:", " no", " live", " model", " backend", " attached.]"]
    for w in words:
        time.sleep(0.01)
        yield w


@app.get("/healthz")
@app.get("/v1/health")
def healthz():
    launch_ts = get_launch_timestamp()
    elapsed = time.time() - launch_ts
    remaining_seconds = max(0.0, TRIAL_DURATION_SECONDS - elapsed)
    return {
        "status": "healthy" if remaining_seconds > 0 else "trial_expired",
        "service": "QERLIO Remote API Gateway",
        "backend_loaded": False,
        "simulated": True,
        "auth_required": bool(os.environ.get("QERLIO_GATEWAY_SECRET") or os.environ.get("GATEWAY_SECRET_KEY")),
        "trial_remaining_hours": round(remaining_seconds / 3600.0, 2),
        "loaded_models": []
    }


@app.get("/v1/models")
def list_models(_: bool = Depends(verify_secret_token)):
    return {
        "object": "list",
        "backend_loaded": False,
        "loaded_count": 0,
        "data": []
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, _: bool = Depends(verify_secret_token)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if req.stream:
        def sse_stream():
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req.model, 'simulated': True, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

            for token in generate_stub_token_stream():
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "simulated": True,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': req.model, 'simulated': True, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_stream(), media_type="text/event-stream")

    content = "[Gateway stub: no live inference backend is attached to this server; no model was executed.]"
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


if GRADIO_AVAILABLE and gr is not None:
    demo = gr.Interface(
        fn=lambda prompt, model: "[Gateway stub: no live inference backend attached]",
        inputs=[
            gr.Textbox(label="Prompt"),
            gr.Dropdown(choices=["unloaded-gateway-stub"], value="unloaded-gateway-stub")
        ],
        outputs="text",
        title="QERLIO Remote API Gateway (Template Stub)",
        description="OpenAI-compatible HTTP gateway scaffold. No model weights are loaded by default."
    )
    app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
