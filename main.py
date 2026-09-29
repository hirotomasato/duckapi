"""DuckAI2API - OpenAI + Anthropic compatible relay for Duck.ai chat & images.

Duck.ai gates its backend behind a behavioral anti-bot challenge that cannot be
replayed server-side, so the relay drives a real headless Chrome (see duckai.py).
Endpoints:
  - POST /v1/chat/completions     (OpenAI)
  - POST /v1/responses            (OpenAI Responses)
  - POST /v1/messages             (Anthropic)
  - POST /v1/images/generations   (OpenAI Images; Luna's native GenerateImage)
  - GET  /v1/images/content/{id}  (relay-cached image bytes)
  - GET  /v1/models, GET /health

Env:
  DUCKAI_API_KEY       bearer token required on requests (empty = open)
  DUCKAI_BASE          duck.ai host (default https://duck.ai)
  DUCKAI_PROXY         optional proxy for the browser (http/https/socks5)
  DUCKAI_PROXIES       comma-separated proxy pool (overrides DUCKAI_PROXY)
  DUCKAI_MODEL         default model when the client omits one (gpt-5.6-luna)
  DUCKAI_NEW_CHAT      "1" to start a fresh chat per request (default: reuse session)
  DUCKAI_TOOL_ROUTING  "1" to enable experimental intent->tool synthesis (default off)
  DUCKAI_CHROME_PATH   override the Chrome binary path
  PORT                 server port (default 8080)
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from collections import deque
from typing import Any, List, Optional, Union
from uuid import uuid4

from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from dotenv import load_dotenv
from pydantic import BaseModel
from duckai import (
    DEFAULT_MODEL,
    DuckAIError,
    DuckAIRateLimit,
    DuckAISession,
    MODEL_LABELS,
    fetch_model_catalog,
    resolve_model,
)
from tools import split_text_and_tool
from toolrouter import has_tool_result, route_intent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("duckai2api")

# ---------------------------------------------------------------------------
# In-memory log tail, for the dashboard's Logs tab. Keeps the last 500 records
# so a failure is inspectable in the browser without tailing server.log. The
# ring buffer is the only copy - server.log remains the durable record.
# ---------------------------------------------------------------------------
LOG_BUFFER: deque = deque(maxlen=500)


class _TailHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            LOG_BUFFER.append({
                "ts": time.time(),
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            })
        except Exception:  # noqa: BLE001 - logging must never raise
            pass


_tail = _TailHandler()
# Attached to uvicorn's loggers too: the failures that matter (ban responses,
# transport death) surface there, not under "duckai2api".
for _name in ("duckai2api", "uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_name).addHandler(_tail)
logging.getLogger().addHandler(_tail)

API_KEY = os.getenv("DUCKAI_API_KEY", "").strip()
# Single proxy (DUCKAI_PROXY) or a comma-separated pool (DUCKAI_PROXIES).
# A pool lets the relay rotate past Duck.ai's per-IP ERR_BN_LIMIT bans.
_proxy_pool = os.getenv("DUCKAI_PROXIES", "").strip() or os.getenv("DUCKAI_PROXY", "").strip()
PROXIES = [p.strip() for p in _proxy_pool.split(",") if p.strip()] or None
DEFAULT = os.getenv("DUCKAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
NEW_CHAT = os.getenv("DUCKAI_NEW_CHAT", "0").strip() == "1"
# Intent->tool synthesis is a gamble for agent clients (WorkBuddy/Claude Code
# always send tools + a huge context; a regex mis-hit returns content:null and
# the IDE reports "no response from model"). Off by default; opt in per deploy.
TOOL_ROUTING = os.getenv("DUCKAI_TOOL_ROUTING", "0").strip() == "1"
PORT = int(os.getenv("PORT", "8080"))
_BOOT_TS = time.time()
_active_requests = 0


def _id() -> str:
    return f"chatcmpl-{uuid4().hex[:24]}"


def _created() -> int:
    return int(time.time())


# resolve_model is imported from duckai (single source of truth for model mapping).


# ---------------------------------------------------------------------------
# Anthropic message helpers
# ---------------------------------------------------------------------------
def _anthropic_block_text(content: Any) -> str:
    """Flatten an Anthropic content field (str or list of blocks) to text.

    Image/tool blocks are skipped - Duck.ai only consumes text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "image":
                    # cannot be forwarded to duck.ai; ignore
                    continue
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        parts.append(f"[tool_result {block.get('tool_use_id','')}]: {inner}")
                    elif isinstance(inner, list):
                        parts.append(f"[tool_result {block.get('tool_use_id','')}]: {_anthropic_block_text(inner)}")
                elif btype == "tool_use":
                    # replay the model's prior call so context stays consistent
                    parts.append(
                        f'[tool_call name="{block.get("name","")}"] '
                        f"{json.dumps(block.get('input',{}), ensure_ascii=False)}"
                    )
                elif btype == "input_json_delta" and "partial_json" in block:
                    parts.append(block.get("partial_json", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _role_label(role: str) -> str:
    return {"user": "Human", "assistant": "Assistant", "system": "System"}.get(role, "Human")


def flatten_conversation(system: Any, messages: List[dict]) -> str:
    """Flatten system + EVERY turn into one duck.ai prompt string.

    Duck.ai is single-turn (one textarea send), so multi-turn context must be
    re-sent in full on every request. Claude Code ships the entire messages[]
    each turn; we preserve all of it (role-labelled) so the model stays coherent.
    """
    parts: List[str] = []
    if system:
        sys_text = _anthropic_block_text(system)
        if sys_text:
            parts.append(sys_text)
    for m in messages:
        role = m.get("role", "user")
        text = _anthropic_block_text(m.get("content", ""))
        if role == "tool":
            # OpenAI tool-result message: feed the tool output as context
            tool_id = m.get("tool_call_id", "")
            label = f"[tool_result {tool_id}]"
        else:
            label = _role_label(role)
        if text.strip():
            parts.append(f"{label}: {text}")
    return "\n\n".join(parts).strip()


def _to_anthropic_tool(openai_tool: dict) -> dict:
    """Convert an OpenAI tool spec ({type:function, function:{...}}) to Anthropic shape."""
    if openai_tool.get("type") == "function" and "function" in openai_tool:
        fn = openai_tool["function"]
        return {
            "name": fn.get("name", "tool"),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        }
    return openai_tool


def build_anthropic_prompt(system: Any, messages: List[dict]) -> str:
    """Combine system + full conversation into a single duck.ai prompt string."""
    return flatten_conversation(system, messages)


# ---------------------------------------------------------------------------
# App + session
# ---------------------------------------------------------------------------
app = FastAPI(title="DuckAI2API")

import dashboard  # noqa: E402  - dashboard reads main's globals, so import after them
app.include_router(dashboard.router)

_sessions: dict = {}


async def _drain_then_exit() -> None:
    """Wait for in-flight requests, close the browsers, then exit.

    This is what the dashboard's Stop button drives. It does the work uvicorn's
    own Ctrl-C path would do, but without reaching into uvicorn: the app object
    is imported from inside Server.serve(), so by the time this module runs
    there is no Server instance to set should_exit on, and signalling our own
    PID is not a substitute (on Windows os.kill SIGINT is TerminateProcess,
    which skips the shutdown hook and strands Chrome).

    Bounded so a wedged stream cannot hold the process open forever; the exit is
    forced once the grace period lapses, which is better than never exiting.
    """
    import os
    deadline = time.monotonic() + 10
    while _active_requests > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    if _active_requests:
        logger.warning("shutdown: %d request(s) still active, exiting anyway",
                       _active_requests)
    await _close_sessions()
    os._exit(0)


class _ActiveRequestTracker:
    """Count requests that have started but not finished sending.

    Pure ASGI on purpose. An @app.middleware("http") would not do: Starlette's
    BaseHTTPMiddleware returns from call_next as soon as the response object
    exists, so its finally fires while a StreamingResponse is still going out -
    measured at 2.45s early on a 6-chunk stream. _drain_then_exit would read a
    count of zero and exit mid-answer, cutting the user off mid-sentence.
    Watching the final http.response.body instead marks the real end of a
    request, streaming or not.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        global _active_requests
        _active_requests += 1
        released = False

        async def counting_send(message) -> None:
            # `global` is needed here too, not just in __call__: a nested
            # function that assigns _active_requests makes it local to itself,
            # and the += would raise UnboundLocalError.
            global _active_requests
            nonlocal released
            # more_body is True for every chunk but the last; that last one is
            # the only reliable "this request is done" signal.
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                released = True
                _active_requests -= 1
            await send(message)

        try:
            await self.app(scope, receive, counting_send)
        finally:
            if not released:
                # The request errored or the client vanished before the last
                # body, so nothing above released the slot.
                _active_requests -= 1


app.add_middleware(_ActiveRequestTracker)


async def get_session(model: str) -> DuckAISession:
    global _sessions
    if model not in _sessions:
        _sessions[model] = DuckAISession(model=model, proxies=PROXIES, new_chat=NEW_CHAT)
    return _sessions[model]


@app.on_event("startup")
async def _prewarm_default_model() -> None:
    """Warm a browser+page during boot so the first request skips ~10s of setup.

    Runs against the default model only. A prewarm failure is logged and
    ignored: the lazy path in get_session still works, so the server stays up
    even if Chrome is missing or the network is down.
    """
    try:
        sess = await get_session(DEFAULT)
        await sess.prewarm()
    except Exception as e:  # never block boot on a warmup problem
        logger.warning("startup prewarm skipped: %s", e)


@app.on_event("shutdown")
async def _close_sessions() -> None:
    """Close every browser on the way out.

    Without this the Playwright driver and its Chrome processes are orphaned:
    _BrowserSession.close() only ever ran on an explicit close() that nothing
    called. Browsers are per-model, so this closes them all.
    """
    for model, sess in list(_sessions.items()):
        try:
            await sess.close()
            logger.info("closed session %s", model)
        except Exception as e:  # noqa: BLE001 - shutdown must always complete
            logger.warning("close failed for %s: %s", model, e)


# Live catalog from GET /duckchat/v1/models (public). Falls back to the static
# MODEL_LABELS snapshot when Duck.ai is unreachable; ids resolve_model() has never
# seen are passed through, so new models work before this file is updated.
CATALOG_TTL = 3600.0
_catalog: List[dict] = []
_catalog_ts = 0.0


async def get_catalog() -> List[dict]:
    global _catalog, _catalog_ts
    now = time.time()
    if _catalog and now - _catalog_ts < CATALOG_TTL:
        return _catalog
    try:
        models = await asyncio.to_thread(fetch_model_catalog)
        if models:
            _catalog, _catalog_ts = models, now
    except Exception as e:  # noqa: BLE001 - catalog is best-effort, never fatal
        logger.warning("model catalog fetch failed: %s", e)
    return _catalog


def require_key(authorization: str = Header(default="")) -> None:
    if not API_KEY:
        return
    if not authorization.startswith("Bearer ") or authorization.split(" ", 1)[1] != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


# Duck.ai's chat body carries native fields the free web UI toggles expose:
#   reasoningEffort  ("none"|"low"|"medium", per model's supportedReasoningEffort)
#   canUseTools / metadata.toolChoice        (WebSearch / image gen etc.)
# We patch them onto the app's own outgoing request (duckai.body rewrite), which
# auto-disables process-wide if the server ever binds its challenge to the body.
_REASONING_MAP = {"minimal": "none", "none": "none", "low": "low", "medium": "medium", "high": "medium"}


def build_rewrite(model: Optional[str] = None, reasoning_effort: Optional[str] = None, thinking: bool = False, duckai: Optional[dict] = None) -> Optional[dict]:
    rw: dict = {}
    if model:
        # Force the page's own chat body onto the requested model. The UI ships
        # Luna by default, so without this the model arg was purely decorative.
        rw["model"] = model
    if isinstance(duckai, dict):
        if duckai.get("reasoningEffort"):
            rw["reasoningEffort"] = str(duckai["reasoningEffort"])
        if duckai.get("canUseTools") is not None:
            rw["canUseTools"] = bool(duckai["canUseTools"])
        if isinstance(duckai.get("toolChoice"), dict):
            rw["toolChoice"] = duckai["toolChoice"]
    if reasoning_effort:
        mapped = _REASONING_MAP.get(str(reasoning_effort).strip().lower())
        if mapped:
            rw.setdefault("reasoningEffort", mapped)
    if thinking:
        rw.setdefault("reasoningEffort", "low")
    return rw or None


# ---------------------------------------------------------------------------
# OpenAI protocol
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any]]


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[dict]] = None
    reasoning_effort: Optional[str] = None
    duckai: Optional[dict] = None


async def _openai_stream(session: DuckAISession, prompt: str, model: str, chat_id: str, rewrite: Optional[dict] = None):
    def chunk_obj(delta: dict, finish):
        return {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": _created(),
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    try:
        # Strict OpenAI clients expect the first chunk's delta to carry role.
        yield f"data: {json.dumps(chunk_obj({'role': 'assistant', 'content': ''}, None))}\n\n"
        async for token in session.send_stream(prompt, rewrite=rewrite):
            yield f"data: {json.dumps(chunk_obj({'content': token}, None))}\n\n"
        yield f"data: {json.dumps(chunk_obj({}, 'stop'))}\n\n"
        yield "data: [DONE]\n\n"
    except (DuckAIRateLimit, DuckAIError) as e:
        err = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(err)}\n\n"
    except Exception as e:  # noqa: BLE001 - never let the SSE connection die uncleanly
        err = {"error": {"message": f"stream failed: {e}", "type": "server_error"}}
        yield f"data: {json.dumps(err)}\n\n"

@app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
async def chat_completions(request: ChatCompletionRequest):
    system = None
    turns = []
    for m in request.messages:
        if m.role == "system":
            system = m.content
        else:
            turns.append({"role": m.role, "content": m.content})
    prompt = flatten_conversation(system, turns)
    if not prompt.strip() and not request.tools:
        raise HTTPException(status_code=400, detail="empty prompt")
    model = resolve_model(request.model)

    def _tname(t: dict) -> str:
        return (t.get("function") or {}).get("name") or t.get("name") or "?"

    logger.info(
        "REQ /v1/chat/completions model=%s->%s stream=%s msgs=%d prompt_len=%d tools=[%s] last_user=%r",
        request.model, model, request.stream, len(turns), len(prompt),
        ",".join(_tname(t) for t in (request.tools or [])),
        (json.dumps(turns[-1]["content"], ensure_ascii=False)[:160] if turns else ""),
    )

    # Relay-side tool routing: synthesise a tool_calls block from the user's
    # intent (Duck.ai cannot emit one). Skip when a tool_result is already present.
    if TOOL_ROUTING and request.tools and not has_tool_result(turns):
        routed = route_intent(turns, request.tools)
        if routed is not None:
            logger.info("ROUTED tool=%s input=%s (Duck.ai not called)", routed.name, routed.input)
            return {
                "id": _id(),
                "object": "chat.completion",
                "created": _created(),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": routed.name,
                                "arguments": json.dumps(routed.input, ensure_ascii=False),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    session = await get_session(model)
    rewrite = build_rewrite(model=model, reasoning_effort=request.reasoning_effort, duckai=request.duckai)

    if request.stream:
        return StreamingResponse(
            _openai_stream(session, prompt, model, _id(), rewrite), media_type="text/event-stream"
        )

    try:
        result = await session.send(prompt, rewrite=rewrite)
    except DuckAIRateLimit as e:
        raise HTTPException(status_code=429, detail=str(e))
    except DuckAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    msg = {"role": "assistant", "content": result.strip()}
    finish = "stop"
    preamble, tc = split_text_and_tool(result)
    if tc is not None:
        msg["content"] = preamble or None
        msg["tool_calls"] = [{
            "id": f"call_{uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": tc.name, "arguments": json.dumps(tc.input, ensure_ascii=False)},
        }]
        finish = "tool_calls"
    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _created(),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# OpenAI Responses API (/v1/responses)
# ---------------------------------------------------------------------------
def _responses_input_text(inp):
    """Flatten a Responses API `input` into a full multi-turn duck.ai prompt.

    `input` may be a string, or a list of items:
      - {"role": "user"|"system"|"assistant", "content": str | [blocks]}
      - {"type": "message", "role": ..., "content": str | [blocks]}
    Every turn is preserved (role-labelled) so multi-turn stays coherent.
    """
    if inp is None:
        return ""
    if isinstance(inp, str):
        return inp.strip()
    if isinstance(inp, list):
        parts: List[str] = []
        for item in inp:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or "user"
            content = item.get("content", "")
            text = _anthropic_block_text(content) if not isinstance(content, str) else content
            if role == "tool":
                label = f"[tool_result {item.get('tool_call_id', '')}]"
            else:
                label = _role_label(role)
            if text.strip():
                parts.append(f"{label}: {text}")
        return "\n\n".join(parts).strip()
    return str(inp).strip()

@app.post("/v1/responses", dependencies=[Depends(require_key)])
async def openai_responses(request: Request):
    raw = await request.json()
    model = resolve_model(raw.get("model"))
    tools = raw.get("tools") or []
    prompt = _responses_input_text(raw.get("input"))
    # Relay-side tool routing: synthesise a function_call from the user's intent.
    if TOOL_ROUTING and tools:
        resp_items = raw.get("input") or []
        route_msgs = []
        if isinstance(resp_items, list):
            for it in resp_items:
                if isinstance(it, dict):
                    role = it.get("role") or ("user" if it.get("type") == "message" else "user")
                    content = it.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                        )
                    route_msgs.append({"role": role, "content": content})
        if not has_tool_result(route_msgs):
            routed = route_intent(route_msgs, tools)
            if routed is not None:
                return {
                    "id": f"resp_{uuid4().hex[:24]}",
                    "object": "response",
                    "created_at": _created(),
                    "model": model,
                    "status": "completed",
                    "output": [{
                        "type": "function_call",
                        "id": f"fc_{uuid4().hex[:24]}",
                        "name": routed.name,
                        "arguments": json.dumps(routed.input, ensure_ascii=False),
                    }],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                }
    if not prompt:
        raise HTTPException(status_code=400, detail="empty input")
    stream = bool(raw.get("stream", False))
    session = await get_session(model)
    reasoning = raw.get("reasoning") or {}
    rewrite = build_rewrite(
        model=model,
        reasoning_effort=reasoning.get("effort") if isinstance(reasoning, dict) else None,
        duckai=raw.get("duckai"),
    )
    resp_id = f"resp_{uuid4().hex[:24]}"

    if not stream:
        try:
            result = await session.send(prompt, rewrite=rewrite)
        except DuckAIRateLimit as e:
            raise HTTPException(status_code=429, detail=str(e))
        except DuckAIError as e:
            raise HTTPException(status_code=502, detail=str(e))
        preamble, tc = split_text_and_tool(result)
        output = []
        if preamble:
            output.append({
                "type": "message",
                "id": f"msg_{uuid4().hex[:24]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": preamble}],
            })
        if tc is not None:
            output.append({
                "type": "function_call",
                "id": f"fc_{uuid4().hex[:24]}",
                "name": tc.name,
                "arguments": json.dumps(tc.input, ensure_ascii=False),
            })
        if not output:
            output.append({
                "type": "message",
                "id": f"msg_{uuid4().hex[:24]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result.strip()}],
            })
        return {
            "id": resp_id,
            "object": "response",
            "created_at": _created(),
            "model": model,
            "status": "completed",
            "output": output,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }

    async def _resp_stream():
        yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': resp_id, 'object': 'response', 'status': 'in_progress', 'model': model, 'output': []}})}\n\n"
        try:
            async for token in session.send_stream(prompt, rewrite=rewrite):
                yield f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': token})}\n\n"
        except (DuckAIRateLimit, DuckAIError) as e:
            yield f"data: {json.dumps({'type': 'error', 'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            return
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'error': {'message': f'stream failed: {e}', 'type': 'server_error'}})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'response.completed', 'response': {'id': resp_id, 'object': 'response', 'status': 'completed', 'model': model, 'output': []}})}\n\n"

    return StreamingResponse(_resp_stream(), media_type="text/event-stream")


def _build_anthropic_message(msg_id: str, model: str, result: str, tool_use=None) -> dict:
    """Build an Anthropic message response.

    - If `tool_use` (a RoutedToolCall from the relay router) is given, emit a
      tool_use content block and stop_reason="tool_use" WITHOUT calling Duck.ai.
    - Else if the model emitted a <tool_call> envelope (never happens on Duck.ai,
      kept for completeness), parse it.
    - Else return a plain text block.
    """
    if tool_use is not None:
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{
                "type": "tool_use",
                "id": f"toolu_{uuid4().hex[:24]}",
                "name": tool_use.name,
                "input": tool_use.input,
            }],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    preamble, tc = split_text_and_tool(result)
    if tc is None:
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": result.strip()}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
    content = []
    if preamble:
        content.append({"type": "text", "text": preamble})
    content.append({
        "type": "tool_use",
        "id": f"toolu_{uuid4().hex[:24]}",
        "name": tc.name,
        "input": tc.input,
    })
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Anthropic protocol
# ---------------------------------------------------------------------------
@app.get("/v1/models", dependencies=[Depends(require_key)])
async def list_models():
    catalog = await get_catalog()
    data = []
    for m in catalog:
        tiers = m.get("accessTier") or []
        data.append(
            {
                "id": m.get("id"),
                "object": "model",
                "created": _created(),
                "owned_by": "duck.ai",
                "display_name": m.get("name") or m.get("id"),
                "access": "free" if "free" in tiers else "paid",
                "capabilities": {
                    "image_upload": bool(m.get("supportsImageUpload")),
                    "reasoning_effort": m.get("supportedReasoningEffort") or [],
                    "tools": m.get("supportedTools") or [],
                    "file_types": m.get("supportedFileTypes") or [],
                },
            }
        )
    if not data:
        data = [
            {
                "id": mid,
                "object": "model",
                "created": _created(),
                "owned_by": "duck.ai",
                "display_name": label,
                "access": "free",
                "capabilities": {},
            }
            for mid, label in MODEL_LABELS.items()
        ]
    return {"object": "list", "data": data}


@app.post("/v1/messages", dependencies=[Depends(require_key)])
async def anthropic_messages(request: Request):
    raw = await request.json()
    model = resolve_model(raw.get("model"))
    system = raw.get("system")
    tools = raw.get("tools") or []
    messages = raw.get("messages") or []
    prompt = build_anthropic_prompt(system, messages)
    # Relay-side tool routing: Duck.ai cannot emit tool calls, so the relay
    # synthesizes a tool_use block from the user's intent. Skip routing when the
    # client is already returning a tool_result (mid-loop) - then we just answer.
    if TOOL_ROUTING and tools and not has_tool_result(messages):
        routed = route_intent(messages, tools)
        if routed is not None:
            return _build_anthropic_message(
                f"msg_{uuid4().hex[:24]}", model, "", tool_use=routed
            )
    stream = bool(raw.get("stream", False))
    session = await get_session(model)
    thinking = raw.get("thinking")
    rewrite = build_rewrite(
        model=model,
        thinking=isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive"),
        duckai=raw.get("duckai"),
    )
    msg_id = f"msg_{uuid4().hex[:24]}"
    if not stream:
        try:
            result = await session.send(prompt, rewrite=rewrite)
        except DuckAIRateLimit as e:
            return _anthropic_error(429, "rate_limit_error", str(e))
        except DuckAIError as e:
            return _anthropic_error(502, "api_error", str(e))
        return _build_anthropic_message(msg_id, model, result)

    async def _anthropic_stream():
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        try:
            async for token in session.send_stream(prompt, rewrite=rewrite):
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": token},
                })
        except (DuckAIRateLimit, DuckAIError) as e:
            yield _sse("error", {"type": "error", "error": {"type": "rate_limit_error", "message": str(e)}})
            return
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": f"stream failed: {e}"}})
            return
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })
        yield _sse("message_stop", {"type": "message_stop"})

    return StreamingResponse(_anthropic_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _anthropic_error(status: int, etype: str, message: str):
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": etype, "message": message}},
    )


# ---------------------------------------------------------------------------
# Images protocol (OpenAI-compatible /v1/images/generations)
# Duck.ai has no dedicated image API; GPT-5.6 Luna's chat turn natively invokes a
# GenerateImage tool (backend "GPT Image 2"). We drive that turn and return the
# base64 JPEG the SSE success-frame carries. Generated bytes are cached in-memory
# so `response_format:"url"` can hand back a stable local URL.
# ---------------------------------------------------------------------------
_IMAGES: dict = {}          # img_id -> {"bytes": bytes, "mime": str}
_IMAGE_CAP = 50


def _store_image(b64: str) -> str:
    raw = base64.b64decode(b64)
    mime = "image/jpeg" if b64.startswith("/9j/") else "image/png" if b64.startswith("iVBOR") else "image/png"
    img_id = f"img_{uuid4().hex[:24]}"
    _IMAGES[img_id] = {"bytes": raw, "mime": mime}
    while len(_IMAGES) > _IMAGE_CAP:  # FIFO eviction
        _IMAGES.pop(next(iter(_IMAGES)))
    return img_id


class ImageRequest(BaseModel):
    model: str = DEFAULT
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = None
    response_format: Optional[str] = "url"


@app.post("/v1/images/generations", dependencies=[Depends(require_key)])
async def images_generations(request: ImageRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="empty prompt")
    model = resolve_model(request.model)
    count = max(1, min(int(request.n or 1), 4))
    session = await get_session(model)
    data = []
    for _ in range(count):
        try:
            result = await session.send_image(request.prompt, size=request.size)
        except DuckAIRateLimit as e:
            raise HTTPException(status_code=429, detail=str(e))
        except DuckAIError as e:
            raise HTTPException(status_code=502, detail=str(e))
        b64 = result["b64"]
        entry = {}
        if request.response_format == "b64_json":
            entry["b64_json"] = b64
        else:
            img_id = _store_image(b64)
            entry["url"] = f"/v1/images/content/{img_id}"
        if result.get("gen_prompt"):
            entry["revised_prompt"] = result["gen_prompt"]
        data.append(entry)
    return {"created": _created(), "data": data}


@app.get("/v1/images/content/{img_id}")
async def images_content(img_id: str, request: Request,
                         authorization: str = Header(default="")):
    """Serve a generated image by id.

    Takes the key from the Authorization header, falling back to a `key` query
    param. The fallback exists because a browser cannot attach a header to an
    <img src>: the dashboard renders results that way, and requiring the header
    would leave the UI unable to show its own images. An img_id is a uuid4
    prefix, so it is not guessable - the param is a second path to the same
    secret, not a hole in it.
    """
    if API_KEY:
        ok = authorization.startswith("Bearer ") and authorization.split(" ", 1)[1] == API_KEY
        if not ok:
            if request.query_params.get("key") != API_KEY:
                raise HTTPException(status_code=401, detail="invalid API key")
    item = _IMAGES.get(img_id)
    if not item:
        raise HTTPException(status_code=404, detail="image not found (expired from cache)")
    return Response(content=item["bytes"], media_type=item["mime"])


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
