"""Dashboard router: status, logs, playground, and a self-shutdown button.

Scope note - the UI can stop the server but not start it. A page served by the
process it controls disappears when that process dies, so there is nothing left
to click. Starting is `python run.py`; this page is for what you can do while
the server is alive.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import duckai
from duckai import DuckAIRateLimit, DuckAIError

HERE = Path(__file__).resolve().parent


def _require_key(authorization: str = Header(default="")) -> None:
    """Same auth as the API surface.

    Header() is load-bearing: a bare `str` parameter is a *query* parameter to
    FastAPI, so the header was never read and every dashboard route 401'd on a
    correct key. That only showed up once DUCKAI_API_KEY was set, which it
    isn't in the default .env - so it went unnoticed.

    Imported lazily rather than at module scope: main imports this module, so
    a top-level `from main import require_key` would be circular. Open access
    when DUCKAI_API_KEY is unset, matching every other endpoint.
    """
    from fastapi import HTTPException
    from main import API_KEY
    if not API_KEY:
        return
    if not authorization.startswith("Bearer ") or authorization.split(" ", 1)[1] != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


router = APIRouter(dependencies=[Depends(_require_key)])


def _sessions() -> dict:
    """main._sessions, read lazily: importing main here would be circular."""
    import main
    return main._sessions


def _mask_proxy(proxy: Optional[str]) -> str:
    """host:port only. DuckAISession.proxies may carry http://user:pass@host and
    this payload goes to a browser - credentials must never leave the process."""
    if not proxy:
        return "direct"
    m = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", proxy.strip())
    if "@" in m:
        m = "***@" + m.rsplit("@", 1)[1]
    return m


def _session_state() -> List[dict]:
    """Flatten main._sessions -> DuckAISession -> _BrowserSession into JSON."""
    # _page_opened_at is asyncio loop time (monotonic), so age must be measured
    # against time.monotonic() - time.time() would yield a nonsense value.
    now = time.monotonic()
    out: List[dict] = []
    for model, pool in _sessions().items():
        for idx, s in getattr(pool, "_sessions", {}).items():
            proxy = pool.proxies[idx] if idx < len(pool.proxies) else None
            age = round(now - s._page_opened_at, 1) if s._page_opened_at else None
            out.append({
                "model": model,
                "proxy_index": idx,
                "proxy": _mask_proxy(proxy),
                "ready": bool(s._ready),
                "banned": bool(s.banned),
                "page_open": s.page is not None,
                "page_age_s": age,
                "new_chat": bool(s.new_chat),
            })
    return out


@router.get("/")
async def index():
    f = HERE / "dashboard.html"
    if not f.exists():  # pragma: no cover - file ships with the project
        raise HTTPException(status_code=500, detail="dashboard.html is missing")
    return FileResponse(f)


@router.get("/api/status")
async def status():
    import main
    pools = _sessions()
    return {
        "uptime_s": round(time.time() - main._BOOT_TS, 1),
        "port": main.PORT,
        "auth_required": bool(main.API_KEY),
        "config": {
            "default_model": main.DEFAULT,
            "new_chat": main.NEW_CHAT,
            "tool_routing": main.TOOL_ROUTING,
            "proxy_count": len(main.PROXIES or []) or 1,
        },
        "warm": {
            "min_s": duckai.WARM_MIN,
            "max_s": duckai.WARM_MAX,
            "prewarm": duckai.PREWARM,
        },
        "sessions": _session_state(),
    }


@router.get("/api/logs")
async def logs(limit: int = 200):
    import main
    return {"records": list(main.LOG_BUFFER)[-max(1, min(limit, 500)):]}


class ChatTurn(BaseModel):
    model: str
    prompt: str
    stream: bool = True


@router.post("/api/chat")
async def chat(turn: ChatTurn):
    """Stream a turn to the real model and report first-token vs total time.

    The two timings are the point: yesterday's 7s warmup fix was never
    benchmarked, and this makes the numbers visible per turn.
    """
    import main
    from duckai import DuckAISession, resolve_model

    if not turn.prompt.strip():
        raise HTTPException(status_code=400, detail="empty prompt")

    model = resolve_model(turn.model)
    session = await main.get_session(model)
    t0 = time.perf_counter()

    async def gen():
        first = None
        chunks = 0
        yield f"data: {json.dumps({'type': 'meta', 'model': model})}\n\n"
        try:
            async for tok in session.send_stream(turn.prompt):
                now = time.perf_counter()
                if first is None:
                    first = now - t0
                    yield f"data: {json.dumps({'type': 'first_token', 'seconds': round(first, 2)})}\n\n"
                chunks += 1
                yield f"data: {json.dumps({'type': 'token', 'text': tok})}\n\n"
            total = time.perf_counter() - t0
            yield f"data: {json.dumps({'type': 'done', 'first_token_s': round(first or 0, 2), 'total_s': round(total, 2), 'tokens': chunks})}\n\n"
            yield "data: [DONE]\n\n"
        except DuckAIRateLimit as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except DuckAIError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:  # noqa: BLE001 - surface it in the UI, never 500 the stream
            yield f"data: {json.dumps({'type': 'error', 'message': f'stream failed: {e}'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


class ImageTurn(BaseModel):
    prompt: str
    size: Optional[str] = None


@router.post("/api/images")
async def images(turn: ImageTurn):
    """Generate one image and hand it back as base64.

    b64 rather than a URL on purpose: the browser would have to fetch that URL
    from an <img> tag, which cannot carry the Authorization header. Inlining the
    bytes sidesteps the whole question. Image gen takes ~15-20s, so this is a
    plain JSON response, not a stream - there is nothing to show until the
    GenerateImage tool returns.
    """
    import main
    from duckai import resolve_model

    if not turn.prompt.strip():
        raise HTTPException(status_code=400, detail="empty prompt")

    session = await main.get_session(resolve_model("gpt-image-1"))
    t0 = time.perf_counter()
    try:
        result = await session.send_image(turn.prompt, size=turn.size)
    except DuckAIRateLimit as e:
        raise HTTPException(status_code=429, detail=str(e))
    except DuckAIError as e:
        raise HTTPException(status_code=502, detail=str(e))

    b64 = result.get("b64")
    if not b64:
        raise HTTPException(status_code=502, detail=result.get("text") or "no image returned")
    return {
        "b64": b64,
        "revised_prompt": result.get("gen_prompt"),
        "seconds": round(time.perf_counter() - t0, 1),
    }


@router.get("/api/models")
async def models():
    import main
    catalog = await main.get_catalog()
    if catalog:
        return {"data": [
            {
                "id": m.get("id"),
                "name": m.get("name") or m.get("id"),
                "access": "free" if "free" in (m.get("accessTier") or []) else "paid",
                "reasoning_effort": m.get("supportedReasoningEffort") or [],
                "image_upload": bool(m.get("supportsImageUpload")),
            }
            for m in catalog
        ]}
    return {"data": [
        {"id": mid, "name": label, "access": "unknown", "reasoning_effort": [], "image_upload": False}
        for mid, label in duckai.MODEL_LABELS.items()
    ]}


@router.post("/api/shutdown")
async def shutdown(request: Request):
    """Stop this server. Only works while it's serving this page - which is the
    point: a dead server has no button either.

    Delegates to main._drain_then_exit(), which waits for in-flight requests,
    closes the browsers, and exits. It does not signal our own PID: run.py
    spawns uvicorn as a child process so there is no Server instance to ask for
    should_exit, and on Windows os.kill(pid, SIGINT) is TerminateProcess rather
    than a catchable signal - it would skip the shutdown hook and orphan Chrome.
    """
    import main
    request.app.state._stopping = True
    # Deferred: this response must flush before the process goes away.
    asyncio.get_running_loop().call_later(0.25, asyncio.create_task, main._drain_then_exit())
    return {"status": "stopping"}
