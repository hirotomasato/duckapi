"""Parsing of the tool-call XML envelope a chat model may emit.

Duck.ai is chat-only (no function-calling API). When a reply contains the
envelope, main.py turns it into a proper tool_use / tool_calls block so agents
can execute the call and send a tool_result back; otherwise the text flows
through unchanged. (Teaching the model the protocol is the caller's job;
intent-based synthesis lives in toolrouter.py.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

_TOOL_CALL_RE = re.compile(
    r"<tool_call\s+name=\"([^\"]+)\"\s*>(.*?)</tool_call>", re.DOTALL
)


@dataclass
class ToolCall:
    name: str
    input: dict


def parse_tool_call(text: str) -> Optional[ToolCall]:
    """Extract the first tool-call envelope from model output.

    Returns None if the model replied in plain text (no tool call).
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    name = m.group(1)
    raw = m.group(2).strip()
    # strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    try:
        inp = json.loads(raw)
    except json.JSONDecodeError:
        # best-effort: if not JSON, pass as a single "input" string
        inp = {"input": raw}
    if not isinstance(inp, dict):
        inp = {"input": raw}
    return ToolCall(name=name, input=inp)


def split_text_and_tool(text: str):
    """Split model output into (preamble_text, tool_call).

    preamble_text is any natural-language text before the tool envelope (may be "").
    """
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return text.strip(), None
    preamble = text[: m.start()].strip()
    tc = parse_tool_call(text)
    return preamble, tc
