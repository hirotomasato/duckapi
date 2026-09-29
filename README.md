<div align="center">

# DuckAI2API

**An OpenAI- and Anthropic-compatible API for Duck.ai — driven by a real Chrome browser.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Playwright](https://img.shields.io/badge/Playwright-1.49-2ead63.svg)](https://playwright.dev/)

</div>

---

## What this is

Duck.ai has no public API. This project opens the real site in a real Chrome
browser via [Playwright](https://playwright.dev/), types your prompt into the
composer, and streams the answer back — then repackages it as a standard
`/v1/chat/completions` endpoint. Any OpenAI-compatible client (or Claude Code,
via the Anthropic-compatible route) can talk to it with nothing but a base URL.

It also generates images, and ships with a web dashboard so you can watch
latency, sessions, and logs live.

---

## Before you use this

Read this section before you install anything.

**This is an unofficial project. It is not affiliated with, endorsed by, or
sponsored by Duck.ai or DuckDuckGo.**

Duck.ai publishes no public API. This project works by driving the ordinary
web interface in a real browser — it types into the composer and reads the
page. That is automation of a consumer site, which is exactly the kind of thing
terms of service tend to prohibit. Whether you may do it is between you and
Duck.ai, not something this README can grant you. **If using it would violate
Duck.ai's Terms of Service, or the law where you live, don't use it.**

Concretely, you accept that:

- **You can be banned, and the ban is permanent.** Duck.ai bans by IP. A ban
  means `418 ERR_BN_LIMIT` on every future request, with no `Retry-After` and
  no cooldown. Nobody can lift it for you. If Duck.ai bans the IP you run this
  from, that is your problem to solve, and it may be unsolvable.
- **Bans can land on people who did nothing.** On shared, mobile, campus, or
  corporate IPs, someone else's volume can get your address blocked.
- **The browser is the real product.** If Duck.ai changes its page, this breaks.
  It breaks on their schedule, not yours, and there is no support SLA.
- **"Free" is not a licence.** That Duck.ai costs you nothing to use does not
  make automated use permitted. Don't read an absence of restriction as consent.
- **The maintainers accept no liability** for bans, blocked accounts, lost
  access, or any claim arising from your use of this. See [License](#license).

If any of that is a problem for you, the honest answer is: don't run this at
volume, and don't run it on an IP you can't afford to lose. If you need a
supported API, use a provider that offers one.

---

## Features

- **OpenAI-compatible** — `POST /v1/chat/completions`, `GET /v1/models`, streaming
  and non-streaming
- **Anthropic-compatible** — `POST /v1/messages` and `POST /v1/responses`, so
  Claude Code and other Anthropic-shaped clients work unchanged
- **Image generation** — `POST /v1/images/generations` via Luna's native
  GenerateImage tool
- **Web dashboard** — live status, a chat/image playground, log tail, and model
  catalog at `http://localhost:8080/`
- **Session reuse** — keeps the Duck.ai conversation alive between turns and
  types only the *delta*, which is roughly **2.5× faster** than resending
  (measured ~3s/turn vs ~10s/turn)
- **Challenge-aware warmup** — polls until Duck.ai is actually ready instead of
  sleeping a fixed 7 seconds; first token drops from ~10.2s to ~5.5s
- **Proxy pool** — rotates across a list of clean exits when one gets banned
- **Credential masking** — proxy passwords are never exposed via the API
- **Graceful shutdown** — the Stop button drains in-flight streams before
  closing browsers
- **Zero build step** — pure HTML/CSS/JS, no npm, no CDN

---

## Requirements

| | |
|---|---|
| **Python** | 3.10 or newer (developed on 3.13) |
| **Chrome** | Installed. Playwright drives your existing Chrome, it does not bundle a browser |
| **OS** | Windows, macOS, or Linux |

---

## Installation

### 1. Clone and set up

```bash
git clone https://github.com/hirotomasato/duckapi.git
cd duckapi
python run.py
```

That is the whole install. `run.py` creates a virtualenv, installs
`requirements.txt`, seeds `.env` from `example.env`, and starts the server —
then opens the dashboard when it's healthy.

The venv is created on **first run only**; later runs reuse it untouched.

### 2. Configure

`.env` was copied from `example.env`. The defaults work out of the box. The one
setting worth changing:

```ini
DUCKAI_API_KEY=pick-a-long-random-string
```

> **Set this before exposing the server to a network.** The default is empty,
> which means open access to chat, images, and the dashboard. The server binds
> `0.0.0.0`, so anything on your LAN can reach it. See
> [Security](#security).

### 3. Use it

Point any OpenAI-compatible client at `http://localhost:8080/v1`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DUCKAI_API_KEY" \
  -d '{
    "model": "gpt-5.6-luna",
    "messages": [{"role": "user", "content": "Say PONG"}],
    "stream": true
  }'
```

Or open the dashboard at <http://localhost:8080/> and use the playground.

---

## Running

```bash
python run.py                  # start, open the dashboard when healthy
python run.py --no-browser     # start headless
python run.py --port 9000      # different port
python run.py --reload         # uvicorn autoreload, for development
python run.py --check          # verify setup and port, don't boot
```

Output is teed to `server.log` **and** the console. The log is truncated on
each start, so one run's log is one run's story.

Manual alternative, if you prefer:

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## Configuration

All variables live in `.env`. Copy from `example.env` — it documents each one.

| Variable | Default | Meaning |
|---|---|---|
| `DUCKAI_API_KEY` | *(empty)* | Bearer token for all `/v1/*` and `/api/*`. **Empty = open access** |
| `DUCKAI_BASE` | `https://duck.ai` | Upstream host |
| `DUCKAI_PROXIES` | *(empty)* | Comma-separated proxy pool, rotated on ban |
| `DUCKAI_PROXY` | *(empty)* | Single proxy, if you don't need a pool |
| `DUCKAI_WARM_MIN` | `2.0` | Hard floor, in seconds, for page warmup |
| `DUCKAI_WARM_MAX` | `7.0` | Deadline for the readiness poll |
| `DUCKAI_PREWARM` | `1` | Warm a browser at startup instead of on first request |
| `DUCKAI_CHROME_PATH` | *(auto)* | Override the Chrome binary |
| `DUCKAI_MODEL` | `gpt-5.6-luna` | Default model when the client omits one |
| `DUCKAI_NEW_CHAT` | `0` | `0` reuses the session (fast); `1` forces a fresh chat per request |
| `DUCKAI_TOOL_ROUTING` | `0` | Experimental intent→tool synthesis. **Off** — see below |
| `PORT` | `8080` | Server port |

### Proxies and bans

Duck.ai issues a **persistent per-IP ban** — you get `418 ERR_BN_LIMIT` on the
very next request, with no `Retry-After` and no cooldown. The only real fix is
a pool of clean exits. Residential and SOCKS5 proxies work best; datacenter IPs
get banned fast.

This section is about diagnosing an IP that is *already* blocked, not about
avoiding bans. Read [Before you use this](#before-you-use-this) first — a proxy
pool changes whose problem a ban is, not whether you have one.

```ini
DUCKAI_PROXIES=http://user:pass@host1:8080,socks5://host2:1080
```

Credentials in this string are masked everywhere they're echoed back
(`***@host:port`), so they never leak through the API or the dashboard.

> **Turning `DUCKAI_TOOL_ROUTING` on is a gamble.** It returns synthesized
> `tool_calls` without calling Duck.ai at all. Agent clients (Claude Code,
> WorkBuddy) always send a tools array plus a large context, and a mis-hit
> yields `content: null` — which the IDE reports as "no response from model".
> It stays off by default for that reason.

---

## API

### OpenAI-compatible

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/chat/completions` | Streaming and non-streaming |
| `GET` | `/v1/models` | Live catalog, falling back to a static list |
| `POST` | `/v1/images/generations` | See [Images](#image-generation) |
| `GET` | `/v1/images/content/{id}` | Fetch a generated image by id |

### Anthropic-compatible

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/messages` | Anthropic Messages shape |
| `POST` | `/v1/responses` | OpenAI Responses shape |

### Dashboard

| Method | Path | Notes |
|---|---|---|
| `GET` | `/` | The dashboard |
| `GET` | `/api/status` | Uptime, sessions, ban flags, page age |
| `GET` | `/api/logs` | Ring-buffer tail, last 500 records |
| `POST` | `/api/chat` | Streaming playground chat |
| `POST` | `/api/images` | Playground image generation |
| `GET` | `/api/models` | Catalog for the UI |
| `POST` | `/api/shutdown` | Stop the server from the UI |
| `GET` | `/health` | Liveness probe |

Interactive API docs are at `/docs` while the server is running.

### Client configuration

<details>
<summary><b>OpenAI SDK (Python)</b></summary>

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="your-key")
stream = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=[{"role": "user", "content": "Say PONG"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```
</details>

<details>
<summary><b>Claude Code</b></summary>

```bash
ANTHROPIC_BASE_URL=http://localhost:8080
ANTHROPIC_AUTH_TOKEN=your-key
```
</details>

<details>
<summary><b>OpenAI-compatible GUI (Chatbox, LobeChat, …)</b></summary>

Set the provider's API host to `http://localhost:8080/v1` and paste your key.
Because sessions are reused by default, the conversation is kept server-side
per model, so "new chat" in the GUI maps to a fresh Duck.ai page.
</details>

---

## Models

Models resolve through a live catalog with a static snapshot as fallback. Any
alias below works as input; the real id is what `/v1/models` returns.

| Model | Label |
|---|---|
| `gpt-5.6-luna` | GPT-5.6 Luna *(default)* |
| `gpt-5.6-terra` | GPT-5.6 Terra |
| `gpt-5.6-sol` | GPT-5.6 Sol |
| `gpt-5.4-mini` | GPT-5.4 mini |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 |
| `claude-haiku-4-5` | Claude Haiku 4.5 |
| `claude-opus-4-8` | Claude Opus 4.8 |
| `mistral-small-2603` | Mistral Small 4 |
| `tinfoil/gpt-oss-120b` | gpt-oss 120B |
| `tinfoil/gemma4-31b` | Gemma 4 31B |

**Aliases.** Common OpenAI and Anthropic names map onto whatever is closest, so
existing configs keep working: `gpt-4o` → `gpt-5.4-mini`, `gpt-4o-mini` →
`gpt-5.4-mini`, `o3-mini` → `gpt-5.4-mini`, `claude-3-5-sonnet` →
`claude-sonnet-4-6`, `claude-3-opus` → `claude-opus-4-8`, `claude-3-haiku` →
`claude-haiku-4-5`, `gpt-oss-120b` → `tinfoil/gpt-oss-120b`, `gemma4-31b` →
`tinfoil/gemma4-31b`.

> `gpt-5.4` left the catalog on 2026-09-18; the alias now resolves to
> `gpt-5.4-mini`, its surviving sibling.

`GET /v1/models` reflects whatever Duck.ai currently offers — the table above is
the snapshot, not a guarantee.

---

## Image generation

Duck.ai has no standalone image API. This drives an ordinary chat turn that
makes the model invoke its **native GenerateImage tool** (backend "GPT Image 2"),
then returns the resulting base64 JPEG. `gpt-image-1`, `gpt-image-2`, and
`dall-e-3` all alias onto `gpt-5.6-luna`.

```bash
curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DUCKAI_API_KEY" \
  -d '{
    "model": "gpt-image-1",
    "prompt": "a red apple on a wooden table, soft natural light",
    "n": 1,
    "size": "1024x1024",
    "response_format": "url"
  }'
```

Returns `url` (a local id you can `GET`) or `b64_json` depending on
`response_format`. **`n` is capped at 4**, and each image takes **15–20s** —
they are generated sequentially.

Images are cached in memory (last 50) purely so `response_format: "url"` can
hand back a stable local URL. Restarting the server drops them.

In the dashboard, switch the composer to **Image** mode. The model picker is
replaced by a size picker, and results render inline in the conversation.

---

## Performance

Measured live against Duck.ai on 2026-09-29:

| | Before | After |
|---|---|---|
| First token (cold) | 10.2s | **5.5s** |
| Per turn, session reuse | 10.3s | **~3s** |

Two changes get you there. **Warmup** polls until the page is genuinely ready
instead of sleeping a fixed 7s, so a fast page skips straight ahead. **Session
reuse** keeps the Duck.ai conversation alive and types only the delta of your
prompt rather than the whole flattened transcript — which is also why the
12,000-character composer cap is rarely hit in practice.

Set `DUCKAI_PREWARM=1` to move browser startup into server boot, so the very
first caller skips the ~10s setup too.

---

## Troubleshooting

**`418 ERR_BN_LIMIT` on the first request** — your IP is banned, and it is
persistent. Configure `DUCKAI_PROXIES`. There is no cooldown to wait out.

**Chrome fails to launch** — Playwright drives your installed Chrome. Set
`DUCKAI_CHROME_PATH` if it lives somewhere non-standard, or verify it runs as
your user.

**Port 8080 already in use** — `run.py` refuses to start and tells you. Use
`python run.py --port 8081`.

**Empty or garbled answers** — check the **Logs** tab in the dashboard. A ban
looks like an empty response, not an error.

**The dashboard can't stop the server after a crash** — expected, and not a
bug. A page served by a process disappears when that process dies, so there is
nothing left to click. Restart with `python run.py`. This is a direct
consequence of having no external launcher.

**First request is slow** — that's prewarm. Set `DUCKAI_PREWARM=1`, or just send
a throwaway message first.

---

## Security

- Set `DUCKAI_API_KEY` before putting this on a network. The default is empty,
  which is open access.
- The server binds `0.0.0.0`. If you don't need LAN access, run it behind a
  reverse proxy, or bind to localhost.
- Proxy credentials are masked in every API response and in the dashboard.
- **There is no rate limiting.** Do not expose this to the public internet —
  you would be proxying Duck.ai through your IP, and you would be the one
  paying for the bans.
- The dashboard's Stop button and in-flight drain are intended for local use.

---

## Project layout

```
duckapi/
├── run.py            # the only entry point: venv, deps, .env, uvicorn
├── main.py           # FastAPI app, /v1/* routes, session lifecycle
├── duckai.py         # Playwright driver, model catalog, warmup
├── dashboard.py      # dashboard + playground routes
├── dashboard.html    # the UI — no build step
├── tools.py          # tool-call extraction
├── toolrouter.py     # experimental intent→tool synthesis
├── example.env       # documented config template
└── requirements.txt
```

---

## How it works

1. A request hits `/v1/chat/completions` and is flattened into a prompt.
2. A per-model browser session is created on demand — a real Chrome, launched
   with a throwaway profile so your own browsing is untouched.
3. The page is warmed until Duck.ai has minted its challenge token.
4. The prompt is typed into the composer and the SSE stream is parsed back into
   tokens.
5. If the previous turn is a prefix of this one, only the **delta** is typed —
   the rest stays as native Duck.ai page history. That is the 2.5× speedup.

---

## License

MIT — see [LICENSE](LICENSE).

The MIT license covers the source code, and nothing else. It is not a grant of
rights from Duck.ai, and it cannot be — Duck.ai is not a party to it. Read
[Before you use this](#before-you-use-this): the code is MIT-licensed, your use
of Duck.ai through it is not something this project can license to you. You are
responsible for complying with Duck.ai's terms of service and with the law in
your jurisdiction. The maintainers accept no liability for bans, blocked
accounts, or any use of this project.
