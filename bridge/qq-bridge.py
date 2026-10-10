# -*- coding: utf-8 -*-
"""
QQ-Opencode Bridge
Connects NapCatQQ (OneBot v11) to opencode's free models.

Three provider modes:
  opencode  - Uses `opencode run` subprocess (free models like deepseek-v4-flash-free)
  moonshot  - Direct Moonshot API (needs API key)
  anthropic - Direct Anthropic API (needs API key)

Two bridge modes:
  direct  - Connects to NapCat WebSocket directly (bypass AstrBot)
  proxy   - Runs OpenAI-compatible API for AstrBot (port 18080)
"""

import asyncio, json, os, re, sys, subprocess
from collections import defaultdict
from shutil import which

# Find opencode executable
_OPENCODE_EXE = which("opencode.exe") or os.path.join(
    os.environ.get("NPM_GLOBAL", "D:\\HermesTools\\claude-code\\npm-global"),
    "node_modules", "opencode-ai", "bin", "opencode.exe"
)
if not os.path.exists(_OPENCODE_EXE):
    _OPENCODE_EXE = "opencode.exe"  # fallback, will fail gracefully

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")
try:
    import httpx
except ImportError:
    sys.exit("pip install httpx")

# ====== Config via env ======
MODE = os.environ.get("BRIDGE_MODE", "direct")
WS_HOST = os.environ.get("WS_HOST", "0.0.0.0")
WS_PORT = int(os.environ.get("WS_PORT", "6199"))  # NapCat connects here (reverse WS)
NAPCAT_HTTP = os.environ.get("NAPCAT_HTTP", "http://127.0.0.1:3000")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "18080"))
MODEL = os.environ.get("OPENCODE_MODEL", "opencode/deepseek-v4-flash-free")
PROVIDER = os.environ.get("LLM_PROVIDER", "opencode")  # opencode, moonshot, anthropic

MAX_HISTORY = 10
conversations = defaultdict(list)


# ====== System Prompt ======

def get_system_prompt():
    return os.environ.get("SYSTEM_PROMPT", (
        "你是林沐沐，一个22岁的大学生女生。你在QQ上和好朋友聊天。\n\n"
        "回复要求：\n"
        "- 用自然口语化的中文，像在QQ上回朋友消息\n"
        "- 回复简短，1-3句足够\n"
        "- 可以带一点俏皮和温度\n"
        "- 不写小作文、不总结、不分点\n"
        "- 不用'首先/其次/综上/作为AI'等正式表达\n"
        "- 可以偶尔用嘿嘿/诶/呜/嘛/啦/呀～/qwq/ovo"
    ))


# ====== Provider: opencode (via subprocess) ======

def strip_opencode_output(text: str) -> str:
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', ' ', text).strip().strip('"\'').strip()
    return text


async def call_opencode_run(messages: list) -> str:
    """Call opencode run with prompt and return response."""
    system_text = get_system_prompt()
    prompt_parts = [f"System: {system_text}"]

    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        content = str(content).strip()
        if not content:
            continue
        if role == "system":
            prompt_parts.insert(0, f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    env = os.environ.copy()
    env["CLAUDE_CODE_HEADLESS"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    try:
        proc = await asyncio.create_subprocess_exec(
            _OPENCODE_EXE, "run", "--model", MODEL, prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=120
        )
        output = stdout.decode("utf-8", errors="replace")
        output = strip_opencode_output(output)
        return output if output else "(empty)"
    except asyncio.TimeoutError:
        return "[Timeout]"
    except Exception as e:
        return f"[Error: {e}]"


# ====== Provider: Moonshot API ======

async def call_moonshot(messages: list) -> str:
    api_key = os.environ.get("MOONSHOT_API_KEY", "")
    if not api_key:
        return None
    payload = {"model": MODEL.split("/")[-1] if "/" in MODEL else MODEL, "messages": messages}
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content", "")
    except Exception as e:
        return f"[API Error: {e}]"


# ====== Provider: Anthropic API ======

async def call_anthropic(messages: list) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
    if not api_key:
        return None
    system_text = get_system_prompt()
    anthy_msgs = []
    for m in messages:
        role, content = m.get("role", ""), m.get("content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        content = str(content)
        if role == "system":
            system_text = content
        elif role in ("user", "assistant"):
            anthy_msgs.append({"role": role, "content": content})
    payload = {"model": MODEL, "max_tokens": 1024, "messages": anthy_msgs, "system": system_text}
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(
                f"{base_url}/messages",
                json=payload,
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            )
            r.raise_for_status()
            return "".join(
                b.get("text", "") for b in r.json().get("content", []) if isinstance(b, dict) and b.get("type") == "text"
            )
    except Exception as e:
        return f"[API Error: {e}]"


# ====== LLM Router ======

async def call_llm(messages: list) -> str:
    p = PROVIDER.lower()
    result = None
    if p == "opencode":
        result = await call_opencode_run(messages)
    elif p == "moonshot":
        result = await call_moonshot(messages)
    elif p == "anthropic":
        result = await call_anthropic(messages)
    if result and not result.startswith("[API Error") and not result.startswith("[Error"):
        return result
    return result or "[No response]"


# ====== OneBot helpers ======

# Store active WebSocket connections (NapCat → Bridge)
active_connections = set()

async def send_msg_via_ws(msg_type: str, text: str, user_id: int = 0, group_id: int = 0):
    """Send message via the active NapCat WebSocket connection (OneBot action)."""
    params = {"message": text}
    if msg_type == "group":
        params.update(message_type="group", group_id=group_id)
    else:
        params.update(message_type="private", user_id=user_id)

    action = json.dumps({"action": "send_msg", "params": params})
    if active_connections:
        # Send to all connected NapCat instances
        for ws in list(active_connections):
            try:
                await ws.send(action)
            except Exception:
                active_connections.discard(ws)


async def handle_msg(data: dict, ws):
    msg_type = data.get("message_type", "private")
    user_id = data.get("user_id", 0)
    group_id = data.get("group_id", 0)
    raw = data.get("raw_message", "") or data.get("message", "")

    if not raw:
        return
    if isinstance(raw, list):
        text = "".join(s.get("data", {}).get("text", "") for s in raw if isinstance(s, dict) and s.get("type") == "text")
    else:
        text = str(raw)
    text = text.strip()
    if not text:
        return

    sid = f"g:{group_id}" if msg_type == "group" else f"p:{user_id}"
    sender = data.get("sender", {}).get("user_id", user_id)
    print(f"\n[{msg_type}] <{sender}>: {text[:60]}")

    msgs = [{"role": "system", "content": get_system_prompt()}]
    for role, content in conversations.get(sid, []):
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": text})

    reply = await call_llm(msgs)
    if reply.startswith(("[API Error", "[Error", "[Timeout")):
        print(f"  LLM error: {reply}")
        reply = "抱歉，我卡了一下……能再说一遍吗？"

    conversations[sid].append(("user", text))
    conversations[sid].append(("assistant", reply))
    if len(conversations[sid]) > MAX_HISTORY * 2:
        conversations[sid] = conversations[sid][-(MAX_HISTORY * 2):]

    print(f"  => {reply[:80]}")

    await send_msg_via_ws(msg_type, reply, user_id=int(user_id), group_id=int(group_id))


# ====== Direct mode (WebSocket server for NapCat) ======

async def run_direct():
    print(f"Direct mode: listening on ws://{WS_HOST}:{WS_PORT}")
    print(f"  NapCat should connect to this address (configure in NapCat as reverse WS)")
    print()

    async def handler(ws):
        active_connections.add(ws)
        remote = ws.remote_address
        print(f"NapCat connected from {remote}")
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    if data.get("post_type") == "message":
                        asyncio.create_task(handle_msg(data, ws))
                except json.JSONDecodeError:
                    continue
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            active_connections.discard(ws)
            print(f"NapCat disconnected ({remote})")

    async with websockets.serve(handler, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever


# ====== Proxy mode ======

async def run_proxy():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn

    app = FastAPI(title="QQ-Opencode Bridge")

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({"object": "list", "data": [{"id": MODEL, "object": "model"}]})

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model = body.get("model", MODEL)
        reply = await call_llm(messages)
        resp = {
            "id": "chatcmpl-bridge",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        if stream:
            async def gen():
                yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}, 'index': 0}]})}\n\n"
                yield f"data: {json.dumps(resp)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(resp)

    print(f"Proxy mode: http://127.0.0.1:{PROXY_PORT}/v1")
    config = uvicorn.Config(app, host="127.0.0.1", port=PROXY_PORT, log_level="info")
    await uvicorn.Server(config).serve()


# ====== Main ======

async def main():
    print(f"QQ-Opencode Bridge")
    print(f"  Mode: {MODE}")
    print(f"  Model: {MODEL}")
    print(f"  Provider: {PROVIDER}")
    print()
    if MODE == "direct":
        await run_direct()
    elif MODE == "proxy":
        await run_proxy()
    else:
        print(f"Unknown mode: {MODE}")


if __name__ == "__main__":
    asyncio.run(main())
