"""
QQ-Opencode Bridge
Connects NapCatQQ (OneBot v11) ↔ opencode model.

Two modes:
1. proxy: Acts as OpenAI-compatible API server for AstrBot (port 18080)
2. direct: Connects to NapCat WebSocket directly to send/receive messages

Uses the same API keys/models as opencode.
"""

import asyncio
import json
import os
import re
import sys
from collections import defaultdict

# --- Dependencies ---
try:
    import websockets
except ImportError:
    print("Missing: pip install websockets")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("Missing: pip install httpx")
    sys.exit(1)

# --- Configuration from env ---
MODE = os.environ.get("BRIDGE_MODE", "direct")  # "proxy" or "direct"
NAPCAT_WS = os.environ.get("NAPCAT_WS", "ws://127.0.0.1:3001")
NAPCAT_HTTP = os.environ.get("NAPCAT_HTTP", "http://127.0.0.1:3000")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "18080"))
ASTROBOT_API_KEY = os.environ.get("ASTROBOT_API_KEY", "sk-opencode-bridge")

# Model to use (must be an Anthropic model or Moonshot model that opencode has access to)
MODEL = os.environ.get("OPENCODE_MODEL", "claude-sonnet-4-0")

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "你是林沐沐，一个22岁的大学生女生。你在QQ上和好朋友聊天。\n\n"
    "回复要求：\n"
    "- 用自然口语化的中文，像在QQ上回朋友消息\n"
    "- 回复简短，1-3句足够\n"
    "- 可以带一点俏皮和温度\n"
    "- 不写小作文、不总结、不分点\n"
    "- 不用'首先/其次/综上/作为AI'等正式表达\n"
    "- 可以偶尔用嘿嘿/诶/呜/嘛/啦/呀～/qwq/ovo"
))

# Conversation history per session
MAX_HISTORY = 10
conversations = defaultdict(list)


# ========== Anthropic API ==========

async def call_anthropic(messages: list, model: str = None) -> str:
    """Call Anthropic API directly (same API key opencode uses)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

    if not api_key:
        return "[ERROR: ANTHROPIC_API_KEY not set]"

    # Build Anthropic-format messages
    anthropic_messages = []
    system_text = SYSTEM_PROMPT

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            text = ""
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    text += c.get("text", "")
            content = text

        if role == "system":
            system_text = content
        elif role in ("user", "assistant"):
            anthropic_messages.append({"role": role, "content": str(content)})

    payload = {
        "model": model or MODEL,
        "max_tokens": 1024,
        "messages": anthropic_messages,
        "system": system_text,
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Handle custom base URL (e.g. proxy)
    url = f"{base_url.rstrip('/')}/messages"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content_blocks = data.get("content", [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "".join(text_parts)
    except Exception as e:
        return f"[API error: {e}]"


# ========== Moonshot/Kimi API (fallback) ==========

async def call_moonshot(messages: list, model: str = None) -> str:
    """Call Moonshot AI API directly."""
    api_key = os.environ.get("MOONSHOT_API_KEY", "")
    if not api_key:
        return None  # skip, not configured

    payload = {
        "model": model or "kimi-k2.5",
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json=payload, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Moonshot error: {e}]"


async def call_llm(messages: list) -> str:
    """Try Anthropic first, fallback to Moonshot, then error."""
    # Try Anthropic
    result = await call_anthropic(messages)
    if result and not result.startswith("[API error"):
        return result

    # Try Moonshot as fallback
    result = await call_moonshot(messages)
    if result and not result.startswith("[Moonshot error"):
        return result

    return result  # return the error message


# ========== OneBot / NapCat Direct Mode ==========

async def send_msg(msg_type: str, target_id: int, text: str, group_id: int = None):
    """Send message via NapCat HTTP API."""
    params = {"message": text}
    if msg_type == "group" and group_id:
        params["message_type"] = "group"
        params["group_id"] = group_id
    else:
        params["message_type"] = "private"
        params["user_id"] = target_id

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{NAPCAT_HTTP}/send_msg",
                json=params,
            )
            return resp.status_code == 200
    except Exception as e:
        print(f"  Send error: {e}")
        return False


async def handle_onebot_message(data: dict):
    """Process an incoming OneBot message."""
    msg_type = data.get("message_type", "private")
    user_id = data.get("user_id", 0)
    group_id = data.get("group_id", 0)
    raw = data.get("raw_message", "") or data.get("message", "")

    if not raw:
        return

    if isinstance(raw, list):
        text = "".join(
            s.get("data", {}).get("text", "")
            for s in raw if isinstance(s, dict) and s.get("type") == "text"
        )
    else:
        text = str(raw)

    text = text.strip()
    if not text:
        return

    if msg_type == "group":
        session_id = f"g:{group_id}"
        sender_id = data.get("sender", {}).get("user_id", user_id)
        print(f"\n[Group {group_id}] <{sender_id}> {text[:60]}...")
    else:
        session_id = f"p:{user_id}"
        print(f"\n[Private] <{user_id}> {text[:60]}...")

    # Build messages with history for the API call
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, content in conversations.get(session_id, []):
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": text})

    reply = await call_llm(msgs)
    if not reply or reply.startswith("[API error"):
        reply = "抱歉，我卡了一下……能再说一遍吗？"

    # Save history
    conversations[session_id].append(("user", text))
    conversations[session_id].append(("assistant", reply))
    if len(conversations[session_id]) > MAX_HISTORY * 2:
        conversations[session_id] = conversations[session_id][-(MAX_HISTORY * 2):]

    print(f"  => {reply[:80]}...")

    if msg_type == "group":
        await send_msg("group", 0, reply, group_id=int(group_id))
    else:
        await send_msg("private", int(user_id), reply)


async def run_direct():
    """Direct mode: connect to NapCat WebSocket and handle messages."""
    print(f"Direct mode: connecting to {NAPCAT_WS}")
    retries = 0
    while retries < 30:
        try:
            async with websockets.connect(NAPCAT_WS) as ws:
                print("Connected to NapCatQQ!")
                retries = 0
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                        if data.get("post_type") == "message":
                            asyncio.create_task(handle_onebot_message(data))
                    except json.JSONDecodeError:
                        continue
        except (ConnectionRefusedError, OSError) as e:
            retries += 1
            wait = min(retries * 2, 30)
            print(f"Retry {retries}/30 in {wait}s: {e}")
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            break
    print("Bridge stopped.")


# ========== OpenAI-Compatible Proxy Mode (for AstrBot) ==========

async def handle_openai_request(body: dict):
    """Handle an OpenAI chat completion request from AstrBot."""
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    reply = await call_llm(messages)

    resp_body = {
        "id": "chatcmpl-opencode-bridge",
        "object": "chat.completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": body.get("model", MODEL),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    if stream:
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}, 'index': 0}]})}\n\n",
        ]
        lines.append(f"data: {json.dumps(resp_body)}\n\n")
        lines.append("data: [DONE]\n\n")
        return httpx.Response(200, text="".join(lines), headers={"Content-Type": "text/event-stream"})
    return httpx.Response(200, json=resp_body)


async def run_proxy():
    """Proxy mode: run OpenAI-compatible API server for AstrBot."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn

    app = FastAPI(title="Opencode Bridge")

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "object": "list",
            "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "opencode-bridge"}]
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model = body.get("model", MODEL)

        # Build messages and call LLM
        reply = await call_llm(messages)

        resp = {
            "id": "chatcmpl-bridge",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        if stream:
            async def stream_gen():
                yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': ''}, 'index': 0}]})}\n\n"
                yield f"data: {json.dumps(resp)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_gen(), media_type="text/event-stream")

        return JSONResponse(resp)

    print(f"Proxy mode: OpenAI-compatible API at http://127.0.0.1:{PROXY_PORT}/v1")
    config = uvicorn.Config(app, host="127.0.0.1", port=PROXY_PORT, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


# ========== Main ==========

async def main():
    print(f"QQ-Opencode Bridge v1.0")
    print(f"  Mode: {MODE}")
    print(f"  Model: {MODEL}")
    print()

    if MODE == "direct":
        await run_direct()
    elif MODE == "proxy":
        await run_proxy()
    else:
        print(f"Unknown mode: {MODE}. Use 'direct' or 'proxy'.")


if __name__ == "__main__":
    asyncio.run(main())
