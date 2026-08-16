# -*- coding: utf-8 -*-
"""
xpander.ai (chat.xpander.ai) -> OpenAI 兼容 API 网关

上游协议（真实浏览器抓包确认，2026-08）：
  - 鉴权: Supabase cookie (sb-svc-sb-auth-token.*) 内含 access_token(1h) + refresh_token(轮换式)
  - 刷新: POST {SUPABASE_URL}/auth/v1/token?grant_type=refresh_token  (Header: apikey)
  - 发消息: POST {CHAT_BACKEND}/{agent_id}/invoke  (SSE, text/event-stream)
      body: {"input":{"text":...,"files":[],"user":null}, "id": null|<conversation_id>,
             "llm_model_provider": "amazon_bedrock", "llm_model_name": "global.anthropic.claude-opus-5",
             "think_mode": "default"|"harder", "deep_planning": {"enabled": bool, ...}}
      SSE 事件: connected / task_created / chunk(正文增量) / tool_call_request(含 reasoning.title)
                / tool_call_result / inline_card / context_status / task_finished(result+tokens)
  - 会话历史: GET {CHAT_BACKEND}/{agent_id}/conversations/{conv_id}
  - 余额: POST {SUPABASE_URL}/functions/v1/credits-overview
  - 模型列表: GET {SUPABASE_URL}/rest/v1/model_prices?select=provider,model_id&is_active=eq.true
  - 智能体列表: POST {SUPABASE_URL}/functions/v1/list-agents
"""
import asyncio
import base64
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
REQ_LOG = LOG_DIR / "requests.jsonl"

SUPABASE_URL = os.environ.get("XP_SUPABASE_URL", "https://svc-sb.app.xpander.ai")
CHAT_BACKEND = os.environ.get("XP_CHAT_BACKEND", "https://chat-backend.xpander.ai")
# Supabase anon key（公开值，前端 JS 中即可见，用于刷新 token / 调 rest 接口）
DEFAULT_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imxnam5ianV1a2p2a3V1dXByZnFxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTI3NDMyNzQsImV4cCI6MjAyODMxOTI3NH0.siARiYg4JJnGWEVkWgsnU6lMp3r-zUJKjoIzQoN1XCg"
)
MODEL_CACHE_TTL = 3600
TOKEN_REFRESH_MARGIN = 300  # 过期前 5 分钟刷新


# ---------------------------------------------------------------------------
# .env 读写（读: 简单解析；写: 全量重写，保留注释头）
# ---------------------------------------------------------------------------
def load_env_file() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV_HEADER = """# xpander OpenAI 兼容网关配置（WebUI 的修改会自动写回本文件）
# 多值分隔: 账号用 ||| 或换行分隔；停用列表/Key 用英文逗号分隔
"""


def save_env_file(env: dict):
    lines = [_ENV_HEADER]
    for k, v in env.items():
        v = str(v).replace("\n", "\\n")
        lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


CFG = load_env_file()


def cfg(key, default=""):
    return os.environ.get(key) or CFG.get(key) or default


def cfg_set(key, value):
    CFG[key] = str(value)
    save_env_file(CFG)


# ---------------------------------------------------------------------------
# 账号池（Supabase cookie，线程安全，失败剔除，token 自动刷新并持久化）
# ---------------------------------------------------------------------------
def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def parse_cookie_token(raw: str) -> dict:
    """从用户粘贴的内容中解析 Supabase token JSON。
    支持: 1) 整个 Cookie 请求头  2) sb-svc-sb-auth-token(.N)=... 片段
          3) base64-xxx 单值  4) 直接的 JSON
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("空内容")
    # 1) 找分片 cookie
    parts = re.findall(r"sb-svc-sb-auth-token(?:\.(\d+))?=[\"']?([^;\"'\s]+)", raw)
    if parts:
        parts.sort(key=lambda p: int(p[0] or 0))
        val = "".join(p[1] for p in parts)
    else:
        val = raw
        if "=" in val and val.split("=", 1)[0].strip().startswith("sb-"):
            val = val.split("=", 1)[1]
    val = val.strip().strip('"')
    if val.startswith("{"):
        tok = json.loads(val)
    else:
        if val.startswith("base64-"):
            val = val[7:]
        tok = json.loads(_b64url_decode(val))
    if "access_token" not in tok or "refresh_token" not in tok:
        raise ValueError("缺少 access_token / refresh_token 字段")
    return tok


def encode_cookie_token(tok: dict) -> str:
    """编码回 base64- 形式（与浏览器 cookie 值一致）。"""
    return "base64-" + base64.urlsafe_b64encode(
        json.dumps(tok, separators=(",", ":")).encode()
    ).decode().rstrip("=")


class Account:
    def __init__(self, raw: str):
        tok = parse_cookie_token(raw)
        self.token = tok
        # 稳定 id：refresh_token 会轮换，用 user.id（其次 email）保证重启/刷新后不变
        user = tok.get("user") or {}
        stable = user.get("id") or user.get("email") or tok["refresh_token"]
        self.id = hashlib.sha256(str(stable).encode()).hexdigest()[:12]
        self.email = user.get("email", "")
        self.disabled = False
        self.fail_count = 0
        self.dead_until = 0.0
        self.last_error = ""
        self.last_test = None  # dict: 测试结果(余额等)

    @property
    def access_token(self):
        return self.token.get("access_token", "")

    @property
    def expires_at(self):
        return int(self.token.get("expires_at") or 0)

    def preview(self):
        at = self.access_token
        return f"{at[:14]}…{at[-6:]}" if len(at) > 24 else "****"

    def info(self):
        return {
            "id": self.id,
            "email": self.email,
            "preview": self.preview(),
            "disabled": self.disabled,
            "fail_count": self.fail_count,
            "dead": time.time() < self.dead_until,
            "dead_until": int(self.dead_until),
            "last_error": self.last_error,
            "expires_at": self.expires_at,
            "expires_in": max(0, self.expires_at - int(time.time())),
            "last_test": self.last_test,
        }


class AccountPool:
    def __init__(self):
        self.lock = threading.RLock()
        self.accounts: list[Account] = []
        self.rr = 0
        self.anon_key = cfg("XP_SUPABASE_ANON_KEY", DEFAULT_ANON_KEY)
        self._load()

    # ---- 持久化 ----
    def _load(self):
        raw_all = cfg("XP_COOKIES")
        raws = [r for r in re.split(r"\|\|\||\\n", raw_all) if r.strip()]
        disabled_ids = {x for x in cfg("XP_DISABLED").split(",") if x}
        with self.lock:
            self.accounts = []
            for raw in raws:
                try:
                    acc = Account(raw)
                    acc.disabled = acc.id in disabled_ids
                    self.accounts.append(acc)
                except Exception:
                    continue

    def persist(self):
        with self.lock:
            raw = "|||".join(encode_cookie_token(a.token) for a in self.accounts)
            cfg_set("XP_COOKIES", raw)
            cfg_set("XP_DISABLED", ",".join(a.id for a in self.accounts if a.disabled))

    # ---- 增删 ----
    def add(self, text: str) -> dict:
        """批量添加：每行一个或 ||| 分隔，自动去重。"""
        raws = []
        for chunk in re.split(r"\|\|\|", text):
            for line in chunk.splitlines():
                line = line.strip()
                if line:
                    raws.append(line)
        added, dup, bad = 0, 0, 0
        errors = []
        with self.lock:
            have = {a.id for a in self.accounts}
            for raw in raws:
                try:
                    acc = Account(raw)
                except Exception as e:
                    bad += 1
                    errors.append(str(e))
                    continue
                if acc.id in have:
                    dup += 1
                    continue
                have.add(acc.id)
                self.accounts.append(acc)
                added += 1
        if added:
            self.persist()
        return {"added": added, "duplicate": dup, "invalid": bad, "errors": errors[:5]}

    def remove(self, ids: list[str]) -> int:
        with self.lock:
            before = len(self.accounts)
            self.accounts = [a for a in self.accounts if a.id not in set(ids)]
            n = before - len(self.accounts)
        if n:
            self.persist()
        return n

    def set_disabled(self, aid: str, disabled: bool) -> bool:
        with self.lock:
            for a in self.accounts:
                if a.id == aid:
                    a.disabled = disabled
                    self.persist()
                    return True
        return False

    # ---- 轮询取用 ----
    def acquire(self) -> Account | None:
        with self.lock:
            live = [a for a in self.accounts
                    if not a.disabled and time.time() >= a.dead_until]
            if not live:
                return None
            self.rr = (self.rr + 1) % len(live)
            return live[self.rr]

    def active_count(self):
        with self.lock:
            return sum(1 for a in self.accounts
                       if not a.disabled and time.time() >= a.dead_until)

    # ---- 失败处理 ----
    def report_failure(self, acc: Account, msg: str, auth_error: bool):
        with self.lock:
            acc.fail_count += 1
            acc.last_error = msg[:300]
            # 鉴权失败先尝试刷新（刷新成功不清死）；连续失败则临时剔除
            if auth_error or acc.fail_count >= 2:
                acc.dead_until = time.time() + 600  # 剔除 10 分钟

    def report_success(self, acc: Account):
        with self.lock:
            acc.fail_count = 0
            acc.last_error = ""
            acc.dead_until = 0.0

    # ---- token 刷新（轮换式 refresh_token，必须持久化新值）----
    async def ensure_token(self, acc: Account, client: httpx.AsyncClient) -> bool:
        if time.time() < acc.expires_at - TOKEN_REFRESH_MARGIN:
            return True
        return await self.refresh(acc, client)

    async def refresh(self, acc: Account, client: httpx.AsyncClient) -> bool:
        try:
            r = await client.post(
                f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
                headers={"apikey": self.anon_key, "Content-Type": "application/json"},
                json={"refresh_token": acc.token["refresh_token"]},
                timeout=20,
            )
            if r.status_code != 200:
                acc.last_error = f"刷新失败 {r.status_code}: {r.text[:200]}"
                return False
            d = r.json()
            with self.lock:
                # refresh_token 轮换：账号 id 基于 user.id，轮换不影响 id 与停用状态
                acc.token.update({
                    "access_token": d["access_token"],
                    "refresh_token": d.get("refresh_token", acc.token["refresh_token"]),
                    "expires_at": int(time.time()) + int(d.get("expires_in", 3600)),
                    "expires_in": d.get("expires_in", 3600),
                })
            self.persist()  # 新 refresh_token 写回 .env（关键：重启后仍可用）
            return True
        except Exception as e:
            acc.last_error = f"刷新异常: {e}"
            return False


POOL = AccountPool()

# ---------------------------------------------------------------------------
# 工具开关（默认全关，WebUI 开关，持久化）
# ---------------------------------------------------------------------------
TOOLS = {
    "think_harder": {
        "name": "深度思考 (think_mode=harder)",
        "desc": "上游 think_mode 设为 harder，模型推理更深入（响应更慢）",
        "default": False,
    },
    "deep_planning": {
        "name": "深度规划 (deep_planning)",
        "desc": "启用 deep_planning，长任务先出计划再执行（对应网页端 Deep work）",
        "default": False,
    },
    "auto_delete": {
        "name": "阅后即焚（自动删除上游会话）",
        "desc": "回答完成后自动 DELETE 上游会话，官网会话列表不保留聊天记录（默认开启）",
        "default": True,
    },
}


def tool_enabled(key: str) -> bool:
    raw = cfg("XP_TOOLS", "")
    m = dict(kv.split("=", 1) for kv in raw.split(",") if "=" in kv)
    return m.get(key, "1" if TOOLS[key]["default"] else "0") == "1"


def set_tool(key: str, on: bool):
    raw = cfg("XP_TOOLS", "")
    m = dict(kv.split("=", 1) for kv in raw.split(",") if "=" in kv)
    m[key] = "1" if on else "0"
    cfg_set("XP_TOOLS", ",".join(f"{k}={v}" for k, v in m.items()))


# ---------------------------------------------------------------------------
# 网关调用 Key
# ---------------------------------------------------------------------------
def api_keys() -> list[str]:
    keys = [k.strip() for k in cfg("API_KEYS", "123456").split(",") if k.strip()]
    return keys or ["123456"]


def check_auth(req: Request) -> bool:
    """收集所有候选凭证，任一匹配即通过。
    注意：公网预览代理会用平台会话 token 改写 Authorization 头，
    因此不能只看 Authorization——X-Api-Key 与 ?key= 同等有效。"""
    cands = []
    auth = req.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        cands.append(auth[7:].strip())
    xak = req.headers.get("x-api-key", "").strip()
    if xak:
        cands.append(xak)
    qk = req.query_params.get("key", "").strip()
    if qk:
        cands.append(qk)
    keys = api_keys()
    return any(c in keys for c in cands)


# ---------------------------------------------------------------------------
# 上游客户端
# ---------------------------------------------------------------------------
def auth_headers(acc: Account, sse: bool = False) -> dict:
    h = {
        "authorization": f"Bearer {acc.access_token}",
        "apikey": POOL.anon_key,
        "content-type": "application/json",
    }
    if sse:
        h["accept"] = "text/event-stream"
    return h


class UpstreamError(Exception):
    def __init__(self, msg, status=0, auth=False):
        super().__init__(msg)
        self.status = status
        self.auth = auth  # 是否鉴权类错误（401/403）


async def fetch_agent_id(client: httpx.AsyncClient, acc: Account) -> str:
    """自动发现 agent id（取第一个 agent，一般是 Omni）；.env XP_AGENT_ID 可覆盖。"""
    fixed = cfg("XP_AGENT_ID")
    if fixed:
        return fixed
    cached = cfg("_AGENT_ID_CACHE")
    if cached:
        return cached
    r = await client.post(f"{SUPABASE_URL}/functions/v1/list-agents",
                          headers=auth_headers(acc), json={}, timeout=20)
    if r.status_code != 200:
        raise UpstreamError(f"list-agents {r.status_code}: {r.text[:200]}", r.status_code,
                            auth=r.status_code in (401, 403))
    data = r.json()
    agents = data if isinstance(data, list) else data.get("data") or data.get("agents") or []
    if not agents:
        raise UpstreamError("该账号下没有可用智能体")
    aid = agents[0].get("id")
    cfg_set("_AGENT_ID_CACHE", aid)
    return aid


def build_invoke_payload(prompt: str, conv_id: str | None,
                         model_pair: tuple[str, str] | None) -> dict:
    payload = {
        "input": {"text": prompt, "files": [], "user": None},
        "id": conv_id,  # null = 新会话
    }
    if model_pair:
        payload["llm_model_provider"], payload["llm_model_name"] = model_pair
    if tool_enabled("think_harder"):
        payload["think_mode"] = "harder"
    if tool_enabled("deep_planning"):
        payload["deep_planning"] = {"enabled": True, "enforce": False, "started": False,
                                    "question_raised": False, "tasks": []}
    return payload


async def invoke_stream(client: httpx.AsyncClient, acc: Account, agent_id: str,
                        payload: dict):
    """调用 /invoke，异步迭代解析后的 SSE 事件 dict。首个事件时间用于 TTFT。"""
    url = f"{CHAT_BACKEND}/{agent_id}/invoke"
    async with client.stream("POST", url, headers=auth_headers(acc, sse=True),
                             json=payload, timeout=httpx.Timeout(600, connect=20)) as r:
        if r.status_code != 200:
            body = (await r.aread()).decode("utf-8", "replace")[:400]
            raise UpstreamError(f"上游 {r.status_code}: {body}", r.status_code,
                                auth=r.status_code in (401, 403))
        buf = ""
        async for chunk_bytes in r.aiter_bytes():
            buf += chunk_bytes.decode("utf-8", "replace").replace("\r\n", "\n")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                block = block.strip()
                if not block.startswith("data:"):
                    continue
                try:
                    yield json.loads(block[5:].strip())
                except Exception:
                    continue
        # 流结束时冲刷残余
        tail = buf.strip()
        if tail.startswith("data:"):
            try:
                yield json.loads(tail[5:].strip())
            except Exception:
                pass


async def fetch_credits(client: httpx.AsyncClient, acc: Account) -> dict:
    """账号体检（免费，不消耗对话额度）：余额/订阅/试用信息。"""
    out = {"ok": False}
    r = await client.post(f"{SUPABASE_URL}/functions/v1/credits-overview",
                          headers=auth_headers(acc), json={}, timeout=20)
    if r.status_code != 200:
        raise UpstreamError(f"credits-overview {r.status_code}: {r.text[:200]}",
                            r.status_code, auth=r.status_code in (401, 403))
    d = r.json().get("data", {})
    bal = d.get("balance", {})
    lic = d.get("license", {})
    out["ok"] = True
    out["balance_credits"] = bal.get("balance_credits")
    out["balance_usd"] = bal.get("usd_equivalent")
    out["blocked"] = bal.get("blocked")
    out["lifetime_granted"] = (bal.get("lifetime_granted_millicredits") or 0) / 1000
    out["lifetime_burned"] = (bal.get("lifetime_burned_millicredits") or 0) / 1000
    out["license_tier"] = lic.get("tier")
    out["license_active"] = lic.get("active")
    out["free_grant_claimed_at"] = d.get("free_grant_claimed_at")
    out["token_expires_in"] = max(0, acc.expires_at - int(time.time()))
    out["email"] = acc.email
    return out


async def delete_conversation(acc: Account, agent_id: str, conv_id: str):
    """阅后即焚：删除上游会话（官网列表不再保留）。软删除，失败静默。"""
    if not conv_id:
        return
    try:
        async with httpx.AsyncClient() as client:
            await POOL.ensure_token(acc, client)
            await client.delete(f"{CHAT_BACKEND}/{agent_id}/conversations/{conv_id}",
                                headers=auth_headers(acc), timeout=15)
    except Exception:
        pass


# 模型列表缓存
MODELS_CACHE = {"ts": 0.0, "items": []}
MODELS_LOCK = threading.Lock()

MODEL_ALIASES = {  # 便捷别名 -> (provider, model_id)
    "sonnet-5": ("amazon_bedrock", "global.anthropic.claude-sonnet-5"),
    "opus-5": ("amazon_bedrock", "global.anthropic.claude-opus-5"),
    "sonnet-4.6": ("amazon_bedrock", "global.anthropic.claude-sonnet-4-6"),
    "opus-4.7": ("amazon_bedrock", "global.anthropic.claude-opus-4-7"),
    "haiku-4.5": ("amazon_bedrock", "global.anthropic.claude-haiku-4-5-20251001-v1:0"),
    "deepseek-v3.2": ("amazon_bedrock", "deepseek-v3-2"),
}


async def fetch_models(force=False) -> list[dict]:
    """从上游真实抓取模型列表（rest/v1/model_prices, is_active=true），缓存 1 小时。"""
    with MODELS_LOCK:
        if not force and MODELS_CACHE["items"] and time.time() - MODELS_CACHE["ts"] < MODEL_CACHE_TTL:
            return MODELS_CACHE["items"]
    acc = POOL.acquire()
    if not acc:
        raise UpstreamError("无可用账号，无法抓取模型列表")
    async with httpx.AsyncClient() as client:
        await POOL.ensure_token(acc, client)
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/model_prices?select=provider,model_id&is_active=eq.true",
            headers=auth_headers(acc), timeout=30)
        if r.status_code != 200:
            raise UpstreamError(f"model_prices {r.status_code}: {r.text[:200]}", r.status_code,
                                auth=r.status_code in (401, 403))
        rows = r.json()
    items, seen = [], set()
    for row in rows:
        p, m = row.get("provider"), row.get("model_id")
        if not p or not m:
            continue
        key = f"{p}/{m}"
        if key in seen:
            continue
        seen.add(key)
        items.append({"id": key, "object": "model", "created": 0,
                      "owned_by": p, "provider": p, "model_id": m})
    items.sort(key=lambda x: (x["provider"] != "amazon_bedrock", x["id"]))
    with MODELS_LOCK:
        MODELS_CACHE["items"] = items
        MODELS_CACHE["ts"] = time.time()
    return items


def resolve_model(model: str, models: list[dict]) -> tuple[str, str] | None:
    """OpenAI model 字段 -> (provider, model_id)。支持别名、精确 id、模糊匹配。"""
    if not model:
        return None
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    for it in models:
        if it["id"] == model:
            return it["provider"], it["model_id"]
    # 模糊: "claude-opus-5" / "opus-5" 等
    low = model.lower().replace("claude-", "").replace("claude", "")
    cand = [it for it in models if low in it["model_id"].lower()]
    if len(cand) >= 1:
        bed = [c for c in cand if c["provider"] == "amazon_bedrock"]
        pick = (bed or cand)[0]
        return pick["provider"], pick["model_id"]
    return None


# ---------------------------------------------------------------------------
# 提示词组装（单条 user 直接透传；多轮压缩用清晰角色标签）
# ---------------------------------------------------------------------------
def messages_to_prompt(messages: list[dict]) -> str:
    """单条 user 直接透传；多轮压缩为带清晰角色标签的单条 prompt。
    上游 invoke 只接受单条 text（协议边界），system 用强指令包装置首以提高遵从度。"""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return _content_text(messages[0].get("content"))  # 直接透传，不加前缀
    parts = []
    systems = [_content_text(m.get("content")) for m in messages if m.get("role") == "system"]
    if systems:
        parts.append("【系统指令，最高优先级，必须严格遵守】\n" + "\n".join(systems))
    label = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue
        parts.append(f"{label.get(role, 'User')}: {_content_text(m.get('content'))}")
    return "\n\n".join(parts)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI 多段内容
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content or "")


# ---------------------------------------------------------------------------
# 请求日志
# ---------------------------------------------------------------------------
RECENT_LOGS: list[dict] = []
LOGS_LOCK = threading.Lock()


def write_log(entry: dict):
    entry["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOGS_LOCK:
        RECENT_LOGS.insert(0, entry)
        del RECENT_LOGS[200:]
    try:
        with open(REQ_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(title="xpander OpenAI 兼容网关", docs_url=None, redoc_url=None)


def _unauth():
    return JSONResponse({"error": {"message": "鉴权失败：请提供 Authorization: Bearer <API key>",
                                   "type": "auth_error"}}, status_code=401)


def _no_account():
    return JSONResponse({"error": {"message": "账号池为空或全部停用/剔除，请在 WebUI 添加或启用账号",
                                   "type": "no_account"}}, status_code=503)


@app.get("/v1/models")
async def list_models(req: Request):
    if not check_auth(req):
        return _unauth()
    try:
        items = await fetch_models()
    except UpstreamError as e:
        items = MODELS_CACHE["items"] or []
        if not items:
            return JSONResponse({"error": {"message": f"模型列表抓取失败: {e}"}}, status_code=502)
    data = [{"id": it["id"], "object": "model", "created": 0, "owned_by": it["owned_by"]}
            for it in items]
    for alias in MODEL_ALIASES:
        data.append({"id": alias, "object": "model", "created": 0, "owned_by": "alias"})
    default_model = cfg("DEFAULT_MODEL", "amazon_bedrock/global.anthropic.claude-sonnet-5")
    if not any(d["id"] == default_model for d in data):
        p, _, m = default_model.partition("/")
        data.insert(0, {"id": default_model, "object": "model", "created": 0, "owned_by": p})
    return {"object": "list", "data": data}


def _oai_chunk(cid, model, delta=None, finish=None):
    c = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
         "model": model, "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"


async def _run_upstream(acc, agent_id, payload):
    """执行一次 invoke，异步生成 SSE 事件（client 生命周期覆盖整个流）。"""
    async with httpx.AsyncClient() as client:
        if not await POOL.ensure_token(acc, client):
            raise UpstreamError("token 刷新失败（refresh_token 可能已失效，请重新粘贴 Cookie）",
                                401, auth=True)
        async for ev in invoke_stream(client, acc, agent_id, payload):
            yield ev


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    if not check_auth(req):
        return _unauth()
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": {"message": "请求体不是合法 JSON"}}, status_code=400)
    messages = body.get("messages") or []
    if not messages:
        return JSONResponse({"error": {"message": "messages 不能为空"}}, status_code=400)
    stream = bool(body.get("stream"))
    model_req = body.get("model") or cfg("DEFAULT_MODEL",
                                         "amazon_bedrock/global.anthropic.claude-sonnet-5")
    prompt = messages_to_prompt(messages)
    t0 = time.time()

    acc = POOL.acquire()
    if not acc:
        write_log({"account": "-", "model": model_req, "input": prompt[:500],
                   "output": "", "reasoning": "", "ttft_ms": 0,
                   "total_ms": int((time.time() - t0) * 1000), "ok": False,
                   "error": "无可用账号"})
        return _no_account()

    try:
        models = await fetch_models()
    except Exception:
        models = []
    model_pair = resolve_model(model_req, models)

    try:
        agent_id = await _get_agent_id(acc)
    except UpstreamError as e:
        POOL.report_failure(acc, str(e), e.auth)
        return JSONResponse({"error": {"message": str(e)}}, status_code=502)

    payload = build_invoke_payload(prompt, None, model_pair)

    if stream:
        return await _stream_response(acc, agent_id, payload, model_req, prompt, t0)
    return await _full_response(acc, agent_id, payload, model_req, prompt, t0)


async def _get_agent_id(acc):
    async with httpx.AsyncClient() as client:
        await POOL.ensure_token(acc, client)
        return await fetch_agent_id(client, acc)


REASONING_TYPES = {"thinking", "reasoning", "thought", "reasoning_content"}  # 协议兼容预留


def _map_event(ev: dict):
    """上游 SSE 事件 -> (kind, text)。kind: content / reasoning / done / meta / ignore"""
    t = ev.get("type")
    if t == "chunk":
        d = ev.get("data")
        return ("content", d if isinstance(d, str) else str(d))
    if t == "tool_call_request":
        d = ev.get("data") or {}
        rs = d.get("reasoning") or {}
        title = rs.get("title") or f"调用工具 {d.get('tool_name', '')}"
        return ("reasoning", f"🔧 {title}\n")
    if t == "tool_call_result":
        d = ev.get("data") or {}
        return ("reasoning", f"✅ 工具 {d.get('tool_name', '')} 执行完成\n")
    if t in REASONING_TYPES:
        d = ev.get("data")
        return ("reasoning", d if isinstance(d, str) else json.dumps(d, ensure_ascii=False))
    if t == "task_finished":
        return ("done", ev.get("data") or {})
    if t == "task_created":
        return ("meta", ev.get("data") or {})
    return ("ignore", None)


async def _stream_response(acc, agent_id, payload, model, prompt, t0):
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    state = {"ttft": 0, "content": "", "reasoning": "", "err": None, "usage": None}

    async def gen():
        first = True
        conv_id = None
        yield _oai_chunk(cid, model, {"role": "assistant"})
        try:
            aiter = _run_upstream(acc, agent_id, payload).__aiter__()
            while True:
                try:
                    ev = await asyncio.wait_for(aiter.__anext__(), timeout=25)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # SSE 心跳：防代理/客户端空闲断连
                    continue
                except StopAsyncIteration:
                    break
                kind, val = _map_event(ev)
                if kind in ("content", "reasoning") and val:
                    if not state["ttft"]:
                        state["ttft"] = int((time.time() - t0) * 1000)
                    field = "content" if kind == "content" else "reasoning_content"
                    state["content" if kind == "content" else "reasoning"] += val
                    yield _oai_chunk(cid, model, {field: val})
                elif kind == "done":
                    tk = val.get("tokens") or {}
                    state["usage"] = {
                        "prompt_tokens": tk.get("prompt_tokens", 0),
                        "completion_tokens": tk.get("completion_tokens", 0),
                        "total_tokens": tk.get("total_tokens", 0),
                    }
                    # 兜底：上游偶发不流式吐 chunk，正文只在 result 里 —— 补发
                    result = val.get("result")
                    if isinstance(result, str) and result and not state["content"]:
                        state["content"] = result
                        if not state["ttft"]:
                            state["ttft"] = int((time.time() - t0) * 1000)
                        yield _oai_chunk(cid, model, {"content": result})
                elif kind == "ignore" and ev.get("type") in ("task_failed", "error"):
                    raise UpstreamError(f"上游任务失败: {json.dumps(ev.get('data'), ensure_ascii=False)[:300]}")
                elif kind == "meta":
                    conv_id = (val or {}).get("id") or conv_id
            POOL.report_success(acc)
            tail = _oai_chunk(cid, model, {}, "stop")
            if state["usage"]:
                c = json.loads(tail[6:])
                c["usage"] = state["usage"]
                tail = f"data: {json.dumps(c, ensure_ascii=False)}\n\n"
            yield tail
            yield "data: [DONE]\n\n"
        except UpstreamError as e:
            state["err"] = str(e)
            POOL.report_failure(acc, str(e), e.auth)
            yield _oai_chunk(cid, model, {"content": f"\n\n[网关错误] {e}"}, "stop")
            yield "data: [DONE]\n\n"
        except Exception as e:
            state["err"] = str(e)
            POOL.report_failure(acc, str(e), False)
            yield _oai_chunk(cid, model, {"content": f"\n\n[网关异常] {e}"}, "stop")
            yield "data: [DONE]\n\n"
        finally:
            write_log({"account": f"{acc.email or acc.preview()}({acc.id})",
                       "model": model, "input": prompt[:500],
                       "output": state["content"][-2000:], "reasoning": state["reasoning"][-2000:],
                       "ttft_ms": state["ttft"],
                       "total_ms": int((time.time() - t0) * 1000),
                       "ok": state["err"] is None, "error": state["err"]})
            # 阅后即焚：删除上游会话（官网列表不保留）
            if conv_id and tool_enabled("auto_delete"):
                asyncio.create_task(delete_conversation(acc, agent_id, conv_id))
            _ = first

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _full_response(acc, agent_id, payload, model, prompt, t0):
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    content, reasoning, result, usage, ttft = "", "", None, None, 0
    conv_id = None
    try:
        async for ev in _run_upstream(acc, agent_id, payload):
            kind, val = _map_event(ev)
            if kind == "content" and val:
                if not ttft:
                    ttft = int((time.time() - t0) * 1000)
                content += val
            elif kind == "reasoning" and val:
                if not ttft:
                    ttft = int((time.time() - t0) * 1000)
                reasoning += val
            elif kind == "done":
                result = val.get("result")
                tk = val.get("tokens") or {}
                usage = {"prompt_tokens": tk.get("prompt_tokens", 0),
                         "completion_tokens": tk.get("completion_tokens", 0),
                         "total_tokens": tk.get("total_tokens", 0)}
            elif kind == "meta":
                conv_id = (val or {}).get("id") or conv_id
        POOL.report_success(acc)
    except UpstreamError as e:
        POOL.report_failure(acc, str(e), e.auth)
        write_log({"account": f"{acc.email or acc.preview()}({acc.id})", "model": model,
                   "input": prompt[:500], "output": "", "reasoning": "",
                   "ttft_ms": ttft, "total_ms": int((time.time() - t0) * 1000),
                   "ok": False, "error": str(e)})
        if conv_id and tool_enabled("auto_delete"):
            asyncio.create_task(delete_conversation(acc, agent_id, conv_id))
        return JSONResponse({"error": {"message": f"上游错误: {e}", "type": "upstream_error"}},
                            status_code=502 if not e.auth else 401)
    final = result if isinstance(result, str) and result else content
    msg = {"role": "assistant", "content": final}
    if reasoning:
        msg["reasoning_content"] = reasoning
    resp = {"id": cid, "object": "chat.completion", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    write_log({"account": f"{acc.email or acc.preview()}({acc.id})", "model": model,
               "input": prompt[:500], "output": final[-2000:], "reasoning": reasoning[-2000:],
               "ttft_ms": ttft, "total_ms": int((time.time() - t0) * 1000),
               "ok": True, "error": None})
    if conv_id and tool_enabled("auto_delete"):
        asyncio.create_task(delete_conversation(acc, agent_id, conv_id))
    return resp


# ---------------------------------------------------------------------------
# 管理 API（WebUI 用，同样需要 Bearer key 鉴权）
# ---------------------------------------------------------------------------
@app.get("/api/overview")
async def api_overview(req: Request):
    with POOL.lock:
        accs = [a.info() for a in POOL.accounts]
    return {
        "status": "ok",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "default_model": cfg("DEFAULT_MODEL", "amazon_bedrock/global.anthropic.claude-sonnet-5"),
        "accounts_total": len(accs),
        "accounts_active": POOL.active_count(),
        "models_cached": len(MODELS_CACHE["items"]),
        "models_cache_age": int(time.time() - MODELS_CACHE["ts"]) if MODELS_CACHE["ts"] else None,
        "tools": {k: tool_enabled(k) for k in TOOLS},
        "agent_id_cache": cfg("_AGENT_ID_CACHE"),
    }


@app.get("/api/accounts")
async def api_accounts(req: Request):
    with POOL.lock:
        return {"accounts": [a.info() for a in POOL.accounts]}


def _clean_cookie_piece(s: str) -> str:
    """清理用户粘贴的分片：去掉 name= 前缀、引号、空白。"""
    s = (s or "").strip().strip('"').strip("'").strip()
    if "=" in s and s.split("=", 1)[0].strip().startswith("sb-svc-sb-auth-token"):
        s = s.split("=", 1)[1].strip()
    return s.rstrip(";").strip()


@app.post("/api/accounts")
async def api_add_accounts(req: Request):
    body = await req.json()
    # 方式一：分片输入（sb-svc-sb-auth-token.0 / .1 两个值，后台拼接）
    p0, p1 = _clean_cookie_piece(body.get("part0", "")), _clean_cookie_piece(body.get("part1", ""))
    if p0 or p1:
        if not p0:
            return JSONResponse({"error": "缺少 token.0 分片"}, status_code=400)
        text = p0 + p1  # 拼接为完整 base64- 值
    else:
        # 方式二：批量/整行粘贴
        text = body.get("text", "")
    if not text.strip():
        return JSONResponse({"error": "内容为空"}, status_code=400)
    result = POOL.add(text)
    return result


@app.delete("/api/accounts")
async def api_del_accounts(req: Request):
    body = await req.json()
    ids = body.get("ids") or []
    n = POOL.remove(ids)
    return {"removed": n}


@app.post("/api/accounts/toggle")
async def api_toggle_account(req: Request):
    body = await req.json()
    ok = POOL.set_disabled(body.get("id", ""), bool(body.get("disabled")))
    if not ok:
        return JSONResponse({"error": "账号不存在"}, status_code=404)
    return {"ok": True}


@app.post("/api/accounts/test")
async def api_test_account(req: Request):
    """单账号免费体检：验证 Cookie 有效性 + 余额/订阅/会话有效期。"""
    body = await req.json()
    aid = body.get("id", "")
    with POOL.lock:
        acc = next((a for a in POOL.accounts if a.id == aid), None)
    if not acc:
        return JSONResponse({"error": "账号不存在"}, status_code=404)
    async with httpx.AsyncClient() as client:
        ok = await POOL.ensure_token(acc, client)
        if not ok:
            with POOL.lock:
                acc.last_test = {"ok": False, "error": "token 刷新失败，Cookie 可能已过期",
                                 "time": time.strftime("%H:%M:%S")}
            return {"ok": False, "error": "token 刷新失败，Cookie 可能已过期，请重新获取"}
        try:
            info = await fetch_credits(client, acc)
            info["time"] = time.strftime("%H:%M:%S")
            with POOL.lock:
                acc.last_test = info
                POOL.report_success(acc)
            return info
        except UpstreamError as e:
            with POOL.lock:
                acc.last_test = {"ok": False, "error": str(e), "time": time.strftime("%H:%M:%S")}
            POOL.report_failure(acc, str(e), e.auth)
            return {"ok": False, "error": str(e)}


@app.get("/api/keys")
async def api_list_keys(req: Request):
    return {"keys": api_keys()}


@app.post("/api/keys")
async def api_add_key(req: Request):
    body = await req.json()
    k = (body.get("key") or "").strip()
    if not k:
        return JSONResponse({"error": "key 不能为空"}, status_code=400)
    keys = api_keys()
    if k not in keys:
        keys.append(k)
        cfg_set("API_KEYS", ",".join(keys))
    return {"keys": keys}


@app.delete("/api/keys")
async def api_del_key(req: Request):
    body = await req.json()
    k = body.get("key", "")
    keys = [x for x in api_keys() if x != k]
    if not keys:
        return JSONResponse({"error": "至少保留一个 key"}, status_code=400)
    cfg_set("API_KEYS", ",".join(keys))
    return {"keys": keys}


@app.get("/api/tools")
async def api_tools(req: Request):
    return {"tools": [{"key": k, "name": v["name"], "desc": v["desc"], "enabled": tool_enabled(k)}
                      for k, v in TOOLS.items()]}


@app.post("/api/tools")
async def api_set_tool(req: Request):
    body = await req.json()
    key = body.get("key", "")
    if key not in TOOLS:
        return JSONResponse({"error": "未知工具"}, status_code=404)
    set_tool(key, bool(body.get("enabled")))
    return {"ok": True, "key": key, "enabled": tool_enabled(key)}


@app.get("/api/models")
async def api_models(req: Request):
    try:
        items = await fetch_models()
    except UpstreamError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return {"models": items, "cache_age": int(time.time() - MODELS_CACHE["ts"]),
            "default_model": cfg("DEFAULT_MODEL", "amazon_bedrock/global.anthropic.claude-sonnet-5")}


@app.post("/api/models/refresh")
async def api_models_refresh(req: Request):
    try:
        items = await fetch_models(force=True)
    except UpstreamError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return {"count": len(items)}


@app.post("/api/config")
async def api_config(req: Request):
    body = await req.json()
    if "default_model" in body:
        cfg_set("DEFAULT_MODEL", body["default_model"])
    if "agent_id" in body:
        cfg_set("_AGENT_ID_CACHE", body["agent_id"])
    return {"ok": True}


@app.get("/api/logs")
async def api_logs(req: Request):
    with LOGS_LOCK:
        return {"logs": RECENT_LOGS[:100]}


# ---------------------------------------------------------------------------
# WebUI（全中文，单页，免登录直接访问）
# ---------------------------------------------------------------------------
WEBUI_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>xpander OpenAI 网关管理后台</title>
<style>
:root{--bg:#0f1115;--card:#1a1e26;--card2:#222836;--border:#2e3646;--fg:#e6e9f0;--dim:#8b93a7;--acc:#7c5cff;--ok:#34c98e;--bad:#ff6b6b;--warn:#f5a623}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;font-size:14px}
header{padding:16px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;flex-wrap:wrap}
header h1{font-size:18px}
header .sub{color:var(--dim);font-size:12px}
nav{display:flex;gap:4px;padding:10px 24px;border-bottom:1px solid var(--border);flex-wrap:wrap}
nav button{background:none;border:1px solid transparent;color:var(--dim);padding:8px 14px;border-radius:8px;cursor:pointer;font-size:14px}
nav button.on{background:var(--card2);color:var(--fg);border-color:var(--border)}
main{padding:20px 24px;max-width:1200px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 20px;margin-bottom:16px}
.card h3{font-size:15px;margin-bottom:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.stat{background:var(--card2);border-radius:10px;padding:14px}
.stat .v{font-size:22px;font-weight:600}
.stat .k{color:var(--dim);font-size:12px;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
th{color:var(--dim);font-weight:500}
.btn{background:var(--acc);color:#fff;border:none;border-radius:8px;padding:7px 14px;cursor:pointer;font-size:13px}
.btn.sm{padding:4px 10px;font-size:12px}
.btn.gray{background:var(--card2);color:var(--fg);border:1px solid var(--border)}
.btn.red{background:transparent;color:var(--bad);border:1px solid var(--bad)}
.btn:disabled{opacity:.5;cursor:not-allowed}
input[type=text],textarea,select{background:var(--card2);border:1px solid var(--border);color:var(--fg);border-radius:8px;padding:8px 10px;font-size:13px;width:100%}
textarea{min-height:110px;font-family:monospace;resize:vertical}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px}
.tag.ok{background:rgba(52,201,142,.15);color:var(--ok)}
.tag.bad{background:rgba(255,107,107,.15);color:var(--bad)}
.tag.warn{background:rgba(245,166,35,.15);color:var(--warn)}
.switch{position:relative;width:36px;height:20px;display:inline-block}
.switch input{display:none}
.switch i{position:absolute;inset:0;background:var(--border);border-radius:20px;transition:.2s;cursor:pointer}
.switch i:before{content:"";position:absolute;width:14px;height:14px;background:#fff;border-radius:50%;top:3px;left:3px;transition:.2s}
.switch input:checked+i{background:var(--acc)}
.switch input:checked+i:before{left:19px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.muted{color:var(--dim);font-size:12px}
pre{background:#0a0c10;border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto;font-size:12px;line-height:1.6}
#chatBox{display:flex;flex-direction:column;gap:12px;min-height:200px;max-height:55vh;overflow:auto;padding:4px}
.msg{max-width:85%;padding:10px 14px;border-radius:12px;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.msg.user{align-self:flex-end;background:var(--acc);color:#fff}
.msg.bot{align-self:flex-start;background:var(--card2)}
details.rs{margin-top:8px;border-top:1px dashed var(--border);padding-top:6px}
details.rs summary{color:var(--warn);cursor:pointer;font-size:12px}
details.rs div{color:var(--dim);font-size:12px;white-space:pre-wrap;margin-top:4px}
.step{margin-bottom:14px;padding-left:8px;border-left:3px solid var(--acc)}
.step b{display:block;margin-bottom:4px}
.toast{position:fixed;top:16px;right:16px;background:var(--card2);border:1px solid var(--border);padding:10px 18px;border-radius:10px;z-index:99;display:none}
.hidden{display:none}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid var(--border);border-top-color:var(--acc);border-radius:50%;animation:spin 1s linear infinite;vertical-align:-2px;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.msg.bot.loading{color:var(--dim)}
</style>
</head>
<body>
<header>
  <h1>🛰️ xpander OpenAI 兼容网关</h1>
  <span class="sub">chat.xpander.ai → /v1/chat/completions</span>
  <span class="sub" id="clock"></span>
</header>
<nav>
  <button data-tab="overview" class="on">概览</button>
  <button data-tab="accounts">账号池管理</button>
  <button data-tab="tools">工具开关</button>
  <button data-tab="chat">在线对话</button>
  <button data-tab="keys">调用 Key</button>
  <button data-tab="logs">请求日志</button>
  <button data-tab="help">如何获取 Cookie</button>
</nav>
<main>
<section id="tab-overview">
  <div class="grid" id="statGrid"></div>
  <div class="card">
    <h3>接口信息</h3>
    <pre>POST {BASE}/v1/chat/completions   （stream=true/false 均支持）
GET  {BASE}/v1/models
鉴权: Authorization: Bearer &lt;调用 Key&gt;（默认 123456，可在「调用 Key」页修改）</pre>
  </div>
  <div class="card">
    <h3>默认模型</h3>
    <div class="row">
      <select id="defaultModelSel" style="max-width:420px"></select>
      <button class="btn sm" onclick="saveDefaultModel()">保存</button>
      <button class="btn sm gray" onclick="refreshModels()">刷新模型列表</button>
      <span class="muted" id="modelCacheInfo"></span>
    </div>
  </div>
</section>

<section id="tab-accounts" class="hidden">
  <div class="card">
    <h3>添加账号（分片粘贴）</h3>
    <p class="muted" style="margin-bottom:10px">浏览器 F12 → Application → Cookies → https://chat.xpander.ai，分别复制 <code>sb-svc-sb-auth-token.0</code> 和 <code>sb-svc-sb-auth-token.1</code> 的 Value 粘贴到下面两框（带不带名称前缀都可以，后台自动拼接）</p>
    <div style="margin-bottom:10px">
      <label class="muted">sb-svc-sb-auth-token.0 的值（以 base64- 开头）</label>
      <textarea id="cookiePart0" style="min-height:64px" placeholder="base64-eyJhY2Nlc3NfdG9rZW4iOi..."></textarea>
    </div>
    <div>
      <label class="muted">sb-svc-sb-auth-token.1 的值</label>
      <textarea id="cookiePart1" style="min-height:64px" placeholder="DI2ODU0In0sImlkZW50aXRpZXMiOi..."></textarea>
    </div>
    <div class="row" style="margin-top:10px">
      <button class="btn" onclick="addAccountParts()">添加账号</button>
      <button class="btn gray" onclick="loadAccounts()">刷新列表</button>
      <span class="muted" id="addResult"></span>
    </div>
    <details style="margin-top:14px">
      <summary class="muted" style="cursor:pointer">批量 / 整行粘贴（高级：一次多个账号，每行一个或 ||| 分隔）</summary>
      <textarea id="cookieInput" style="margin-top:8px" placeholder="支持整行 Cookie 头直接粘贴；多账号每行一个"></textarea>
      <div class="row" style="margin-top:8px">
        <button class="btn sm" onclick="addAccounts()">批量添加</button>
      </div>
    </details>
  </div>
  <div class="card">
    <h3>账号列表</h3>
    <div class="row" style="margin-bottom:10px">
      <button class="btn sm red" onclick="delSelected()">删除选中</button>
      <button class="btn sm gray" onclick="toggleSelectAll()">全选/取消</button>
    </div>
    <table>
      <thead><tr><th></th><th>账号</th><th>Token 预览</th><th>余额</th><th>状态</th><th>启用</th><th>操作</th></tr></thead>
      <tbody id="accBody"></tbody>
    </table>
  </div>
</section>

<section id="tab-tools" class="hidden">
  <div class="card"><h3>上游工具 / 参数开关（默认全部关闭，开启状态持久化到 .env）</h3>
    <div id="toolList"></div>
  </div>
</section>

<section id="tab-chat" class="hidden">
  <div class="card">
    <div class="row" style="margin-bottom:10px">
      <select id="chatModel" style="max-width:340px"></select>
      <span class="muted">阅后即焚：对话只存在浏览器内存，刷新页面即清空</span>
      <button class="btn sm gray" onclick="clearChat()">清空</button>
    </div>
    <div id="chatBox"></div>
    <div class="row" style="margin-top:12px">
      <textarea id="chatInput" style="min-height:46px" placeholder="输入消息，Ctrl+Enter 发送" onkeydown="if(event.ctrlKey&&event.key==='Enter')sendChat()"></textarea>
      <button class="btn" id="chatSendBtn" onclick="sendChat()">发送</button>
    </div>
  </div>
</section>

<section id="tab-keys" class="hidden">
  <div class="card">
    <h3>调用 Key 管理（/v1 接口的 Bearer Key）</h3>
    <div class="row" style="margin-bottom:10px">
      <input type="text" id="newKey" placeholder="新 key" style="max-width:280px">
      <button class="btn sm" onclick="addKey()">添加</button>
    </div>
    <table><thead><tr><th>Key</th><th>操作</th></tr></thead><tbody id="keyBody"></tbody></table>
  </div>
</section>

<section id="tab-logs" class="hidden">
  <div class="card">
    <h3>最近请求日志（内存中保留 200 条，全量见 logs/requests.jsonl）</h3>
    <button class="btn sm gray" onclick="loadLogs()">刷新</button>
    <div style="overflow:auto;margin-top:10px">
    <table>
      <thead><tr><th>时间</th><th>账号</th><th>模型</th><th>TTFT</th><th>总耗时</th><th>状态</th><th>详情</th></tr></thead>
      <tbody id="logBody"></tbody>
    </table>
    </div>
  </div>
</section>

<section id="tab-help" class="hidden">
  <div class="card"><h3>如何获取 Cookie（Supabase 登录态）</h3>
    <div class="step"><b>1. 登录 chat.xpander.ai</b>用 Google 账号正常登录网页端。</div>
    <div class="step"><b>2. 打开开发者工具</b>按 F12（或右键 → 检查），切到「应用 / Application」标签。</div>
    <div class="step"><b>3. 找到 Cookie</b>左侧 Cookies → https://chat.xpander.ai，找到 <code>sb-svc-sb-auth-token.0</code> 和 <code>sb-svc-sb-auth-token.1</code> 两项。</div>
    <div class="step"><b>4. 复制完整值</b>把两项的 Value 按顺序拼接（或直接到「网络/Network」里任选一个发往 svc-sb 的请求，复制请求头里的整行 Cookie），粘贴到「账号池管理」页的输入框。</div>
    <div class="step"><b>5. 添加并测试</b>点击「添加」，然后在列表里点「测试」验证余额与有效期。网关会用 refresh_token 自动续期，理论上粘贴一次长期有效；若 Google 侧登出或改密，需重新获取。</div>
  </div>
  <div class="card"><h3>关于网页端 Settings → API keys 里的 Key</h3>
    <p class="muted" style="line-height:1.8">
    那是 <b>xpander 官方 REST API</b>（api.xpander.ai/v1）的调用凭证，例如 GET /v1/agents、POST /v1/agents/{id}/invoke 同步调用，<b>不能</b>用于网页聊天后端（chat-backend），也无法替代 Cookie 接入本网关。它适合自建应用直连 xpander 平台；本网关走的是网页端真实协议（SSE 流式 + 会话），所以账号池仍使用 Cookie。</p>
  </div>
</section>
</main>
<div class="toast" id="toast"></div>
<script>
const $ = s => document.querySelector(s);
const BASE = location.origin;
document.getElementById('statGrid').innerHTML = '';
function toast(t){const e=$('#toast');e.textContent=t;e.style.display='block';setTimeout(()=>e.style.display='none',2500)}
async function api(path, opt){const r=await fetch('/api'+path,opt);return r.json()}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
  document.querySelectorAll('main section').forEach(s=>s.classList.add('hidden'));
  $('#tab-'+b.dataset.tab).classList.remove('hidden');
  if(b.dataset.tab==='accounts')loadAccounts();
  if(b.dataset.tab==='keys')loadKeys();
  if(b.dataset.tab==='tools')loadTools();
  if(b.dataset.tab==='logs')loadLogs();
  if(b.dataset.tab==='overview')loadOverview();
});
async function loadOverview(){
  const d = await api('/overview');
  $('#statGrid').innerHTML = [
    ['服务状态','🟢 运行中'],['账号总数',d.accounts_total],['可用账号',d.accounts_active],
    ['默认模型',d.default_model.split('/').pop()],['模型缓存',d.models_cached+' 个'],
    ['缓存时间',d.models_cache_age==null?'未抓取':d.models_cache_age+' 秒前'],
  ].map(([k,v])=>`<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join('');
  document.querySelectorAll('#tab-overview pre')[0].innerHTML =
    document.querySelectorAll('#tab-overview pre')[0].innerHTML.replaceAll('{BASE}', BASE);
  loadModelList(d.default_model);
  $('#modelCacheInfo').textContent = d.models_cache_age==null?'':'缓存于 '+d.models_cache_age+' 秒前（TTL 3600s）';
}
async function loadModelList(selected){
  const d = await api('/models');
  const sels = [$('#defaultModelSel'), $('#chatModel')];
  sels.forEach(sel=>{ if(!sel) return; sel.innerHTML=''; (d.models||[]).forEach(m=>{
    const o=document.createElement('option'); o.value=m.id; o.textContent=m.id; sel.appendChild(o);});
    sel.value = selected || (d.default_model||''); });
}
async function refreshModels(){const d=await api('/models/refresh',{method:'POST'});toast(d.count?'已刷新，共 '+d.count+' 个模型':('刷新失败: '+(d.error||'')));loadOverview()}
async function saveDefaultModel(){await api('/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({default_model:$('#defaultModelSel').value})});toast('已保存默认模型')}

let ACCS=[];
async function loadAccounts(){
  const d=await api('/accounts');ACCS=d.accounts;
  $('#accBody').innerHTML=ACCS.map(a=>{
    const bal=a.last_test&&a.last_test.ok?('$'+(a.last_test.balance_usd??'?')):'-';
    let st='';
    if(a.disabled)st='<span class="tag warn">已停用</span>';
    else if(a.dead)st='<span class="tag bad">已剔除</span>';
    else st='<span class="tag ok">正常</span>';
    return `<tr><td><input type="checkbox" class="accSel" data-id="${a.id}"></td>
    <td>${a.email||'-'}<div class="muted">ID ${a.id} · 会话剩 ${Math.round(a.expires_in/60)} 分钟</div></td>
    <td class="muted">${a.preview}</td><td>${bal}</td><td>${st}${a.last_error?'<div class="muted" style="color:var(--bad)">'+a.last_error.slice(0,60)+'</div>':''}</td>
    <td><label class="switch"><input type="checkbox" ${a.disabled?'':'checked'} onchange="toggleAcc('${a.id}',this.checked)"><i></i></label></td>
    <td><button class="btn sm gray" onclick="testAcc('${a.id}',this)">测试</button>
    <button class="btn sm red" onclick="delAcc('${a.id}')">删除</button></td></tr>`}).join('')||'<tr><td colspan="7" class="muted">暂无账号，请在上方粘贴 Cookie 添加</td></tr>';
}
async function addAccounts(){
  const text=$('#cookieInput').value;if(!text.trim())return toast('请先粘贴 Cookie');
  const d=await api('/accounts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  $('#addResult').textContent=`新增 ${d.added}，重复 ${d.duplicate}，无效 ${d.invalid}`;
  $('#cookieInput').value='';loadAccounts();
}
async function addAccountParts(){
  const part0=$('#cookiePart0').value, part1=$('#cookiePart1').value;
  if(!part0.trim())return toast('请先粘贴 token.0 分片');
  const d=await api('/accounts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({part0,part1})});
  if(d.error)return toast(d.error);
  $('#addResult').textContent=d.added?'✅ 添加成功':(d.duplicate?'账号已存在（重复）':'添加失败');
  $('#cookiePart0').value='';$('#cookiePart1').value='';loadAccounts();
}
function toggleSelectAll(){const cbs=[...document.querySelectorAll('.accSel')];const all=cbs.every(c=>c.checked);cbs.forEach(c=>c.checked=!all)}
async function delSelected(){const ids=[...document.querySelectorAll('.accSel:checked')].map(c=>c.dataset.id);if(!ids.length)return toast('未选中账号');
  await api('/accounts',{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids})});loadAccounts();toast('已删除')}
async function delAcc(id){await api('/accounts',{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:[id]})});loadAccounts()}
async function toggleAcc(id,on){await api('/accounts/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,disabled:!on})});toast(on?'已启用':'已停用（重启保留）')}
async function testAcc(id,btn){btn.disabled=true;btn.textContent='检测中…';
  const d=await api('/accounts/test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
  btn.disabled=false;btn.textContent='测试';
  if(d.ok)alert(`✅ Cookie 有效\n邮箱: ${d.email}\n余额: $${d.balance_usd}（${d.balance_credits} credits）\n订阅: ${d.license_tier||'免费版'}（激活: ${d.license_active}）\n免费额度领取: ${d.free_grant_claimed_at||'-'}\n累计消耗: ${d.lifetime_burned} credits\nToken 剩余有效期: ${Math.round(d.token_expires_in/60)} 分钟（自动续期）\n账户封禁: ${d.blocked?'是':'否'}`);
  else alert('❌ 检测失败: '+(d.error||'未知错误'));
  loadAccounts();}

async function loadTools(){const d=await api('/tools');
  $('#toolList').innerHTML=d.tools.map(t=>`<div class="row" style="justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border)">
  <div><b>${t.name}</b><div class="muted">${t.desc}</div></div>
  <label class="switch"><input type="checkbox" ${t.enabled?'checked':''} onchange="setTool('${t.key}',this.checked)"><i></i></label></div>`).join('')}
async function setTool(k,on){await api('/tools',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k,enabled:on})});toast('已保存')}

async function loadKeys(){const d=await api('/keys');
  $('#keyBody').innerHTML=d.keys.map(k=>`<tr><td><code>${k}</code></td>
  <td><button class="btn sm red" onclick="delKey('${k}')">删除</button></td></tr>`).join('')}
async function addKey(){const k=$('#newKey').value.trim();if(!k)return;
  await api('/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k})});$('#newKey').value='';loadKeys();toast('已添加')}
async function delKey(k){const d=await api('/keys',{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k})});
  if(d.error)return toast(d.error);loadKeys();toast('已删除')}

async function loadLogs(){const d=await api('/logs');
  $('#logBody').innerHTML=d.logs.map(l=>`<tr><td class="muted">${l.ts}</td><td>${l.account}</td>
  <td class="muted">${(l.model||'').split('/').pop()}</td><td>${l.ttft_ms}ms</td><td>${l.total_ms}ms</td>
  <td>${l.ok?'<span class="tag ok">成功</span>':'<span class="tag bad">失败</span>'}</td>
  <td><button class="btn sm gray" onclick='alert("输入:\\n"+${JSON.stringify(JSON.stringify(l.input||""))}+"\\n\\n输出:\\n"+${JSON.stringify(JSON.stringify((l.output||"").slice(-800)))}+"\\n\\n推理:\\n"+${JSON.stringify(JSON.stringify((l.reasoning||"").slice(-500)))}+"\\n\\n错误: "+${JSON.stringify(JSON.stringify(l.error||"-"))})'>查看</button></td></tr>`).join('')||'<tr><td colspan="7" class="muted">暂无日志</td></tr>'}

// ---- 在线对话（阅后即焚：只在内存） ----
let CHAT=[];
let chatWaitStart=0, chatTimer=null;
function renderChat(){
  $('#chatBox').innerHTML=CHAT.map(m=>{
    if(m.role==='user')return `<div class="msg user">${esc(m.content)}</div>`;
    if(m.loading)return `<div class="msg bot loading"><span class="spinner"></span>上游思考中… 已等待 <b>${m.waitSec||0}s</b><span class="muted">（复杂任务可能要 1-2 分钟，请稍候）</span>${m.reasoning?`<details class="rs"><summary>🧠 推理过程</summary><div>${esc(m.reasoning)}</div></details>`:''}</div>`;
    return `<div class="msg bot">${esc(m.content)}${m.reasoning?`<details class="rs"><summary>🧠 推理过程（${m.reasoning.length} 字）</summary><div>${esc(m.reasoning)}</div></details>`:''}</div>`}).join('');
  $('#chatBox').scrollTop=$('#chatBox').scrollHeight}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>')}
function clearChat(){CHAT=[];renderChat()}
async function sendChat(){
  const text=$('#chatInput').value.trim();if(!text)return;
  $('#chatInput').value='';$('#chatSendBtn').disabled=true;
  CHAT.push({role:'user',content:text});
  CHAT.push({role:'assistant',content:'',reasoning:'',loading:true,waitSec:0});
  renderChat();
  chatWaitStart=Date.now();
  chatTimer=setInterval(()=>{const m=CHAT[CHAT.length-1];if(m&&m.loading){m.waitSec=Math.round((Date.now()-chatWaitStart)/1000);renderChat()}},1000);
  const model=$('#chatModel').value;
  const firstKey=(await api('/keys')).keys[0];
  try{
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+firstKey,'X-Api-Key':firstKey},
      body:JSON.stringify({model,messages:CHAT.slice(0,-1).map(m=>({role:m.role,content:m.content})),stream:true})});
    if(!r.ok){const t=await r.text();throw new Error(r.status+' '+t.slice(0,200))}
    const rd=r.body.getReader();const dec=new TextDecoder();let buf='';
    while(true){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf('\n\n'))>=0){const blk=buf.slice(0,i);buf=buf.slice(i+2);
        if(!blk.startsWith('data:'))continue;const js=blk.slice(5).trim();if(js==='[DONE]')continue;
        try{const d=JSON.parse(js).choices[0].delta;const m=CHAT[CHAT.length-1];
          if(d.content){m.loading=false;m.content+=d.content}
          if(d.reasoning_content){m.reasoning+=d.reasoning_content;renderChat()}
          if(!m.loading)renderChat()}catch(e){}}}
    const m=CHAT[CHAT.length-1];
    if(m.loading){m.loading=false;m.content=m.content||'（上游返回了空内容，可换模型重试）'}
  }catch(e){const m=CHAT[CHAT.length-1];m.loading=false;m.content='[请求失败] '+e.message}
  clearInterval(chatTimer);
  $('#chatSendBtn').disabled=false;renderChat();
}
loadOverview();setInterval(()=>{$('#clock').textContent=new Date().toLocaleString('zh-CN')},1000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def webui():
    return WEBUI_HTML


def main():
    port = int(cfg("PORT", "8300"))
    print(f"[*] xpander OpenAI 网关启动: http://0.0.0.0:{port}")
    print(f"[*] WebUI: http://127.0.0.1:{port}/  （免登录直接访问）")
    print(f"[*] API:   http://127.0.0.1:{port}/v1/chat/completions  (Bearer key 鉴权)")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
