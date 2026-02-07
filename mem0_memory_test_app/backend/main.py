# -*- coding: utf-8 -*-
"""
FastAPI 入口：POST /chat 接收玩家消息，召回记忆、按所选 LLM 生成 NPC 回复、写入记忆。
并提供 GET /logs/stream 用于前端实时展示服务端日志。
"""

import asyncio
import json
import logging
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import (
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
)
from db_init import run_all as db_init_run_all
from memory_backends import create_memory_for_backend

logger = logging.getLogger(__name__)
_memory_cache: Dict[str, Any] = {}

# 日志缓冲：保留最近 N 行，供 SSE 推送给前端
LOG_BUFFER_MAX = 1000
_log_buffer: Deque[str] = deque(maxlen=LOG_BUFFER_MAX)
_log_lock = threading.Lock()


class LogBufferHandler(logging.Handler):
    """将日志写入 _log_buffer，便于 SSE 推送。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _log_lock:
                _log_buffer.append(msg)
        except Exception:
            pass


def _setup_log_stream():
    """把 LogBufferHandler 挂到 root logger，确保 INFO 级别日志能流入前端。"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)  # 否则 root 默认 WARNING，会过滤掉 mem0/httpx 的 INFO
    h = LogBufferHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(h)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_log_stream()
    db_init_run_all()
    yield
    _memory_cache.clear()


app = FastAPI(title="NPC 对话记忆测试", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: str = Field(default="player-1")
    npc_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None, description="当前会话 ID，用于短期记忆召回与写入")
    backend: str = Field(default="mongodb", description="mongodb | postgres")
    llm_provider: str = Field(default="deepseek", description="deepseek | azure_openai")


class ChatResponse(BaseModel):
    reply: str
    recalled_memories: list = Field(default_factory=list)
    memory_type_this_turn: str = Field(default="long_term", description="本轮判定：long_term 或 short_term")
    backend: str
    llm_provider: str = Field(default="deepseek")


# ---------- 记忆管理 API 的请求/响应模型 ----------
class MemorySearchQuery(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str = Field(default="player-1")
    agent_id: Optional[str] = Field(default=None)
    backend: str = Field(default="mongodb", description="mongodb | postgres")
    limit: int = Field(default=10, ge=1, le=100)


class MemoryAddBody(BaseModel):
    """添加记忆（支持短期会话 session_id、短期过期、过程记忆）。"""
    content: str = Field(..., min_length=1)
    user_id: str = Field(default="player-1")
    agent_id: Optional[str] = Field(default=None)
    backend: str = Field(default="mongodb")
    session_id: Optional[str] = Field(default=None, description="短期记忆（会话）：用 session_id 区分，会话结束可清空")
    expiration_days: Optional[int] = Field(default=None, ge=1, le=365, description="短期记忆（过期）：N 天后过期，不传则长期")
    memory_type: Optional[str] = Field(default=None, description="procedural_memory 表示过程记忆")


class MemoryUpdateBody(BaseModel):
    new_content: str = Field(..., min_length=1)


class MemoryDeleteAllQuery(BaseModel):
    user_id: str = Field(default="player-1")
    agent_id: Optional[str] = Field(default=None)
    backend: str = Field(default="mongodb")


def _get_memory(backend: str):
    if backend not in _memory_cache:
        _memory_cache[backend] = create_memory_for_backend(backend)
    return _memory_cache[backend]


def _npc_reply_deepseek(user_message: str, recalled: list[str]) -> str:
    if not DEEPSEEK_API_KEY:
        return "[未配置 DEEPSEEK_API_KEY]"
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    system = "你是一个开放世界游戏中的 NPC，根据与玩家的对话和「召回的记忆」自然回复。若没有记忆，就按当前对话回答。回复简洁、口语化。"
    user_content = "【本轮召回的记忆】\n" + "\n".join("- " + m for m in recalled) + "\n\n【玩家说】\n" + user_message if recalled else user_message
    resp = client.chat.completions.create(model=DEEPSEEK_MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}])
    return (resp.choices[0].message.content or "").strip()


def _npc_reply_azure(user_message: str, recalled: list[str]) -> str:
    if not OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        return "[未配置 Azure OpenAI（OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT）]"
    from openai import AzureOpenAI
    client = AzureOpenAI(api_key=OPENAI_API_KEY, api_version=AZURE_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT.rstrip("/"))
    system = "你是一个开放世界游戏中的 NPC，根据与玩家的对话和「召回的记忆」自然回复。若没有记忆，就按当前对话回答。回复简洁、口语化。"
    user_content = "【本轮召回的记忆】\n" + "\n".join("- " + m for m in recalled) + "\n\n【玩家说】\n" + user_message if recalled else user_message
    resp = client.chat.completions.create(model=AZURE_DEPLOYMENT, messages=[{"role": "system", "content": system}, {"role": "user", "content": user_content}])
    return (resp.choices[0].message.content or "").strip()


def _npc_reply(user_message: str, recalled: list[str], llm_provider: str) -> str:
    if (llm_provider or "").strip().lower() == "azure_openai":
        return _npc_reply_azure(user_message, recalled)
    return _npc_reply_deepseek(user_message, recalled)


# ---------- 长短记忆分类：根据本轮对话内容判定写入长期还是短期 ----------
CLASSIFY_SYSTEM = """你只输出一个词：long_term 或 short_term。
- long_term：本轮对话涉及 用户偏好、账户信息、重要事实（需永久保存）。
- short_term：本轮为 当前上下文、多步骤任务、临时提醒（会话级即可）。
只回复 long_term 或 short_term，不要其他内容。"""


def _classify_memory_type_deepseek(user_message: str, assistant_reply: str) -> str:
    if not DEEPSEEK_API_KEY:
        return "long_term"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        content = f"【玩家】{user_message}\n【NPC】{assistant_reply}"
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=20,
        )
        out = (resp.choices[0].message.content or "").strip().lower()
        return "short_term" if "short" in out else "long_term"
    except Exception as e:
        logger.warning("分类长短记忆失败，默认 long_term: %s", e)
        return "long_term"


def _classify_memory_type_azure(user_message: str, assistant_reply: str) -> str:
    if not OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        return "long_term"
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=OPENAI_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT.rstrip("/"),
        )
        content = f"【玩家】{user_message}\n【NPC】{assistant_reply}"
        resp = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=20,
        )
        out = (resp.choices[0].message.content or "").strip().lower()
        return "short_term" if "short" in out else "long_term"
    except Exception as e:
        logger.warning("分类长短记忆失败，默认 long_term: %s", e)
        return "long_term"


def _classify_memory_type(user_message: str, assistant_reply: str, llm_provider: str) -> str:
    if (llm_provider or "").strip().lower() == "azure_openai":
        return _classify_memory_type_azure(user_message, assistant_reply)
    return _classify_memory_type_deepseek(user_message, assistant_reply)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    backend = (req.backend or "mongodb").strip().lower()
    if backend not in ("mongodb", "postgres"):
        raise HTTPException(status_code=400, detail="backend 只能是 mongodb 或 postgres")
    llm_provider = (req.llm_provider or "deepseek").strip().lower()
    if llm_provider not in ("deepseek", "azure_openai"):
        llm_provider = "deepseek"
    user_id = (req.user_id or "player-1").strip()
    npc_id = (req.npc_id or "npc-default").strip()
    session_id = (req.session_id or "").strip() or None
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    memory = _get_memory(backend)
    # 召回：长期记忆（无 run_id）+ 当前会话短期记忆（run_id=session_id），合并去重
    recalled_set: set = set()
    try:
        long_result = memory.search(query=message, user_id=user_id, agent_id=npc_id, limit=5)
        for r in long_result.get("results", []):
            m = r.get("memory", "")
            if m:
                recalled_set.add(m)
    except Exception as e:
        logger.warning("长期记忆检索失败: %s", e)
    if session_id:
        try:
            short_result = memory.search(
                query=message, user_id=user_id, agent_id=npc_id, run_id=session_id, limit=5
            )
            for r in short_result.get("results", []):
                m = r.get("memory", "")
                if m:
                    recalled_set.add(m)
        except Exception as e:
            logger.warning("短期记忆检索失败: %s", e)
    recalled = list(recalled_set)

    reply = _npc_reply(message, recalled, llm_provider)

    # 判定本轮写入长期还是短期
    memory_type_this_turn = _classify_memory_type(message, reply, llm_provider)
    messages = [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    add_kwargs: Dict[str, Any] = {"user_id": user_id, "agent_id": npc_id}
    if memory_type_this_turn == "short_term" and session_id:
        add_kwargs["run_id"] = session_id
    try:
        memory.add(messages, infer=True, **add_kwargs)
    except (json.JSONDecodeError, Exception) as e:
        logger.info("infer=True 失败，回退 infer=False: %s", e)
        try:
            memory.add(messages, infer=False, **add_kwargs)
        except Exception as e2:
            logger.warning("记忆写入失败: %s", e2)

    return ChatResponse(
        reply=reply,
        recalled_memories=recalled,
        memory_type_this_turn=memory_type_this_turn,
        backend=backend,
        llm_provider=llm_provider,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- 记忆管理：搜索、按 ID 操作 ----------
def _norm_backend(backend: str) -> str:
    b = (backend or "mongodb").strip().lower()
    if b not in ("mongodb", "postgres"):
        raise HTTPException(status_code=400, detail="backend 只能是 mongodb 或 postgres")
    return b


@app.get("/memory/by-session")
def memory_get_by_session(
    session_id: str = Query(..., min_length=1),
    user_id: Optional[str] = Query(None),
    backend: str = Query("mongodb"),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    """按 session_id（会话）获取短期记忆；可选 user_id 限定用户。mem0 内部用 run_id 存储。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    try:
        kwargs = {"run_id": session_id.strip(), "limit": limit}
        if user_id and str(user_id).strip():
            kwargs["user_id"] = str(user_id).strip()
        result = memory.get_all(**kwargs)
    except Exception as e:
        logger.warning("memory.get_all(by-session) 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return result if isinstance(result, dict) else {"results": result}


@app.get("/memory/by-user")
def memory_get_by_user(
    user_id: str = Query("player-1"),
    agent_id: Optional[str] = Query(None),
    backend: str = Query("mongodb"),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    """按玩家 ID（及可选 agent_id）获取该范围下所有记忆。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    try:
        kwargs = {"user_id": (user_id or "player-1").strip(), "limit": limit}
        if agent_id and str(agent_id).strip():
            kwargs["agent_id"] = str(agent_id).strip()
        result = memory.get_all(**kwargs)
    except Exception as e:
        logger.warning("memory.get_all 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return result if isinstance(result, dict) else {"results": result}


@app.post("/memory/search")
def memory_search(body: MemorySearchQuery) -> Dict[str, Any]:
    """搜索记忆：query + user_id + agent_id + limit。"""
    backend = _norm_backend(body.backend)
    memory = _get_memory(backend)
    agent_id = (body.agent_id or "").strip() or None
    try:
        result = memory.search(
            query=body.query,
            user_id=(body.user_id or "player-1").strip(),
            agent_id=agent_id,
            limit=body.limit,
        )
    except Exception as e:
        logger.warning("memory.search 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return result if isinstance(result, dict) else {"results": result}


@app.get("/memory/{memory_id}")
def memory_get(
    memory_id: str,
    backend: str = Query("mongodb", description="mongodb | postgres"),
) -> Dict[str, Any]:
    """按 ID 获取单条记忆。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    if not memory_id or not memory_id.strip():
        raise HTTPException(status_code=400, detail="memory_id 不能为空")
    try:
        out = memory.get(memory_id.strip())
    except Exception as e:
        logger.warning("memory.get 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    if out is None:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return out if isinstance(out, dict) else {"memory": out}


@app.patch("/memory/{memory_id}")
def memory_update(
    memory_id: str,
    body: MemoryUpdateBody,
    backend: str = Query("mongodb"),
) -> Dict[str, Any]:
    """按 ID 更新记忆内容（OSS 支持）。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    if not memory_id or not memory_id.strip():
        raise HTTPException(status_code=400, detail="memory_id 不能为空")
    if not hasattr(memory, "update"):
        raise HTTPException(status_code=501, detail="当前后端不支持 update")
    try:
        memory.update(memory_id.strip(), body.new_content)
    except Exception as e:
        logger.warning("memory.update 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "memory_id": memory_id}


@app.delete("/memory/{memory_id}")
def memory_delete(
    memory_id: str,
    backend: str = Query("mongodb"),
) -> Dict[str, Any]:
    """按 ID 删除单条记忆。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    if not memory_id or not memory_id.strip():
        raise HTTPException(status_code=400, detail="memory_id 不能为空")
    try:
        memory.delete(memory_id.strip())
    except Exception as e:
        logger.warning("memory.delete 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "memory_id": memory_id}


@app.delete("/memory")
def memory_delete_all(
    user_id: str = Query("player-1"),
    agent_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None, description="短期会话：按 session_id 清空该会话记忆"),
    backend: str = Query("mongodb"),
) -> Dict[str, Any]:
    """按 user_id/agent_id 或 session_id 删除该范围下所有记忆。"""
    backend = _norm_backend(backend)
    memory = _get_memory(backend)
    try:
        kwargs: Dict[str, Any] = {"limit": 500}
        if session_id and str(session_id).strip():
            kwargs["run_id"] = str(session_id).strip()
            if user_id and str(user_id).strip():
                kwargs["user_id"] = str(user_id).strip()
        else:
            kwargs["user_id"] = (user_id or "player-1").strip()
            if agent_id and str(agent_id).strip():
                kwargs["agent_id"] = str(agent_id).strip()
        # mem0 delete_all 与 MongoDB vector_store.list() 返回格式不兼容（list 返回单层列表，
        # delete_all 却用 [0] 当列表，导致迭代到 tuple 报 'tuple' object has no attribute 'id'）
        # 改为：get_all 取回 id 列表，再逐条 delete。
        result = memory.get_all(**kwargs)
        results_list = result.get("results", result) if isinstance(result, dict) else result
        ids: List[str] = []
        for item in results_list or []:
            if isinstance(item, dict):
                ids.append(item.get("id"))
            elif hasattr(item, "id"):
                ids.append(getattr(item, "id"))
        for mid in ids:
            if mid:
                try:
                    memory.delete(mid)
                except Exception as e:
                    logger.warning("memory.delete(%s) 失败: %s", mid, e)
    except Exception as e:
        logger.warning("memory delete_all 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "deleted_count": len(ids)}


@app.post("/memory/add")
def memory_add(body: MemoryAddBody) -> Dict[str, Any]:
    """添加记忆；支持短期会话（session_id，内部传 run_id）、短期过期（expiration_days）、过程记忆（memory_type）。"""
    backend = _norm_backend(body.backend)
    memory = _get_memory(backend)
    user_id = (body.user_id or "player-1").strip()
    agent_id = (body.agent_id or "").strip() or None
    messages = [{"role": "user", "content": body.content}]
    kwargs: Dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "infer": False}
    if body.session_id and body.session_id.strip():
        kwargs["run_id"] = body.session_id.strip()
    if body.expiration_days is not None:
        expires_at = (datetime.now() + timedelta(days=body.expiration_days)).strftime("%Y-%m-%d")
        kwargs["metadata"] = kwargs.get("metadata") or {}
        kwargs["metadata"]["expiration_date"] = expires_at
    if body.memory_type:
        kwargs["memory_type"] = body.memory_type.strip()
    try:
        result = memory.add(messages, **kwargs)
    except Exception as e:
        logger.warning("memory.add 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return result if isinstance(result, dict) else {"results": [result]}


async def _logs_sse_generator():
    """SSE 流：先发历史缓冲，再轮询新行并推送。"""
    with _log_lock:
        lines = list(_log_buffer)
    last_n = len(lines)
    for line in lines:
        yield f"data: {json.dumps({'line': line})}\n\n"
    while True:
        await asyncio.sleep(0.3)
        with _log_lock:
            new_lines = list(_log_buffer)
        if len(new_lines) > last_n:
            for line in new_lines[last_n:]:
                yield f"data: {json.dumps({'line': line})}\n\n"
            last_n = len(new_lines)


@app.get("/logs/stream")
async def logs_stream():
    """Server-Sent Events：实时推送服务端日志，供前端右侧面板展示。"""
    return StreamingResponse(
        _logs_sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
