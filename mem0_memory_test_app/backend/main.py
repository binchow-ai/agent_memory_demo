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
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, HTTPException
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
    backend: str = Field(default="mongodb", description="mongodb | postgres")
    llm_provider: str = Field(default="deepseek", description="deepseek | azure_openai")


class ChatResponse(BaseModel):
    reply: str
    recalled_memories: list = Field(default_factory=list)
    backend: str
    llm_provider: str = Field(default="deepseek")


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
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message 不能为空")

    memory = _get_memory(backend)
    try:
        search_result = memory.search(query=message, user_id=user_id, agent_id=npc_id, limit=5)
    except Exception as e:
        logger.warning("记忆检索失败: %s", e)
        search_result = {}
    recalled = [r.get("memory", "") for r in search_result.get("results", []) if r.get("memory")]

    reply = _npc_reply(message, recalled, llm_provider)

    messages = [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    try:
        memory.add(messages, user_id=user_id, agent_id=npc_id, infer=True)
    except (json.JSONDecodeError, Exception) as e:
        logger.info("infer=True 失败，回退 infer=False: %s", e)
        try:
            memory.add(messages, user_id=user_id, agent_id=npc_id, infer=False)
        except Exception as e2:
            logger.warning("记忆写入失败: %s", e2)

    return ChatResponse(reply=reply, recalled_memories=recalled, backend=backend, llm_provider=llm_provider)


@app.get("/health")
def health():
    return {"status": "ok"}


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
