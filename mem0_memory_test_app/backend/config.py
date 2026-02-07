# -*- coding: utf-8 -*-
"""
配置文件模块（config.py）
========================
从环境变量或 .env 读取配置，供 main / memory_backends / db_init 使用。
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    root_env = Path(__file__).resolve().parents[2] / ".env"
    if root_env.exists():
        load_dotenv(root_env)
    else:
        load_dotenv()
except ImportError:
    pass

# --- 向量库 ---
MONGODB_URI = os.getenv("MONGODB_URI", "")
POSTGRES_URI = os.getenv("POSTGRES_URI", "")
EMBEDDING_DIMS = 1024

# --- NPC 回复与 mem0 可选的 LLM：DeepSeek ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# --- NPC 回复可选的 LLM：Azure OpenAI ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# --- 嵌入（Voyage）---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
