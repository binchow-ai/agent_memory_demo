# -*- coding: utf-8 -*-
"""
记忆后端：按 backend（mongodb / postgres）构造 mem0 Memory。
mem0 内部用 LLM 做事实抽取，这里统一用 DeepSeek；NPC 回复用哪家 LLM 由 main.py 按前端 llm_provider 决定。
"""

import os
from datetime import datetime
from typing import Optional

os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "true")

from langchain_voyageai import VoyageAIEmbeddings
from config import (
    MONGODB_URI,
    POSTGRES_URI,
    VOYAGE_API_KEY,
    EMBEDDING_DIMS,
    DEEPSEEK_API_KEY,
)


def _voyage_embedder():
    return VoyageAIEmbeddings(model="voyage-4-lite", voyage_api_key=VOYAGE_API_KEY)


CUSTOM_FACT_EXTRACTION_PROMPT = """You extract facts from conversations (e.g. player and NPC). Output only what should be remembered.

RULES:
- Reply with ONLY a single JSON object. No markdown, no code fences, no explanation.
- The JSON must have exactly one key: "facts". The value must be an array of strings. If nothing to remember: {"facts": []}
- Examples: {"facts": []} or {"facts": ["Player prefers swords", "NPC name is Merida"]}
- Use the same language as the conversation. Today's date: """ + datetime.now().strftime("%Y-%m-%d") + """

Output ONLY the JSON object."""


def get_mongodb_config():
    return {
        "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT,
        "vector_store": {
            "provider": "mongodb",
            "config": {
                "db_name": "mem0_agent_memory",
                "collection_name": "extracted_memories",
                "mongo_uri": MONGODB_URI,
                "embedding_model_dims": EMBEDDING_DIMS,
            },
        },
        "embedder": {
            "provider": "langchain",
            "config": {"model": _voyage_embedder(), "embedding_dims": EMBEDDING_DIMS},
        },
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": 1.0,
                "api_key": DEEPSEEK_API_KEY,
            },
        },
    }


def get_postgres_config():
    connection_string = POSTGRES_URI or ""
    if connection_string and "sslmode=" not in connection_string:
        connection_string = connection_string.rstrip("/") + "?sslmode=require"
    return {
        "custom_fact_extraction_prompt": CUSTOM_FACT_EXTRACTION_PROMPT,
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string": connection_string,
                "collection_name": "mem0_memories",
                "embedding_model_dims": EMBEDDING_DIMS,
            },
        },
        "embedder": {
            "provider": "langchain",
            "config": {"model": _voyage_embedder(), "embedding_dims": EMBEDDING_DIMS},
        },
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": 1.0,
                "api_key": DEEPSEEK_API_KEY,
            },
        },
    }


def create_memory_for_backend(backend: str):
    from mem0 import Memory
    if backend == "mongodb":
        config = get_mongodb_config()
    elif backend == "postgres":
        config = get_postgres_config()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return Memory.from_config(config)
