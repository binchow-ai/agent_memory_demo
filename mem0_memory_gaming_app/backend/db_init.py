# -*- coding: utf-8 -*-
"""
数据库初始化模块（db_init.py）
==============================

应用启动时自动检查并创建「存记忆」所需的库、表、索引：
- MongoDB：确保有向量搜索索引，否则无法按相似度查记忆
- PostgreSQL：确保安装了 pgvector 扩展，否则无法存向量

逻辑：存在就跳过，不存在才创建，避免重复创建报错。
"""

import logging

# 从 config 里拿「是否配置了数据库」以及「向量维度」
from config import MONGODB_URI, POSTGRES_URI, EMBEDDING_DIMS

# 用 Python 自带的 logging 打日志，方便排查问题；__name__ 会显示是 db_init 模块在打
logger = logging.getLogger(__name__)

# =============================================================================
# 常量：和 memory_backends 里用的名字保持一致，避免写错
# =============================================================================
MONGODB_DB_NAME = "mem0_agent_memory"  # MongoDB 数据库名
MONGODB_COLLECTION_NAME = "extracted_memories"  # 存「抽取出的记忆」的集合名
# 与 mem0 约定一致：向量索引名 = {collection}_vector_index；全文索引供混合检索
MONGODB_VECTOR_INDEX_NAME = f"{MONGODB_COLLECTION_NAME}_vector_index"
MONGODB_TEXT_INDEX_NAME = f"{MONGODB_COLLECTION_NAME}_text_index"


def ensure_mongodb_indexes() -> None:
    """
    确保 MongoDB 里存在「向量搜索索引」。
    - 没配置 MONGODB_URI 则直接返回，不报错
    - 已有索引则只打日志「已存在」
    - 没有索引则创建（Atlas 上创建后可能要等约 1 分钟才真正可用）
    """
    # 未配置或配成空字符串，就不做任何事
    if not MONGODB_URI or not MONGODB_URI.strip():
        logger.info("MONGODB_URI 未配置，跳过 MongoDB 资源初始化")
        return
    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel

    # 连上 MongoDB（Atlas 会按 URI 自动选集群）
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB_NAME]
    col = db[MONGODB_COLLECTION_NAME]
    # 向量索引名与 mem0 MongoDBVectorStore 一致，否则 mem0 检索会报「索引不存在」
    index_name = MONGODB_VECTOR_INDEX_NAME

    existing = list(col.list_search_indexes(name=index_name))
    if existing:
        logger.info("MongoDB 向量索引已存在: %s", index_name)
    else:
        # mem0 文档形态：{ embedding, payload:{ user_id, agent_id, run_id, data, ... } }
        col.create_search_index(
            SearchIndexModel(
                name=index_name,
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": EMBEDDING_DIMS,
                            "similarity": "cosine",
                        },
                        {"type": "filter", "path": "payload.user_id"},
                        {"type": "filter", "path": "payload.agent_id"},
                        {"type": "filter", "path": "payload.run_id"},
                    ]
                },
                type="vectorSearch",
            )
        )
        logger.info(
            "MongoDB 向量索引已创建: %s（Atlas 上可能需要约 1 分钟就绪）", index_name
        )

    # 记忆集合暂时不要全文索引
    # Atlas Search 全文索引：对 payload.data 建索引，供混合检索 $search 分支使用
    # text_existing = list(col.list_search_indexes(name=MONGODB_TEXT_INDEX_NAME))
    # if text_existing:
    #     logger.info("MongoDB Atlas Search 全文索引已存在: %s", MONGODB_TEXT_INDEX_NAME)
    # else:
    #     col.create_search_index(
    #         SearchIndexModel(
    #             name=MONGODB_TEXT_INDEX_NAME,
    #             definition={
    #                 "mappings": {
    #                     "dynamic": False,
    #                     "fields": {
    #                         "payload": {
    #                             "type": "document",
    #                             "fields": {
    #                                 "data": {"type": "string", "analyzer": "lucene.standard"},
    #                             },
    #                         }
    #                     },
    #                 }
    #             },
    #             type="search",
    #         )
    #     )
    #     logger.info(
    #         "MongoDB 全文索引已创建: %s（READY 后混合检索可用；约 1 分钟）",
    #         MONGODB_TEXT_INDEX_NAME,
    #     )
    client.close()


def ensure_postgres_resources() -> None:
    """
    确保 PostgreSQL 里已启用 pgvector 扩展。
    - 没配置 POSTGRES_URI 则直接返回
    - 执行 SQL：CREATE EXTENSION IF NOT EXISTS vector;
    - 「IF NOT EXISTS」表示已有则什么都不做，没有才创建
    """
    if not POSTGRES_URI or not POSTGRES_URI.strip():
        # logger.info("POSTGRES_URI 未配置，跳过 PostgreSQL 资源初始化")
        return
    import psycopg2

    conn_str = POSTGRES_URI.strip()
    # 云上的 Postgres（如 Supabase）通常要求 SSL；连接串里没写 sslmode 就自动加上
    if "sslmode=" not in conn_str:
        conn_str = (
            conn_str.rstrip("/")
            + ("?" if "?" not in conn_str else "&")
            + "sslmode=require"
        )

    conn = psycopg2.connect(conn_str)
    conn.autocommit = True  # 执行 CREATE EXTENSION 这类 DDL 不需要事务，直接提交
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.close()
    logger.info("PostgreSQL pgvector 扩展已就绪（已存在则跳过）")


def run_all() -> None:
    """
    执行「全部」初始化：先 MongoDB，再 PostgreSQL。
    应用启动时在 main.py 的 lifespan 里调用一次即可。
    """
    ensure_mongodb_indexes()
    ensure_postgres_resources()
