# 开放世界 NPC 对话记忆测试

基于 **mem0** 的开放世界 NPC 对话记忆 Demo：支持长期/短期记忆、过程记忆，可切换 **MongoDB** 或 **PostgreSQL（pgvector）** 作为向量库，便于对比不同后端的存储与召回表现。

---

## 功能概览

- **对话记忆**：玩家与 NPC 对话 → 自动召回记忆、生成回复、按内容判定写入长期或短期记忆；备注中展示「系统判定为短期/长期记忆」与「本轮 NPC 召回的记忆」。
- **记忆管理**：按关键词搜索记忆、按玩家 ID 获取/删除该玩家（及可选 NPC）下的全部记忆；右侧面板实时展示服务端日志（所有 Tab 共用）。
- **短期与过程记忆**：  
  - **短期（按会话 session_id）**：会话级记忆，可获取、清空当前会话。  
  - **短期（按过期时间）**：设置 N 天后过期，适合临时提醒。  
  - **过程记忆**：步骤、流程类内容（mem0 `procedural_memory`）。

---

## 记忆类型说明（Mem0）

| 类型 | 区分方式 | 适用场景 | 存储/检索 |
|------|----------|----------|------------|
| **长期记忆** | `user_id`，不设过期 | 用户偏好、账户信息、重要事实 | 永久保存，对话按 `user_id` + `agent_id` 检索 |
| **短期记忆（会话）** | `session_id`（mem0 内部用 `run_id`） | 当前会话上下文、多步任务 | 会话结束可清空，对话时按 `session_id` 一并检索 |
| **短期记忆（过期）** | `expiration_date`（如 7 天后） | 临时提醒 | 过期后不再被检索 |
| **过程记忆** | `memory_type=procedural_memory` | 步骤、流程、操作说明 | 与事实记忆分开存储，便于按「怎么做」召回 |

对话时：系统根据本轮内容用 LLM 判定写入**长期**还是**短期（session）**；若请求带 `session_id` 且判定为短期，则写入带 `run_id` 的会话记忆，否则写入长期。

---

## 技术栈

- **后端**：FastAPI、mem0（OSS）、Voyage 嵌入、DeepSeek / Azure OpenAI（NPC 回复与长短记忆分类）
- **向量库**：MongoDB Atlas 或 PostgreSQL（pgvector）
- **前端**：单页 HTML + 原生 JS，左侧内容区 + 右侧可拖拽日志面板

---

## 目录结构

```
mem0_memory_test_app/
├── backend/
│   ├── config.py           # 环境变量与配置
│   ├── db_init.py          # 启动时创建库/表/索引（存在则跳过）
│   ├── memory_backends.py  # mem0 双后端封装（MongoDB / PostgreSQL）
│   ├── main.py             # 对话、记忆管理、日志流等 API
│   ├── requirements.txt
│   ├── run.sh              # 推荐启动方式
│   └── .env.example        # 环境变量示例
├── frontend/
│   └── index.html          # 三 Tab 界面 + 日志面板
└── README.md
```

---

## 环境变量

在项目根目录或 `backend/` 下放置 `.env`，参考 `backend/.env.example`：

| 变量 | 说明 |
|------|------|
| `MONGODB_URI` | MongoDB 连接串（至少配一个向量库） |
| `POSTGRES_URI` | PostgreSQL 连接串 |
| `VOYAGE_API_KEY` | Voyage 嵌入（mem0 用） |
| `DEEPSEEK_API_KEY` | DeepSeek（NPC 回复 + 长短记忆分类） |
| `DEEPSEEK_MODEL` | 可选，默认 `deepseek-chat` |
| `AZURE_OPENAI_ENDPOINT` | 可选，用 Azure 做 NPC 回复时必填 |
| `OPENAI_API_KEY` | 可选，Azure 时必填 |
| `AZURE_DEPLOYMENT` / `AZURE_API_VERSION` | 可选，默认 gpt-4o 等 |

---

## 本地运行

### 1. 后端（必须）

```bash
cd mem0_memory_test_app/backend
./run.sh
```

或手动：

```bash
cd mem0_memory_test_app/backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API 文档：<http://127.0.0.1:8000/docs>

### 2. 前端

```bash
cd mem0_memory_test_app/frontend
python -m http.server 5500
```

浏览器打开 <http://127.0.0.1:5500>，默认请求 `http://127.0.0.1:8000` 作为 API。

---

## 使用说明

### 页头

- **记忆后端**：MongoDB / PostgreSQL 切换。
- **NPC 回复 LLM**：DeepSeek / Azure OpenAI。
- **玩家 ID**：同一 ID 下记忆共用；默认 `player-1`。
- **当前 NPC**：填写则记忆按「玩家 + 该 NPC」隔离；留空则使用默认 `npc-default`，与「添加短期/过程记忆」时的默认一致，便于对话召回。

### Tab 1：对话记忆

- 输入消息发送 → 后端召回长期 + 当前会话短期记忆 → 生成 NPC 回复 → 判定本轮为长期/短期并写入。
- 每条 NPC 回复下方显示：「系统判定为短期记忆，本轮 NPC 召回的记忆」或「系统判定为长期记忆，本轮 NPC 召回的记忆」，以及召回内容列表。

### Tab 2：记忆管理

- **搜索记忆**：关键词 + 条数，按当前玩家/NPC 搜索。
- **按玩家 ID 获取记忆**：可选 NPC、条数，列出该范围记忆。
- **按玩家 ID 删除记忆**：清空该玩家（及可选 NPC）下全部记忆。
- 右侧为服务端日志（SSE 实时，所有 Tab 共用）。

### Tab 3：短期与过程记忆

- **短期记忆（按会话 session_id）**：输入内容 + 会话 ID（留空用当前页会话）→ 添加；可「获取当前会话记忆」「清空当前会话记忆」。
- **短期记忆（按过期时间）**：输入内容 + 过期天数（如 7）→ 添加，N 天后自动失效。
- **过程记忆**：输入步骤/流程文本 → 添加，类型为 `procedural_memory`。
- 添加时均会带上当前玩家 ID 与 `agent_id`（与对话默认一致），保证对话能召回。

---

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 对话：召回记忆、生成回复、判定长短记忆并写入。Body 含 `message`, `user_id`, `npc_id`, `session_id`(可选), `backend`, `llm_provider`。返回 `reply`, `recalled_memories`, `memory_type_this_turn`, `backend`, `llm_provider`。 |
| GET | `/health` | 健康检查。 |
| GET | `/logs/stream` | SSE 服务端日志流。 |
| POST | `/memory/search` | 语义搜索。Body: `query`, `user_id`, `agent_id`, `backend`, `limit`。 |
| GET | `/memory/by-user` | 按玩家（及可选 `agent_id`）列出记忆。Query: `user_id`, `agent_id`, `backend`, `limit`。 |
| GET | `/memory/by-session` | 按会话列出短期记忆。Query: `session_id`, `user_id`, `backend`, `limit`。 |
| GET | `/memory/{memory_id}` | 按 ID 获取单条。Query: `backend`。 |
| PATCH | `/memory/{memory_id}` | 按 ID 更新内容。Body: `new_content`。Query: `backend`。 |
| DELETE | `/memory/{memory_id}` | 按 ID 删除单条。Query: `backend`。 |
| DELETE | `/memory` | 按范围删除：Query `user_id`, `agent_id` 或 `session_id`（清空该会话）。 |
| POST | `/memory/add` | 添加记忆。Body: `content`, `user_id`, `agent_id`, `backend`, 以及可选 `session_id` / `expiration_days` / `memory_type`。 |

---

## 库表与索引（自动）

- **MongoDB**：数据库 `mem0_agent_memory`、集合 `extracted_memories`，启动时检查并创建向量搜索索引（维度与 Voyage 一致）；新索引可能需约 1 分钟就绪。
- **PostgreSQL**：启动时执行 `CREATE EXTENSION IF NOT EXISTS vector;`，mem0 在首次写入时按配置建表。

若只测一种后端，仅配置对应 `MONGODB_URI` 或 `POSTGRES_URI` 即可。

---

## 依赖（backend）

- mem0ai、langchain-voyageai、pymongo、pgvector、psycopg2-binary  
- fastapi、uvicorn、openai、python-dotenv  

见 `backend/requirements.txt`。
