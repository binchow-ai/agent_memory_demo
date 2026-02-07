# 开放世界 NPC 对话记忆测试

业务场景：**开放世界游戏中 NPC 与玩家的对话**。  
本应用用于**对比 mem0 在 MongoDB Atlas 与 PostgreSQL（pgvector）上，对「玩家–NPC」对话记忆的存储与召回表现**。  
同一套流程：按玩家（及可选 NPC）检索记忆 → NPC 生成回复 → 将本轮对话写入记忆；仅切换向量库后端，便于观察召回质量与一致性差异。

## 资源与环境

- 环境变量见项目根目录 `.env`（本应用会优先读取 `agent_memory_demo/.env`）。  
- 若在其它目录运行后端，可在 `backend/` 下放 `.env` 或复制根目录 `.env`。  
- 所需变量见 `backend/.env.example`。

## 目录结构

```
mem0_memory_test_app/
├── backend/           # FastAPI + mem0（MongoDB / PostgreSQL）
│   ├── config.py      # 环境变量与配置
│   ├── db_init.py     # 启动时自动创建库/表/索引（存在则跳过）
│   ├── memory_backends.py  # mem0 双后端封装
│   ├── main.py        # 聊天与记忆搜索 API
│   ├── requirements.txt
│   └── .env.example
├── frontend/          # 静态前端
│   └── index.html     # 聊天页（可切换后端、玩家 ID、当前 NPC）
└── README.md
```

## 库表与索引（自动创建）

应用**启动时**会自动初始化存储资源，无需手动建库建表：

- **MongoDB Atlas**：若配置了 `MONGODB_URI`，会确保数据库 `mem0_agent_memory`、集合 `extracted_memories` 存在，并为该集合创建向量搜索索引（维度与 Voyage 一致）；若索引已存在则跳过。Atlas 上新建索引可能需要约 1 分钟才就绪。
- **PostgreSQL**：若配置了 `POSTGRES_URI`，会执行 `CREATE EXTENSION IF NOT EXISTS vector;`，确保 pgvector 可用。mem0 会按配置的 `collection_name` 在首次写入时自动建表。

## 本地运行

### 1. 后端（必须）

**方式一（推荐，使用项目根 venv）**：若项目根目录已有 `venv` 且已安装 mem0ai 等依赖，可直接：

```bash
cd mem0_memory_test_app/backend
./run.sh
```

`run.sh` 会自动使用 `agent_memory_demo/venv` 并设置 `PYTHONPATH`，读取根目录 `.env`。

**方式二（本目录独立 venv）**：

```bash
cd mem0_memory_test_app/backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API 文档：<http://127.0.0.1:8000/docs>

### 2. 前端

用本地静态服务器打开 `frontend/index.html`，例如：

```bash
cd mem0_memory_test_app/frontend
python -m http.server 5500
```

浏览器访问：<http://127.0.0.1:5500>（默认会请求 `http://127.0.0.1:8000` 作为 API）。

或用 VS Code / Cursor 的 “Open with Live Server” 打开 `frontend/index.html`，若端口不是 8000，前端会尽量根据当前页的 origin 推断 API 地址（见 `index.html` 内 `API_BASE`）。

## 使用方式

1. **选择后端**：页面上切换「MongoDB Atlas」或「PostgreSQL」。
2. **选择 LLM**：页面上切换「DeepSeek」或「Azure OpenAI」，用于生成 NPC 回复（记忆抽取仍由 mem0 内配置的 DeepSeek 完成）。
3. **玩家 ID**：同一玩家 ID 下的对话共用记忆；改 ID 可模拟不同玩家。
4. **当前 NPC**（可选）：填写后，记忆按「该玩家 + 该 NPC」隔离，不同 NPC 拥有各自与这名玩家的记忆；留空则不分 NPC，所有对话记在同一玩家下。
5. **发消息**：以玩家身份输入，NPC 会结合召回的记忆回复；回复下方会显示「本轮 NPC 召回的记忆」。
6. **对比测试**：  
   - 同一玩家 ID、同一 NPC 下，先用 MongoDB 聊几轮（如告诉 NPC 自己的名字、喜好、任务选择），再问“你还记得我是谁吗？”“我之前说过喜欢什么？”观察召回。  
   - 切到 PostgreSQL，同样玩家 + NPC 再聊几轮并提问，对比两种后端的记忆一致性与召回效果。

## API 说明

- **POST /chat**  
  - Body: `{ "message": "玩家输入", "user_id": "player-1", "npc_id": "villager-01" 或 null, "backend": "mongodb" | "postgres", "llm_provider": "deepseek" | "azure_openai" }`  
  - 返回: `{ "reply", "recalled_memories", "backend", "llm_provider" }`  
  - 行为：按玩家（及可选 `npc_id`）检索记忆 → 按 `llm_provider` 用 DeepSeek 或 Azure OpenAI 生成 NPC 回复 → 将本轮对话写入 mem0。

- **POST /memories/search**  
  - Body: `{ "query": "检索语句", "user_id": "player-1", "npc_id": "villager-01" 或 null, "backend": "mongodb" | "postgres", "limit": 5 }`  
  - 用于直接测试记忆检索（不生成回复）。

- **GET /health**  
  - 健康检查。

## 依赖说明

- **MongoDB**：使用 `.env` 中的 `MONGODB_URI`；启动时会自动创建向量索引（存在则跳过）。  
- **PostgreSQL**：使用 `POSTGRES_URI`（如 Supabase）；启动时会自动执行 `CREATE EXTENSION IF NOT EXISTS vector;`，表由 mem0 在首次写入时创建。  
- **LLM / Embedding**：Azure OpenAI（gpt-4o）+ Voyage（voyage-4-large，1024 维），与现有 mem0 配置一致。

## 可选：只测某一后端

若暂时只连 MongoDB 或只连 PostgreSQL，保留对应 URI 即可；未配置的后端在首次切换时会报错，可按错误信息检查环境变量或网络。
