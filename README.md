# Agent Memory Demo

基于多种记忆方案与 RAG 的 Agent 记忆能力演示项目，支持 Azure OpenAI、Voyage 嵌入与 MongoDB 持久化。

## 技术栈

- **Python 3.12** · **Jupyter**
- **LLM / 嵌入**：Azure OpenAI（Chat）、Voyage AI（Embeddings）
- **存储**：MongoDB（向量 + 元数据）、LangGraph Store / Checkpoint
- **记忆框架**：mem0、LangMem（LangGraph）、Memory Bank、Memorizz

## 项目结构

```
agent_memory_demo/
├── .env.sample          # 环境变量模板（复制为 .env 并填写）
├── README.md
├── mem0/                # mem0 记忆 + MongoDB
│   ├── memory_augmented_agent_with_mem0_mongodb.ipynb
│   └── demo.ipynb
├── langmem/             # LangMem (LangGraph) 记忆 + MongoDB
│   ├── memory_augmented_agent_with_mongodb.ipynb
│   ├── memory_augmented_agent_with_mongodb_zh.ipynb
│   └── test.ipynb
├── memorizz/            # Memorizz 多智能体与知识库
│   ├── knowledge_base.ipynb
│   ├── memagent_single_agent.ipynb
│   ├── memagent_summarisation.ipynb
│   ├── memagents_multi_agents.ipynb
│   ├── persona.ipynb
│   ├── toolbox.ipynb
│   └── workflow.ipynb
├── memory_bank/         # 本地记忆 Agent
│   └── memory_augmented_agent_with_local_memory.ipynb
├── information_retrieval/  # RAG / 检索增强
│   ├── zero_to_hero_with_genai_with_mongodb_azure_openai.ipynb
│   └── zero_to_hero_with_genai_with_mongodb_openai.ipynb
└── utilities/           # 公共工具
    └── pdf_chunker.py   # PDF 下载、分块，供语义记忆摄入
```

## 环境配置

### 1. 克隆与虚拟环境

```bash
cd agent_memory_demo
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. 环境变量

复制模板并填写密钥与连接串（勿提交 `.env`）：

```bash
cp .env.sample .env
```

`.env` 中需配置：

| 变量 | 说明 |
|------|------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI 端点，如 `https://xxx.openai.azure.com/` |
| `OPENAI_API_KEY` | Azure OpenAI API Key（与 endpoint 对应） |
| `VOYAGE_API_KEY` | Voyage AI API Key（用于嵌入） |
| `MONGODB_URI` | MongoDB 连接串（Atlas 或自建） |

### 3. 安装依赖

各 notebook 内通常含 `%pip install ...`，按需运行即可。常用依赖示例：

```bash
pip install -U mem0ai langmem langgraph langchain-voyageai langgraph-checkpoint-mongodb langgraph-store-mongodb pymongo openai python-dotenv azure-identity
pip install -U langchain-openai langchain-community pypdf requests
```

## 使用说明

### mem0 + MongoDB

- **Notebook**：`mem0/memory_augmented_agent_with_mem0_mongodb.ipynb`
- 使用 **mem0** 的 `Memory`，向量存 MongoDB，嵌入为 **Voyage**，LLM 为 **Azure OpenAI**。
- 含 PDF 摄入语义记忆、对话记忆（`chat_with_memories`）及索引维度说明（1024 / 1536）。
- 若遇 “Index does not exist”，先运行「确保向量索引存在」的 cell，再运行 `Memory.from_config`。

### LangMem + MongoDB

- **Notebook**：`langmem/memory_augmented_agent_with_mongodb_zh.ipynb`（或同目录下其他版本）
- 使用 **LangGraph** + **LangMem** 的 manage/search memory 工具，存储为 **MongoDBStore**，嵌入为 **Voyage**。
- 适合与 LangGraph Agent 集成、做对话与知识记忆。

### Information Retrieval（RAG）

- **Notebook**：`information_retrieval/zero_to_hero_with_genai_with_mongodb_azure_openai.ipynb`
- RAG 流程：文档加载、分块、嵌入、存 MongoDB，检索后调用 Azure OpenAI 生成回答。

### 其他

- **memory_bank**：本地记忆增强 Agent 示例。
- **memorizz**：多智能体、知识库、persona、工作流等示例，可按 notebook 名称自选运行。

### 公共工具

- **`utilities/pdf_chunker.py`**：`ingest_pdf_and_chunk(url)` 从 URL 下载 PDF、分块，返回适合写入语义记忆的列表。在 mem0 等 notebook 中通过 `sys.path` 或项目根引用。

## 常见问题

- **401 / Incorrect API key**：确认使用 Azure 时，LLM 与嵌入均配置为 Azure 或 Voyage，不要用同一 key 调 OpenAI 官方 API。
- **向量维度 1536 vs 1024**：若曾用 OpenAI 嵌入（1536）建过 MongoDB 向量索引，改用 Voyage（1024）时需删除旧索引并重建，或运行 notebook 中「删除旧索引 / 确保索引存在」的 cell。
- **记不住对话**：mem0 示例中 `chat_with_memories` 默认使用 `CURRENT_USER_ID`，保证写入与检索为同一 `user_id`；若写入失败会打印 Warning。

## License

本仓库为演示与学习用途，按项目原有许可或 MIT 使用即可。
