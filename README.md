# ScholarQA

ScholarQA 是一个面向论文初学者的轻量级 Agentic RAG 系统。用户上传论文 PDF 后，可以围绕论文中的公式、实验、方法、段落和结论进行自然语言问答，也可以逐段翻译、生成总结，并支持多篇论文之间的概览与对比。

系统的核心目标是：像一位耐心的论文阅读助教一样，基于论文原文回答问题，给出页码和段落出处；证据不足时明确说明不确定，避免胡编乱造。

## 核心能力

- **单篇论文问答**：解释公式、定位概念、回答实验和方法相关问题。
- **逐段翻译**：按页码或段落返回原文与中文译文对照，保留公式和引用标记。
- **一键总结**：内置“总结创新点”“梳理实验”“全文摘要”等快捷指令。
- **多篇论文 QA**：选择多篇论文后，可询问“分别讲什么”“对比方法和实验结果”等问题。
- **多轮记忆**：保留最近 10 轮对话，旧对话自动压缩为摘要，用于连续追问和指代消解。
- **混合检索**：ChromaDB 向量检索 + 本地 BM25 + RRF 融合，并支持 parent-child 证据回溯。
- **引用溯源**：回答附带页码、段落号和原文片段，前端可折叠查看。
- **SSE 流式输出**：单篇问答、翻译和总结均支持流式输出；多篇最终答案以 chunk 方式流式返回。

## 技术栈

- Python 3.10+
- FastAPI + Server-Sent Events
- Streamlit
- ChromaDB 本地持久化
- PyMuPDF
- BAAI/bge-m3
- OpenAI 兼容 LLM 接口
- LangGraph，仅用于多篇论文 QA 分支

所有 LLM 配置均从 `.env` 读取，不在代码中硬编码。

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd scholar-rag-main
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

打开 `.env`，至少填写：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
LLM_MODEL=gpt-4o-mini
```

如果使用 DeepSeek、通义、智谱、本地 vLLM 等 OpenAI 兼容接口，只需要把 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 改成对应服务。

### 3. 一键启动

Mac / Linux：

```bash
chmod +x run.sh
./run.sh
```

Windows：

```bat
run.bat
```

启动后访问：

[http://localhost:8501](http://localhost:8501)

左侧上传 PDF，点击“解析并索引”，完成后即可在主界面提问、翻译或进行多篇论文 QA。

## 环境变量

主要配置见 `.env.example`：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048
COMPARE_MAX_TOKENS=4096

EMBEDDING_MODEL=BAAI/bge-m3
HF_ENDPOINT=https://hf-mirror.com
CHROMA_DIR=./chroma_data
CHROMA_COLLECTION=scholarqa_papers_bge_m3
TOP_K=6
RRF_K=60
RERANKER_MODEL=
DEBUG_RETRIEVAL=0

UPLOAD_DIR=./uploads
OUTPUT_DIR=./output
HOST=0.0.0.0
PORT=8000
SCHOLARQA_API_URL=http://127.0.0.1:8000
```

说明：

- `COMPARE_MAX_TOKENS`：多篇论文综合回答更长，单独设置输出上限。
- `RERANKER_MODEL`：可选 CrossEncoder reranker，例如 `BAAI/bge-reranker-base`。
- `DEBUG_RETRIEVAL=1`：返回多篇论文对比时的检索调试信息。
- `HF_ENDPOINT=https://hf-mirror.com`：国内网络环境下下载 HuggingFace 模型更稳定。

## 项目结构

```text
ScholarQA/
├── backend/
│   ├── main.py                 # FastAPI 入口，负责 /chat /translate /summary 路由
│   ├── agent.py                # 单篇论文 Agent：记忆、意图路由、生成器、单篇 RAG
│   ├── multi_paper_graph.py    # 多篇论文 LangGraph 工作流
│   ├── answer_composer.py      # 单篇 QA 的回答 Prompt 组织器
│   ├── pdf_parser.py           # PDF 解析，输出 text/page/section/block_id
│   ├── nodes.py                # parent-child 节点构建
│   ├── retriever.py            # Chroma + BM25 + RRF + parent 回溯
│   ├── prompts.py              # 单篇与多篇 Prompt 模板
│   ├── eval/
│   │   ├── ground_truth.json   # 自定义检索 benchmark 标注
│   │   └── eval_retrieval.py   # Recall@k / Precision@k / MRR 评测脚本
│   └── test/
│       ├── test_answer_quality.py
│       ├── test_multi_paper_compare.py
│       └── test_single_agent_regression.py
├── app.py                      # Streamlit 前端
├── requirements.txt
├── .env.example
├── run.sh
└── run.bat
```

仓库中还保留了原企业版项目的部分目录，例如 `backend/agent/`、`backend/rag/`、`frontend/`。当前轻量版 ScholarQA 的运行入口是 `backend/main.py` 和 `app.py`；保留目录主要用于参考，不参与默认启动链路。

## 系统架构

### 单篇论文流程

```text
Streamlit
  -> /chat
  -> ScholarQAAgent
     -> IntentRouter
        -> qa / translation / summary
     -> QueryRewriter
     -> HybridRetriever
        -> dense search
        -> BM25 search
        -> RRF fusion
        -> optional CrossEncoder rerank
        -> parent chunk expansion
     -> Generator
     -> citations + SSE answer
```

单篇流程不使用 LangGraph，保持简单稳定。

### 多篇论文流程

当 `/chat` 收到 `paper_ids` 且数量大于 1 时，进入 LangGraph 多篇论文分支：

```text
MultiPaperCompareGraph
  -> decompose_query
  -> dispatch_paper_workers
     -> paper_worker × N
  -> prepare_synthesis
  -> synthesize_comparison / overview
```

设计原则：

- 每篇论文一个 `paper_worker`。
- `paper_worker` 只看自己负责的论文，不做跨论文比较。
- `prepare_synthesis` 统一整理证据，并生成 `[P1-C1]` 形式的引用编号。
- 最终由 `synthesize_comparison` 综合生成对比表格或多篇论文导读。

多篇分支有两种模式：

- **compare mode**：适合“对比这几篇论文的方法、实验、局限性”。
- **overview mode**：适合“这三篇分别讲什么”“每篇论文讲什么”。该模式会优先读取每篇论文的 Abstract、Introduction、Method、Experiment、Conclusion 等代表性片段，再生成导读式总结。

## API 简表

### 健康检查

```http
GET /health
```

### 上传论文

```http
POST /upload
```

上传 PDF，解析文本，构建 parent-child 节点，并写入 ChromaDB。

### 单篇问答

```http
POST /chat
Content-Type: application/json

{
  "query": "这篇论文的核心方法是什么？",
  "session_id": "optional-session-id",
  "paper_id": "paper-id"
}
```

### 多篇论文 QA

```http
POST /chat
Content-Type: application/json

{
  "query": "对比这三篇论文的方法和实验结果",
  "session_id": "optional-session-id",
  "paper_ids": ["paper-1", "paper-2", "paper-3"]
}
```

### 逐段翻译

```http
POST /translate
Content-Type: application/json

{
  "paper_id": "paper-id",
  "page": 2,
  "paragraph_index": 3,
  "query": "请翻译第2页第3段"
}
```

### 总结

```http
POST /summary
```

## 运行数据

以下目录为本地运行数据，已加入 `.gitignore`：

- `uploads/`：上传的 PDF
- `chroma_data/`：ChromaDB 本地向量库
- `output/`：后端日志和导出的论文解析结果
- `.env`：本地密钥配置

## 项目定位

ScholarQA 不是通用聊天机器人，而是面向论文阅读场景的轻量 Agentic RAG：

- 以论文原文为唯一事实来源。
- 用检索和引用保证可追溯。
- 用单篇 Agent 解决深度阅读。
- 用 LangGraph 多篇分支解决多论文概览和对比。
- 尽量保持单进程、零部署、易启动。
