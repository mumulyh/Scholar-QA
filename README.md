# ScholarQA

ScholarQA 是一个面向论文阅读的轻量 RAG 系统。用户上传 PDF 后，可以围绕论文中的方法、公式、实验、结论进行问答，也可以逐段翻译、生成总结，并支持多篇论文的概览与对比。

系统目标很简单：**只基于论文原文回答问题，给出可追溯引用；证据不足时明确说明无法确定。**

> 当前 GitHub `main` 分支是核心可运行版本：单篇论文使用 Query-Rewrite RAG 主链路；多篇论文对比使用 LangGraph workflow。此前实验过的完整 Tool-Routing Agentic RAG 模块没有放进这个精简运行版，避免把实验代码和核心代码混在一起。

## 核心能力

- **单篇论文问答**：解释公式、定位概念、回答实验和方法相关问题。
- **逐段翻译**：按页码或段落返回中文译文，尽量保留公式和引用标记。
- **一键总结**：内置“总结创新点”“梳理实验”“全文摘要”等快捷入口。
- **多篇论文 QA**：选择多篇论文后，可询问“分别讲什么”“对比方法和实验结果”等问题。
- **多轮记忆**：保留最近对话，用于连续追问和指代消解。
- **混合检索**：ChromaDB dense 检索 + 本地 BM25 + RRF 融合，并支持 parent-child 证据回溯。
- **可选重排**：通过 `RERANKER_MODEL` 开启 CrossEncoder rerank。
- **引用溯源**：回答附带页码、段落号和原文片段。
- **SSE 流式输出**：问答、翻译、总结均支持流式返回。

## 当前运行链路

### 1. 单篇论文 QA：Query-Rewrite RAG

单篇论文问答不走 LangGraph，保持简单稳定：

```text
Streamlit
  -> FastAPI /chat
  -> ScholarQAAgent
     -> IntentRouter
        -> qa / translation / summary
     -> QueryRewriter
        -> 将用户问题改写成更适合论文检索的 query
        -> 原问题始终保留为兜底 query
     -> HybridRetriever
        -> Dense Search
        -> BM25 Search
        -> RRF Fusion
        -> optional CrossEncoder Rerank
        -> Parent / Neighbor Context Expansion
     -> optional Evidence Reflection
        -> 如果开启 ENABLE_AGENTIC_RETRIEVAL，则检查证据是否足够并补搜
     -> Generator
     -> citations + SSE answer
```

关键点：

- `QueryRewriter` 由 LLM 负责，适合处理中英文表达差异、连续追问和实体消解。
- 原始 query 永远保留，避免 rewrite 失败导致完全搜偏。
- `HybridRetriever` 同时使用向量检索和 BM25，最后用 RRF 融合。
- `ENABLE_AGENTIC_RETRIEVAL=1` 时会开启轻量 evidence reflection，但默认关闭，因为它会增加 LLM 调用和检索耗时。

### 2. 多篇论文 QA：LangGraph Workflow

当 `/chat` 收到多个 `paper_ids` 时，进入多篇论文工作流：

```text
Streamlit
  -> FastAPI /chat
  -> MultiPaperCompareRunner
     -> LangGraph
        -> decompose_query
        -> dispatch_paper_workers
           -> paper_worker × N
        -> prepare_synthesis
        -> synthesize_comparison / overview
     -> citations + SSE answer
```

设计原则：

- 每篇论文一个 `paper_worker`，只检索自己负责的论文。
- `prepare_synthesis` 汇总每篇论文证据，并统一生成 `[P1-C1]` 形式的引用。
- 最终节点根据用户问题生成对比表格、共同点差异点，或多篇论文导读。

### 3. Tool-Routing Agentic RAG 实验设计

完整的 Tool-Routing Agentic RAG 设计用于复杂问题，例如：

- “方法、实验、局限分别是什么？”
- “Lemma 如何推出主定理？”
- “实验如何验证理论结论？”
- “论文是否明确使用了某种方法？”

设计链路如下：

```text
ComplexityRouter
  -> 判断 simple_rag / agentic_rag
QuestionAnalyzer
  -> 保留原问题 q0
  -> 拆 method / experiment / formula / background / limitation 子任务
ToolRouter
  -> 根据 query_type 选择 semantic_search / keyword_search / section_search / neighbor_expand
SubAgent Loop
  -> 每个子任务调用 RAG Core 检索
  -> EvidenceVerifier 判断 supported / partial / missing
  -> 如果 partial / missing，最多 retry
Synthesizer
  -> 合并各子任务证据
  -> 去重、全局重排
  -> 生成带引用的最终回答
```

这个实验设计的核心思想是：复杂论文问题不是一次检索就结束，而是需要 **拆解问题、按证据类型检索、检查证据完整性、必要时补搜**。

当前 `main` 分支保留了核心运行链路和轻量 evidence reflection；完整 Tool-Routing Agentic RAG 模块可以作为后续分支继续合并。

## 技术栈

- Python 3.10+
- FastAPI + Server-Sent Events
- Streamlit
- ChromaDB 本地持久化
- PyMuPDF
- sentence-transformers
- BM25 + RRF
- OpenAI-compatible LLM API
- LangGraph（用于多篇论文 QA workflow）

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/mumulyh/Scholar-QA.git
cd Scholar-QA
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

至少填写：

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

```text
http://localhost:8501
```

左侧上传 PDF，点击“解析并索引”，完成后即可提问、翻译或进行多篇论文 QA。

## 日常开发启动

如果依赖已经装好，不建议每次都跑 `run.sh`，可以分两个终端启动：

后端：

```bash
cd /path/to/Scholar-QA
./venv/bin/python backend/main.py
```

前端：

```bash
cd /path/to/Scholar-QA
./venv/bin/python -m streamlit run app.py --server.port 8501
```

如果要排查检索耗时，可以临时关闭检索前 LLM rewrite：

```bash
ENABLE_QUERY_REWRITE=0 \
ENABLE_AGENTIC_RETRIEVAL=0 \
./venv/bin/python backend/main.py
```

## 环境变量

主要配置见 `.env.example`：

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048

EMBEDDING_MODEL=BAAI/bge-m3
HF_ENDPOINT=https://hf-mirror.com
CHROMA_DIR=./chroma_data
CHROMA_COLLECTION=scholarqa_papers_bge_m3
TOP_K=6
RRF_K=60
RERANKER_MODEL=

ENABLE_QUERY_REWRITE=1
QUERY_REWRITE_MAX_ATTEMPTS=1
QUERY_REWRITE_TIMEOUT_SECONDS=8
ENABLE_AGENTIC_RETRIEVAL=0
RETRIEVAL_REFLECTION_MAX_ATTEMPTS=1
RETRIEVAL_REFLECTION_TIMEOUT_SECONDS=8

UPLOAD_DIR=./uploads
OUTPUT_DIR=./output
HOST=0.0.0.0
PORT=8000
SCHOLARQA_API_URL=http://127.0.0.1:8000
```

说明：

- `EMBEDDING_MODEL`：默认推荐 `BAAI/bge-m3`，也可以换成多语言模型。
- `RERANKER_MODEL`：为空则不开 CrossEncoder；设置模型名后开启 rerank。
- `ENABLE_QUERY_REWRITE=0`：最快检索模式，跳过 LLM query rewrite。
- `ENABLE_AGENTIC_RETRIEVAL=1`：开启轻量 evidence reflection，会增加 LLM 调用和耗时。
- `QUERY_REWRITE_TIMEOUT_SECONDS`：限制 rewrite 等待时间，避免检索阶段被 LLM 卡住。

## 项目结构

```text
Scholar-QA/
├── backend/
│   ├── main.py                # FastAPI 入口，/chat /upload /translate /summary
│   ├── agent.py               # 单篇 QA、QueryRewriter、Memory、Generator
│   ├── retriever.py           # Dense + BM25 + RRF + optional CrossEncoder
│   ├── pdf_parser.py          # PDF 解析
│   ├── nodes.py               # parent-child chunk 构建
│   ├── answer_composer.py     # QA prompt 组织
│   ├── multi_paper_graph.py   # 多篇论文 LangGraph workflow
│   ├── prompts.py             # Prompt 模板
│   ├── config.py              # 环境变量配置
│   └── requirements.txt
├── app.py                     # Streamlit 前端
├── requirements.txt
├── .env.example
├── run.sh
└── run.bat
```

## API 简表

### 健康检查

```http
GET /health
```

### 上传论文

```http
POST /upload
```

上传 PDF，解析文本，构建 parent-child chunks，并写入 ChromaDB。

### 问答

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

## 本地运行数据

以下目录不会上传 GitHub：

- `.env`：本地密钥配置
- `uploads/`：上传的 PDF
- `chroma_data/`：ChromaDB 本地向量库
- `output/`：日志与解析结果

## 项目定位

ScholarQA 不是通用聊天机器人，而是面向论文阅读场景的 RAG 系统：

- 以论文原文为唯一事实来源。
- 用检索和引用保证可追溯。
- 简单问题走稳定普通 RAG。
- 多篇论文走 LangGraph workflow。
- 复杂问题的 Tool-Routing Agentic RAG 是后续可合并的增强方向。
