"""FastAPI entry point for the lightweight ScholarQA backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

# embedding 模型会读取 HuggingFace 环境变量，所以要先加载 .env。
from agent import Generator, IntentRouter, MemoryManager, ScholarQAAgent
from pdf_parser import PDFParser
from retriever import ChromaRetriever, PaperLibrary


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


UPLOAD_DIR = _env_path("UPLOAD_DIR", PROJECT_ROOT / "uploads")
OUTPUT_DIR = _env_path("OUTPUT_DIR", PROJECT_ROOT / "output")
CHROMA_DIR = _env_path("CHROMA_DIR", PROJECT_ROOT / "chroma_data")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "scholarqa_papers")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)
TOP_K = int(os.getenv("TOP_K", "6"))
RRF_K = int(os.getenv("RRF_K", "60"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "").strip() or None
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pdf_parser = PDFParser()
paper_library = PaperLibrary()
retriever = ChromaRetriever(
    persist_dir=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_model_name=EMBEDDING_MODEL,
    rrf_k=RRF_K,
    reranker_model_name=RERANKER_MODEL,
)
memory_manager = MemoryManager(window_size=10)
intent_router = IntentRouter()
generator = Generator()
scholarqa_agent = ScholarQAAgent(
    retriever=retriever,
    paper_library=paper_library,
    memory_manager=memory_manager,
    intent_router=intent_router,
    generator=generator,
    top_k=TOP_K,
)
multi_paper_compare_runner = None

app = FastAPI(title="ScholarQA", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request body for streamed chat."""

    query: str = Field(..., min_length=1)
    session_id: str | None = None
    paper_id: str | None = None
    paper_ids: list[str] | None = None


class TranslateRequest(BaseModel):
    """Request body for streamed translation."""

    paper_id: str
    session_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    paragraph_index: int | None = Field(default=None, ge=1)
    block_id: str | None = None
    query: str = "请翻译指定段落"


class SummaryRequest(BaseModel):
    """Request body for streamed summary presets."""

    instruction: str = "请生成这篇论文的全文摘要，覆盖研究问题、方法、实验和结论。"
    session_id: str | None = None
    paper_id: str | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return backend health and lightweight runtime configuration.

    Returns:
        Service status for the Streamlit frontend.
    """
    return {
        "status": "ok",
        "papers": len(paper_library.list_papers()),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model_configured": bool(os.getenv("LLM_MODEL")),
    }


@app.get("/papers")
async def list_papers() -> dict[str, list[dict[str, Any]]]:
    """List uploaded papers.

    Returns:
        Uploaded paper summaries.
    """
    return {"papers": paper_library.list_papers()}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload, parse, and index a PDF.

    Args:
        file: Uploaded PDF file.

    Returns:
        Paper metadata after indexing.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件。")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="PDF 文件为空。")

    paper_id = hashlib.sha256(content).hexdigest()[:16]
    safe_filename = _safe_filename(file.filename)
    file_path = UPLOAD_DIR / f"{paper_id}_{safe_filename}"

    await _write_bytes(file_path, content)
    chunks = await pdf_parser.parse(file_path, paper_id)
    if not chunks:
        raise HTTPException(status_code=400, detail="无法从 PDF 中解析出文本。")

    # 先写入内存库，再写入 Chroma，确保上传后立即可检索、可翻译。
    paper_library.upsert(paper_id=paper_id, filename=file.filename, chunks=chunks)
    await retriever.add_document(
        paper_id=paper_id,
        chunks=paper_library.get_child_nodes(paper_id),
    )
    retriever.register_parent_nodes(paper_library.get_parent_nodes(paper_id))
    await paper_library.export_json(OUTPUT_DIR / "papers.json")

    return {
        "paper_id": paper_id,
        "filename": file.filename,
        "chunk_count": len(chunks),
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """Stream a routed chat answer through SSE.

    Args:
        request: Chat payload containing query, session id, and paper id.

    Returns:
        Server-Sent Events response.
    """
    session_id = request.session_id or str(uuid.uuid4())
    paper_ids = _normalize_paper_ids(request.paper_id, request.paper_ids)

    # 多篇对比是新增旁路：只有显式传入多篇 paper_ids 时才进入 LangGraph。
    if len(paper_ids) > 1:
        try:
            compare_runner = _get_multi_paper_compare_runner()
            event_source = compare_runner.stream_compare(
                session_id=session_id,
                query=request.query,
                paper_ids=paper_ids,
            )
        except RuntimeError as exc:
            event_source = _friendly_stream(str(exc))
    elif request.paper_ids is not None and intent_router.is_compare_query(request.query):
        event_source = _friendly_stream(
            "请至少选择两篇论文进行多篇对比。"
        )
    else:
        event_source = scholarqa_agent.stream_chat(
            session_id=session_id,
            query=request.query,
            paper_id=paper_ids[0] if paper_ids else request.paper_id,
        )

    return EventSourceResponse(
        _stream_events(
            event_source,
            session_id=session_id,
        ),
        media_type="text/event-stream",
    )


@app.post("/translate")
async def translate(request: TranslateRequest) -> EventSourceResponse:
    """Stream bilingual paragraph translation through SSE.

    Args:
        request: Translation target payload.

    Returns:
        Server-Sent Events response.
    """
    session_id = request.session_id or str(uuid.uuid4())
    return EventSourceResponse(
        _stream_events(
            scholarqa_agent.stream_translation_request(
                session_id=session_id,
                query=request.query,
                paper_id=request.paper_id,
                page=request.page,
                paragraph_index=request.paragraph_index,
                block_id=request.block_id,
            ),
            session_id=session_id,
        ),
        media_type="text/event-stream",
    )


@app.post("/summary")
async def summary(request: SummaryRequest) -> EventSourceResponse:
    """Stream a paper summary through SSE.

    Args:
        request: Summary instruction, session id, and optional paper id.

    Returns:
        Server-Sent Events response.
    """
    session_id = request.session_id or str(uuid.uuid4())
    return EventSourceResponse(
        _stream_events(
            scholarqa_agent.stream_chat(
                session_id=session_id,
                query=request.instruction,
                paper_id=request.paper_id,
            ),
            session_id=session_id,
        ),
        media_type="text/event-stream",
    )


async def _stream_events(
    events: AsyncGenerator[dict[str, Any], None],
    session_id: str,
) -> AsyncGenerator[str, None]:
    yield _json_event("session_id", session_id)
    try:
        async for event in events:
            yield _json_event(event["type"], event.get("data"))
    except Exception as exc:
        yield _json_event("error", str(exc))


def _json_event(event_type: str, data: Any) -> str:
    return json.dumps({"type": event_type, "data": data}, ensure_ascii=False)


def _normalize_paper_ids(
    paper_id: str | None,
    paper_ids: list[str] | None,
) -> list[str]:
    """Normalize old paper_id and new paper_ids fields.

    Args:
        paper_id: Backward-compatible single paper id.
        paper_ids: Optional multi-paper ids.

    Returns:
        Deduplicated paper ids preserving user selection order.
    """
    raw_paper_ids = (
        paper_ids
        if paper_ids is not None
        else ([paper_id] if paper_id else [])
    )
    return list(dict.fromkeys(pid for pid in raw_paper_ids if pid))


def _get_multi_paper_compare_runner():
    """Lazy-load the LangGraph compare runner only for multi-paper requests."""
    global multi_paper_compare_runner
    if multi_paper_compare_runner is None:
        from multi_paper_graph import MultiPaperCompareRunner

        multi_paper_compare_runner = MultiPaperCompareRunner(
            retriever=retriever,
            paper_library=paper_library,
            memory_manager=memory_manager,
            generator=generator,
            top_k=TOP_K,
        )
    return multi_paper_compare_runner


async def _friendly_stream(
    message: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream a short friendly answer without invoking retrieval.

    Args:
        message: Message shown to the user.

    Yields:
        SSE-ready event dictionaries.
    """
    yield {"type": "intent", "data": "compare"}
    yield {"type": "answer", "data": message}
    yield {"type": "done", "data": None}


async def _write_bytes(file_path: Path, content: bytes) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(file_path.write_bytes, content)


def _safe_filename(filename: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    return sanitized[:120] or "paper.pdf"


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
