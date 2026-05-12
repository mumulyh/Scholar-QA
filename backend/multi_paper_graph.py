"""LangGraph workflow for multi-paper comparison in ScholarQA."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Annotated, Any, AsyncGenerator, TypedDict

from openai import AsyncOpenAI

from prompts import (
    SYSTEM_PROMPT_COMPARE_SYNTHESIS,
    SYSTEM_PROMPT_DECOMPOSE,
    SYSTEM_PROMPT_MULTI_PAPER_OVERVIEW,
    SYSTEM_PROMPT_PAPER_WORKER,
)
from retriever import ChromaRetriever, PaperLibrary

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send
except ImportError as exc:  # pragma: no cover - exercised only without dependency.
    raise RuntimeError(
        "多篇论文对比需要安装 langgraph，请先运行 pip install -r requirements.txt。"
    ) from exc


logger = logging.getLogger(__name__)

DEFAULT_COMPARISON_DIMENSIONS = [
    "研究问题",
    "核心方法",
    "数据集/实验设置",
    "评价指标",
    "实验结果",
    "局限性",
]
OVERVIEW_DIMENSIONS = [
    "论文概览",
]


def merge_worker_results(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge parallel paper worker outputs.

    Args:
        left: Existing accumulated worker results.
        right: Newly produced worker results.

    Returns:
        Ordered concatenation of worker results.
    """
    return [*(left or []), *(right or [])]


class MultiPaperCompareState(TypedDict, total=False):
    """State passed through the multi-paper comparison graph."""

    query: str
    rewritten_query: str
    paper_ids: list[str]
    paper_id: str
    session_id: str
    memory_context: str
    chat_history: list[dict[str, str]]
    paper_titles: dict[str, str]
    answer_mode: str
    comparison_dimensions: list[str]
    worker_results: Annotated[list[dict[str, Any]], merge_worker_results]
    evidence_blocks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    final_answer: str
    debug_retrieval: dict[str, Any] | None
    error: str | None


class MultiPaperLLM:
    """Small OpenAI-compatible helper used only by the compare graph."""

    def __init__(self) -> None:
        """Initialize from the same environment variables as the single agent."""
        self._base_url = os.getenv("LLM_BASE_URL")
        self._api_key = os.getenv("LLM_API_KEY")
        self._model = os.getenv("LLM_MODEL")
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self._client: AsyncOpenAI | None = None

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 1200,
    ) -> str:
        """Run a non-streaming completion.

        Args:
            system_prompt: System instruction.
            user_prompt: User content.
            temperature: Optional temperature override.
            max_tokens: Maximum tokens to generate.

        Returns:
            Model response text.
        """
        response = await self._client_or_raise().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def _client_or_raise(self) -> AsyncOpenAI:
        if not self._base_url or not self._api_key or not self._model:
            raise RuntimeError(
                "LLM_BASE_URL、LLM_API_KEY、LLM_MODEL 必须在 .env 中配置。"
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client


def build_multi_paper_compare_graph(
    *,
    retriever: ChromaRetriever,
    paper_library: PaperLibrary,
    llm: MultiPaperLLM,
    top_k: int = 5,
):
    """Build the LangGraph workflow for compare-only routing.

    Args:
        retriever: Existing ScholarQA hybrid retriever.
        paper_library: In-memory paper registry.
        llm: OpenAI-compatible helper for graph nodes.
        top_k: Retrieval count for each dimension query.

    Returns:
        A compiled LangGraph graph.
    """

    async def decompose_query(state: MultiPaperCompareState) -> dict[str, Any]:
        return await _decompose_query(state, llm)

    async def paper_worker(state: MultiPaperCompareState) -> dict[str, Any]:
        return await _paper_worker(
            state=state,
            retriever=retriever,
            paper_library=paper_library,
            llm=llm,
            top_k=top_k,
        )

    def dispatch_paper_workers(
        state: MultiPaperCompareState,
    ) -> dict[str, Any]:
        logger.info(
            "MultiPaperCompare dispatch: papers=%s dimensions=%s",
            len(state.get("paper_ids", [])),
            state.get("comparison_dimensions", []),
        )
        return {}

    def send_paper_workers(state: MultiPaperCompareState) -> list[Send]:
        return [
            Send(
                "paper_worker",
                {
                    "query": state["query"],
                    "rewritten_query": state.get("rewritten_query", ""),
                    "paper_id": paper_id,
                    "paper_ids": state["paper_ids"],
                    "paper_titles": state.get("paper_titles", {}),
                    "comparison_dimensions": state["comparison_dimensions"],
                    "memory_context": state.get("memory_context", ""),
                    "session_id": state.get("session_id", ""),
                },
            )
            for paper_id in state.get("paper_ids", [])
        ]

    def prepare_synthesis(state: MultiPaperCompareState) -> dict[str, Any]:
        return _prepare_synthesis(state, paper_library)

    async def synthesize_comparison(
        state: MultiPaperCompareState,
    ) -> dict[str, Any]:
        return await _synthesize_comparison(state, llm)

    graph = StateGraph(MultiPaperCompareState)
    graph.add_node("decompose_query", decompose_query)
    graph.add_node("dispatch_paper_workers", dispatch_paper_workers)
    graph.add_node("paper_worker", paper_worker)
    graph.add_node("prepare_synthesis", prepare_synthesis)
    graph.add_node("synthesize_comparison", synthesize_comparison)

    graph.add_edge(START, "decompose_query")
    graph.add_edge("decompose_query", "dispatch_paper_workers")
    graph.add_conditional_edges(
        "dispatch_paper_workers",
        send_paper_workers,
        ["paper_worker"],
    )
    graph.add_edge("paper_worker", "prepare_synthesis")
    graph.add_edge("prepare_synthesis", "synthesize_comparison")
    graph.add_edge("synthesize_comparison", END)

    return graph.compile()


class MultiPaperCompareRunner:
    """SSE-friendly runner for the compare-only LangGraph workflow."""

    def __init__(
        self,
        *,
        retriever: ChromaRetriever,
        paper_library: PaperLibrary,
        memory_manager: Any,
        generator: Any,
        top_k: int = 5,
    ) -> None:
        """Initialize the runner without touching the single-paper agent.

        Args:
            retriever: Existing ScholarQA retriever.
            paper_library: Existing in-memory paper library.
            memory_manager: Existing memory manager.
            generator: Existing generator, reused for query rewriting and memory.
            top_k: Retrieval count per dimension.
        """
        self._retriever = retriever
        self._paper_library = paper_library
        self._memory_manager = memory_manager
        self._generator = generator
        self._llm = MultiPaperLLM()
        self._graph = build_multi_paper_compare_graph(
            retriever=retriever,
            paper_library=paper_library,
            llm=self._llm,
            top_k=top_k,
        )

    async def stream_compare(
        self,
        *,
        session_id: str,
        query: str,
        paper_ids: list[str],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream multi-paper comparison events.

        Args:
            session_id: Conversation session id.
            query: User comparison question.
            paper_ids: Selected paper ids.

        Yields:
            SSE-ready event dictionaries.
        """
        selected_paper_ids = list(dict.fromkeys(pid for pid in paper_ids if pid))
        paper_titles = _paper_title_map(self._paper_library)
        valid_paper_ids = [pid for pid in selected_paper_ids if pid in paper_titles]

        if len(valid_paper_ids) < 2:
            yield {
                "type": "answer",
                "data": "请至少选择两篇论文进行多篇对比。",
            }
            yield {"type": "done", "data": None}
            return

        memory_context = self._memory_manager.build_context(session_id)
        yield {"type": "intent", "data": "compare"}
        yield {"type": "status", "data": "decomposing"}

        rewritten_queries = await self._generator.rewrite_retrieval_queries(
            query,
            memory_context,
        )
        rewritten_query = " ".join(rewritten_queries[:3])

        graph_input: MultiPaperCompareState = {
            "query": query,
            "rewritten_query": rewritten_query,
            "paper_ids": valid_paper_ids,
            "session_id": session_id,
            "memory_context": memory_context,
            "paper_titles": paper_titles,
            "answer_mode": _answer_mode_for_query(query),
            "worker_results": [],
        }

        yield {"type": "status", "data": "retrieving"}
        result = await self._graph.ainvoke(graph_input)
        citations = result.get("citations", [])
        debug_retrieval = result.get("debug_retrieval")
        final_answer = result.get("final_answer", "") or "根据论文当前提供的内容，无法确定"

        yield {"type": "citations", "data": citations}
        if _debug_enabled() and debug_retrieval:
            yield {"type": "debug_retrieval", "data": debug_retrieval}

        yield {"type": "status", "data": "generating"}
        answer_parts: list[str] = []
        async for chunk in _chunk_text(final_answer):
            answer_parts.append(chunk)
            yield {"type": "answer", "data": chunk}

        await self._memory_manager.remember(
            session_id=session_id,
            user_query=query,
            assistant_answer="".join(answer_parts),
            intent="compare",
            generator=self._generator,
        )
        yield {"type": "done", "data": None}


async def _decompose_query(
    state: MultiPaperCompareState,
    llm: MultiPaperLLM,
) -> dict[str, Any]:
    """Decompose the user query into comparison dimensions."""
    if state.get("answer_mode") == "overview":
        logger.info("MultiPaperCompare mode=overview dimensions=%s", OVERVIEW_DIMENSIONS)
        return {
            "answer_mode": "overview",
            "comparison_dimensions": OVERVIEW_DIMENSIONS,
        }

    user_prompt = "\n\n".join(
        [
            "【对话记忆】",
            state.get("memory_context", "") or "无",
            "【用户问题】",
            state["query"],
        ]
    )
    try:
        content = await llm.complete(
            SYSTEM_PROMPT_DECOMPOSE,
            user_prompt,
            temperature=0.0,
            max_tokens=420,
        )
        dimensions = _parse_dimensions(content)
    except Exception as exc:
        logger.warning("decompose_query failed, using defaults: %s", exc)
        dimensions = DEFAULT_COMPARISON_DIMENSIONS

    logger.info("MultiPaperCompare dimensions: %s", dimensions)
    return {
        "answer_mode": state.get("answer_mode", "compare"),
        "comparison_dimensions": dimensions,
    }


async def _paper_worker(
    *,
    state: MultiPaperCompareState,
    retriever: ChromaRetriever,
    paper_library: PaperLibrary,
    llm: MultiPaperLLM,
    top_k: int,
) -> dict[str, Any]:
    """Retrieve and summarize evidence for one paper only."""
    paper_id = state["paper_id"]
    paper_title = state.get("paper_titles", {}).get(paper_id, paper_id)
    base_query = state.get("rewritten_query") or state["query"]
    dimensions = state.get("comparison_dimensions") or DEFAULT_COMPARISON_DIMENSIONS

    logger.info("PaperWorker start: paper=%s dimensions=%s", paper_id, len(dimensions))

    try:
        if state.get("answer_mode") == "overview":
            worker_result = await _paper_overview_worker(
                state=state,
                paper_id=paper_id,
                paper_title=paper_title,
                paper_library=paper_library,
                llm=llm,
            )
            logger.info("PaperWorker overview done: paper=%s", paper_id)
            return {"worker_results": [worker_result]}

        dimension_evidence = []
        worker_errors = []
        for dimension in dimensions:
            dimension_query = f"{base_query} {dimension}".strip()
            try:
                chunks = await retriever.search(
                    query=dimension_query,
                    paper_id=paper_id,
                    top_k=top_k,
                )
                evidence = [
                    _normalize_evidence(chunk, paper_title)
                    for chunk in chunks[:3]
                    if chunk.get("text")
                ]
                status = "found" if evidence else "not_mentioned"
                summary = _fallback_dimension_summary(status, evidence)
            except Exception as exc:
                logger.exception("PaperWorker retrieval failed: %s", paper_id)
                worker_errors.append(f"{dimension}: {exc}")
                evidence = []
                status = "not_mentioned"
                summary = f"检索失败：{exc}"

            dimension_evidence.append(
                {
                    "dimension": dimension,
                    "query": dimension_query,
                    "summary": summary,
                    "status": status,
                    "evidence": evidence,
                }
            )

        dimension_evidence = await _summarize_worker_dimensions(
            llm=llm,
            query=state["query"],
            paper_id=paper_id,
            paper_title=paper_title,
            dimension_evidence=dimension_evidence,
        )

        worker_result = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "dimension_evidence": dimension_evidence,
            "error": "; ".join(worker_errors) if worker_errors else None,
        }
        logger.info("PaperWorker done: paper=%s", paper_id)
    except Exception as exc:
        logger.exception("PaperWorker failed: %s", paper_id)
        worker_result = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "dimension_evidence": [
                {
                    "dimension": dimension,
                    "summary": "未提及",
                    "status": "not_mentioned",
                    "evidence": [],
                }
                for dimension in dimensions
            ],
            "error": str(exc),
        }

    return {"worker_results": [worker_result]}


async def _paper_overview_worker(
    *,
    state: MultiPaperCompareState,
    paper_id: str,
    paper_title: str,
    paper_library: PaperLibrary,
    llm: MultiPaperLLM,
) -> dict[str, Any]:
    """Collect broad paper-level evidence for overview questions.

    Args:
        state: Current graph state.
        paper_id: Current paper id.
        paper_title: Display title.
        paper_library: Existing in-memory paper library.
        llm: OpenAI-compatible helper.

    Returns:
        Worker result containing representative evidence for one paper.
    """
    representative_chunks = paper_library.representative_chunks(
        paper_id,
        max_chunks=12,
    )
    evidence = [
        _normalize_evidence(chunk, paper_title)
        for chunk in representative_chunks
        if chunk.get("text")
    ]
    status = "found" if evidence else "not_mentioned"
    dimension_evidence = [
        {
            "dimension": "论文概览",
            "query": state["query"],
            "summary": _fallback_dimension_summary(status, evidence),
            "status": status,
            "evidence": evidence[:8],
        }
    ]
    dimension_evidence = await _summarize_worker_dimensions(
        llm=llm,
        query=state["query"],
        paper_id=paper_id,
        paper_title=paper_title,
        dimension_evidence=dimension_evidence,
    )
    return {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "dimension_evidence": dimension_evidence,
        "error": None,
    }


async def _summarize_worker_dimensions(
    *,
    llm: MultiPaperLLM,
    query: str,
    paper_id: str,
    paper_title: str,
    dimension_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use one LLM call to summarize all dimensions for a single paper."""
    compact_payload = {
        "query": query,
        "paper_id": paper_id,
        "paper_title": paper_title,
        "dimensions": [
            {
                "dimension": item["dimension"],
                "status": item["status"],
                "evidence": [
                    {
                        "chunk_id": evidence.get("chunk_id"),
                        "page": evidence.get("page"),
                        "text": _preview(evidence.get("text", ""), 700),
                    }
                    for evidence in item.get("evidence", [])
                ],
            }
            for item in dimension_evidence
        ],
    }

    try:
        content = await llm.complete(
            SYSTEM_PROMPT_PAPER_WORKER,
            json.dumps(compact_payload, ensure_ascii=False),
            temperature=0.0,
            max_tokens=1000,
        )
        summaries = _parse_worker_summaries(content)
    except Exception as exc:
        logger.warning("PaperWorker summary failed for %s: %s", paper_id, exc)
        summaries = {}

    for item in dimension_evidence:
        summary_item = summaries.get(item["dimension"])
        if not summary_item:
            continue
        status = summary_item.get("status")
        summary = summary_item.get("summary")
        if status in {"found", "not_mentioned"}:
            item["status"] = status
        if isinstance(summary, str) and summary.strip():
            item["summary"] = summary.strip()
    return dimension_evidence


def _prepare_synthesis(
    state: MultiPaperCompareState,
    paper_library: PaperLibrary,
) -> dict[str, Any]:
    """Normalize worker outputs into evidence blocks and citations."""
    dimensions = state.get("comparison_dimensions") or DEFAULT_COMPARISON_DIMENSIONS
    paper_ids = state.get("paper_ids", [])
    worker_by_paper = {
        result.get("paper_id"): result for result in state.get("worker_results", [])
    }
    paper_titles = state.get("paper_titles") or _paper_title_map(paper_library)
    paper_indexes = {paper_id: index + 1 for index, paper_id in enumerate(paper_ids)}
    citation_counters = {paper_id: 0 for paper_id in paper_ids}

    citations: list[dict[str, Any]] = []
    evidence_blocks: list[dict[str, Any]] = []

    for dimension in dimensions:
        paper_entries = []
        for paper_id in paper_ids:
            worker = worker_by_paper.get(paper_id, {})
            dimension_item = _find_dimension_item(
                worker.get("dimension_evidence", []),
                dimension,
            )
            status = dimension_item.get("status", "not_mentioned")
            summary = dimension_item.get("summary", "未提及")
            evidence_with_ids = []
            citation_ids = []

            evidence_limit = 5 if state.get("answer_mode") == "overview" else 3
            for evidence in dimension_item.get("evidence", [])[:evidence_limit]:
                citation_counters[paper_id] += 1
                citation_id = (
                    f"[P{paper_indexes.get(paper_id, 0)}-"
                    f"C{citation_counters[paper_id]}]"
                )
                citation = _build_citation(
                    citation_id=citation_id,
                    paper_id=paper_id,
                    paper_title=paper_titles.get(paper_id, paper_id),
                    evidence=evidence,
                )
                citations.append(citation)
                citation_ids.append(citation_id)
                evidence_with_ids.append({**evidence, "citation_id": citation_id})

            paper_entries.append(
                {
                    "paper_id": paper_id,
                    "paper_title": paper_titles.get(paper_id, paper_id),
                    "status": status,
                    "summary": summary,
                    "citations": citation_ids,
                    "evidence": evidence_with_ids,
                    "error": worker.get("error"),
                }
            )

        evidence_blocks.append(
            {
                "dimension": dimension,
                "papers": paper_entries,
            }
        )

    debug_retrieval = _build_debug_retrieval(
        state=state,
        citations=citations,
        evidence_blocks=evidence_blocks,
    )
    logger.info(
        "prepare_synthesis: dimensions=%s citations=%s",
        len(evidence_blocks),
        len(citations),
    )
    return {
        "evidence_blocks": evidence_blocks,
        "citations": citations,
        "debug_retrieval": debug_retrieval,
    }


async def _synthesize_comparison(
    state: MultiPaperCompareState,
    llm: MultiPaperLLM,
) -> dict[str, Any]:
    """Generate the final Markdown comparison answer."""
    payload = {
        "query": state["query"],
        "answer_mode": state.get("answer_mode", "compare"),
        "paper_ids": state.get("paper_ids", []),
        "paper_titles": state.get("paper_titles", {}),
        "evidence_blocks": _compact_evidence_blocks(state.get("evidence_blocks", [])),
        "citations": [
            {
                "citation_id": citation.get("citation_id"),
                "paper_title": citation.get("filename"),
                "page": citation.get("page"),
                "chunk_id": citation.get("block_id"),
                "quote": citation.get("quote"),
            }
            for citation in state.get("citations", [])
        ],
    }
    logger.info("synthesize_comparison: citations=%s", len(state.get("citations", [])))
    prompt = (
        SYSTEM_PROMPT_MULTI_PAPER_OVERVIEW
        if state.get("answer_mode") == "overview"
        else SYSTEM_PROMPT_COMPARE_SYNTHESIS
    )
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    compare_max_tokens = int(
        os.getenv("COMPARE_MAX_TOKENS", str(max(llm_max_tokens, 4096)))
    )
    answer = await llm.complete(
        prompt,
        json.dumps(payload, ensure_ascii=False),
        temperature=0.2,
        max_tokens=compare_max_tokens,
    )
    return {"final_answer": _evidence_check(answer, state.get("citations", []))}


def _parse_dimensions(content: str) -> list[str]:
    parsed = _parse_json(content)
    raw_dimensions = parsed.get("comparison_dimensions", [])
    dimensions = []
    for raw_dimension in raw_dimensions:
        dimension = str(raw_dimension).strip()
        if not dimension or dimension == "其他":
            continue
        if dimension not in dimensions:
            dimensions.append(dimension)
        if len(dimensions) >= 6:
            break
    return dimensions or DEFAULT_COMPARISON_DIMENSIONS


def _parse_worker_summaries(content: str) -> dict[str, dict[str, Any]]:
    parsed = _parse_json(content)
    summaries = parsed.get("dimension_summaries", [])
    if not isinstance(summaries, list):
        return {}
    return {
        str(item.get("dimension", "")).strip(): item
        for item in summaries
        if isinstance(item, dict) and item.get("dimension")
    }


def _parse_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    json_text = match.group(0) if match else cleaned
    parsed = json.loads(json_text)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_evidence(
    chunk: dict[str, Any],
    paper_title: str,
) -> dict[str, Any]:
    block_id = chunk.get("block_id") or chunk.get("parent_id") or chunk.get("child_id")
    return {
        "chunk_id": block_id,
        "block_id": block_id,
        "page": chunk.get("page"),
        "paragraph_index": chunk.get("paragraph_index"),
        "section": chunk.get("section", "Unknown"),
        "text": chunk.get("text", ""),
        "score": chunk.get("rerank_score", chunk.get("score")),
        "paper_id": chunk.get("paper_id"),
        "paper_title": paper_title,
        "filename": chunk.get("filename", paper_title),
        "node_type": chunk.get("node_type", "paragraph"),
    }


def _fallback_dimension_summary(
    status: str,
    evidence: list[dict[str, Any]],
) -> str:
    if status == "not_mentioned" or not evidence:
        return "未提及"
    first = evidence[0]
    return (
        f"相关证据位于第 {first.get('page')} 页、"
        f"段落 {first.get('block_id')}：{_preview(first.get('text', ''), 120)}"
    )


def _find_dimension_item(
    items: list[dict[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    for item in items:
        if item.get("dimension") == dimension:
            return item
    return {
        "dimension": dimension,
        "summary": "未提及",
        "status": "not_mentioned",
        "evidence": [],
    }


def _build_citation(
    *,
    citation_id: str,
    paper_id: str,
    paper_title: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    text = evidence.get("text", "")
    return {
        "citation_id": citation_id,
        "paper_id": paper_id,
        "filename": paper_title,
        "page": evidence.get("page"),
        "paragraph_index": evidence.get("paragraph_index"),
        "section": evidence.get("section", "Unknown"),
        "block_id": evidence.get("block_id") or evidence.get("chunk_id"),
        "node_type": evidence.get("node_type", "paragraph"),
        "text": text,
        "quote": _preview(text, 320),
    }


def _build_debug_retrieval(
    *,
    state: MultiPaperCompareState,
    citations: list[dict[str, Any]],
    evidence_blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    papers: dict[str, dict[str, Any]] = {
        paper_id: {"top_chunks": []} for paper_id in state.get("paper_ids", [])
    }
    for block in evidence_blocks:
        dimension = block.get("dimension")
        for paper in block.get("papers", []):
            paper_debug = papers.setdefault(paper["paper_id"], {"top_chunks": []})
            for evidence in paper.get("evidence", []):
                paper_debug["top_chunks"].append(
                    {
                        "dimension": dimension,
                        "citation_id": evidence.get("citation_id"),
                        "block_id": evidence.get("block_id"),
                        "page": evidence.get("page"),
                        "section": evidence.get("section"),
                        "score": evidence.get("score"),
                        "preview": _preview(evidence.get("text", ""), 160),
                    }
                )

    return {
        "original_query": state.get("query", ""),
        "rewritten_query": state.get("rewritten_query", ""),
        "papers": papers,
        "used_citations": [citation.get("citation_id") for citation in citations],
    }


def _compact_evidence_blocks(
    evidence_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact_blocks = []
    for block in evidence_blocks:
        compact_papers = []
        for paper in block.get("papers", []):
            compact_papers.append(
                {
                    "paper_id": paper.get("paper_id"),
                    "paper_title": paper.get("paper_title"),
                    "status": paper.get("status"),
                    "summary": paper.get("summary"),
                    "citations": paper.get("citations", []),
                    "evidence": [
                        {
                            "citation_id": evidence.get("citation_id"),
                            "page": evidence.get("page"),
                            "chunk_id": evidence.get("chunk_id"),
                            "text": _preview(evidence.get("text", ""), 900),
                        }
                        for evidence in paper.get("evidence", [])
                    ],
                }
            )
        compact_blocks.append(
            {
                "dimension": block.get("dimension"),
                "papers": compact_papers,
            }
        )
    return compact_blocks


def _evidence_check(answer: str, citations: list[dict[str, Any]]) -> str:
    if not citations:
        return answer
    citation_ids = [
        re.escape(str(citation.get("citation_id", "")))
        for citation in citations
        if citation.get("citation_id")
    ]
    has_citation = any(re.search(citation_id, answer) for citation_id in citation_ids)
    if has_citation:
        return answer
    return (
        answer.rstrip()
        + "\n\n注意：部分综合判断缺少直接引用支持，已按证据不足处理。"
    )


def _paper_title_map(paper_library: PaperLibrary) -> dict[str, str]:
    return {
        paper["paper_id"]: paper["filename"]
        for paper in paper_library.list_papers()
    }


def _answer_mode_for_query(query: str) -> str:
    normalized_query = query.strip().lower()
    overview_terms = [
        "分别讲什么",
        "分别在讲什么",
        "分别讲了什么",
        "每篇讲什么",
        "每篇论文讲什么",
        "这几篇讲什么",
        "这三篇讲什么",
        "介绍这几篇",
        "概览这几篇",
        "各自讲什么",
        "summarize each",
        "what each paper",
        "about each paper",
    ]
    return (
        "overview"
        if any(term in normalized_query for term in overview_terms)
        else "compare"
    )


def _preview(text: str, max_length: int) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip() + "..."


async def _chunk_text(text: str) -> AsyncGenerator[str, None]:
    for index in range(0, len(text), 24):
        yield text[index : index + 24]
        await asyncio.sleep(0)


def _debug_enabled() -> bool:
    return os.getenv("DEBUG_RETRIEVAL", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
