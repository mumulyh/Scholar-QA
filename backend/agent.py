"""Agentic RAG core for ScholarQA."""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

from answer_composer import AnswerComposer
from prompts import (
    INTENT_ROUTER_HINT,
    MEMORY_SUMMARY_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    RETRIEVAL_REWRITE_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    TRANSLATION_SYSTEM_PROMPT,
    format_context,
)
from retriever import ChromaRetriever, PaperLibrary

UNKNOWN_ANSWER = "根据论文当前提供的内容，无法确定"
MAX_CHAT_CITATIONS = 5


@dataclass(slots=True)
class ConversationTurn:
    """One complete user-assistant exchange."""

    user: str
    assistant: str
    intent: str


@dataclass(slots=True)
class SessionMemory:
    """Memory state for a chat session."""

    summary: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)


class MemoryManager:
    """Sliding-window memory with LLM-compressed long-term summary."""

    def __init__(self, window_size: int = 10) -> None:
        """Initialize memory storage.

        Args:
            window_size: Number of recent complete conversation turns to keep.
        """
        self._window_size = window_size
        self._sessions: dict[str, SessionMemory] = {}

    def build_context(self, session_id: str) -> str:
        """Build context for pronoun resolution and follow-up understanding.

        Args:
            session_id: Conversation session id.

        Returns:
            A compact memory string.
        """
        memory = self._sessions.get(session_id)
        if not memory:
            return ""

        recent_lines = []
        for turn in memory.turns[-self._window_size :]:
            recent_lines.append(f"用户：{turn.user}")
            recent_lines.append(f"助手：{turn.assistant}")

        parts = []
        if memory.summary:
            parts.append(f"长期压缩记忆：{memory.summary}")
        if recent_lines:
            parts.append("最近对话：\n" + "\n".join(recent_lines))
        return "\n\n".join(parts)

    async def remember(
        self,
        session_id: str,
        user_query: str,
        assistant_answer: str,
        intent: str,
        generator: "Generator",
    ) -> None:
        """Store a completed conversation turn and compress overflow.

        Args:
            session_id: Conversation session id.
            user_query: User message.
            assistant_answer: Assistant response.
            intent: Routed intent.
            generator: LLM generator used for memory compression.
        """
        memory = self._sessions.setdefault(session_id, SessionMemory())
        memory.turns.append(
            ConversationTurn(
                user=user_query,
                assistant=assistant_answer,
                intent=intent,
            )
        )

        if len(memory.turns) <= self._window_size:
            return

        overflow_turns = memory.turns[: -self._window_size]
        memory.turns = memory.turns[-self._window_size :]
        memory.summary = await generator.compress_memory(memory.summary, overflow_turns)


class IntentRouter:
    """Rule-based intent router for translation, summary, QA, and compare hints."""

    _COMPARE_KEYWORDS = [
        "对比",
        "比较",
        "区别",
        "不同",
        "相同点",
        "差异",
        "哪篇",
        "哪个方法更好",
        "哪个效果更好",
        "优势",
        "劣势",
        "局限",
        "实验更充分",
        "方法更先进",
        "compare",
        "comparison",
        "difference",
        "different",
        "better",
        "advantage",
        "disadvantage",
        "limitation",
        "stronger",
        "weaker",
    ]

    def route(self, user_input: str) -> str:
        """Route user input to one of the supported task types.

        Args:
            user_input: Raw user query.

        Returns:
            One of ``translation``, ``summary``, or ``qa``.
        """
        normalized_input = user_input.strip().lower()

        translation_keywords = [
            "翻译",
            "译成中文",
            "中文意思",
            "这段什么意思",
            "逐段",
            "translate",
        ]
        summary_keywords = [
            "总结",
            "归纳",
            "创新点",
            "方法概述",
            "实验设置",
            "全文摘要",
            "梳理",
            "summary",
            "summarize",
        ]

        if any(keyword in normalized_input for keyword in translation_keywords):
            return "translation"
        if any(keyword in normalized_input for keyword in summary_keywords):
            return "summary"
        return "qa"

    def is_compare_query(self, user_input: str) -> bool:
        """Check whether a query is explicitly asking for paper comparison.

        Args:
            user_input: Raw user query.

        Returns:
            True when the query contains comparison intent keywords.
        """
        normalized_input = user_input.strip().lower()
        return any(keyword in normalized_input for keyword in self._COMPARE_KEYWORDS)

    def route_with_papers(
        self,
        user_input: str,
        paper_ids: list[str] | None,
    ) -> str:
        """Route with selected paper count for API-level branching.

        Args:
            user_input: Raw user query.
            paper_ids: Selected paper ids.

        Returns:
            ``compare`` when multiple papers are selected, otherwise the
            original single-paper route.
        """
        selected_count = len([paper_id for paper_id in paper_ids or [] if paper_id])
        if selected_count > 1:
            return "compare"
        return self.route(user_input)


class Generator:
    """OpenAI-compatible streaming generator."""

    def __init__(self) -> None:
        """Initialize an OpenAI-compatible async client from environment variables."""
        self._base_url = os.getenv("LLM_BASE_URL")
        self._api_key = os.getenv("LLM_API_KEY")
        self._model = os.getenv("LLM_MODEL")
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self._client: AsyncOpenAI | None = None
        self._rewrite_cache: dict[str, list[str]] = {}
        self._answer_composer = AnswerComposer()

    async def stream_answer(
        self,
        query: str,
        intent: str,
        chunks: list[dict[str, Any]],
        memory_context: str,
        rewritten_queries: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a grounded answer token by token.

        Args:
            query: User query.
            intent: Routed task intent.
            chunks: Retrieved paper chunks used as the only evidence.
            memory_context: Conversation memory for follow-up understanding.
            rewritten_queries: Retrieval rewrites that produced the evidence.

        Yields:
            Generated text deltas.
        """
        if not chunks:
            yield UNKNOWN_ANSWER
            return

        system_prompt = self._system_prompt_for_intent(intent)
        if intent == "qa":
            user_prompt = self._answer_composer.compose_user_prompt(
                query=query,
                chunks=chunks,
                memory_context=memory_context,
                rewritten_queries=rewritten_queries,
            )
        else:
            user_prompt = self._compose_basic_user_prompt(
                query=query,
                chunks=chunks,
                memory_context=memory_context,
            )

        async for token in self._stream_chat(system_prompt, user_prompt):
            yield token

    async def stream_translation(self, source_text: str) -> AsyncGenerator[str, None]:
        """Stream a Chinese translation for one source paragraph.

        Args:
            source_text: Original paper paragraph.

        Yields:
            Translation text deltas.
        """
        user_prompt = "\n".join(
            [
                "请只输出中文译文，不要添加额外说明。",
                "原文如下：",
                source_text,
            ]
        )
        async for token in self._stream_chat(TRANSLATION_SYSTEM_PROMPT, user_prompt):
            yield token

    async def compress_memory(
        self,
        existing_summary: str,
        overflow_turns: list[ConversationTurn],
    ) -> str:
        """Compress old conversation turns into a durable summary.

        Args:
            existing_summary: Previous long-term memory summary.
            overflow_turns: Turns falling out of the short-term window.

        Returns:
            Updated memory summary.
        """
        overflow_text = "\n".join(
            f"用户：{turn.user}\n助手：{turn.assistant}\n意图：{turn.intent}"
            for turn in overflow_turns
        )
        user_prompt = "\n\n".join(
            [
                f"已有摘要：{existing_summary or '无'}",
                "需要压缩的旧对话：",
                overflow_text,
            ]
        )
        try:
            response = await self._client_or_raise().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": MEMORY_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            content = response.choices[0].message.content or existing_summary
            return content.strip()
        except Exception:
            return existing_summary

    async def rewrite_retrieval_queries(
        self,
        query: str,
        memory_context: str,
    ) -> list[str]:
        """Rewrite a user question into English retrieval queries.

        Args:
            query: Current user question.
            memory_context: Conversation memory for pronoun resolution.

        Returns:
            Query rewrites used only for retrieval.
        """
        user_prompt = "\n\n".join(
            [
                "【对话记忆】",
                memory_context or "无",
                "【当前问题】",
                query,
            ]
        )
        cache_key = json.dumps(
            {"query": query, "memory_context": memory_context},
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self._rewrite_cache:
            return list(self._rewrite_cache[cache_key])

        try:
            response = await self._client_or_raise().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": RETRIEVAL_REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=480,
            )
            content = response.choices[0].message.content or ""
            rewritten_queries = self._parse_rewrite_queries(content, query)
            self._rewrite_cache[cache_key] = rewritten_queries
            return list(rewritten_queries)
        except Exception:
            return []

    def _parse_rewrite_queries(self, content: str, original_query: str) -> list[str]:
        cleaned_content = content.strip()
        cleaned_content = re.sub(r"^```(?:json)?|```$", "", cleaned_content).strip()
        json_match = re.search(r"\{.*\}", cleaned_content, flags=re.DOTALL)
        json_content = json_match.group(0) if json_match else cleaned_content

        queries: list[str] = []
        try:
            parsed = json.loads(json_content)
            raw_queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
        except json.JSONDecodeError:
            raw_queries = cleaned_content.splitlines()

        for raw_query in raw_queries:
            query = str(raw_query).strip(" -*0123456789.\t")
            if query and query != original_query and query not in queries:
                queries.append(query)
            if len(queries) >= 6:
                break
        return queries

    def _system_prompt_for_intent(self, intent: str) -> str:
        if intent == "summary":
            return SUMMARY_SYSTEM_PROMPT
        if intent == "translation":
            return TRANSLATION_SYSTEM_PROMPT
        return QA_SYSTEM_PROMPT

    async def _stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> AsyncGenerator[str, None]:
        client = self._client_or_raise()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )

        async for chunk in response:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                yield token

    def _compose_basic_user_prompt(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        memory_context: str,
    ) -> str:
        """Compose the original compact prompt for non-QA generation.

        Args:
            query: User query.
            chunks: Retrieved paper chunks used as evidence.
            memory_context: Conversation memory for follow-up understanding.

        Returns:
            Prompt text for summary-style generation.
        """
        context_block = format_context(chunks)
        return "\n\n".join(
            [
                "【对话记忆】",
                memory_context or "无",
                "注意：对话记忆只能用于理解指代，不能作为论文事实来源。",
                "【论文片段】",
                context_block,
                "【用户问题】",
                query,
                "请基于论文片段回答，并附上页码和段落号。",
            ]
        )

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


class ScholarQAAgent:
    """Single-path retrieval-generation agent with intent routing."""

    def __init__(
        self,
        retriever: ChromaRetriever,
        paper_library: PaperLibrary,
        memory_manager: MemoryManager,
        intent_router: IntentRouter,
        generator: Generator,
        top_k: int = 6,
    ) -> None:
        """Initialize the agent.

        Args:
            retriever: ChromaDB retriever.
            paper_library: In-memory paper store.
            memory_manager: Conversation memory manager.
            intent_router: User intent router.
            generator: Streaming LLM generator.
            top_k: Default retrieval count.
        """
        self._retriever = retriever
        self._paper_library = paper_library
        self._memory_manager = memory_manager
        self._intent_router = intent_router
        self._generator = generator
        self._top_k = top_k

    async def stream_chat(
        self,
        session_id: str,
        query: str,
        paper_id: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a routed ScholarQA chat response.

        Args:
            session_id: Conversation session id.
            query: User query.
            paper_id: Optional selected paper id.

        Yields:
            SSE-ready event dictionaries.
        """
        selected_paper_id = paper_id or self._paper_library.latest_paper_id()
        if not selected_paper_id:
            yield {"type": "answer", "data": "请先上传论文 PDF。"}
            yield {"type": "done", "data": None}
            return

        intent = self._intent_router.route(query)
        memory_context = self._memory_manager.build_context(session_id)
        yield {"type": "intent", "data": intent}

        if intent == "translation":
            async for event in self.stream_translation_request(
                session_id=session_id,
                query=query,
                paper_id=selected_paper_id,
            ):
                yield event
            return

        yield {"type": "status", "data": "retrieving"}
        retrieval_queries: list[str] | None = None
        if intent == "summary":
            chunks = await self._summary_chunks(query, selected_paper_id)
        else:
            retrieval_queries = await self._build_retrieval_queries(
                query,
                memory_context,
            )
            chunks = await self._qa_chunks(
                query=query,
                paper_id=selected_paper_id,
                memory_context=memory_context,
                retrieval_queries=retrieval_queries,
            )

        citations = self._build_citations(
            chunks,
            query=query,
            max_citations=MAX_CHAT_CITATIONS,
        )
        yield {"type": "citations", "data": citations}
        yield {"type": "status", "data": "generating"}

        answer_parts: list[str] = []
        async for token in self._generator.stream_answer(
            query=query,
            intent=intent,
            chunks=chunks,
            memory_context=memory_context,
            rewritten_queries=retrieval_queries,
        ):
            answer_parts.append(token)
            yield {"type": "answer", "data": token}

        await self._memory_manager.remember(
            session_id=session_id,
            user_query=query,
            assistant_answer="".join(answer_parts),
            intent=intent,
            generator=self._generator,
        )
        yield {"type": "done", "data": None}

    async def stream_translation_request(
        self,
        session_id: str,
        query: str,
        paper_id: str,
        page: int | None = None,
        paragraph_index: int | None = None,
        block_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream paragraph-by-paragraph translation events.

        Args:
            session_id: Conversation session id.
            query: Natural language translation request.
            paper_id: Selected paper id.
            page: Optional page number.
            paragraph_index: Optional paragraph number within the page.
            block_id: Optional exact block id.

        Yields:
            SSE-ready translation events.
        """
        target = self._extract_translation_target(query)
        selected_page = page if page is not None else target.get("page")
        selected_paragraph = (
            paragraph_index if paragraph_index is not None else target.get("paragraph")
        )
        chunks = self._paper_library.select_chunks(
            paper_id=paper_id,
            page=selected_page,
            paragraph_index=selected_paragraph,
            block_id=block_id,
            limit=None if selected_page is not None else 3,
        )

        # 用户没有指定页码时，用检索结果找到最可能需要翻译的原文段落。
        if not chunks:
            chunks = await self._retriever.search(query, paper_id=paper_id, top_k=3)

        citations = self._build_citations(
            chunks,
            query=query,
            max_citations=MAX_CHAT_CITATIONS,
            rank=False,
        )
        yield {"type": "citations", "data": citations}

        if not chunks:
            yield {"type": "answer", "data": UNKNOWN_ANSWER}
            yield {"type": "done", "data": None}
            return

        answer_parts: list[str] = []
        for chunk in chunks:
            yield {"type": "translation_segment_start", "data": chunk}
            translation_parts: list[str] = []
            async for token in self._generator.stream_translation(chunk["text"]):
                translation_parts.append(token)
                answer_parts.append(token)
                yield {
                    "type": "translation_delta",
                    "data": {
                        "block_id": chunk["block_id"],
                        "token": token,
                    },
                }
            yield {
                "type": "translation_segment_done",
                "data": {
                    "block_id": chunk["block_id"],
                    "translation": "".join(translation_parts),
                },
            }

        await self._memory_manager.remember(
            session_id=session_id,
            user_query=query,
            assistant_answer="".join(answer_parts),
            intent="translation",
            generator=self._generator,
        )
        yield {"type": "done", "data": None}

    async def _summary_chunks(
        self,
        query: str,
        paper_id: str,
    ) -> list[dict[str, Any]]:
        representative_chunks = self._paper_library.representative_chunks(paper_id)
        retrieved_chunks = await self._retriever.search(
            query,
            paper_id=paper_id,
            top_k=self._top_k,
        )
        merged_chunks: list[dict[str, Any]] = []
        seen_block_ids: set[str] = set()
        for chunk in [*retrieved_chunks, *representative_chunks]:
            block_id = chunk.get("block_id")
            if block_id and block_id not in seen_block_ids:
                merged_chunks.append(chunk)
                seen_block_ids.add(block_id)
        return merged_chunks[:14]

    async def _qa_chunks(
        self,
        query: str,
        paper_id: str,
        memory_context: str,
        retrieval_queries: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if retrieval_queries is None:
            retrieval_queries = await self._build_retrieval_queries(
                query,
                memory_context,
            )
        retrieval_scope = "\n".join([query, memory_context, *retrieval_queries])
        formula_chunks = self._formula_context_chunks(retrieval_scope, paper_id)
        per_query_top_k = max(self._top_k * 2, 12)
        candidate_records: list[dict[str, Any]] = []
        for rank, chunk in enumerate(formula_chunks, start=1):
            candidate_records.append(
                {
                    "chunk": chunk,
                    "source": "formula",
                    "rank": rank,
                    "query_index": -1,
                    "query": query,
                }
            )

        has_rewritten_query = len(retrieval_queries) > 1
        for query_index, retrieval_query in enumerate(retrieval_queries):
            is_fallback_query = (
                has_rewritten_query
                and query_index == len(retrieval_queries) - 1
            )
            semantic_results = await self._retriever.search(
                retrieval_query,
                paper_id=paper_id,
                top_k=per_query_top_k,
            )
            for rank, chunk in enumerate(semantic_results, start=1):
                candidate_records.append(
                    {
                        "chunk": chunk,
                        "source": "semantic",
                        "rank": rank,
                        "query_index": query_index,
                        "query": retrieval_query,
                        "is_fallback_query": is_fallback_query,
                    }
                )

            keyword_results = self._keyword_chunks(
                retrieval_query,
                paper_id,
                limit=min(per_query_top_k, 8),
            )
            for rank, chunk in enumerate(keyword_results, start=1):
                candidate_records.append(
                    {
                        "chunk": chunk,
                        "source": "keyword",
                        "rank": rank,
                        "query_index": query_index,
                        "query": retrieval_query,
                        "is_fallback_query": is_fallback_query,
                    }
                )

        seed_chunks = [record["chunk"] for record in candidate_records]
        context_chunks = self._neighbor_chunks(
            paper_id=paper_id,
            seed_chunks=seed_chunks,
            window=3,
            limit=18,
        )
        for rank, chunk in enumerate(context_chunks, start=1):
            candidate_records.append(
                {
                    "chunk": chunk,
                    "source": "context",
                    "rank": rank,
                    "query_index": len(retrieval_queries),
                    "query": query,
                    "is_fallback_query": True,
                }
            )

        # 候选池统一排序，避免关键词命中段或公式碎片压过真正的证据段。
        ranked_chunks = self._rank_qa_candidates(
            query=query,
            retrieval_queries=retrieval_queries,
            candidate_records=candidate_records,
            limit=max(self._top_k * 2, 14),
        )
        # PDF 公式和器件说明经常被解析成相邻短块；追加邻段给生成器做证据拼接，
        # 但保留已排序结果在前，避免影响检索指标里的 Top-k 排名。
        return self._append_neighbor_evidence(
            paper_id=paper_id,
            ranked_chunks=ranked_chunks,
            window=12,
            limit=max(self._top_k * 4, 28),
        )

    def _rank_qa_candidates(
        self,
        query: str,
        retrieval_queries: list[str],
        candidate_records: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rank merged retrieval candidates for answer generation.

        Args:
            query: Original user query.
            retrieval_queries: Query rewrites produced from explicit question aspects.
            candidate_records: Candidate chunks with retrieval source metadata.
            limit: Maximum number of chunks to return.

        Returns:
            Ranked, deduplicated chunks for the LLM context.
        """
        global_keywords = self._expanded_keywords(
            " ".join([query, *retrieval_queries])
        )
        formula_numbers = self._extract_formula_numbers(
            " ".join([query, *retrieval_queries])
        )
        merged: dict[str, dict[str, Any]] = {}

        for order, record in enumerate(candidate_records):
            chunk = record["chunk"]
            block_id = chunk.get("block_id")
            if not block_id or self._is_reference_chunk(chunk):
                continue

            score = self._qa_candidate_score(
                record=record,
                global_keywords=global_keywords,
                formula_numbers=formula_numbers,
                order=order,
            )
            existing = merged.get(block_id)
            if existing is None:
                merged[block_id] = {
                    "chunk": chunk,
                    "score": score,
                    "best_score": score,
                    "best_order": order,
                    "sources": {record["source"]},
                    "query_indexes": self._score_query_indexes(record),
                }
                continue

            existing["score"] = max(existing["score"], score)
            existing["sources"].add(record["source"])
            existing["query_indexes"].update(self._score_query_indexes(record))
            if score > existing["best_score"]:
                existing["chunk"] = chunk
                existing["best_score"] = score
            existing["best_order"] = min(existing["best_order"], order)

        for item in merged.values():
            # 多查询命中只能小幅加分，避免宽泛词在多个子查询中重复出现而淹没精确证据。
            item["score"] += min(len(item["sources"]) - 1, 2) * 0.7
            item["score"] += min(len(item["query_indexes"]) - 1, 3) * 0.6
            if formula_numbers and "formula" in item["sources"]:
                item["score"] += 1.5

        ranked_items = sorted(
            merged.values(),
            key=lambda item: (item["score"], -item["best_order"]),
            reverse=True,
        )
        ranked_chunks: list[dict[str, Any]] = []
        for item in ranked_items[:limit]:
            ranked_chunks.append(
                {
                    **item["chunk"],
                    "answer_score": float(item["score"]),
                    "retrieval_source": ",".join(sorted(item["sources"])),
                }
            )
        return ranked_chunks

    def _score_query_indexes(self, record: dict[str, Any]) -> set[int]:
        source = record["source"]
        if source not in {"semantic", "keyword"}:
            return set()
        query_index = int(record.get("query_index", -1))
        return {query_index} if query_index >= 0 else set()

    def _qa_candidate_score(
        self,
        record: dict[str, Any],
        global_keywords: list[str],
        formula_numbers: list[str],
        order: int,
    ) -> float:
        chunk = record["chunk"]
        source = record["source"]
        rank = max(int(record["rank"]), 1)
        retrieval_query = str(record["query"])
        text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        normalized_text = text.lower()
        token_count = len(
            re.findall(r"[a-zA-Z0-9_+-]+|[α-ωΑ-Ω]+|[\u4e00-\u9fff]", text)
        )

        source_weights = {
            "semantic": 12.0,
            "formula": 3.0,
            "keyword": 1.5,
            "context": 0.8,
        }
        score = source_weights.get(source, 1.0) / (rank ** 0.5)
        if record.get("is_fallback_query"):
            score *= 0.55
        if formula_numbers and source == "formula":
            score += 12.0

        aspect_keywords = self._expanded_keywords(retrieval_query)
        aspect_hits = sum(
            1 for keyword in aspect_keywords if keyword in normalized_text
        )
        global_hits = sum(
            1 for keyword in global_keywords if keyword in normalized_text
        )
        score += min(aspect_hits, 8) * 0.32
        score += min(global_hits, 10) * 0.12

        formula_hit = any(f"({number})" in text for number in formula_numbers)
        if formula_hit:
            score += 2.5

        if token_count < 8:
            score -= 4.0 if (aspect_hits or global_hits or formula_hit) else 10.0
        elif token_count < 20:
            score -= 1.5

        if chunk.get("node_type") == "formula":
            score -= 2.0 if formula_hit else 4.0
        if "authorized licensed use" in normalized_text:
            score -= 12.0
        if re.search(r"\.{6,}|…{2,}", text):
            score -= 8.0
        if re.match(r"^(fig\.|table)\b", normalized_text) and not global_hits:
            score -= 3.0

        explanatory_markers = [
            "where",
            "denote",
            "denotes",
            "represent",
            "represents",
            "respectively",
            "defined",
            "definition",
            "set to",
            "coefficient",
            "coefficients",
            "parameter",
            "parameters",
            "in contrast",
            "compared",
            "difference",
            "distinction",
            "advantage",
            "performance",
            "limitation",
            "however",
            "lack",
            "lacks",
        ]
        if any(marker in normalized_text for marker in explanatory_markers):
            score += 1.8
            if formula_numbers:
                score += 1.5

        retrieval_score = chunk.get("rerank_score", chunk.get("score"))
        if isinstance(retrieval_score, (int, float)):
            score += min(max(float(retrieval_score), -5.0), 5.0) * 0.35

        return score - order * 0.001

    def _formula_context_chunks(
        self,
        query: str,
        paper_id: str,
    ) -> list[dict[str, Any]]:
        formula_numbers = self._extract_formula_numbers(query)
        if not formula_numbers:
            return []

        chunks: list[dict[str, Any]] = []
        seen_block_ids: set[str] = set()
        for formula_number in formula_numbers:
            for chunk in self._paper_library.formula_context_chunks(
                paper_id=paper_id,
                formula_number=formula_number,
                window=4,
            ):
                block_id = chunk.get("block_id")
                if block_id and block_id not in seen_block_ids:
                    chunks.append(chunk)
                    seen_block_ids.add(block_id)
        return chunks

    def _keyword_chunks(
        self,
        query: str,
        paper_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        keywords = self._expanded_keywords(query)
        if not keywords:
            return []

        scored_chunks: list[tuple[int, dict[str, Any]]] = []
        for chunk in self._paper_library.get_chunks(paper_id):
            if self._is_reference_chunk(chunk):
                continue
            text = chunk.get("text", "")
            normalized_text = text.lower()
            score = sum(1 for keyword in keywords if keyword in normalized_text)
            if score <= 0:
                continue
            scored_chunks.append((score, {**chunk, "score": float(score)}))

        scored_chunks.sort(
            key=lambda item: (
                -item[0],
                item[1].get("page", 0),
                item[1].get("paragraph_index", 0),
            )
        )
        return [chunk for _, chunk in scored_chunks[:limit]]

    def _expanded_keywords(self, query: str) -> list[str]:
        raw_keywords = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}|[α-ωΑ-Ω]+", query)
        keywords: list[str] = []
        for raw_keyword in raw_keywords:
            keyword = raw_keyword.lower()
            keywords.append(keyword)
            if len(keyword) > 3 and keyword.endswith("s"):
                keywords.append(keyword[:-1])
        chinese_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        for phrase in chinese_phrases:
            keywords.append(phrase)
        return list(dict.fromkeys(keywords))

    def _extract_formula_numbers(self, query: str) -> list[str]:
        patterns = [
            r"公式\s*[\(（]?\s*(\d+)\s*[\)）]?",
            r"式\s*[\(（]?\s*(\d+)\s*[\)）]?",
            r"equation\s*[\(（]?\s*(\d+)\s*[\)）]?",
            r"eq\.?\s*[\(（]?\s*(\d+)\s*[\)）]?",
        ]
        formula_numbers: list[str] = []
        for pattern in patterns:
            formula_numbers.extend(re.findall(pattern, query, flags=re.IGNORECASE))
        return list(dict.fromkeys(formula_numbers))

    def _is_reference_chunk(self, chunk: dict[str, Any]) -> bool:
        section = str(chunk.get("section", "")).strip().lower()
        return section in {"references", "reference", "bibliography"}

    def _neighbor_chunks(
        self,
        paper_id: str,
        seed_chunks: list[dict[str, Any]],
        window: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        all_chunks = self._paper_library.get_chunks(paper_id)
        index_by_block_id = {
            chunk.get("block_id"): index
            for index, chunk in enumerate(all_chunks)
            if chunk.get("block_id")
        }

        selected_chunks: list[dict[str, Any]] = []
        seen_block_ids: set[str] = set()
        for seed_chunk in seed_chunks:
            seed_block_id = seed_chunk.get("block_id")
            seed_index = index_by_block_id.get(seed_block_id)
            if seed_index is None:
                continue

            start_index = max(0, seed_index - window)
            end_index = min(len(all_chunks), seed_index + window + 1)
            for chunk in all_chunks[start_index:end_index]:
                block_id = chunk.get("block_id")
                if not block_id or block_id == seed_block_id:
                    continue
                if block_id in seen_block_ids or self._is_reference_chunk(chunk):
                    continue
                selected_chunks.append(chunk)
                seen_block_ids.add(block_id)
                if len(selected_chunks) >= limit:
                    return selected_chunks
        return selected_chunks

    def _append_neighbor_evidence(
        self,
        paper_id: str,
        ranked_chunks: list[dict[str, Any]],
        window: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Append nearby evidence chunks after ranked retrieval results.

        Args:
            paper_id: Selected paper id.
            ranked_chunks: Already ranked chunks used as primary evidence.
            window: Number of neighboring chunks to append around top evidence.
            limit: Maximum total chunk count.

        Returns:
            Ranked chunks followed by deduplicated nearby chunks.
        """
        if not ranked_chunks or len(ranked_chunks) >= limit:
            return ranked_chunks[:limit]

        all_chunks = self._paper_library.get_chunks(paper_id)
        index_by_block_id = {
            chunk.get("block_id"): index
            for index, chunk in enumerate(all_chunks)
            if chunk.get("block_id")
        }
        seen_block_ids = {
            chunk.get("block_id")
            for chunk in ranked_chunks
            if chunk.get("block_id")
        }
        appended_chunks = list(ranked_chunks)

        for seed_chunk in ranked_chunks[: min(len(ranked_chunks), self._top_k)]:
            seed_block_id = seed_chunk.get("block_id")
            seed_index = index_by_block_id.get(seed_block_id)
            if seed_index is None:
                continue

            # 公式、变量解释、器件作用通常紧跟在引导句后面，所以优先追加后文。
            for direction in (1, -1):
                for offset in range(1, window + 1):
                    neighbor_index = seed_index + direction * offset
                    if neighbor_index < 0 or neighbor_index >= len(all_chunks):
                        continue
                    chunk = all_chunks[neighbor_index]
                    block_id = chunk.get("block_id")
                    if (
                        not block_id
                        or block_id in seen_block_ids
                        or self._is_reference_chunk(chunk)
                    ):
                        continue
                    appended_chunks.append(
                        {
                            **chunk,
                            "retrieval_source": "neighbor_evidence",
                        }
                    )
                    seen_block_ids.add(block_id)
                    if len(appended_chunks) >= limit:
                        return appended_chunks
        return appended_chunks

    async def _build_retrieval_queries(
        self,
        query: str,
        memory_context: str,
    ) -> list[str]:
        base_query = query
        if memory_context:
            base_query = "\n".join(
                [
                    "以下历史只用于消解代词和连续追问：",
                    memory_context,
                    "当前问题：",
                    query,
                    INTENT_ROUTER_HINT,
                ]
            )

        rewritten_queries = await self._generator.rewrite_retrieval_queries(
            query,
            memory_context,
        )
        # 英文论文优先使用英文改写检索；中文原问题保留为兜底，避免丢失缩写/公式号。
        queries = [*rewritten_queries, base_query] if rewritten_queries else [base_query]
        return list(dict.fromkeys(query_text for query_text in queries if query_text))

    def _build_citations(
        self,
        chunks: list[dict[str, Any]],
        query: str = "",
        max_citations: int | None = None,
        rank: bool = True,
    ) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        seen_block_ids: set[str] = set()
        citation_chunks = self._rank_citation_chunks(query, chunks) if rank else chunks
        for chunk in citation_chunks:
            block_id = chunk.get("block_id")
            if not block_id or block_id in seen_block_ids:
                continue
            seen_block_ids.add(block_id)
            citations.append(
                {
                    "paper_id": chunk.get("paper_id"),
                    "filename": chunk.get("filename", ""),
                    "page": chunk.get("page"),
                    "paragraph_index": chunk.get(
                        "paragraph_index",
                        self._paragraph_from_block_id(block_id),
                    ),
                    "section": chunk.get("section", "Unknown"),
                    "block_id": block_id,
                    "node_type": chunk.get("node_type", "paragraph"),
                    "text": chunk.get("text", ""),
                    "quote": self._citation_quote(chunk.get("text", "")),
                }
            )
            if max_citations is not None and len(citations) >= max_citations:
                break
        return citations

    def _rank_citation_chunks(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        query_keywords = self._expanded_keywords(query)
        formula_numbers = self._extract_formula_numbers(query)

        def citation_score(index: int, chunk: dict[str, Any]) -> float:
            text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
            normalized_text = text.lower()
            token_count = len(
                re.findall(r"[a-zA-Z0-9_+-]+|[α-ωΑ-Ω]+|[\u4e00-\u9fff]", text)
            )
            score = 0.0
            keyword_hits = sum(
                1 for keyword in query_keywords if keyword in normalized_text
            )
            formula_hit = any(f"({number})" in text for number in formula_numbers)

            # 引用要服务于“出处说明”，极短公式碎片和下载水印不适合作为主要证据。
            if token_count < 8:
                score -= 4.0 if keyword_hits or formula_hit else 12.0
            elif token_count < 20:
                score -= 4.0
            if "authorized licensed use" in normalized_text:
                score -= 20.0
            if chunk.get("node_type") == "formula":
                score -= 2.0

            score += keyword_hits * 3.0

            if formula_hit:
                score += 10.0

            explanatory_markers = [
                "where",
                "denote",
                "denotes",
                "represent",
                "represents",
                "respectively",
                "defined",
                "definition",
                "set to",
                "coefficient",
                "coefficients",
                "parameter",
                "parameters",
            ]
            has_explanatory_marker = any(
                re.search(
                    rf"(?<![a-zA-Z]){re.escape(marker)}(?![a-zA-Z])",
                    normalized_text,
                )
                for marker in explanatory_markers
            )
            if has_explanatory_marker:
                score += 5.0
            elif query_keywords and keyword_hits == 0 and not formula_hit:
                score -= 6.0

            retrieval_score = chunk.get(
                "answer_score",
                chunk.get("rerank_score", chunk.get("score")),
            )
            if isinstance(retrieval_score, (int, float)):
                score += min(max(float(retrieval_score), -5.0), 5.0) * 0.2

            return score - index * 0.01

        return [
            chunk
            for _, chunk in sorted(
                (
                    (citation_score(index, chunk), chunk)
                    for index, chunk in enumerate(chunks)
                ),
                key=lambda item: item[0],
                reverse=True,
            )
        ]

    def _citation_quote(self, text: str, max_length: int = 320) -> str:
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if len(cleaned_text) <= max_length:
            return cleaned_text
        return cleaned_text[:max_length].rstrip() + "..."

    def _paragraph_from_block_id(self, block_id: str) -> int | None:
        match = re.search(r"-b(\d+)$", block_id)
        if not match:
            return None
        return int(match.group(1))

    def _extract_translation_target(self, query: str) -> dict[str, int]:
        target: dict[str, int] = {}
        page_match = re.search(r"第?([一二三四五六七八九十百\d]+)页", query)
        paragraph_match = re.search(r"第?([一二三四五六七八九十百\d]+)段", query)
        if page_match:
            target["page"] = self._parse_number(page_match.group(1))
        if paragraph_match:
            target["paragraph"] = self._parse_number(paragraph_match.group(1))
        return target

    def _parse_number(self, raw_number: str) -> int:
        if raw_number.isdigit():
            return int(raw_number)

        digit_map = {
            "零": 0,
            "一": 1,
            "二": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if raw_number == "十":
            return 10
        if raw_number.startswith("十"):
            return 10 + digit_map.get(raw_number[-1], 0)
        if "十" in raw_number:
            tens, ones = raw_number.split("十", maxsplit=1)
            return digit_map.get(tens, 1) * 10 + digit_map.get(ones, 0)
        return digit_map.get(raw_number, 1)
