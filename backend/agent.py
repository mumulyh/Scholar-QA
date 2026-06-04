"""Agentic RAG core for ScholarQA."""

from __future__ import annotations

import asyncio
import os
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from openai import AsyncOpenAI

from answer_composer import AnswerComposer
from prompts import (
    INTENT_ROUTER_HINT,
    MEMORY_SUMMARY_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    RETRIEVAL_REFLECTION_SYSTEM_PROMPT,
    RETRIEVAL_REWRITE_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    TRANSLATION_SYSTEM_PROMPT,
    format_context,
)
from retriever import ChromaRetriever, PaperLibrary

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

UNKNOWN_ANSWER = "根据论文当前提供的内容，无法确定"
MAX_CHAT_CITATIONS = 5
ENABLE_FINAL_RERANK = os.getenv("ENABLE_FINAL_RERANK", "false").lower() == "true"
ENABLE_AGENTIC_RETRIEVAL = (
    os.getenv("ENABLE_AGENTIC_RETRIEVAL", "false").lower() == "true"
)
ENABLE_AGENTIC_FALLBACK_REPAIR = (
    os.getenv("ENABLE_AGENTIC_FALLBACK_REPAIR", "false").lower() == "true"
)
AGENTIC_RETRIEVAL_MAX_REPAIR_QUERIES = int(
    os.getenv("AGENTIC_RETRIEVAL_MAX_REPAIR_QUERIES", "3")
)
AGENTIC_REPAIR_INSERT_LIMIT = int(os.getenv("AGENTIC_REPAIR_INSERT_LIMIT", "2"))
AGENTIC_REPAIR_PROTECT_TOP_N = int(os.getenv("AGENTIC_REPAIR_PROTECT_TOP_N", "2"))


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
        self._last_rewrite_diagnostics: dict[str, Any] = {"attempted": False}
        self._answer_composer = AnswerComposer()

    @property
    def last_rewrite_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics for the most recent retrieval rewrite call."""
        return dict(self._last_rewrite_diagnostics)

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
        paper_context: str = "",
    ) -> list[str]:
        """Rewrite a user question into retrieval queries.

        Args:
            query: Current user question.
            memory_context: Conversation memory for pronoun resolution.
            paper_context: Compact paper vocabulary/representative snippets.

        Returns:
            Query rewrites used only for retrieval.
        """
        user_prompt = "\n\n".join(
            [
                "【对话记忆】",
                memory_context or "无",
                "【论文词表/代表片段】",
                paper_context or "无",
                "【当前问题】",
                query,
            ]
        )
        cache_key = json.dumps(
            {
                "query": query,
                "memory_context": memory_context,
                "paper_context": paper_context,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self._rewrite_cache:
            cached_queries = list(self._rewrite_cache[cache_key])
            self._last_rewrite_diagnostics = {
                "attempted": True,
                "from_cache": True,
                "raw": "",
                "queries": cached_queries,
                "error": "",
            }
            return list(self._rewrite_cache[cache_key])

        self._last_rewrite_diagnostics = {
            "attempted": True,
            "from_cache": False,
            "raw": "",
            "queries": [],
            "error": "",
        }
        last_error = ""
        last_content = ""
        for attempt in range(3):
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
                last_content = content
                if not content.strip():
                    last_error = "empty_response"
                rewritten_queries = self._parse_rewrite_queries(content, query)
                if rewritten_queries:
                    self._rewrite_cache[cache_key] = rewritten_queries
                    self._last_rewrite_diagnostics = {
                        "attempted": True,
                        "from_cache": False,
                        "raw": content[:1200],
                        "queries": list(rewritten_queries),
                        "error": "",
                        "attempts": attempt + 1,
                    }
                    return list(rewritten_queries)
                if not last_error:
                    last_error = "parse_empty"
            except Exception as exc:
                last_error = exc.__class__.__name__
            if attempt < 2:
                await asyncio.sleep(0.4 * (attempt + 1))

        self._last_rewrite_diagnostics = {
            "attempted": True,
            "from_cache": False,
            "raw": last_content[:1200],
            "queries": [],
            "error": last_error,
            "attempts": 3,
        }
        return []

    def _fallback_rewrite_queries(
        self,
        query: str,
        paper_context: str,
    ) -> list[str]:
        """Generate conservative paper-language fallback queries after LLM failure."""
        if (
            not query.strip()
            or not self._looks_like_english_paper(paper_context)
            or not self._needs_fallback_rewrite(query)
        ):
            return []

        paper_terms = self._paper_context_terms(paper_context)
        query_terms = self._query_bridge_terms(query, paper_terms)
        theorem_terms = re.findall(
            r"\b(?:Theorem|Lemma|Proposition|Corollary)\s+\d+(?:\.\d+)?\b",
            query,
            flags=re.IGNORECASE,
        )
        queries: list[str] = []

        if theorem_terms:
            queries.append(" ".join([*theorem_terms, *query_terms[:8]]))
        if "proof" in query_terms or "derive" in query_terms:
            queries.append(
                " ".join(
                    [
                        *theorem_terms,
                        "proof",
                        "lemma",
                        "estimate",
                        "bound",
                        "condition",
                        "assumption",
                        "result",
                    ]
                )
            )
        if "spectral information" in query_terms or "graph quantities" in query_terms:
            graph_terms = [
                term
                for term in [
                    "spectral information",
                    "graph",
                    "diameter",
                    "complement edge",
                    "neighbor set",
                    "graph constant",
                ]
                if term in paper_terms or term in query_terms
            ]
            queries.append(" ".join(graph_terms))
        if "relation" in query_terms or len(theorem_terms) >= 2:
            queries.append(
                " ".join([*theorem_terms, "relation", "assumptions", "result"])
            )
        if query_terms:
            queries.append(" ".join(query_terms[:10]))

        cleaned_queries: list[str] = []
        for raw_query in queries:
            cleaned_query = self._clean_rewrite_query(raw_query, query)
            if cleaned_query and cleaned_query not in cleaned_queries:
                cleaned_queries.append(cleaned_query)
        return cleaned_queries[:4]

    def _needs_fallback_rewrite(self, query: str) -> bool:
        """Use fallback rewrites only for questions likely to need bridge terms."""
        return bool(
            re.search(
                r"(Theorem|Lemma|Proposition|Corollary|不依赖|谱信息|图量|哪些图|"
                r"关系|比较|区别|差别|哪些|有哪些|优缺点|弊端|如何从|推出|推导)",
                query,
                flags=re.IGNORECASE,
            )
        )

    def _looks_like_english_paper(self, paper_context: str) -> bool:
        text = paper_context.strip()
        if not text:
            return False
        ascii_letters = len(re.findall(r"[A-Za-z]", text))
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        return ascii_letters > max(cjk_chars * 2, 80)

    def _paper_context_terms(self, paper_context: str) -> set[str]:
        lowered = paper_context.lower()
        candidates = [
            "assumption",
            "definition",
            "lemma",
            "proof",
            "theorem",
            "corollary",
            "remark",
            "result",
            "lyapunov functional",
            "spectral information",
            "network structure",
            "graph",
            "diameter",
            "complement edge",
            "neighbor set",
            "flocking",
            "pattern formation",
            "numerical experiment",
            "simulation",
            "parameter",
            "initial data",
        ]
        return {term for term in candidates if term in lowered}

    def _query_bridge_terms(self, query: str, paper_terms: set[str]) -> list[str]:
        bridge_rules = [
            (("定理",), "theorem"),
            (("引理",), "lemma"),
            (("推论",), "corollary"),
            (("证明", "如何从", "推出"), "proof"),
            (("推出", "得到", "推导"), "derive"),
            (("能量估计", "能量"), "energy estimate"),
            (("谱信息", "谱"), "spectral information"),
            (("图量", "图的量"), "graph quantities"),
            (("直径",), "diameter"),
            (("邻居", "邻接"), "neighbor set"),
            (("关系",), "relation"),
            (("假设", "条件"), "assumption"),
            (("结论", "结果"), "result"),
            (("数值", "实验", "仿真"), "numerical experiment"),
            (("参数", "设置"), "parameter setting"),
            (("网络结构", "网络"), "network structure"),
            (("模式形成", "队形"), "pattern formation"),
            (("哪些", "有哪些", "优缺点", "弊端"), "coverage"),
        ]
        terms: list[str] = []
        for pattern in re.findall(
            r"\b(?:Theorem|Lemma|Proposition|Corollary)\s+\d+(?:\.\d+)?\b",
            query,
            flags=re.IGNORECASE,
        ):
            terms.append(pattern)
        for needles, term in bridge_rules:
            if any(needle in query for needle in needles):
                terms.append(term)
        for term in paper_terms:
            if term in {
                "flocking",
                "pattern formation",
                "lyapunov functional",
            } and term.lower() in query.lower():
                terms.append(term)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]*", query):
            terms.append(token)

        deduped: list[str] = []
        for term in terms:
            if term and term not in deduped:
                deduped.append(term)
        return deduped

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
            query = self._clean_rewrite_query(str(raw_query), original_query)
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= 6:
                break
        return queries

    def _clean_rewrite_query(self, raw_query: str, original_query: str) -> str:
        query = re.sub(r"\s+", " ", raw_query).strip(" -*0123456789.\t\"'`,，、")
        query = query.strip()
        query = re.sub(r'^\s*["\']|["\'],?\s*$', "", query).strip()
        if not query or query == original_query:
            return ""
        if query in {"{", "}", "[", "]"} or len(query) < 4:
            return ""
        lowered_query = query.lower().strip()
        if re.search(r"^\{?\\?\"?qu(?:e|$)", lowered_query):
            return ""
        if re.search(r"[\{\}\[\]]", query) and len(query) < 40:
            return ""
        blocked_fragments = [
            '"queries"',
            "queries:",
            "queries",
            "对话记忆",
            "当前问题",
            "任务类型",
            "translation",
            "summary",
            "qa：",
            "qa:",
        ]
        if any(fragment in lowered_query for fragment in blocked_fragments):
            return ""
        if len(query) > 180:
            return ""
        return query

    async def reflect_retrieval_coverage(
        self,
        query: str,
        memory_context: str,
        retrieval_queries: list[str],
        chunks: list[dict[str, Any]],
        paper_context: str = "",
    ) -> dict[str, Any]:
        """Evaluate whether retrieved evidence covers the current question.

        Args:
            query: Current user question.
            memory_context: Conversation memory for follow-up resolution.
            retrieval_queries: Queries already used by the first retrieval pass.
            chunks: Ranked evidence chunks from the first retrieval pass.

        Returns:
            Parsed reflection JSON with sufficiency and targeted repair queries.
        """
        user_prompt = "\n\n".join(
            [
                "【对话记忆】",
                memory_context or "无",
                "【当前问题】",
                query,
                "【已用检索 query】",
                "\n".join(f"- {item}" for item in retrieval_queries) or "无",
                "【论文词表/代表片段】",
                paper_context or "无",
                "【当前证据片段】",
                self._format_reflection_chunks(chunks),
            ]
        )
        last_error = "reflection_failed"
        last_content = ""
        last_finish_reason = ""
        for attempt in range(3):
            try:
                response = await self._client_or_raise().chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": RETRIEVAL_REFLECTION_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1000,
            )
                choice = response.choices[0]
                last_finish_reason = str(getattr(choice, "finish_reason", "") or "")
                content = choice.message.content or ""
                last_content = content
                parsed = self._parse_retrieval_reflection(content)
                parsed["attempts"] = attempt + 1
                if content.strip() and parsed.get("parse_ok"):
                    return parsed
                last_error = (
                    f"empty_response finish_reason={last_finish_reason}"
                    if not content.strip()
                    else f"parse_failed finish_reason={last_finish_reason}"
                )
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {str(exc)[:240]}"
            if attempt < 2:
                await asyncio.sleep(0.4 * (attempt + 1))

        parsed = self._parse_retrieval_reflection(last_content)
        parsed["attempts"] = 3
        if not parsed.get("parse_ok"):
            parsed["reason"] = last_error
            parsed["last_error"] = last_error
            if last_finish_reason:
                parsed["finish_reason"] = last_finish_reason
        else:
            parsed["reason"] = parsed.get("reason") or last_error
        return parsed

    def _format_reflection_chunks(
        self,
        chunks: list[dict[str, Any]],
        max_chunks: int = 10,
        max_text_chars: int = 760,
    ) -> str:
        lines: list[str] = []
        for index, chunk in enumerate(chunks[:max_chunks], start=1):
            text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
            if len(text) > max_text_chars:
                text = f"{text[:max_text_chars]}..."
            lines.append(
                "\n".join(
                    [
                        f"[{index}] block_id={chunk.get('block_id', '')}",
                        (
                            f"page={chunk.get('page', '')}; "
                            f"section={chunk.get('section', 'Unknown')}; "
                            f"block_type={chunk.get('block_type', chunk.get('node_type', 'paragraph'))}"
                        ),
                        f"text={text}",
                    ]
                )
            )
        return "\n\n".join(lines) if lines else "无"

    def _parse_retrieval_reflection(self, content: str) -> dict[str, Any]:
        cleaned_content = content.strip()
        cleaned_content = re.sub(r"^```(?:json)?|```$", "", cleaned_content).strip()
        json_match = re.search(r"\{.*\}", cleaned_content, flags=re.DOTALL)
        json_content = json_match.group(0) if json_match else cleaned_content

        parse_ok = True
        try:
            parsed = json.loads(json_content)
        except json.JSONDecodeError:
            parse_ok = False
            parsed = {}
        if not isinstance(parsed, dict):
            parse_ok = False
            parsed = {}

        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in {"complete", "partial", "wrong_direction", "insufficient"}:
            legacy_sufficient = parsed.get("is_sufficient")
            decision = "complete" if legacy_sufficient is True else "insufficient"

        coverage_score = self._bounded_float(parsed.get("coverage_score", 0.0))
        covered_slots = self._coverage_slot_list(parsed.get("covered_slots", []))
        missing_slots = self._missing_slot_list(parsed.get("missing_slots", []))
        structural_actions = self._structural_action_list(
            parsed.get("structural_actions", [])
        )
        retry_queries = self._string_list(parsed.get("retry_queries", []))

        # Backward compatibility for older reflection responses.
        legacy_slots = self._slot_coverage_list(parsed.get("slot_coverage", []))
        for slot in legacy_slots:
            if slot.get("status") == "covered":
                covered_slots.append(
                    {
                        "slot": slot.get("slot", ""),
                        "evidence_block_ids": slot.get("supporting_blocks", []),
                        "reason": "legacy covered slot",
                    }
                )
            elif slot.get("slot"):
                missing_slots.append(
                    {
                        "slot": slot.get("slot", ""),
                        "needed_evidence_type": "supporting evidence",
                        "reason": slot.get("missing_reason", ""),
                    }
                )
            repair_query = slot.get("repair_query", "")
            if repair_query:
                retry_queries.append(repair_query)

        missing_aspects = [
            slot.get("slot", "")
            for slot in missing_slots
            if slot.get("slot")
        ]
        if not missing_aspects:
            missing_aspects = self._string_list(parsed.get("missing_aspects", []))

        should_retry = parsed.get("should_retry")
        if not isinstance(should_retry, bool):
            should_retry = decision != "complete"
        if structural_actions:
            should_retry = True
        is_sufficient = decision == "complete" and not should_retry

        return {
            "parse_ok": parse_ok,
            "decision": decision,
            "coverage_score": coverage_score,
            "covered_slots": covered_slots,
            "missing_slots": missing_slots,
            "structural_actions": structural_actions,
            "should_retry": should_retry,
            "is_sufficient": is_sufficient,
            "slot_coverage": legacy_slots,
            "covered_aspects": [
                slot.get("slot", "")
                for slot in covered_slots
                if slot.get("slot")
            ],
            "missing_aspects": missing_aspects,
            "retry_queries": self._string_list(retry_queries)[
                :AGENTIC_RETRIEVAL_MAX_REPAIR_QUERIES
            ],
            "reason": str(
                parsed.get("decision_reason", parsed.get("reason", ""))
            ).strip()[:500]
            or ("reflection_parse_failed" if not parse_ok else ""),
            "raw": content[:1200],
        }

    def _bounded_float(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    def _coverage_slot_list(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        slots: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            slot = re.sub(r"\s+", " ", str(item.get("slot", ""))).strip()
            if not slot:
                continue
            slots.append(
                {
                    "slot": slot[:180],
                    "evidence_block_ids": self._string_list(
                        item.get("evidence_block_ids", [])
                    )[:8],
                    "reason": re.sub(
                        r"\s+",
                        " ",
                        str(item.get("reason", "")),
                    ).strip()[:300],
                }
            )
        return slots[:6]

    def _missing_slot_list(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        slots: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            slot = re.sub(r"\s+", " ", str(item.get("slot", ""))).strip()
            if not slot:
                continue
            slots.append(
                {
                    "slot": slot[:180],
                    "needed_evidence_type": re.sub(
                        r"\s+",
                        " ",
                        str(item.get("needed_evidence_type", "")),
                    ).strip()[:180],
                    "reason": re.sub(
                        r"\s+",
                        " ",
                        str(item.get("reason", "")),
                    ).strip()[:300],
                }
            )
        return slots[:6]

    def _structural_action_list(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        allowed_actions = {
            "heading_expand",
            "table_expand",
            "abstract_expand",
            "neighbor_expand",
        }
        actions: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in value:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip().lower()
            if action not in allowed_actions:
                continue
            target_block_id = re.sub(
                r"\s+",
                " ",
                str(item.get("target_block_id", "")),
            ).strip()
            if not target_block_id:
                continue
            key = (action, target_block_id)
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                {
                    "action": action,
                    "target_block_id": target_block_id[:80],
                    "reason": re.sub(
                        r"\s+",
                        " ",
                        str(item.get("reason", "")),
                    ).strip()[:300],
                }
            )
        return actions[:4]

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            text = re.sub(r"\s+", " ", str(item)).strip()
            if text and text not in items:
                items.append(text)
        return items

    def _slot_coverage_list(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        slots: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).strip().lower()
            if status not in {"covered", "partial", "missing"}:
                status = "missing"
            slot = re.sub(r"\s+", " ", str(item.get("slot", ""))).strip()
            repair_query = re.sub(
                r"\s+",
                " ",
                str(item.get("repair_query", "")),
            ).strip()
            slots.append(
                {
                    "slot": slot,
                    "status": status,
                    "supporting_blocks": self._string_list(
                        item.get("supporting_blocks", [])
                    ),
                    "missing_reason": re.sub(
                        r"\s+",
                        " ",
                        str(item.get("missing_reason", "")),
                    ).strip()[:300],
                    "repair_query": repair_query[:180],
                }
            )
        return slots

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
        self._last_agentic_retrieval: dict[str, Any] = {}

    @property
    def last_agentic_retrieval(self) -> dict[str, Any]:
        """Return diagnostics from the latest agentic retrieval pass."""
        return dict(self._last_agentic_retrieval)

    @property
    def last_query_rewrite(self) -> dict[str, Any]:
        """Return diagnostics from the latest retrieval rewrite call."""
        return self._generator.last_rewrite_diagnostics

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
                paper_id=selected_paper_id,
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
        self._last_agentic_retrieval = {
            "enabled": ENABLE_AGENTIC_RETRIEVAL,
            "initial_top_blocks": [],
            "reflection": None,
            "structural_actions": [],
            "structural_top_blocks": [],
            "inserted_structural_blocks": [],
            "retry_queries": [],
            "repair_top_blocks": [],
            "inserted_repair_blocks": [],
            "final_top_blocks": [],
        }
        if retrieval_queries is None:
            retrieval_queries = await self._build_retrieval_queries(
                query,
                memory_context,
                paper_id=paper_id,
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
        if ENABLE_FINAL_RERANK:
            ranked_chunks = await self._rerank_final_candidates(
                query=query,
                retrieval_queries=retrieval_queries,
                ranked_chunks=ranked_chunks,
                limit=max(self._top_k * 2, 14),
            )
        self._last_agentic_retrieval["initial_top_blocks"] = [
            chunk.get("block_id") for chunk in ranked_chunks[:10]
        ]

        agentic_queries = list(retrieval_queries)
        if ENABLE_AGENTIC_RETRIEVAL:
            paper_context = self._rewrite_paper_context(paper_id)
            reflection = await self._generator.reflect_retrieval_coverage(
                query=query,
                memory_context=memory_context,
                retrieval_queries=retrieval_queries,
                chunks=ranked_chunks[:10],
                paper_context=paper_context,
            )
            structural_actions: list[dict[str, Any]] = []
            structural_chunks: list[dict[str, Any]] = []
            repair_queries = self._clean_agentic_repair_queries(
                raw_queries=reflection.get("retry_queries", []),
                original_query=query,
                existing_queries=retrieval_queries,
            )
            if (
                ENABLE_AGENTIC_FALLBACK_REPAIR
                and not reflection.get("is_sufficient", True)
                and not repair_queries
            ):
                repair_queries = self._fallback_agentic_repair_queries(
                    query=query,
                    existing_queries=retrieval_queries,
                    paper_context=paper_context,
                )
            self._last_agentic_retrieval.update(
                {
                    "reflection": reflection,
                    "structural_actions": structural_actions,
                    "structural_top_blocks": [
                        chunk.get("block_id") for chunk in structural_chunks[:8]
                    ],
                    "inserted_structural_blocks": [
                        chunk.get("block_id") for chunk in structural_chunks
                    ],
                    "retry_queries": repair_queries,
                }
            )

            can_retry = (
                bool(reflection.get("should_retry"))
                and not reflection.get("is_sufficient", True)
                and repair_queries
                and (
                    bool(reflection.get("parse_ok"))
                    or ENABLE_AGENTIC_FALLBACK_REPAIR
                )
            )

            if can_retry:
                repair_start_index = len(retrieval_queries)
                repair_candidate_records: list[dict[str, Any]] = []
                for repair_offset, repair_query in enumerate(repair_queries):
                    query_index = repair_start_index + repair_offset
                    semantic_results = await self._retriever.search(
                        repair_query,
                        paper_id=paper_id,
                        top_k=per_query_top_k,
                    )
                    for rank, chunk in enumerate(semantic_results, start=1):
                        repair_candidate_records.append(
                            {
                                "chunk": chunk,
                                "source": "repair_semantic",
                                "rank": rank,
                                "query_index": query_index,
                                "query": repair_query,
                                "is_fallback_query": False,
                            }
                        )

                    keyword_results = self._keyword_chunks(
                        repair_query,
                        paper_id,
                        limit=min(per_query_top_k, 8),
                    )
                    for rank, chunk in enumerate(keyword_results, start=1):
                        repair_candidate_records.append(
                            {
                                "chunk": chunk,
                                "source": "repair_keyword",
                                "rank": rank,
                                "query_index": query_index,
                                "query": repair_query,
                                "is_fallback_query": False,
                            }
                        )

                agentic_queries = [*retrieval_queries, *repair_queries]
                repair_ranked_chunks = self._rank_qa_candidates(
                    query=query,
                    retrieval_queries=repair_queries,
                    candidate_records=repair_candidate_records,
                    limit=max(AGENTIC_REPAIR_INSERT_LIMIT * 4, 8),
                )
                repair_ranked_chunks = self._filter_agentic_repair_chunks(
                    repair_chunks=repair_ranked_chunks,
                    repair_queries=repair_queries,
                    missing_aspects=reflection.get("missing_aspects", []),
                    limit=AGENTIC_REPAIR_INSERT_LIMIT,
                )
                self._last_agentic_retrieval["repair_top_blocks"] = [
                    chunk.get("block_id") for chunk in repair_ranked_chunks[:8]
                ]
                ranked_chunks = self._merge_agentic_repair_chunks(
                    initial_chunks=ranked_chunks,
                    repair_chunks=repair_ranked_chunks,
                    protect_top_n=AGENTIC_REPAIR_PROTECT_TOP_N,
                    insert_limit=AGENTIC_REPAIR_INSERT_LIMIT,
                )
                self._last_agentic_retrieval["inserted_repair_blocks"] = [
                    chunk.get("block_id")
                    for chunk in repair_ranked_chunks[:AGENTIC_REPAIR_INSERT_LIMIT]
                ]

        # PDF 公式和器件说明经常被解析成相邻短块；追加邻段给生成器做证据拼接，
        # 但保留已排序结果在前，避免影响检索指标里的 Top-k 排名。
        final_chunks = self._append_neighbor_evidence(
            paper_id=paper_id,
            ranked_chunks=ranked_chunks,
            window=12,
            limit=max(self._top_k * 4, 28),
        )
        self._last_agentic_retrieval["final_top_blocks"] = [
            chunk.get("block_id") for chunk in final_chunks[:10]
        ]
        self._last_agentic_retrieval["final_queries"] = agentic_queries
        return final_chunks

    def _structural_action_chunks(
        self,
        paper_id: str,
        structural_actions: list[dict[str, Any]],
        existing_chunks: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Fetch evidence by document structure before issuing new search queries."""
        if not structural_actions or limit <= 0:
            return []

        all_chunks = self._paper_library.get_chunks(paper_id)
        index_by_block_id = {
            chunk.get("block_id"): index
            for index, chunk in enumerate(all_chunks)
            if chunk.get("block_id")
        }
        seen_block_ids = {
            chunk.get("block_id")
            for chunk in existing_chunks
            if chunk.get("block_id")
        }
        selected_chunks: list[dict[str, Any]] = []

        for action_item in structural_actions:
            action = action_item.get("action", "")
            target_block_id = action_item.get("target_block_id", "")
            target_index = index_by_block_id.get(target_block_id)
            if target_index is None:
                continue

            candidate_indexes = self._structural_candidate_indexes(
                action=action,
                target_index=target_index,
                chunk_count=len(all_chunks),
            )
            for candidate_index in candidate_indexes:
                chunk = all_chunks[candidate_index]
                block_id = chunk.get("block_id")
                if (
                    not block_id
                    or block_id in seen_block_ids
                    or self._is_reference_chunk(chunk)
                ):
                    continue
                if action in {"heading_expand", "abstract_expand"} and self._looks_like_heading_chunk(chunk):
                    continue
                selected_chunks.append(
                    {
                        **chunk,
                        "retrieval_source": f"structural_{action}",
                    }
                )
                seen_block_ids.add(block_id)
                if len(selected_chunks) >= limit:
                    return selected_chunks
        return selected_chunks

    def _structural_candidate_indexes(
        self,
        action: str,
        target_index: int,
        chunk_count: int,
    ) -> list[int]:
        if action == "neighbor_expand":
            start_index = max(0, target_index - 2)
            end_index = min(chunk_count, target_index + 3)
            return [
                index
                for index in range(start_index, end_index)
                if index != target_index
            ]
        if action == "table_expand":
            start_index = max(0, target_index - 1)
            end_index = min(chunk_count, target_index + 5)
            return [
                index
                for index in range(start_index, end_index)
                if index != target_index
            ]
        if action in {"heading_expand", "abstract_expand"}:
            return list(range(target_index + 1, min(chunk_count, target_index + 7)))
        return []

    def _looks_like_heading_chunk(self, chunk: dict[str, Any]) -> bool:
        node_type = str(
            chunk.get("block_type", chunk.get("node_type", ""))
        ).strip().lower()
        if node_type in {
            "heading",
            "title",
            "section_header",
            "section_title",
            "table_title",
            "abstract_title",
        }:
            return True
        text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
        if not text:
            return False
        if text.lower() in {
            "abstract",
            "摘要",
            "conclusion",
            "references",
            "参考文献",
        }:
            return True
        if len(text) <= 120 and re.match(
            r"^(?:\d+(?:\.\d+)+|第\s*\d+\s*章)\s+[\w\u4e00-\u9fff][\w\u4e00-\u9fff\- ]+$",
            text,
        ):
            return True
        return bool(
            len(text) <= 120
            and re.match(r"^(?:表|图|table|fig\.|figure)\s*[\dIVXivx]+", text, re.I)
        )

    def _filter_agentic_repair_chunks(
        self,
        repair_chunks: list[dict[str, Any]],
        repair_queries: list[str],
        missing_aspects: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Keep only repair chunks that look aligned with the missing slots."""
        if not repair_chunks or limit <= 0:
            return []

        repair_keywords = self._expanded_keywords(
            " ".join([*repair_queries, *missing_aspects])
        )
        scored_chunks: list[tuple[float, int, dict[str, Any]]] = []
        for index, chunk in enumerate(repair_chunks):
            if self._is_reference_chunk(chunk):
                continue
            text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
            if not text or self._looks_like_navigation_or_boilerplate(text):
                continue
            normalized_text = text.lower()
            keyword_hits = sum(
                1 for keyword in repair_keywords if keyword in normalized_text
            )
            formula_or_metric = self._has_measurement_or_formula_signal(text)
            causal_or_method = self._has_method_or_causal_signal(normalized_text)
            if keyword_hits <= 0 and not (formula_or_metric and causal_or_method):
                continue

            score = float(keyword_hits)
            answer_score = chunk.get("answer_score")
            if isinstance(answer_score, (int, float)):
                score += min(max(float(answer_score), -5.0), 12.0) * 0.08
            if formula_or_metric:
                score += 0.5
            if causal_or_method:
                score += 0.3
            scored_chunks.append((score, -index, chunk))

        scored_chunks.sort(reverse=True)
        return [
            {
                **chunk,
                "retrieval_source": self._append_source_label(
                    chunk.get("retrieval_source", ""),
                    "agentic_repair",
                ),
            }
            for _, _, chunk in scored_chunks[:limit]
        ]

    def _merge_agentic_repair_chunks(
        self,
        initial_chunks: list[dict[str, Any]],
        repair_chunks: list[dict[str, Any]],
        protect_top_n: int,
        insert_limit: int,
    ) -> list[dict[str, Any]]:
        """Insert a small number of repair chunks without reranking the world."""
        if not repair_chunks or insert_limit <= 0:
            return initial_chunks

        protected_count = min(max(protect_top_n, 0), len(initial_chunks))
        merged_chunks: list[dict[str, Any]] = []
        seen_block_ids: set[str] = set()

        for chunk in initial_chunks[:protected_count]:
            block_id = chunk.get("block_id")
            if block_id:
                seen_block_ids.add(block_id)
            merged_chunks.append(chunk)

        inserted = 0
        for chunk in repair_chunks:
            block_id = chunk.get("block_id")
            if not block_id or block_id in seen_block_ids:
                continue
            merged_chunks.append(chunk)
            seen_block_ids.add(block_id)
            inserted += 1
            if inserted >= insert_limit:
                break

        for chunk in initial_chunks[protected_count:]:
            block_id = chunk.get("block_id")
            if block_id and block_id in seen_block_ids:
                continue
            if block_id:
                seen_block_ids.add(block_id)
            merged_chunks.append(chunk)

        return merged_chunks

    def _append_source_label(self, source: Any, label: str) -> str:
        parts = [
            part
            for part in str(source or "").split(",")
            if part
        ]
        if label not in parts:
            parts.append(label)
        return ",".join(parts)

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
            if ENABLE_FINAL_RERANK:
                item["score"] += self._evidence_signal_score(item["chunk"], global_keywords)

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

    async def _rerank_final_candidates(
        self,
        query: str,
        retrieval_queries: list[str],
        ranked_chunks: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Let the CrossEncoder judge the merged QA candidate pool."""
        if not ranked_chunks:
            return []

        rerank_query = self._final_rerank_query(query, retrieval_queries)
        reranked_chunks = await self._retriever.rerank_chunks(
            rerank_query,
            ranked_chunks,
        )
        adjusted_chunks = [
            {
                **chunk,
                "final_rank_score": self._final_rank_score(
                    chunk=chunk,
                    rerank_index=index,
                    rerank_query=rerank_query,
                ),
            }
            for index, chunk in enumerate(reranked_chunks)
        ]
        adjusted_chunks.sort(
            key=lambda chunk: chunk["final_rank_score"],
            reverse=True,
        )
        return adjusted_chunks[:limit]

    def _final_rerank_query(
        self,
        query: str,
        retrieval_queries: list[str],
    ) -> str:
        query_parts: list[str] = []
        for query_part in [query, *retrieval_queries]:
            compact_query = re.sub(r"\s+", " ", query_part).strip()
            if compact_query and compact_query not in query_parts:
                query_parts.append(compact_query)
            if len(query_parts) >= 4:
                break
        return " ".join(query_parts)

    def _final_rank_score(
        self,
        chunk: dict[str, Any],
        rerank_index: int,
        rerank_query: str,
    ) -> float:
        rerank_score = chunk.get("rerank_score")
        if isinstance(rerank_score, (int, float)):
            base_score = float(rerank_score) * 0.65
        else:
            base_score = 0.0

        answer_score = chunk.get("answer_score")
        if isinstance(answer_score, (int, float)):
            base_score += min(max(float(answer_score), -8.0), 24.0) * 0.22

        return (
            base_score
            + self._ranking_quality_adjustment(chunk)
            + self._query_evidence_alignment(rerank_query, chunk)
            - rerank_index * 0.001
        )

    def _score_query_indexes(self, record: dict[str, Any]) -> set[int]:
        source = record["source"]
        if source not in {"semantic", "keyword", "repair_semantic", "repair_keyword"}:
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
            "repair_semantic": 10.5,
            "formula": 3.0,
            "keyword": 1.5,
            "repair_keyword": 1.4,
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
        if ENABLE_FINAL_RERANK:
            score += self._ranking_quality_adjustment(chunk)

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

    def _ranking_quality_adjustment(self, chunk: dict[str, Any]) -> float:
        text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        normalized_text = text.lower()
        adjustment = 0.0

        if self._looks_like_navigation_or_boilerplate(text):
            adjustment -= 3.5
        if self._looks_like_page_header_or_footer(text):
            adjustment -= 2.0
        if self._looks_like_abstract_or_summary_without_specifics(text):
            adjustment -= 1.0

        if self._has_measurement_or_formula_signal(text):
            adjustment += 1.6
        if self._has_table_or_metric_signal(text):
            adjustment += 1.1
        if self._has_method_or_causal_signal(normalized_text):
            adjustment += 0.8

        return adjustment

    def _query_evidence_alignment(
        self,
        query: str,
        chunk: dict[str, Any],
    ) -> float:
        text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        normalized_text = text.lower()
        normalized_query = query.lower()
        section = str(chunk.get("section", "")).lower()
        adjustment = 0.0

        detail_query = any(
            marker in normalized_query
            for marker in [
                "哪些",
                "哪几",
                "分别",
                "相比",
                "比较",
                "达标",
                "超过",
                "指标",
                "标准",
                "规程",
                "which",
                "what",
                "compare",
                "compared",
                "standard",
                "criteria",
                "indicator",
                "grade",
                "class",
            ]
        )
        query_has_standard_code = bool(
            re.search(r"\b[A-Z]{2,}\s*\d{2,}(?:-\d{2,4})?\b", query, flags=re.IGNORECASE)
        )
        text_has_standard_code = bool(
            re.search(r"\b[A-Z]{2,}\s*\d{2,}(?:-\d{2,4})?\b", text, flags=re.IGNORECASE)
        )
        standard_query = query_has_standard_code or any(
            marker in normalized_query
            for marker in ["a级", "a 级", "标准", "规程", "grade", "class"]
        )

        if standard_query and (
            text_has_standard_code
            or re.search(r"检定|规程|a\s*级|grade|class", normalized_text)
        ):
            adjustment += 1.4
        if standard_query and self._has_table_or_metric_signal(text):
            adjustment += 2.2
        if detail_query and self._has_measurement_or_formula_signal(text):
            adjustment += 0.8
        if detail_query and re.search(r"(?:表|table)\s*\d+", text, flags=re.IGNORECASE):
            adjustment += 1.0

        high_level_section = any(
            marker in section
            for marker in ["摘要", "abstract", "总结", "结论", "conclusion", "summary"]
        )
        if detail_query and high_level_section and not re.search(
            r"(?:表|table)\s*\d+|检定指标|实验组|标准熔化|重复性|示值误差",
            text,
            flags=re.IGNORECASE,
        ):
            adjustment -= 3.0

        return adjustment

    def _evidence_signal_score(
        self,
        chunk: dict[str, Any],
        global_keywords: list[str],
    ) -> float:
        text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
        normalized_text = text.lower()
        if not text:
            return -5.0

        score = 0.0
        if self._has_measurement_or_formula_signal(text):
            score += 1.2
        if self._has_table_or_metric_signal(text):
            score += 0.8
        if self._has_method_or_causal_signal(normalized_text):
            score += 0.6

        keyword_hits = sum(1 for keyword in global_keywords if keyword in normalized_text)
        if keyword_hits and self._has_measurement_or_formula_signal(text):
            score += min(keyword_hits, 4) * 0.25
        return score

    def _looks_like_navigation_or_boilerplate(self, text: str) -> bool:
        compact_text = re.sub(r"\s+", "", text)
        if not compact_text:
            return True
        if re.search(r"\.{6,}|…{2,}", text):
            return True
        if re.fullmatch(r"[\dIVXivx一二三四五六七八九十]+", compact_text):
            return True
        if len(compact_text) < 12 and re.search(r"第.+[章节]|目录|contents", text, re.IGNORECASE):
            return True
        return False

    def _looks_like_page_header_or_footer(self, text: str) -> bool:
        compact_text = re.sub(r"\s+", "", text)
        if len(compact_text) > 45:
            return False
        header_terms = [
            "学位论文",
            "硕士论文",
            "博士论文",
            "journal",
            "proceedings",
            "conference",
        ]
        return any(term in compact_text.lower() for term in header_terms)

    def _looks_like_abstract_or_summary_without_specifics(self, text: str) -> bool:
        compact_text = re.sub(r"\s+", "", text)
        if len(compact_text) < 120:
            return False
        abstract_markers = ["摘要", "abstract", "总结", "conclusion"]
        has_marker = any(marker in text.lower() for marker in abstract_markers)
        return has_marker and not self._has_measurement_or_formula_signal(text)

    def _has_measurement_or_formula_signal(self, text: str) -> bool:
        unit_pattern = (
            r"(?:℃|°c|k/min|℃/min|j/g|mj|mw|w|v|mv|a|ma|hz|khz|mhz|"
            r"ms|s|mm|μm|um|nm|%|倍|次|级|阶)"
        )
        has_number_with_unit = bool(
            re.search(r"\d+(?:\.\d+)?\s*" + unit_pattern, text, flags=re.IGNORECASE)
        )
        has_formula = bool(
            re.search(r"(?:公式|式|equation|eq\.?)\s*[\(（]?\d+[\)）]?", text, flags=re.IGNORECASE)
            or re.search(r"\([0-9]{1,3}\)\s*$", text)
        )
        return has_number_with_unit or has_formula

    def _has_table_or_metric_signal(self, text: str) -> bool:
        metric_terms = [
            "精度",
            "准确度",
            "重复性",
            "误差",
            "灵敏度",
            "分辨率",
            "扫描速率",
            "升温速率",
            "温度范围",
            "性能",
            "指标",
            "parameter",
            "accuracy",
            "precision",
            "resolution",
            "sensitivity",
            "performance",
        ]
        normalized_text = text.lower()
        has_metric_term = any(term in normalized_text for term in metric_terms)
        has_table_hint = bool(re.search(r"^(表|table)\s*\d+", text.strip(), flags=re.IGNORECASE))
        return has_table_hint or (has_metric_term and bool(re.search(r"\d", text)))

    def _has_method_or_causal_signal(self, normalized_text: str) -> bool:
        markers = [
            "因此",
            "所以",
            "从而",
            "为了",
            "通过",
            "采用",
            "基于",
            "实现",
            "解决",
            "原因",
            "because",
            "therefore",
            "thus",
            "by using",
            "based on",
            "implemented",
        ]
        return any(marker in normalized_text for marker in markers)

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
        use_rewrite: bool = True,
        paper_id: str | None = None,
    ) -> list[str]:
        rewritten_queries = []
        if use_rewrite:
            rewritten_queries = await self._generator.rewrite_retrieval_queries(
                query,
                memory_context,
                paper_context=self._rewrite_paper_context(paper_id),
            )
        # 原问题始终作为干净兜底；对话历史只交给 rewrite 做实体消解，避免整段历史污染检索。
        queries = [*rewritten_queries, query] if rewritten_queries else [query]
        return list(dict.fromkeys(query_text for query_text in queries if query_text))

    def _clean_agentic_repair_queries(
        self,
        raw_queries: list[str],
        original_query: str,
        existing_queries: list[str],
    ) -> list[str]:
        existing = {
            re.sub(r"\s+", " ", query).strip().lower()
            for query in [original_query, *existing_queries]
            if query
        }
        cleaned_queries: list[str] = []
        for raw_query in raw_queries:
            query = self._generator._clean_rewrite_query(str(raw_query), original_query)
            normalized_query = query.lower()
            if not query or normalized_query in existing:
                continue
            if normalized_query in {item.lower() for item in cleaned_queries}:
                continue
            cleaned_queries.append(query)
            if len(cleaned_queries) >= AGENTIC_RETRIEVAL_MAX_REPAIR_QUERIES:
                break
        return cleaned_queries

    def _fallback_agentic_repair_queries(
        self,
        query: str,
        existing_queries: list[str],
        paper_context: str,
    ) -> list[str]:
        """Build conservative repair queries from real paper context snippets."""
        if not paper_context:
            return []

        query_keywords = set(self._expanded_keywords(" ".join([query, *existing_queries])))
        existing = {
            re.sub(r"\s+", " ", item).strip().lower()
            for item in [query, *existing_queries]
            if item
        }
        candidates: list[tuple[float, str]] = []

        for raw_line in paper_context.splitlines():
            text_match = re.search(r"\bText:\s*(.+)$", raw_line)
            text = text_match.group(1) if text_match else raw_line
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue

            normalized_text = text.lower()
            line_keywords = set(self._expanded_keywords(text))
            overlap = len(query_keywords & line_keywords)
            score = float(overlap)
            if re.search(
                r"\b(?:Theorem|Lemma|Definition|Corollary|Proposition)\s+\d",
                text,
            ):
                score += 1.5
            if self._has_measurement_or_formula_signal(text):
                score += 0.8
            if self._has_method_or_causal_signal(normalized_text):
                score += 0.5
            if not candidates:
                score += 0.4

            words = re.findall(r"[A-Za-z0-9_+\\().,;:/-]+|[α-ωΑ-Ω]+|[\u4e00-\u9fff]+", text)
            repair_query = " ".join(words[:24]).strip()
            if len(repair_query) > 160:
                repair_query = repair_query[:160].rsplit(" ", 1)[0]
            repair_query = self._generator._clean_rewrite_query(repair_query, query)
            if not repair_query or repair_query.lower() in existing:
                continue
            candidates.append((score, repair_query))

        candidates.sort(key=lambda item: item[0], reverse=True)
        repair_queries: list[str] = []
        for _, repair_query in candidates:
            if repair_query not in repair_queries:
                repair_queries.append(repair_query)
            if len(repair_queries) >= AGENTIC_RETRIEVAL_MAX_REPAIR_QUERIES:
                break
        return repair_queries

    def _rewrite_paper_context(self, paper_id: str | None) -> str:
        """Build compact paper vocabulary for query rewriting.

        The rewrite model should not guess paper-specific terminology from
        pretraining alone. A few representative chunks give it the paper's real
        language, symbols, theorem names, and section vocabulary while keeping
        the prompt small.
        """
        if not paper_id:
            return ""

        non_reference_chunks = [
            chunk
            for chunk in self._paper_library.get_chunks(paper_id)
            if str(chunk.get("section", "")).strip().lower()
            not in {"references", "reference", "bibliography"}
        ]
        front_matter_chunks = non_reference_chunks[:8]
        math_statement_chunks = [
            chunk
            for chunk in non_reference_chunks
            if re.search(
                r"\b(?:Theorem|Lemma|Definition|Corollary|Proposition)\s+\d",
                str(chunk.get("text", "")),
            )
        ][:6]
        representative_chunks = self._paper_library.representative_chunks(
            paper_id,
            max_chunks=4,
        )
        chunks = [*front_matter_chunks, *math_statement_chunks, *representative_chunks]

        context_lines: list[str] = []
        seen_block_ids: set[str] = set()
        for chunk in chunks:
            block_id = str(chunk.get("block_id", ""))
            if not block_id or block_id in seen_block_ids:
                continue
            seen_block_ids.add(block_id)

            section = str(chunk.get("section", "")).strip() or "Unknown"
            text = re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()
            if not text:
                continue
            max_text_chars = 760 if len(context_lines) == 0 else 420
            if len(text) > max_text_chars:
                text = f"{text[:max_text_chars]}..."
            context_lines.append(
                f"[{block_id} p{chunk.get('page', '?')}] Section: {section}; Text: {text}"
            )
            if len(context_lines) >= 12:
                break

        return "\n".join(context_lines)

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
