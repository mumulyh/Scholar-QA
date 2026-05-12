"""Lightweight prompt composer for ScholarQA answer generation."""

from __future__ import annotations

import re
from typing import Any

from prompts import format_context


class AnswerComposer:
    """Build focused QA prompts without making factual conclusions."""

    def compose_user_prompt(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        memory_context: str = "",
        rewritten_queries: list[str] | None = None,
    ) -> str:
        """Compose the user prompt passed to the LLM.

        Args:
            query: Original user question.
            chunks: Retrieved paper evidence chunks.
            memory_context: Conversation memory used for pronoun resolution.
            rewritten_queries: Retrieval rewrites used to collect evidence.

        Returns:
            A structured prompt that asks the LLM to answer like a paper tutor.
        """
        subquestions = self.detect_subquestions(query)
        entities = self.detect_entities(query)
        notation_notes = self.detect_notation_notes(query)
        context_block = format_context(chunks)

        # 这里把“如何回答”从“论文事实”中拆开，避免模型把风格指令当成证据。
        prompt_parts = [
            "【对话记忆】",
            memory_context or "无",
            "注意：对话记忆只能用于理解指代，不能作为论文事实来源。",
            "【用户原问题】",
            query,
            "【拆分后的子问题】",
            self._format_list(subquestions, empty_text="未检测到需要拆分的子问题"),
            "【用户问题中的对象/术语】",
            self._format_list(entities, empty_text="未检测到显式对象或术语"),
            "【可能的输入更正】",
            self._format_list(notation_notes, empty_text="无"),
        ]

        if rewritten_queries:
            prompt_parts.extend(
                [
                    "【检索改写，仅用于说明证据来源，不是论文事实】",
                    self._format_list(rewritten_queries, empty_text="无"),
                ]
            )

        prompt_parts.extend(
            [
                "【论文片段】",
                context_block,
                "【回答风格要求】",
                self.answer_style_instruction(),
                "请现在基于【论文片段】回答【用户原问题】。",
            ]
        )
        return "\n\n".join(prompt_parts)

    def detect_subquestions(self, query: str) -> list[str]:
        """Detect explicit subquestions in a user query.

        Args:
            query: Original user question.

        Returns:
            Ordered subquestions. The original query is returned when splitting is
            not useful.
        """
        normalized_query = re.sub(r"\s+", " ", query).strip()
        if not normalized_query:
            return []

        # 优先按问号/分号切分，因为这类符号通常表示用户真的问了多个点。
        segments = re.split(r"[？?；;]+", normalized_query)
        questions = [segment.strip(" ，,。.") for segment in segments if segment.strip()]

        if len(questions) <= 1:
            questions = self._split_by_connectors(normalized_query)

        if len(questions) <= 1:
            return [normalized_query]
        return list(dict.fromkeys(questions))

    def detect_entities(self, query: str) -> list[str]:
        """Extract visible entities from the user query for answer planning.

        Args:
            query: Original user question.

        Returns:
            Entity-like strings copied from the query only.
        """
        entities: list[str] = []
        english_entities = re.findall(r"\b[A-Za-z]{2,}[A-Za-z0-9_-]*\b", query)
        chinese_entities = re.findall(
            r"[\u4e00-\u9fffA-Za-z0-9_-]{2,}"
            r"(?:芯片|器件|模块|电路|公式|范围|参数|放大器|增益|接口)",
            query,
        )
        for entity in [*english_entities, *chinese_entities]:
            cleaned_entity = entity.strip()
            if cleaned_entity and cleaned_entity not in entities:
                entities.append(cleaned_entity)
        return entities

    def detect_notation_notes(self, query: str) -> list[str]:
        """Detect small user-input hints without turning them into evidence.

        Args:
            query: Original user question.

        Returns:
            Notes that help the LLM understand the question wording.
        """
        notes: list[str] = []
        # 只处理用户原文已经写出的拼写；这里是意图理解，不是论文证据或检索加权。
        if re.search(r"\bpja\b", query, flags=re.IGNORECASE):
            notes.append(
                "用户写了 pja，可能是在问 PGA（可编程增益放大器）；"
                "这只是理解用户意图，不能当作论文证据。"
            )
        return notes

    def answer_style_instruction(self) -> str:
        """Return stable answer-style instructions for single-paper QA.

        Returns:
            Chinese instructions for a tutor-like grounded answer.
        """
        return "\n".join(
            [
                "1. 第一段先直接回答：有、没有直接提到、或文中没有明确说明但相关内容是……。",
                "2. 如果用户有多个子问题，逐项回答，不要混成一个结论。",
                "3. 先说结论，再解释论文里的相关器件、公式、模块或实验设置。",
                "4. 面向论文新手：首次出现缩写时，用括号解释；不要用模糊的“该模块/该器件”代替明确对象。",
                "5. 如果论文直接回答了问题，要给出具体答案、作用和页码段落。",
                "6. 如果片段说某个对象用于采集、模数转换、信号转换或处理，要明确把该对象作为对应功能的答案；不要前文说了对象，结尾又说无法确定。",
                "7. 如果论文没有固定数值但给了公式或条件，要说明没有固定范围，并解释公式/条件决定了什么。",
                "8. 如果用户问的对象不存在或片段未直接支持，要说“论文没有直接说明 X”；随后说明片段里最接近的 Y 是什么、它和 X 的关系是什么、不能确定什么。",
                "9. 如果片段中有多个相似对象，必须用“这里要区分……”明确分开它们，不能把一个对象的属性套到另一个对象上。",
                "10. 每个关键判断都要标注来源，格式为“（页码 X，段落 pX-bYYY）”。",
                "11. 不要补充论文片段之外的器件参数、功能范围、实验数值或外部常识。",
                "12. 结尾用一句“所以：……”回扣用户的每个子问题，确保最终结论和前文证据一致。",
                "13. 简单问题控制在 2-4 句话；多子问题可以分点，但不要写成长篇综述。",
            ]
        )

    def _split_by_connectors(self, query: str) -> list[str]:
        question_markers = (
            "是什么",
            "多少",
            "范围",
            "有没有",
            "是否",
            "哪些",
            "怎么",
            "为什么",
            "区别",
        )
        if sum(marker in query for marker in question_markers) < 2:
            return [query]

        candidates = re.split(r"(?:以及|还有|并且|同时|另外|，|,)\s*", query)
        questions = [candidate.strip(" ？?。.") for candidate in candidates]
        return [question for question in questions if question]

    def _format_list(self, values: list[str], empty_text: str) -> str:
        if not values:
            return empty_text
        return "\n".join(f"- {value}" for value in values)
