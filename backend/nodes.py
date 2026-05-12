"""Lightweight paper node models and parent-child node building."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal["section_header", "paragraph", "formula"]


@dataclass(slots=True)
class PaperNode:
    """Semantic node used by ScholarQA retrieval."""

    node_id: str
    paper_id: str
    filename: str
    node_type: NodeType
    text: str
    retrieval_text: str
    page: int
    paragraph_index: int
    block_id: str
    order: int
    section_path: list[str] = field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert the node to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the node.
        """
        return {
            "node_id": self.node_id,
            "paper_id": self.paper_id,
            "filename": self.filename,
            "node_type": self.node_type,
            "text": self.text,
            "retrieval_text": self.retrieval_text,
            "page": self.page,
            "paragraph_index": self.paragraph_index,
            "block_id": self.block_id,
            "order": self.order,
            "section_path": self.section_path,
            "section": self.section,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @property
    def section(self) -> str:
        """Return the deepest known section title.

        Returns:
            Section title, or ``Unknown``.
        """
        return self.section_path[-1] if self.section_path else "Unknown"


class PaperNodeBuilder:
    """Build parent and child nodes from parsed PDF paragraphs."""

    def __init__(self, child_max_chars: int = 720, child_overlap: int = 90) -> None:
        """Initialize node splitting parameters.

        Args:
            child_max_chars: Maximum characters per child retrieval node.
            child_overlap: Character overlap between adjacent child nodes.
        """
        self._child_max_chars = child_max_chars
        self._child_overlap = child_overlap

    def build(
        self,
        paper_id: str,
        filename: str,
        parsed_blocks: list[dict[str, Any]],
    ) -> tuple[list[PaperNode], list[PaperNode]]:
        """Build parent and child nodes.

        Args:
            paper_id: Paper identifier.
            filename: Original uploaded filename.
            parsed_blocks: Paragraph-like blocks returned by ``PDFParser``.

        Returns:
            A tuple of ``(parent_nodes, child_nodes)``.
        """
        parent_nodes: list[PaperNode] = []
        child_nodes: list[PaperNode] = []

        for order, block in enumerate(parsed_blocks, start=1):
            text = block.get("text", "").strip()
            if not text:
                continue

            section = block.get("section") or "Unknown"
            block_id = block["block_id"]
            node_type = self._guess_node_type(text, section)
            parent_node_id = f"{paper_id}:{block_id}:parent"
            parent_node = PaperNode(
                node_id=parent_node_id,
                paper_id=paper_id,
                filename=filename,
                node_type=node_type,
                text=text,
                retrieval_text=self._retrieval_text(
                    filename=filename,
                    section=section,
                    node_type=node_type,
                    text=text,
                ),
                page=int(block["page"]),
                paragraph_index=int(block["paragraph_index"]),
                block_id=block_id,
                order=order,
                section_path=[section] if section else [],
                parent_id=None,
                metadata={"source": "parent"},
            )
            parent_nodes.append(parent_node)

            for child_index, child_text in enumerate(self._split_child_text(text), start=1):
                child_node = PaperNode(
                    node_id=f"{paper_id}:{block_id}:child-{child_index:02d}",
                    paper_id=paper_id,
                    filename=filename,
                    node_type=node_type,
                    text=child_text,
                    retrieval_text=self._retrieval_text(
                        filename=filename,
                        section=section,
                        node_type=node_type,
                        text=child_text,
                    ),
                    page=int(block["page"]),
                    paragraph_index=int(block["paragraph_index"]),
                    block_id=block_id,
                    order=order,
                    section_path=[section] if section else [],
                    parent_id=parent_node_id,
                    metadata={
                        "source": "child",
                        "child_index": child_index,
                    },
                )
                child_nodes.append(child_node)

        return parent_nodes, child_nodes

    def _retrieval_text(
        self,
        filename: str,
        section: str,
        node_type: NodeType,
        text: str,
    ) -> str:
        return "\n".join(
            [
                f"Paper: {filename}",
                f"Section: {section}",
                f"Node type: {node_type}",
                text,
            ]
        )

    def _split_child_text(self, text: str) -> list[str]:
        if len(text) <= self._child_max_chars:
            return [text]

        # 子节点越小，命中公式/术语越准；保留少量重叠，避免割裂定义。
        pieces: list[str] = []
        start_index = 0
        while start_index < len(text):
            end_index = min(start_index + self._child_max_chars, len(text))
            if end_index < len(text):
                split_index = self._best_split_index(text, start_index, end_index)
                end_index = max(split_index, start_index + self._child_max_chars // 2)
            pieces.append(text[start_index:end_index].strip())
            if end_index >= len(text):
                break
            start_index = max(0, end_index - self._child_overlap)

        return [piece for piece in pieces if piece]

    def _best_split_index(self, text: str, start_index: int, end_index: int) -> int:
        window = text[start_index:end_index]
        split_candidates = [
            window.rfind("\n"),
            window.rfind(". "),
            window.rfind("; "),
            window.rfind("。"),
            window.rfind("；"),
        ]
        best_local_index = max(split_candidates)
        if best_local_index <= 0:
            return end_index
        return start_index + best_local_index + 1

    def _guess_node_type(self, text: str, section: str) -> NodeType:
        normalized_text = text.strip()
        if normalized_text == section and len(normalized_text) <= 120:
            return "section_header"
        if self._looks_like_formula(normalized_text):
            return "formula"
        return "paragraph"

    def _looks_like_formula(self, text: str) -> bool:
        if len(text) > 260:
            return False
        math_marks = set("=≤≥∑∏√∇λθφψαβγ+-*/^_{}[]()")
        math_count = sum(1 for char in text if char in math_marks)
        has_equation_number = bool(re.search(r"\(\d+\)\s*$", text))
        has_latex_command = "\\" in text or "mathbb" in text or "frac" in text
        return has_equation_number or has_latex_command or math_count >= 4
