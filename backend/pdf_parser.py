"""PDF parsing utilities for ScholarQA."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


@dataclass(slots=True)
class _PageTextBlock:
    """One cleaned text block from a PDF page before paragraph merging."""

    text: str
    source_block_id: str
    x0: float
    y0: float
    x1: float
    y1: float


class PDFParser:
    """Parse academic PDFs into paragraph-level blocks with source metadata."""

    _TARGET_PARAGRAPH_CHARS = 520
    _MIN_PARAGRAPH_CHARS = 260
    _MAX_PARAGRAPH_CHARS = 700

    async def parse(self, pdf_path: Path, paper_id: str) -> list[dict[str, Any]]:
        """Parse a PDF file into text blocks.

        Args:
            pdf_path: Local path to the uploaded PDF.
            paper_id: Stable identifier assigned to this paper.

        Returns:
            A list of dictionaries shaped as
            ``[{text, page, section, block_id, paper_id, paragraph_index}]``.
        """
        return await asyncio.to_thread(self._parse_sync, pdf_path, paper_id)

    def _parse_sync(self, pdf_path: Path, paper_id: str) -> list[dict[str, Any]]:
        document_blocks: list[dict[str, Any]] = []
        current_section = "Unknown"
        seen_main_content = False

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                page_blocks = page.get_text("blocks")
                ordered_blocks = sorted(page_blocks, key=lambda item: (item[1], item[0]))
                raw_blocks: list[_PageTextBlock] = []
                paragraph_index = 0

                for raw_block in ordered_blocks:
                    text = self._clean_text(raw_block[4])
                    if not text:
                        continue

                    if self._is_front_matter(text, page_index, seen_main_content):
                        continue

                    paragraph_index += 1
                    raw_blocks.append(
                        _PageTextBlock(
                            text=text,
                            source_block_id=f"p{page_index}-b{paragraph_index:03d}",
                            x0=float(raw_block[0]),
                            y0=float(raw_block[1]),
                            x1=float(raw_block[2]),
                            y1=float(raw_block[3]),
                        )
                    )

                # 论文问答需要稳定出处，因此按页面内自然阅读顺序生成段落号。
                merged_blocks = self._merge_page_blocks(raw_blocks)
                for merged_index, merged_block in enumerate(merged_blocks, start=1):
                    text = merged_block["text"]
                    if self._looks_like_section_heading(text):
                        current_section = text[:120]
                        if current_section.lower() in {"abstract", "introduction"}:
                            seen_main_content = True
                    elif current_section != "Unknown":
                        seen_main_content = True

                    document_blocks.append(
                        {
                            "paper_id": paper_id,
                            "text": text,
                            "page": page_index,
                            "section": current_section,
                            "block_id": f"p{page_index}-b{merged_index:03d}",
                            "paragraph_index": merged_index,
                            "source_block_ids": merged_block["source_block_ids"],
                        }
                    )

        return document_blocks

    def _merge_page_blocks(
        self,
        blocks: list[_PageTextBlock],
    ) -> list[dict[str, Any]]:
        """Merge line-like PDF blocks into semantic paragraphs.

        Args:
            blocks: Cleaned blocks from one page in reading order.

        Returns:
            Paragraph-like dictionaries with merged text and source block ids.
        """
        merged_blocks: list[dict[str, Any]] = []
        buffer: list[_PageTextBlock] = []

        def flush_buffer() -> None:
            if not buffer:
                return
            merged_blocks.append(
                {
                    "text": self._join_buffer_text(buffer),
                    "source_block_ids": [block.source_block_id for block in buffer],
                }
            )
            buffer.clear()

        for block in blocks:
            if self._should_stay_standalone(block.text):
                flush_buffer()
                buffer.append(block)
                flush_buffer()
                continue

            if buffer and self._should_start_new_paragraph(buffer, block):
                flush_buffer()

            buffer.append(block)

        flush_buffer()
        return merged_blocks

    def _join_buffer_text(self, blocks: list[_PageTextBlock]) -> str:
        """Join source lines while preserving readable PDF line boundaries."""
        return "\n".join(block.text for block in blocks).strip()

    def _should_stay_standalone(self, text: str) -> bool:
        """Return whether a block should not be merged into neighbors."""
        normalized_text = re.sub(r"\s+", " ", text).strip()
        if self._looks_like_section_heading(normalized_text):
            return True
        return bool(
            re.match(r"^(图|表|figure|fig\.|table)\s*[\dIVXivx]+", normalized_text)
        )

    def _should_start_new_paragraph(
        self,
        buffer: list[_PageTextBlock],
        next_block: _PageTextBlock,
    ) -> bool:
        """Decide whether the next text block starts a new paragraph."""
        buffer_text = self._join_buffer_text(buffer)
        buffer_length = self._semantic_length(buffer_text)
        previous_text = buffer[-1].text
        next_text = next_block.text

        if self._should_stay_standalone(buffer_text):
            return True

        if buffer_length >= self._MAX_PARAGRAPH_CHARS:
            return True

        if self._starts_new_enumerated_item(next_text) and buffer_length >= self._MIN_PARAGRAPH_CHARS:
            return True

        if buffer_length >= self._TARGET_PARAGRAPH_CHARS and self._ends_sentence(previous_text):
            return True

        vertical_gap = next_block.y0 - buffer[-1].y1
        if (
            buffer_length >= self._MIN_PARAGRAPH_CHARS
            and vertical_gap > self._line_gap_threshold(buffer[-1], next_block)
            and self._ends_sentence(previous_text)
        ):
            return True

        return False

    def _semantic_length(self, text: str) -> int:
        """Count content characters without layout whitespace."""
        return len(re.sub(r"\s+", "", text))

    def _ends_sentence(self, text: str) -> bool:
        """Return whether text ends like a complete sentence or clause."""
        return bool(re.search(r"[。！？；;.!?：:]\s*(?:[\]）】》)]\s*)?$", text.strip()))

    def _starts_new_enumerated_item(self, text: str) -> bool:
        """Return whether text appears to begin a numbered/list item."""
        normalized_text = text.strip()
        return bool(
            re.match(
                r"^(?:\(?\d+[\).、]|[（(][一二三四五六七八九十]+[）)]|[一二三四五六七八九十]+[、.])",
                normalized_text,
            )
        )

    def _line_gap_threshold(
        self,
        previous_block: _PageTextBlock,
        next_block: _PageTextBlock,
    ) -> float:
        """Estimate a conservative gap threshold from neighboring block heights."""
        previous_height = max(previous_block.y1 - previous_block.y0, 1.0)
        next_height = max(next_block.y1 - next_block.y0, 1.0)
        return max(previous_height, next_height) * 1.2

    def _clean_text(self, raw_text: str) -> str:
        """Normalize extracted text while preserving formulas and citations.

        Args:
            raw_text: Text returned by PyMuPDF for a single block.

        Returns:
            Cleaned paragraph text.
        """
        cleaned_lines = []
        for line in raw_text.splitlines():
            normalized_line = re.sub(r"[ \t]+", " ", line).strip()
            if normalized_line:
                cleaned_lines.append(normalized_line)
        return "\n".join(cleaned_lines).strip()

    def _looks_like_section_heading(self, text: str) -> bool:
        """Guess whether a block is a section heading.

        Args:
            text: Cleaned paragraph text.

        Returns:
            True when the paragraph resembles an academic section heading.
        """
        single_line_text = re.sub(r"\s+", " ", text).strip()
        compact_text = re.sub(r"\s+", "", text)
        if len(single_line_text) > 120 or "\n" in text:
            return False

        normalized_text = single_line_text.lower()
        known_headings = {
            "abstract",
            "introduction",
            "related work",
            "background",
            "method",
            "methods",
            "methodology",
            "experiments",
            "experiment",
            "results",
            "discussion",
            "conclusion",
            "references",
            "摘要",
            "abstract",
            "目录",
            "绪论",
            "结论",
            "总结",
            "参考文献",
            "致谢",
        }
        if normalized_text in known_headings or compact_text in known_headings:
            return True

        numbered_heading = re.match(r"^\d+(?:\.\d+)+\s+[\w\u4e00-\u9fff][\w\u4e00-\u9fff\- ]+$", single_line_text)
        roman_heading = re.match(r"^[IVX]+\.\s+[A-Za-z][\w\- ]+$", single_line_text)
        chinese_chapter_heading = re.match(
            r"^第\s*[一二三四五六七八九十百\d]+\s*[章节篇]\s+.+$",
            single_line_text,
        )
        chinese_numbered_heading = re.match(
            r"^\d+(?:\.\d+){1,3}\s+[\u4e00-\u9fffA-Za-z].+$",
            single_line_text,
        )
        return bool(
            numbered_heading
            or roman_heading
            or chinese_chapter_heading
            or chinese_numbered_heading
        )

    def _is_front_matter(
        self,
        text: str,
        page_index: int,
        seen_main_content: bool,
    ) -> bool:
        """Filter title, authors, venues, and copyright metadata.

        Args:
            text: Cleaned block text.
            page_index: 1-based page number.
            seen_main_content: Whether abstract/introduction content has started.

        Returns:
            True when the block should not become a searchable paragraph.
        """
        if page_index != 1 or seen_main_content:
            return False

        normalized_text = re.sub(r"\s+", " ", text).strip().lower()
        if normalized_text in {"abstract", "i. introduction"}:
            return False

        front_matter_markers = [
            "ieee internet of things journal",
            "authorized licensed use",
            "downloaded on",
            "corresponding author",
            "digital object identifier",
            "received ",
            "accepted ",
            "copyright",
            "this work was supported",
            "@",
        ]
        if any(marker in normalized_text for marker in front_matter_markers):
            return True

        looks_like_author_list = "," in text and not text.endswith(".") and len(text) < 260
        looks_like_title = len(text) < 220 and text.count("\n") <= 4 and not text.endswith(".")
        return looks_like_author_list or looks_like_title
