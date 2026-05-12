"""PDF parsing utilities for ScholarQA."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import fitz


class PDFParser:
    """Parse academic PDFs into paragraph-level blocks with source metadata."""

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
                paragraph_index = 0

                for raw_block in ordered_blocks:
                    text = self._clean_text(raw_block[4])
                    if not text:
                        continue

                    if self._is_front_matter(text, page_index, seen_main_content):
                        continue

                    # 论文问答需要稳定出处，因此按页面内自然阅读顺序生成段落号。
                    paragraph_index += 1
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
                            "block_id": f"p{page_index}-b{paragraph_index:03d}",
                            "paragraph_index": paragraph_index,
                        }
                    )

        return document_blocks

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
        if len(text) > 120 or "\n" in text:
            return False

        normalized_text = text.strip().lower()
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
        }
        if normalized_text in known_headings:
            return True

        numbered_heading = re.match(r"^\d+(\.\d+)*\s+[A-Za-z][\w\- ]+$", text)
        roman_heading = re.match(r"^[IVX]+\.\s+[A-Za-z][\w\- ]+$", text)
        return bool(numbered_heading or roman_heading)

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
