"""Streamlit frontend for ScholarQA."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from html import escape
from typing import Any, Generator

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
raw_api_base_url = os.getenv("SCHOLARQA_API_URL", "http://127.0.0.1:8000")
API_BASE_URL = raw_api_base_url.replace(
    "http://localhost",
    "http://127.0.0.1",
    1,
).rstrip("/")


def normalize_latex_markdown(text: str) -> str:
    """Convert common LLM LaTeX delimiters into Streamlit-friendly Markdown.

    Args:
        text: Assistant text that may contain raw LaTeX delimiters.

    Returns:
        Text with math delimiters normalized for Streamlit rendering.
    """
    normalized_text = text.replace("\\[", "$$").replace("\\]", "$$")
    normalized_text = normalized_text.replace("\\(", "$").replace("\\)", "$")

    # 数学公式放进引用块时容易被当作普通文本，这里只移除公式行的引用符号。
    normalized_lines = []
    for line in normalized_text.splitlines():
        is_formula_line = "$" in line and line.lstrip().startswith(">")
        if is_formula_line:
            normalized_lines.append(re.sub(r"^\s*>\s?", "", line))
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


def citation_excerpt(text: str, max_length: int = 320) -> str:
    """Build a compact citation quote for display.

    Args:
        text: Raw citation text.
        max_length: Maximum visible quote length.

    Returns:
        A single-line excerpt.
    """
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if len(cleaned_text) <= max_length:
        return cleaned_text
    return cleaned_text[:max_length].rstrip() + "..."


def init_state() -> None:
    """Initialize Streamlit session state."""
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("multi_messages", [])
    st.session_state.setdefault("selected_paper_id", None)
    st.session_state.setdefault("selected_paper_ids", [])
    st.session_state.setdefault("papers", [])
    st.session_state.setdefault("pending_prompt", None)


def inject_style() -> None:
    """Inject subtle academic styling and lightweight animations."""
    st.markdown(
        """
        <style>
        :root {
            --scholar-border: rgba(128, 128, 128, 0.18);
            --scholar-muted: rgba(128, 128, 128, 0.10);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
        }
        [data-testid="stChatMessage"] {
            animation: fadeInMessage 220ms ease-out;
            border-bottom: 1px solid var(--scholar-border);
        }
        @keyframes fadeInMessage {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .upload-pulse {
            display: inline-block;
            animation: uploadPulse 880ms ease-out;
        }
        @keyframes uploadPulse {
            0% { transform: scale(0.86); opacity: 0.35; }
            45% { transform: scale(1.08); opacity: 1; }
            100% { transform: scale(1); opacity: 1; }
        }
        .translation-card {
            border: 1px solid var(--scholar-border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            background: var(--scholar-muted);
        }
        .citation-text {
            border-left: 3px solid var(--scholar-border);
            padding-left: 0.8rem;
            color: inherit;
            opacity: 0.92;
        }
        .citation-card {
            border: 1px solid var(--scholar-border);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.75rem;
            background: var(--scholar-muted);
        }
        .citation-meta {
            font-weight: 650;
            margin-bottom: 0.35rem;
        }
        .citation-quote {
            border-left: 3px solid var(--scholar-border);
            padding-left: 0.75rem;
            opacity: 0.92;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fetch_papers() -> list[dict[str, Any]]:
    """Fetch uploaded paper list from the backend.

    Returns:
        Paper metadata list.
    """
    try:
        with httpx.Client(timeout=8.0, trust_env=False) as client:
            response = client.get(f"{API_BASE_URL}/papers")
            response.raise_for_status()
            return response.json().get("papers", [])
    except httpx.HTTPError:
        return []


def upload_pdf(uploaded_file: Any) -> dict[str, Any]:
    """Upload a PDF to the backend.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        Backend paper metadata.
    """
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf",
        )
    }
    last_error: httpx.HTTPError | None = None
    for attempt_index in range(5):
        try:
            with httpx.Client(timeout=120.0, trust_env=False) as client:
                response = client.post(f"{API_BASE_URL}/upload", files=files)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code not in {502, 503, 504}:
                raise
        except httpx.HTTPError as exc:
            last_error = exc

        # 后端启动初期会加载 embedding 模型，短暂拒绝连接时自动退避重试。
        time.sleep(1.2 * (attempt_index + 1))

    if last_error:
        raise last_error
    raise RuntimeError("上传失败：后端没有返回有效响应。")


def iter_sse_events(
    endpoint: str,
    payload: dict[str, Any],
) -> Generator[dict[str, Any], None, None]:
    """Iterate over backend SSE events.

    Args:
        endpoint: Backend endpoint path.
        payload: JSON request body.

    Yields:
        Parsed event dictionaries.
    """
    with httpx.Client(timeout=None, trust_env=False) as client:
        with client.stream("POST", f"{API_BASE_URL}{endpoint}", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw_payload = line.removeprefix("data: ").strip()
                if raw_payload:
                    yield json.loads(raw_payload)


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render collapsible source citations.

    Args:
        citations: Citation metadata returned by the backend.
    """
    if not citations:
        return

    with st.expander("主要原文引用与出处", expanded=False):
        for citation in citations:
            page = citation.get("page", "?")
            paragraph = citation.get("paragraph_index") or citation.get("block_id")
            block_id = citation.get("block_id", "")
            citation_id = citation.get("citation_id", "")
            section = citation.get("section") or "Unknown"
            filename = citation.get("filename", "")
            quote = citation_excerpt(citation.get("quote") or citation.get("text", ""))
            citation_prefix = f"{citation_id} · " if citation_id else ""
            st.markdown(
                "\n".join(
                    [
                        "<div class='citation-card'>",
                        "<div class='citation-meta'>",
                        (
                            f"{escape(citation_prefix)}"
                            f"第 {escape(str(page))} 页，第 {escape(str(paragraph))} 段"
                        ),
                        (
                            " <span style='opacity:.62'>"
                            f"({escape(block_id)} · {escape(section)})</span>"
                        ),
                        "</div>",
                        (
                            f"<div style='opacity:.72; margin-bottom:.35rem'>"
                            f"{escape(filename)}</div>"
                            if filename
                            else ""
                        ),
                        f"<div class='citation-quote'>“{escape(quote)}”</div>",
                        "</div>",
                    ]
                ),
                unsafe_allow_html=True,
            )


def render_history(messages_key: str = "messages") -> None:
    """Render chat history.

    Args:
        messages_key: Streamlit session key containing message dictionaries.
    """
    for message in st.session_state[messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(normalize_latex_markdown(message.get("content", "")))
            render_citations(message.get("citations", []))


def stream_chat_response(
    prompt: str,
    paper_ids: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Stream chat response into Streamlit.

    Args:
        prompt: User prompt.
        paper_ids: Optional multi-paper selection. When omitted, the single
            selected paper id is sent for the original single-paper workflow.

    Returns:
        Final answer text and citations.
    """
    citations: list[dict[str, Any]] = []
    final_text_parts: list[str] = []

    def render_stream() -> None:
        payload = {
            "query": prompt,
            "session_id": st.session_state.session_id,
        }
        if paper_ids is None:
            payload["paper_id"] = st.session_state.selected_paper_id
        else:
            payload["paper_ids"] = paper_ids
        placeholder = st.empty()
        try:
            for event in iter_sse_events("/chat", payload):
                event_type = event.get("type")
                data = event.get("data")
                if event_type == "session_id" and data:
                    st.session_state.session_id = data
                elif event_type == "citations":
                    citations.clear()
                    citations.extend(data or [])
                elif event_type == "answer":
                    final_text_parts.append(data or "")
                    placeholder.markdown(
                        normalize_latex_markdown("".join(final_text_parts))
                    )
                elif event_type == "translation_segment_start":
                    segment_header = (
                        f"\n\n**页码 {data.get('page')} · "
                        f"段落 {data.get('block_id')}**\n\n"
                        f"> {data.get('text', '')}\n\n"
                    )
                    final_text_parts.append(segment_header)
                    placeholder.markdown(
                        normalize_latex_markdown("".join(final_text_parts))
                    )
                elif event_type == "translation_delta":
                    token = (data or {}).get("token", "")
                    final_text_parts.append(token)
                    placeholder.markdown(
                        normalize_latex_markdown("".join(final_text_parts))
                    )
                elif event_type == "translation_segment_done":
                    final_text_parts.append("\n\n")
                    placeholder.markdown(
                        normalize_latex_markdown("".join(final_text_parts))
                    )
                elif event_type == "error":
                    error_text = f"\n\n后端错误：{data}"
                    final_text_parts.append(error_text)
                    placeholder.markdown(
                        normalize_latex_markdown("".join(final_text_parts))
                    )
        except httpx.HTTPError as exc:
            error_text = f"无法连接后端：{exc}"
            final_text_parts.append(error_text)
            placeholder.markdown(error_text)

    render_stream()
    return "".join(final_text_parts), citations


def render_sidebar() -> None:
    """Render upload area, paper list, and quick prompts."""
    with st.sidebar:
        st.title("ScholarQA")
        st.caption("学术论文问答助手")

        uploaded_file = st.file_uploader("上传论文 PDF", type=["pdf"])
        if uploaded_file and st.button("解析并索引", use_container_width=True):
            with st.spinner("正在解析 PDF 并写入本地向量库..."):
                try:
                    paper = upload_pdf(uploaded_file)
                    st.session_state.selected_paper_id = paper["paper_id"]
                    st.session_state.selected_paper_ids = [paper["paper_id"]]
                    st.session_state.papers = fetch_papers()
                    st.success(f"已上传：{paper['filename']}")
                    st.markdown(
                        "<span class='upload-pulse'>上传完成，已可提问</span>",
                        unsafe_allow_html=True,
                    )
                except httpx.HTTPError as exc:
                    st.error(f"上传失败：后端可能仍在启动，请稍后重试。{exc}")
                except RuntimeError as exc:
                    st.error(str(exc))

        if st.button("刷新论文列表", use_container_width=True):
            st.session_state.papers = fetch_papers()

        if not st.session_state.papers:
            st.session_state.papers = fetch_papers()

        paper_options = {
            f"{paper['filename']} · {paper['chunk_count']} 段": paper["paper_id"]
            for paper in st.session_state.papers
        }
        if paper_options:
            option_labels = list(paper_options.keys())
            current_single_index = 0
            if st.session_state.selected_paper_id in paper_options.values():
                current_single_index = list(paper_options.values()).index(
                    st.session_state.selected_paper_id
                )
            selected_label = st.selectbox(
                "单篇问答 / 翻译论文",
                option_labels,
                index=current_single_index,
            )
            st.session_state.selected_paper_id = paper_options[selected_label]

            selected_ids = set(st.session_state.selected_paper_ids)
            default_multi_labels = [
                label for label, paper_id in paper_options.items()
                if paper_id in selected_ids
            ] or [selected_label]
            selected_multi_labels = st.multiselect(
                "多篇论文 QA 选择",
                option_labels,
                default=default_multi_labels,
            )
            st.session_state.selected_paper_ids = [
                paper_options[label] for label in selected_multi_labels
            ]
        else:
            st.info("还没有上传论文。")

        st.divider()
        st.subheader("快捷指令")
        quick_prompts = {
            "总结创新点": "请总结这篇论文的方法创新点，按条目给出，并标注页码和段落号。",
            "梳理实验": "请梳理这篇论文的实验设置与主要结果，指出证据所在页码和段落号。",
            "全文摘要": "请生成这篇论文的全文摘要，覆盖研究问题、方法、实验和结论。",
        }
        for label, prompt in quick_prompts.items():
            if st.button(label, use_container_width=True):
                st.session_state.pending_prompt = prompt


def render_chat_tab() -> None:
    """Render the main chat workflow."""
    st.header("单篇论文问答")
    render_history()

    prompt = st.session_state.pending_prompt or st.chat_input("问一个关于论文的问题")
    st.session_state.pending_prompt = None
    if not prompt:
        return

    if not st.session_state.selected_paper_id:
        st.warning("请先在左侧点击“解析并索引”，完成后再提问。")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索论文并生成回答..."):
            answer, citations = stream_chat_response(prompt)
        render_citations(citations)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
        }
    )


def render_multi_paper_tab() -> None:
    """Render the multi-paper comparison workflow."""
    st.header("多篇论文 QA")
    render_history("multi_messages")

    selected_paper_ids = st.session_state.selected_paper_ids
    paper_by_id = {
        paper["paper_id"]: paper
        for paper in st.session_state.papers
    }
    selected_papers = [
        paper_by_id[paper_id]
        for paper_id in selected_paper_ids
        if paper_id in paper_by_id
    ]

    if selected_papers:
        st.markdown(f"当前正在对比 **{len(selected_papers)}** 篇论文：")
        for paper in selected_papers:
            st.markdown(f"- {paper['filename']}")
    else:
        st.info("请先在左侧选择至少两篇论文。")

    prompt = st.chat_input("问一个需要跨论文比较的问题", key="multi_chat_input")
    if not prompt:
        return

    if len(selected_papers) < 2:
        st.warning("多篇论文 QA 需要至少选择两篇论文。")
        return

    st.session_state.multi_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在让每篇论文独立检索，并综合生成对比回答..."):
            answer, citations = stream_chat_response(
                prompt,
                paper_ids=[paper["paper_id"] for paper in selected_papers],
            )
        render_citations(citations)

    st.session_state.multi_messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
        }
    )


def render_translation_tab() -> None:
    """Render the paragraph translation workflow."""
    st.header("逐段翻译")

    if not st.session_state.selected_paper_id:
        st.info("请先在左侧上传或选择论文。")
        return

    left_column, right_column = st.columns(2)
    with left_column:
        page = st.number_input("页码", min_value=1, value=1, step=1)
    with right_column:
        translate_whole_page = st.checkbox("整页翻译", value=False)
        paragraph_index = st.number_input(
            "段落号（页内）",
            min_value=1,
            value=1,
            step=1,
            disabled=translate_whole_page,
        )

    if not st.button("开始翻译", type="primary"):
        return

    payload = {
        "paper_id": st.session_state.selected_paper_id,
        "session_id": st.session_state.session_id,
        "page": int(page),
        "paragraph_index": None if translate_whole_page else int(paragraph_index),
        "query": (
            f"请翻译第{int(page)}页"
            if translate_whole_page
            else f"请翻译第{int(page)}页第{int(paragraph_index)}段"
        ),
    }

    current_segments: dict[str, dict[str, str]] = {}
    container = st.empty()
    with st.spinner("正在逐字翻译..."):
        try:
            for event in iter_sse_events("/translate", payload):
                event_type = event.get("type")
                data = event.get("data")
                if event_type == "translation_segment_start":
                    block_id = data["block_id"]
                    current_segments[block_id] = {
                        "source": data["text"],
                        "translation": "",
                        "page": str(data["page"]),
                    }
                elif event_type == "translation_delta":
                    block_id = data["block_id"]
                    if block_id in current_segments:
                        current_segments[block_id]["translation"] += data["token"]
                elif event_type == "error":
                    st.error(str(data))

                with container.container():
                    for block_id, segment in current_segments.items():
                        st.markdown(f"**页码 {segment['page']} · 段落 {block_id}**")
                        source_column, translation_column = st.columns(2)
                        with source_column:
                            source_text = escape(segment["source"])
                            st.markdown(
                                f"<div class='translation-card'>{source_text}</div>",
                                unsafe_allow_html=True,
                            )
                        with translation_column:
                            translated_text = escape(segment["translation"])
                            st.markdown(
                                f"<div class='translation-card'>{translated_text}</div>",
                                unsafe_allow_html=True,
                            )
        except httpx.HTTPError as exc:
            st.error(f"无法连接后端：{exc}")


def main() -> None:
    """Run the ScholarQA Streamlit app."""
    st.set_page_config(
        page_title="ScholarQA",
        layout="wide",
    )
    init_state()
    inject_style()
    render_sidebar()

    chat_tab, translation_tab, compare_tab = st.tabs(
        ["单篇问答", "翻译模式", "多篇论文 QA"]
    )
    with chat_tab:
        render_chat_tab()
    with translation_tab:
        render_translation_tab()
    with compare_tab:
        render_multi_paper_tab()


if __name__ == "__main__":
    main()
