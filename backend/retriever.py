"""Hybrid retrieval and lightweight paper storage for ScholarQA."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer

from nodes import PaperNode, PaperNodeBuilder


def _tokenize(text: str) -> list[str]:
    """Tokenize English terms and Chinese characters for lightweight BM25.

    Args:
        text: Input text.

    Returns:
        Normalized tokens.
    """
    return re.findall(r"[a-zA-Z0-9_+-]+|[\u4e00-\u9fff]", text.lower())


@dataclass(slots=True)
class RetrievedChunk:
    """A parent paragraph retrieved by the hybrid pipeline."""

    paper_id: str
    filename: str
    text: str
    page: int
    paragraph_index: int
    section: str
    block_id: str
    node_type: str
    score: float
    parent_id: str
    child_id: str

    def as_dict(self) -> dict[str, Any]:
        """Convert the chunk to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the retrieved parent chunk.
        """
        return {
            "paper_id": self.paper_id,
            "filename": self.filename,
            "text": self.text,
            "page": self.page,
            "paragraph_index": self.paragraph_index,
            "section": self.section,
            "block_id": self.block_id,
            "node_type": self.node_type,
            "score": self.score,
            "parent_id": self.parent_id,
            "child_id": self.child_id,
        }


class BM25Index:
    """Small in-memory BM25 index for child nodes."""

    def __init__(self, nodes: list[PaperNode], k1: float = 1.5, b: float = 0.75) -> None:
        """Build a BM25 index.

        Args:
            nodes: Child nodes for a paper.
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self._nodes = nodes
        self._k1 = k1
        self._b = b
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_lengths: dict[str, int] = {}
        self._document_frequencies: Counter[str] = Counter()
        self._average_document_length = 0.0
        self._build()

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Search child nodes with BM25.

        Args:
            query: User query.
            limit: Maximum number of child node ids to return.

        Returns:
            ``[(child_node_id, score), ...]`` sorted by descending score.
        """
        query_tokens = _tokenize(query)
        if not query_tokens or not self._nodes:
            return []

        scores: dict[str, float] = defaultdict(float)
        total_documents = len(self._nodes)
        for token in query_tokens:
            document_frequency = self._document_frequencies.get(token, 0)
            if document_frequency == 0:
                continue
            idf = math.log(
                1 + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for node in self._nodes:
                frequency = self._term_frequencies[node.node_id].get(token, 0)
                if frequency == 0:
                    continue
                document_length = self._document_lengths[node.node_id]
                denominator = frequency + self._k1 * (
                    1 - self._b
                    + self._b
                    * document_length
                    / max(self._average_document_length, 1.0)
                )
                scores[node.node_id] += idf * frequency * (self._k1 + 1) / denominator

        ranked_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_scores[:limit]

    def _build(self) -> None:
        total_length = 0
        for node in self._nodes:
            tokens = _tokenize(node.retrieval_text)
            token_counts = Counter(tokens)
            self._term_frequencies[node.node_id] = token_counts
            self._document_lengths[node.node_id] = len(tokens)
            total_length += len(tokens)
            self._document_frequencies.update(token_counts.keys())

        if self._nodes:
            self._average_document_length = total_length / len(self._nodes)


class PaperLibrary:
    """In-memory paper metadata, parent nodes, child nodes, and JSON export."""

    def __init__(self) -> None:
        """Initialize the paper library."""
        self._papers: dict[str, dict[str, Any]] = {}
        self._latest_paper_id: str | None = None
        self._node_builder = PaperNodeBuilder()

    def upsert(self, paper_id: str, filename: str, chunks: list[dict[str, Any]]) -> None:
        """Create or replace a paper record.

        Args:
            paper_id: Stable paper identifier.
            filename: Original uploaded file name.
            chunks: Parsed paragraph chunks.
        """
        parent_nodes, child_nodes = self._node_builder.build(
            paper_id=paper_id,
            filename=filename,
            parsed_blocks=chunks,
        )
        self._papers[paper_id] = {
            "paper_id": paper_id,
            "filename": filename,
            "chunk_count": len(parent_nodes),
            "chunks": [node.as_dict() for node in parent_nodes],
            "child_nodes": [node.as_dict() for node in child_nodes],
            "parent_nodes_by_id": {node.node_id: node for node in parent_nodes},
            "child_nodes_by_id": {node.node_id: node for node in child_nodes},
        }
        self._latest_paper_id = paper_id

    def latest_paper_id(self) -> str | None:
        """Return the most recently uploaded paper id.

        Returns:
            Latest paper id, or None when no paper has been uploaded.
        """
        return self._latest_paper_id

    def list_papers(self) -> list[dict[str, Any]]:
        """List uploaded papers without embedding the full text.

        Returns:
            Paper summaries for the frontend sidebar.
        """
        return [
            {
                "paper_id": paper["paper_id"],
                "filename": paper["filename"],
                "chunk_count": paper["chunk_count"],
            }
            for paper in self._papers.values()
        ]

    def get_chunks(self, paper_id: str) -> list[dict[str, Any]]:
        """Return parent chunks for a paper.

        Args:
            paper_id: Paper identifier.

        Returns:
            Parent nodes in reading order, represented as dictionaries.
        """
        paper = self._papers.get(paper_id)
        if not paper:
            return []
        return list(paper["chunks"])

    def get_child_nodes(self, paper_id: str) -> list[PaperNode]:
        """Return child nodes for hybrid retrieval.

        Args:
            paper_id: Paper identifier.

        Returns:
            Child nodes for the selected paper.
        """
        paper = self._papers.get(paper_id)
        if not paper:
            return []
        return list(paper["child_nodes_by_id"].values())

    def get_parent_nodes(self, paper_id: str) -> list[PaperNode]:
        """Return parent nodes for parent expansion.

        Args:
            paper_id: Paper identifier.

        Returns:
            Parent nodes for the selected paper.
        """
        paper = self._papers.get(paper_id)
        if not paper:
            return []
        return list(paper["parent_nodes_by_id"].values())

    def get_parent_node(self, paper_id: str, parent_id: str) -> PaperNode | None:
        """Fetch a parent node by id.

        Args:
            paper_id: Paper identifier.
            parent_id: Parent node id.

        Returns:
            Parent node if present.
        """
        paper = self._papers.get(paper_id)
        if not paper:
            return None
        return paper["parent_nodes_by_id"].get(parent_id)

    def formula_context_chunks(
        self,
        paper_id: str,
        formula_number: str,
        window: int = 3,
    ) -> list[dict[str, Any]]:
        """Return chunks around a numbered formula.

        Args:
            paper_id: Paper identifier.
            formula_number: Formula number such as ``13``.
            window: Number of neighboring chunks on each side.

        Returns:
            Parent chunks near the target formula.
        """
        chunks = self._non_reference_chunks(self.get_chunks(paper_id))
        target_indexes = [
            index
            for index, chunk in enumerate(chunks)
            if re.search(rf"\({re.escape(formula_number)}\)\s*$", chunk.get("text", ""))
        ]
        if not target_indexes:
            return []

        selected_chunks: list[dict[str, Any]] = []
        seen_block_ids: set[str] = set()
        for target_index in target_indexes:
            start_index = max(0, target_index - window)
            end_index = min(len(chunks), target_index + window + 1)
            for chunk in chunks[start_index:end_index]:
                block_id = chunk.get("block_id")
                if block_id and block_id not in seen_block_ids:
                    selected_chunks.append(chunk)
                    seen_block_ids.add(block_id)
        return selected_chunks

    def select_chunks(
        self,
        paper_id: str,
        page: int | None = None,
        paragraph_index: int | None = None,
        block_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Select parent chunks by page, paragraph number, or block id.

        Args:
            paper_id: Paper identifier.
            page: Optional 1-based page number.
            paragraph_index: Optional 1-based paragraph number within a page.
            block_id: Optional exact block id such as ``p2-b003``.
            limit: Optional maximum number of chunks.

        Returns:
            Matching parent chunks in reading order.
        """
        chunks = self._non_reference_chunks(self.get_chunks(paper_id))
        if page is not None:
            chunks = [chunk for chunk in chunks if chunk.get("page") == page]
        if paragraph_index is not None:
            chunks = [
                chunk
                for chunk in chunks
                if chunk.get("paragraph_index") == paragraph_index
            ]
        if block_id is not None:
            chunks = [chunk for chunk in chunks if chunk.get("block_id") == block_id]
        if limit is not None:
            chunks = chunks[:limit]
        return chunks

    def representative_chunks(
        self,
        paper_id: str,
        max_chunks: int = 14,
    ) -> list[dict[str, Any]]:
        """Pick broad-coverage parent chunks for paper-level summaries.

        Args:
            paper_id: Paper identifier.
            max_chunks: Maximum number of chunks to return.

        Returns:
            A compact set of parent chunks covering major paper sections.
        """
        chunks = self._non_reference_chunks(self.get_chunks(paper_id))
        if not chunks:
            return []

        picked: list[dict[str, Any]] = []
        seen_block_ids: set[str] = set()
        section_keywords = [
            "abstract",
            "introduction",
            "method",
            "approach",
            "experiment",
            "result",
            "discussion",
            "conclusion",
        ]

        # 总结类问题需要覆盖论文结构，而不是只取语义最相近的局部片段。
        for keyword in section_keywords:
            for chunk in chunks:
                search_area = f"{chunk.get('section', '')}\n{chunk.get('text', '')}"
                if keyword in search_area.lower() and chunk["block_id"] not in seen_block_ids:
                    picked.append(chunk)
                    seen_block_ids.add(chunk["block_id"])
                    break

        for chunk in chunks:
            if len(picked) >= max_chunks:
                break
            if chunk["block_id"] not in seen_block_ids:
                picked.append(chunk)
                seen_block_ids.add(chunk["block_id"])

        return picked[:max_chunks]

    def _non_reference_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            chunk
            for chunk in chunks
            if not _is_reference_section(chunk.get("section", ""))
        ]

    async def export_json(self, output_path: Path) -> None:
        """Export metadata and parsed text to a JSON file.

        Args:
            output_path: Destination JSON path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            paper_id: {
                "paper_id": paper["paper_id"],
                "filename": paper["filename"],
                "chunk_count": paper["chunk_count"],
                "chunks": paper["chunks"],
                "child_nodes": paper["child_nodes"],
            }
            for paper_id, paper in self._papers.items()
        }
        await asyncio.to_thread(
            output_path.write_text,
            json.dumps(payload, ensure_ascii=False, indent=2),
            "utf-8",
        )


class ChromaRetriever:
    """ChromaDB dense retrieval with BM25, RRF fusion, and parent expansion."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_model_name: str,
        rrf_k: int = 60,
        reranker_model_name: str | None = None,
    ) -> None:
        """Initialize ChromaDB, embedding model, and BM25 state.

        Args:
            persist_dir: ChromaDB persistence directory.
            collection_name: Chroma collection name.
            embedding_model_name: SentenceTransformer model name or local path.
            rrf_k: RRF constant used to fuse dense and BM25 ranks.
            reranker_model_name: Optional CrossEncoder reranker model name.
        """
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_model = SentenceTransformer(embedding_model_name)
        self._reranker = (
            CrossEncoder(reranker_model_name) if reranker_model_name else None
        )
        self._rrf_k = rrf_k
        self._child_nodes_by_id: dict[str, PaperNode] = {}
        self._parent_nodes_by_id: dict[str, PaperNode] = {}
        self._bm25_indexes: dict[str, BM25Index] = {}

    async def add_document(self, paper_id: str, chunks: list[PaperNode]) -> None:
        """Embed and upsert child nodes.

        Args:
            paper_id: Paper identifier.
            chunks: Child nodes generated from parsed PDF blocks.
        """
        await asyncio.to_thread(self._add_document_sync, paper_id, chunks)

    async def search(
        self,
        query: str,
        paper_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Retrieve parent chunks using dense + BM25 + RRF.

        Args:
            query: User query, optionally enriched with conversation memory.
            paper_id: Optional paper filter.
            top_k: Number of parent chunks to return.

        Returns:
            Retrieved parent chunks with source metadata.
        """
        return await asyncio.to_thread(self._search_sync, query, paper_id, top_k)

    def _add_document_sync(self, paper_id: str, chunks: list[PaperNode]) -> None:
        if not chunks:
            return

        try:
            self._collection.delete(where={"paper_id": paper_id})
        except Exception:
            pass

        self._drop_paper_from_memory(paper_id)
        for child_node in chunks:
            self._child_nodes_by_id[child_node.node_id] = child_node
            parent_node_id = child_node.parent_id
            if parent_node_id:
                self._parent_nodes_by_id[parent_node_id] = self._parent_from_child(
                    child_node
                )

        documents = [node.retrieval_text for node in chunks]
        embeddings = self._embedding_model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        ids = [node.node_id for node in chunks]
        metadatas = [
            {
                "paper_id": node.paper_id,
                "filename": node.filename,
                "page": int(node.page),
                "paragraph_index": int(node.paragraph_index),
                "section": node.section,
                "block_id": node.block_id,
                "node_type": node.node_type,
                "parent_id": node.parent_id or "",
                "child_id": node.node_id,
            }
            for node in chunks
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self._bm25_indexes[paper_id] = BM25Index(chunks)

    def register_parent_nodes(self, parent_nodes: list[PaperNode]) -> None:
        """Register parent nodes for parent expansion.

        Args:
            parent_nodes: Parent nodes generated from a paper.
        """
        for node in parent_nodes:
            self._parent_nodes_by_id[node.node_id] = node

    def _search_sync(
        self,
        query: str,
        paper_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        include_references = _query_needs_references(query)
        fetch_k = max(top_k * 4, 24)
        dense_hits = self._dense_search(query, paper_id, fetch_k, include_references)
        bm25_hits = self._bm25_search(query, paper_id, fetch_k, include_references)
        fused_child_ids = self._rrf_fuse(dense_hits, bm25_hits)

        retrieved_chunks: list[dict[str, Any]] = []
        seen_parent_ids: set[str] = set()
        for child_id, fused_score in fused_child_ids:
            child_node = self._child_nodes_by_id.get(child_id)
            if child_node is None:
                continue
            parent_id = child_node.parent_id or child_node.node_id
            if parent_id in seen_parent_ids:
                continue
            parent_node = self._parent_nodes_by_id.get(parent_id, child_node)
            if not include_references and _is_reference_section(parent_node.section):
                continue
            seen_parent_ids.add(parent_id)
            retrieved_chunks.append(
                RetrievedChunk(
                    paper_id=parent_node.paper_id,
                    filename=parent_node.filename,
                    text=parent_node.text,
                    page=parent_node.page,
                    paragraph_index=parent_node.paragraph_index,
                    section=parent_node.section,
                    block_id=parent_node.block_id,
                    node_type=parent_node.node_type,
                    score=fused_score,
                    parent_id=parent_id,
                    child_id=child_id,
                ).as_dict()
            )
            if len(retrieved_chunks) >= max(top_k * 2, top_k):
                break

        if self._reranker and retrieved_chunks:
            retrieved_chunks = self._rerank_parents(query, retrieved_chunks)

        return retrieved_chunks[:top_k]

    def _rerank_parents(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self._reranker.predict(pairs)
        ranked_chunks = [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ]
        return sorted(
            ranked_chunks,
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )

    def _dense_search(
        self,
        query: str,
        paper_id: str | None,
        fetch_k: int,
        include_references: bool,
    ) -> list[tuple[str, float]]:
        if self._collection.count() == 0:
            return []

        query_embedding = self._embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()[0]
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": fetch_k * 3,
            "include": ["metadatas", "distances"],
        }
        if paper_id:
            query_kwargs["where"] = {"paper_id": paper_id}

        results = self._collection.query(**query_kwargs)
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        filtered_hits: list[tuple[str, float]] = []
        for child_id, distance in zip(ids, distances):
            child_node = self._child_nodes_by_id.get(str(child_id))
            if child_node is None:
                continue
            if not include_references and _is_reference_section(child_node.section):
                continue
            filtered_hits.append((str(child_id), 1.0 - float(distance)))
            if len(filtered_hits) >= fetch_k:
                break
        return filtered_hits

    def _bm25_search(
        self,
        query: str,
        paper_id: str | None,
        fetch_k: int,
        include_references: bool,
    ) -> list[tuple[str, float]]:
        paper_ids = [paper_id] if paper_id else list(self._bm25_indexes.keys())
        hits: list[tuple[str, float]] = []
        for selected_paper_id in paper_ids:
            if not selected_paper_id:
                continue
            index = self._bm25_indexes.get(selected_paper_id)
            if not index:
                continue
            for child_id, score in index.search(query, fetch_k * 3):
                child_node = self._child_nodes_by_id.get(child_id)
                if child_node is None:
                    continue
                if not include_references and _is_reference_section(child_node.section):
                    continue
                hits.append((child_id, score))

        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:fetch_k]

    def _rrf_fuse(
        self,
        dense_hits: list[tuple[str, float]],
        bm25_hits: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        fused_scores: dict[str, float] = defaultdict(float)
        for rank, (child_id, _score) in enumerate(dense_hits, start=1):
            fused_scores[child_id] += 1.0 / (self._rrf_k + rank)
        for rank, (child_id, _score) in enumerate(bm25_hits, start=1):
            fused_scores[child_id] += 1.0 / (self._rrf_k + rank)

        return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    def _drop_paper_from_memory(self, paper_id: str) -> None:
        self._child_nodes_by_id = {
            node_id: node
            for node_id, node in self._child_nodes_by_id.items()
            if node.paper_id != paper_id
        }
        self._parent_nodes_by_id = {
            node_id: node
            for node_id, node in self._parent_nodes_by_id.items()
            if node.paper_id != paper_id
        }
        self._bm25_indexes.pop(paper_id, None)

    def _parent_from_child(self, child_node: PaperNode) -> PaperNode:
        if child_node.parent_id and child_node.parent_id in self._parent_nodes_by_id:
            return self._parent_nodes_by_id[child_node.parent_id]
        return PaperNode(
            node_id=child_node.parent_id or child_node.node_id,
            paper_id=child_node.paper_id,
            filename=child_node.filename,
            node_type=child_node.node_type,
            text=child_node.text,
            retrieval_text=child_node.retrieval_text,
            page=child_node.page,
            paragraph_index=child_node.paragraph_index,
            block_id=child_node.block_id,
            order=child_node.order,
            section_path=child_node.section_path,
            parent_id=None,
            metadata={"source": "fallback_parent"},
        )


def _is_reference_section(section: str) -> bool:
    return section.strip().lower() in {"references", "reference", "bibliography"}


def _query_needs_references(query: str) -> bool:
    normalized_query = query.lower()
    reference_terms = [
        "参考文献",
        "引用了哪些",
        "文献列表",
        "reference",
        "references",
        "bibliography",
    ]
    return any(term in normalized_query for term in reference_terms)
