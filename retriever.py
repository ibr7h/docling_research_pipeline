"""Docling-native retrieval module (Phase 1.5).

Implements:
- search(query, k=5): vector retrieval from Chroma with metadata
- ask(query): basic RAG flow (retrieve -> construct context -> LLM call)

Design keeps extension points for future hybrid search, reranking, and metadata filtering.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import chromadb

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore[assignment]


DEFAULT_COLLECTION = "docling_chunks"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float | None
    doc_id: str | None
    section_title: str | None
    chunk_type: str | None
    source_path: str | None
    raw_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "doc_id": self.doc_id,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            "source_path": self.source_path,
            "metadata": self.raw_metadata,
        }


class DoclingRetriever:
    """Retriever over the Chroma collection produced by `index_chunks.py`."""

    def __init__(
        self,
        *,
        chroma_dir: str = "chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
        llm_model: str = DEFAULT_MODEL,
        llm_client: Any | None = None,
    ) -> None:
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.llm_model = llm_model
        self.llm_client = llm_client or self._build_default_llm_client()

    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k chunks with metadata.

        Extension points:
        - metadata_filter: forwarded to Chroma `where` clause
        - `_hybrid_search`: reserved for future lexical+dense retrieval
        - `_rerank`: reserved for future reranker integration
        """
        query = query.strip()
        if not query:
            return []

        # For now, dense search only. Hybrid hook retained for future phases.
        base_results = self._vector_search(query=query, k=k, metadata_filter=metadata_filter)

        # Reranker hook retained for future phases.
        reranked = self._rerank(query, base_results)
        return [chunk.to_dict() for chunk in reranked[:k]]

    def ask(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """RAG: retrieve chunks, build context, call LLM, and return answer payload."""
        retrieved = self.search(query, k=k, metadata_filter=metadata_filter)
        context = self._build_context(retrieved)
        prompt = self._build_prompt(query=query, context=context)

        answer = self._llm_answer(
            prompt=prompt,
            system_prompt=system_prompt
            or "You are a helpful assistant. Use the provided context and be explicit about uncertainty.",
        )

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "results": retrieved,
        }

    def _vector_search(
        self,
        *,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        out: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            metadata = meta or {}
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    score=(1.0 - dist) if isinstance(dist, (float, int)) else None,
                    doc_id=metadata.get("doc_id"),
                    section_title=metadata.get("section_title"),
                    chunk_type=metadata.get("chunk_type"),
                    source_path=metadata.get("source_path"),
                    raw_metadata=metadata,
                )
            )
        return out

    def _hybrid_search(self, query: str, k: int) -> Iterable[RetrievedChunk]:
        """Reserved for future hybrid retrieval (dense + lexical)."""
        _ = (query, k)
        return []

    def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Reserved for future reranker integration.

        Current behavior: passthrough.
        """
        _ = query
        return chunks

    @staticmethod
    def _build_context(chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for idx, item in enumerate(chunks, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"[Chunk {idx}]",
                        f"doc_id: {item.get('doc_id')}",
                        f"section_title: {item.get('section_title')}",
                        f"chunk_type: {item.get('chunk_type')}",
                        f"source_path: {item.get('source_path')}",
                        "content:",
                        item.get("text", ""),
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_prompt(*, query: str, context: str) -> str:
        return (
            "Answer the question using only the provided context when possible. "
            "If context is insufficient, say so explicitly.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n"
        )

    def _build_default_llm_client(self) -> Any | None:
        if OpenAI is None:
            return None
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return OpenAI()

    def _llm_answer(self, *, prompt: str, system_prompt: str) -> str:
        if self.llm_client is None:
            return (
                "LLM client is not configured. Set OPENAI_API_KEY and install the openai package "
                "to enable ask()."
            )

        response = self.llm_client.responses.create(
            model=self.llm_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.output_text or "").strip()
