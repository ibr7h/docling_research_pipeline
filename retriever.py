"""Docling-native retriever built on Chroma chunk indexes."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import chromadb


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
    page_numbers: list[int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "doc_id": self.doc_id,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            "source_path": self.source_path,
            "page_numbers": self.page_numbers,
            "metadata": self.metadata,
        }


class DoclingRetriever:
    """Retriever for chunks indexed by `index_chunks.py`."""

    def __init__(
        self,
        *,
        chroma_dir: str = "chroma_db",
        collection_name: str = DEFAULT_COLLECTION,
        llm_model: str = DEFAULT_MODEL,
    ) -> None:
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.llm_model = llm_model

    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k dense retrieval with metadata filter extension point."""
        clean_query = query.strip()
        if not clean_query:
            return []

        result = self.collection.query(
            query_texts=[clean_query],
            n_results=k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        records: list[RetrievedChunk] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            metadata = meta or {}
            pages = self._normalize_page_numbers(metadata.get("page_numbers"))
            records.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text or "",
                    score=(1.0 - dist) if isinstance(dist, (float, int)) else None,
                    doc_id=metadata.get("doc_id"),
                    section_title=metadata.get("section_title"),
                    chunk_type=metadata.get("chunk_type"),
                    source_path=metadata.get("source_path"),
                    page_numbers=pages,
                    metadata=metadata,
                )
            )

        # Future hook: hybrid search + reranking can be inserted here.
        return [record.to_dict() for record in records]

    def ask(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """RAG helper: retrieve evidence, construct context, call LLM, return answer + sources."""
        evidence = self.search(query=query, k=k, metadata_filter=metadata_filter)
        context = self.build_context(evidence)

        prompt = (
            "Answer with clear, grounded statements from the provided context. "
            "If context is insufficient, explicitly say so.\n\n"
            f"Question:\n{query}\n\nContext:\n{context}"
        )
        answer = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt
            or "You are a research assistant that cites available evidence snippets.",
        )

        return {
            "query": query,
            "answer": answer,
            "sources": evidence,
        }

    @staticmethod
    def build_context(chunks: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            lines.append(
                "\n".join(
                    [
                        f"[Evidence {idx}]",
                        f"doc_id: {chunk.get('doc_id')}",
                        f"chunk_id: {chunk.get('chunk_id')}",
                        f"chunk_type: {chunk.get('chunk_type')}",
                        f"section_title: {chunk.get('section_title')}",
                        f"page_numbers: {chunk.get('page_numbers')}",
                        f"source_path: {chunk.get('source_path')}",
                        "text:",
                        chunk.get("text", ""),
                    ]
                )
            )
        return "\n\n".join(lines)

    @staticmethod
    def _normalize_page_numbers(raw: Any) -> list[int]:
        if isinstance(raw, list):
            return [p for p in raw if isinstance(p, int)]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [p for p in parsed if isinstance(p, int)]
            except json.JSONDecodeError:
                return []
        return []

    def _call_llm(self, *, prompt: str, system_prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return (
                "LLM call skipped: set OPENAI_API_KEY to enable answer generation. "
                "Retrieved evidence is returned in `sources`."
            )

        body = {
            "model": self.llm_model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            url="https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="ignore")
            return f"LLM request failed with HTTP {err.code}: {detail[:300]}"
        except urllib.error.URLError as err:
            return f"LLM request failed: {err.reason}"

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        return "LLM returned no textual output."
