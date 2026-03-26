"""Docling-native retriever built on Chroma chunk indexes.

Phase 4 adds retrieval quality hooks while preserving public API:
- metadata filtering
- hybrid search hooks (lexical + vector)
- reranking hooks
- table-aware behavior
- multilingual embedding readiness
"""

from __future__ import annotations

import json
import os
import re
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
        enable_hybrid: bool = False,
        enable_rerank: bool = False,
        prefer_tables_for_table_queries: bool = True,
        multilingual_mode: bool = False,
    ) -> None:
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.llm_model = llm_model
        self.enable_hybrid = enable_hybrid
        self.enable_rerank = enable_rerank
        self.prefer_tables_for_table_queries = prefer_tables_for_table_queries
        self.multilingual_mode = multilingual_mode

    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k retrieval with metadata filtering and quality-improvement hooks."""
        clean_query = query.strip()
        if not clean_query:
            return []

        where = self._build_where_filter(metadata_filter)
        base = self._vector_search(clean_query, k=max(k * 3, 10), where=where)

        if self.enable_hybrid:
            merged = self._hybrid_search(clean_query, base, k=max(k * 3, 10), where=where)
        else:
            merged = base

        filtered = self._filter_results(merged, metadata_filter)

        if self.enable_rerank:
            reranked = self._rerank(clean_query, filtered)
        else:
            reranked = filtered

        table_aware = self._apply_table_bias(clean_query, reranked)
        return [record.to_dict() for record in table_aware[:k]]

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
            "Add short source citations using [doc_id|chunk_id]. "
            "If context is insufficient, explicitly say so.\n\n"
            f"Question:\n{query}\n\nContext:\n{context}"
        )
        answer = self._call_llm(
            prompt=prompt,
            system_prompt=system_prompt
            or "You are a research assistant that provides evidence-grounded answers.",
        )

        return {
            "query": query,
            "answer": answer,
            "sources": evidence,
            "evidence_summary": self._format_evidence_summary(evidence),
        }

    def _vector_search(self, query: str, *, k: int, where: dict[str, Any] | None) -> list[RetrievedChunk]:
        result = self.collection.query(
            query_texts=[self._prepare_query_for_embedding(query)],
            n_results=k,
            where=where,
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
        return records

    def _hybrid_search(
        self,
        query: str,
        vector_results: list[RetrievedChunk],
        *,
        k: int,
        where: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieval hook (lexical BM25-like + vector merge).

        Current implementation is a lightweight lexical scorer over candidates to keep
        dependencies minimal. A full BM25 index can replace this method later.
        """
        _ = where
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return vector_results[:k]

        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in vector_results:
            text_tokens = self._tokenize(chunk.text)
            lexical = self._lexical_score(query_tokens, text_tokens)
            vector = chunk.score or 0.0
            hybrid_score = (0.65 * vector) + (0.35 * lexical)
            scored.append((hybrid_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        merged = [chunk for _, chunk in scored]
        return merged[:k]

    def _rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Reranker hook.

        Current default reranker is heuristic and lightweight:
        - boosts section title token overlap with query
        - slight preference for shorter, denser snippets

        Replace this method with a cross-encoder/model reranker in future phases.
        """
        query_tokens = set(self._tokenize(query))

        def score(item: RetrievedChunk) -> float:
            base = item.score or 0.0
            title_tokens = set(self._tokenize(item.section_title or ""))
            overlap = len(query_tokens & title_tokens)
            length_penalty = min(len(item.text) / 2000.0, 0.2)
            return base + (0.05 * overlap) - length_penalty

        return sorted(results, key=score, reverse=True)

    def _filter_results(
        self,
        results: list[RetrievedChunk],
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """In-memory filter hook (kept even when Chroma where-filter is used)."""
        if not metadata_filter:
            return results

        allowed_keys = {"doc_id", "chunk_type", "section_title", "source_path"}
        active = {k: v for k, v in metadata_filter.items() if k in allowed_keys and v is not None}
        if not active:
            return results

        filtered: list[RetrievedChunk] = []
        for item in results:
            ok = True
            for key, value in active.items():
                field_value = getattr(item, key)
                if str(field_value) != str(value):
                    ok = False
                    break
            if ok:
                filtered.append(item)
        return filtered

    def _apply_table_bias(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Table-aware ranking behavior for tabular intents."""
        if not self.prefer_tables_for_table_queries:
            return results

        table_intent = any(term in query.lower() for term in ["table", "row", "column", "dataset", "csv"])
        if not table_intent:
            return results

        return sorted(results, key=lambda r: (r.chunk_type == "table", r.score or 0.0), reverse=True)

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
    def _format_evidence_summary(chunks: list[dict[str, Any]]) -> list[str]:
        summary: list[str] = []
        for chunk in chunks:
            summary.append(
                " | ".join(
                    [
                        f"doc_id={chunk.get('doc_id')}",
                        f"chunk_id={chunk.get('chunk_id')}",
                        f"type={chunk.get('chunk_type')}",
                        f"section={chunk.get('section_title')}",
                        f"pages={chunk.get('page_numbers')}",
                        f"source={chunk.get('source_path')}",
                    ]
                )
            )
        return summary

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

    @staticmethod
    def _build_where_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
        if not metadata_filter:
            return None
        allowed_keys = {"doc_id", "chunk_type", "section_title", "source_path"}
        where = {k: v for k, v in metadata_filter.items() if k in allowed_keys and v is not None}
        return where or None

    def _prepare_query_for_embedding(self, query: str) -> str:
        """Multilingual embedding readiness hook.

        Currently passes text through unchanged. Future multilingual embedding
        routing (language ID / model selection / translation) can live here.
        """
        _ = self.multilingual_mode
        return query

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _lexical_score(query_tokens: list[str], text_tokens: list[str]) -> float:
        if not text_tokens:
            return 0.0
        q = set(query_tokens)
        t = set(text_tokens)
        overlap = len(q & t)
        return overlap / max(len(q), 1)

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
