"""Phase 3 research tasks built on top of `retriever.py`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retriever import DoclingRetriever


@dataclass
class TaskResult:
    task: str
    query: str
    result_text: str
    evidence: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "query": self.query,
            "result_text": self.result_text,
            "evidence": self.evidence,
        }


class ResearchTasks:
    """High-level research tasks using retriever.search()/ask() only."""

    def __init__(self, retriever: DoclingRetriever) -> None:
        self.retriever = retriever

    def summarize_document(self, doc_id_or_query: str) -> dict[str, Any]:
        metadata_filter = {"doc_id": doc_id_or_query} if doc_id_or_query else None
        query = f"Summarize the document: {doc_id_or_query}"
        rag = self.retriever.ask(
            query=self._prompt_summarize(doc_id_or_query),
            k=8,
            metadata_filter=metadata_filter,
        )
        return TaskResult(
            task="summarize_document",
            query=query,
            result_text=rag["answer"],
            evidence=rag["sources"],
        ).to_dict()

    def compare_documents(self, topic_or_query: str, k: int = 8) -> dict[str, Any]:
        evidence = self.retriever.search(topic_or_query, k=k)
        rag = self.retriever.ask(query=self._prompt_compare(topic_or_query), k=k)
        return TaskResult(
            task="compare_documents",
            query=topic_or_query,
            result_text=rag["answer"],
            evidence=evidence,
        ).to_dict()

    def literature_review(self, topic: str, k: int = 10) -> dict[str, Any]:
        rag = self.retriever.ask(query=self._prompt_literature_review(topic), k=k)
        return TaskResult(
            task="literature_review",
            query=topic,
            result_text=rag["answer"],
            evidence=rag["sources"],
        ).to_dict()

    def extract_references(self, doc_id_or_query: str) -> dict[str, Any]:
        metadata_filter = {"doc_id": doc_id_or_query} if doc_id_or_query else None
        rag = self.retriever.ask(
            query=self._prompt_extract_references(doc_id_or_query),
            k=10,
            metadata_filter=metadata_filter,
        )
        return TaskResult(
            task="extract_references",
            query=doc_id_or_query,
            result_text=rag["answer"],
            evidence=rag["sources"],
        ).to_dict()

    @staticmethod
    def _prompt_summarize(doc_ref: str) -> str:
        return (
            "Create a concise but comprehensive document summary.\n"
            f"Document reference: {doc_ref}\n"
            "Structure output as: (1) purpose, (2) key findings, (3) methods/data, (4) limitations."
        )

    @staticmethod
    def _prompt_compare(topic: str) -> str:
        return (
            "Compare relevant documents on the requested topic.\n"
            f"Topic/query: {topic}\n"
            "Structure output as: similarities, differences, evidence-backed conclusions, open questions."
        )

    @staticmethod
    def _prompt_literature_review(topic: str) -> str:
        return (
            "Write a literature review from retrieved evidence.\n"
            f"Topic: {topic}\n"
            "Structure output as: thematic synthesis, consensus, disagreements, gaps, future directions."
        )

    @staticmethod
    def _prompt_extract_references(doc_ref: str) -> str:
        return (
            "Extract and list references/citations from the evidence.\n"
            f"Document reference: {doc_ref}\n"
            "Return a clean reference list and note ambiguous items separately."
        )


def summarize_document(doc_id_or_query: str, retriever: DoclingRetriever | None = None) -> dict[str, Any]:
    return ResearchTasks(retriever or DoclingRetriever()).summarize_document(doc_id_or_query)


def compare_documents(
    topic_or_query: str,
    k: int = 8,
    retriever: DoclingRetriever | None = None,
) -> dict[str, Any]:
    return ResearchTasks(retriever or DoclingRetriever()).compare_documents(topic_or_query, k=k)


def literature_review(topic: str, k: int = 10, retriever: DoclingRetriever | None = None) -> dict[str, Any]:
    return ResearchTasks(retriever or DoclingRetriever()).literature_review(topic, k=k)


def extract_references(doc_id_or_query: str, retriever: DoclingRetriever | None = None) -> dict[str, Any]:
    return ResearchTasks(retriever or DoclingRetriever()).extract_references(doc_id_or_query)
