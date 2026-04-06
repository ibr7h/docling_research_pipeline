"""Storage layer for annotation records."""

from __future__ import annotations

import json
from pathlib import Path

from annotations.models import Annotation


class AnnotationStore:
    """File-backed annotation storage using JSONL.

    Design goals:
    - append-friendly writes
    - easy ingestion into retrieval/summarization pipelines
    - stable schema for future annotation-aware retrieval
    """

    def __init__(self, root_dir: str | Path = "artifacts/annotations") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_file = self.root_dir / "annotations.jsonl"

    def add(self, annotation: Annotation) -> Annotation:
        with self.annotations_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(annotation.to_dict(), ensure_ascii=False) + "\n")
        return annotation

    def create_and_add(
        self,
        *,
        doc_id: str,
        source_path: str,
        selected_text: str,
        note: str,
        chunk_id: str | None = None,
        page_numbers: list[int] | None = None,
        metadata: dict | None = None,
    ) -> Annotation:
        annotation = Annotation.create(
            doc_id=doc_id,
            source_path=source_path,
            selected_text=selected_text,
            note=note,
            chunk_id=chunk_id,
            page_numbers=page_numbers,
            metadata=metadata,
        )
        return self.add(annotation)

    def list_all(self) -> list[Annotation]:
        if not self.annotations_file.exists():
            return []

        out: list[Annotation] = []
        with self.annotations_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(Annotation.from_dict(json.loads(line)))
        return out

    def find(
        self,
        *,
        doc_id: str | None = None,
        chunk_id: str | None = None,
        source_path: str | None = None,
        text_query: str | None = None,
    ) -> list[Annotation]:
        results = self.list_all()
        if doc_id is not None:
            results = [a for a in results if a.doc_id == doc_id]
        if chunk_id is not None:
            results = [a for a in results if a.chunk_id == chunk_id]
        if source_path is not None:
            results = [a for a in results if a.source_path == source_path]
        if text_query:
            q = text_query.lower()
            results = [
                a
                for a in results
                if q in a.selected_text.lower() or q in a.note.lower()
            ]
        return results

    def as_dicts(self, **filters: str) -> list[dict]:
        return [a.to_dict() for a in self.find(**filters)]
