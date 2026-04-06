"""Data models for scientific PDF annotation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Annotation:
    annotation_id: str
    doc_id: str
    chunk_id: str | None
    source_path: str
    page_numbers: list[int]
    selected_text: str
    note: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        doc_id: str,
        source_path: str,
        selected_text: str,
        note: str,
        chunk_id: str | None = None,
        page_numbers: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Annotation":
        return cls(
            annotation_id=str(uuid4()),
            doc_id=doc_id,
            chunk_id=chunk_id,
            source_path=source_path,
            page_numbers=page_numbers or [],
            selected_text=selected_text,
            note=note,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "page_numbers": self.page_numbers,
            "selected_text": self.selected_text,
            "note": self.note,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Annotation":
        return cls(
            annotation_id=str(raw.get("annotation_id", "")),
            doc_id=str(raw.get("doc_id", "")),
            chunk_id=raw.get("chunk_id"),
            source_path=str(raw.get("source_path", "")),
            page_numbers=[p for p in raw.get("page_numbers", []) if isinstance(p, int)],
            selected_text=str(raw.get("selected_text", "")),
            note=str(raw.get("note", "")),
            created_at=str(raw.get("created_at", "")),
            metadata=raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {},
        )
