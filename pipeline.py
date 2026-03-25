"""Phase 1 ingestion pipeline.

This module converts source documents with Docling and writes per-document artifacts:
- document.json (source of truth)
- document.md (display-only rendering)
- chunks.jsonl (structure-aware chunks)
- metadata.json (pipeline metadata)
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from docling.document_converter import DocumentConverter


SUPPORTED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".md",
    ".html",
    ".txt",
}


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    chunk_type: str  # text | table
    text: str
    section_title: str | None
    page_numbers: list[int]
    source_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "section_title": self.section_title,
            "page_numbers": self.page_numbers,
            "source_path": self.source_path,
        }


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "document"


def find_documents(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    docs = [
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(docs)


def as_dict(doc: Any) -> dict[str, Any]:
    for method_name in ("export_to_dict", "to_dict", "model_dump"):
        method = getattr(doc, method_name, None)
        if callable(method):
            data = method()
            if isinstance(data, dict):
                return data
    if isinstance(doc, dict):
        return doc
    raise TypeError("Unable to serialize Docling document to dict")


def as_markdown(doc: Any) -> str:
    for method_name in ("export_to_markdown", "to_markdown"):
        method = getattr(doc, method_name, None)
        if callable(method):
            md = method()
            if isinstance(md, str):
                return md
    text = _extract_text_fallback(as_dict(doc))
    return text.strip() + "\n"


def _extract_text_fallback(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join(_extract_text_fallback(item) for item in data)
    if isinstance(data, dict):
        bits: list[str] = []
        for key in ("text", "content", "title", "caption"):
            val = data.get(key)
            if isinstance(val, str):
                bits.append(val)
        for val in data.values():
            if isinstance(val, (dict, list)):
                bits.append(_extract_text_fallback(val))
        return "\n".join(bit for bit in bits if bit)
    return ""


def _infer_chunk_type(node: dict[str, Any]) -> str | None:
    node_type = str(node.get("type", "")).lower()
    if "table" in node_type:
        return "table"
    if any(k in node for k in ("rows", "cells", "table")):
        return "table"
    if any(k in node for k in ("text", "content", "paragraph")):
        return "text"
    return None


def _extract_chunk_text(node: dict[str, Any], chunk_type: str) -> str:
    if chunk_type == "table":
        if isinstance(node.get("table"), str):
            return node["table"]
        rows = node.get("rows")
        if isinstance(rows, list):
            rendered: list[str] = []
            for row in rows:
                if isinstance(row, list):
                    rendered.append(" | ".join(str(cell) for cell in row))
                else:
                    rendered.append(str(row))
            return "\n".join(rendered)
    for key in ("text", "content", "paragraph", "caption"):
        val = node.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return _extract_text_fallback(node).strip()


def _extract_pages(node: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for key in ("page", "page_no", "page_number"):
        val = node.get(key)
        if isinstance(val, int):
            pages.append(val)
    for key in ("pages", "page_numbers"):
        val = node.get(key)
        if isinstance(val, list):
            pages.extend(v for v in val if isinstance(v, int))
    return sorted(set(pages))


def iter_structure_aware_chunks(
    data: dict[str, Any],
    *,
    doc_id: str,
    source_path: str,
) -> Iterable[ChunkRecord]:
    """Walk JSON tree and emit structure-aware chunks.

    The logic carries forward the nearest section title to child nodes.
    """

    def walk(node: Any, current_section: str | None) -> Iterable[ChunkRecord]:
        if isinstance(node, list):
            for item in node:
                yield from walk(item, current_section)
            return
        if not isinstance(node, dict):
            return

        section = current_section
        node_type = str(node.get("type", "")).lower()
        if "heading" in node_type or "section" in node_type:
            for title_key in ("text", "title", "content"):
                title = node.get(title_key)
                if isinstance(title, str) and title.strip():
                    section = title.strip()
                    break

        chunk_type = _infer_chunk_type(node)
        if chunk_type in {"text", "table"}:
            text = _extract_chunk_text(node, chunk_type)
            if text:
                yield ChunkRecord(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    chunk_type=chunk_type,
                    text=text,
                    section_title=section,
                    page_numbers=_extract_pages(node),
                    source_path=source_path,
                )

        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from walk(value, section)

    yield from walk(data, None)


def process_document(source_file: Path, output_root: Path, converter: DocumentConverter) -> Path:
    result = converter.convert(str(source_file))
    doc = getattr(result, "document", result)

    doc_id = slugify(source_file.stem)
    out_dir = output_root / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_json = as_dict(doc)
    doc_md = as_markdown(doc)

    document_json_path = out_dir / "document.json"
    document_md_path = out_dir / "document.md"
    chunks_path = out_dir / "chunks.jsonl"
    metadata_path = out_dir / "metadata.json"

    document_json_path.write_text(
        json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    document_md_path.write_text(doc_md, encoding="utf-8")

    chunks = list(
        iter_structure_aware_chunks(
            doc_json,
            doc_id=doc_id,
            source_path=str(source_file.resolve()),
        )
    )

    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")

    metadata = {
        "doc_id": doc_id,
        "source_path": str(source_file.resolve()),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "artifact_paths": {
            "document_json": str(document_json_path),
            "document_md": str(document_md_path),
            "chunks_jsonl": str(chunks_path),
        },
        "chunk_counts": {
            "total": len(chunks),
            "text": sum(1 for c in chunks if c.chunk_type == "text"),
            "table": sum(1 for c in chunks if c.chunk_type == "table"),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Docling-based Phase 1 ingestion pipeline")
    parser.add_argument("input", type=Path, help="Source file or directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory to write per-document artifacts",
    )
    args = parser.parse_args()

    docs = find_documents(args.input)
    if not docs:
        raise SystemExit(f"No supported documents found at: {args.input}")

    converter = DocumentConverter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for doc_path in docs:
        out_dir = process_document(doc_path, args.output_dir, converter)
        print(f"Processed {doc_path} -> {out_dir}")


if __name__ == "__main__":
    main()
