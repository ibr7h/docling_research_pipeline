"""Phase 1 chunk indexing with ChromaDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb


def iter_chunk_files(artifacts_root: Path) -> list[Path]:
    return sorted(artifacts_root.rglob("chunks.jsonl"))


def load_chunks(chunks_file: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    page_numbers = chunk.get("page_numbers", [])
    if not isinstance(page_numbers, list):
        page_numbers = []
    page_numbers = [p for p in page_numbers if isinstance(p, int)]

    return {
        "doc_id": chunk.get("doc_id"),
        "chunk_id": chunk.get("chunk_id"),
        "chunk_type": chunk.get("chunk_type"),
        "section_title": chunk.get("section_title"),
        "page_numbers": ",".join(str(p) for p in page_numbers),
        "page_start": min(page_numbers) if page_numbers else None,
        "page_end": max(page_numbers) if page_numbers else None,
        "source_path": chunk.get("source_path"),
    }


def index_chunks(artifacts_root: Path, chroma_dir: Path, collection_name: str) -> int:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)

    count = 0
    for chunk_file in iter_chunk_files(artifacts_root):
        for chunk in load_chunks(chunk_file):
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            collection.upsert(
                ids=[chunk_id],
                documents=[text],
                metadatas=[build_metadata(chunk)],
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Index chunks.jsonl into Chroma")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root directory containing per-document artifact folders",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("chroma_db"),
        help="Chroma persistent storage directory",
    )
    parser.add_argument(
        "--collection",
        default="docling_chunks",
        help="Chroma collection name",
    )
    args = parser.parse_args()

    indexed = index_chunks(args.artifacts_dir, args.chroma_dir, args.collection)
    print(f"Indexed {indexed} chunks into collection '{args.collection}'")


if __name__ == "__main__":
    main()
