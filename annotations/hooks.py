"""Hooks for integrating annotations into retrieval and task pipelines."""

from __future__ import annotations

from typing import Any

from annotations.store import AnnotationStore


def attach_annotations_to_results(
    results: list[dict[str, Any]],
    store: AnnotationStore,
) -> list[dict[str, Any]]:
    """Attach related annotations to each retrieved evidence item.

    Matching strategy:
    1) exact chunk_id when available
    2) fallback by doc_id/source_path
    """
    enriched: list[dict[str, Any]] = []
    for item in results:
        chunk_id = item.get("chunk_id")
        doc_id = item.get("doc_id")
        source_path = item.get("source_path")

        ann = []
        if chunk_id:
            ann = store.as_dicts(chunk_id=chunk_id)
        if not ann and doc_id:
            ann = store.as_dicts(doc_id=doc_id)
        if not ann and source_path:
            ann = store.as_dicts(source_path=source_path)

        updated = dict(item)
        updated["annotations"] = ann
        enriched.append(updated)
    return enriched


def annotation_context_for_prompt(
    results: list[dict[str, Any]],
    max_annotations_per_result: int = 3,
) -> str:
    """Build concise annotation context for future RAG/task prompts."""
    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        annotations = item.get("annotations") or []
        if not annotations:
            continue

        lines.append(f"[Annotations for Evidence {idx}]")
        for annotation in annotations[:max_annotations_per_result]:
            lines.append(
                " - "
                + " | ".join(
                    [
                        f"annotation_id={annotation.get('annotation_id')}",
                        f"pages={annotation.get('page_numbers')}",
                        f"selected_text={annotation.get('selected_text')}",
                        f"note={annotation.get('note')}",
                    ]
                )
            )
    return "\n".join(lines)
