"""Interactive CLI for the Docling-native research pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

import index_chunks
import pipeline
from retriever import DoclingRetriever
from tasks import ResearchTasks


def _print_evidence(items: list[dict[str, Any]]) -> None:
    if not items:
        print("No evidence returned.")
        return
    for idx, item in enumerate(items, start=1):
        print(f"\n[{idx}] doc_id={item.get('doc_id')} chunk_id={item.get('chunk_id')}")
        print(f"    section={item.get('section_title')} type={item.get('chunk_type')}")
        print(f"    pages={item.get('page_numbers')} source={item.get('source_path')}")
        snippet = (item.get("text") or "").replace("\n", " ")
        print(f"    text={snippet[:240]}{'...' if len(snippet) > 240 else ''}")


def _index_documents() -> tuple[str, str] | None:
    input_path = Path(input("Input file/dir: ").strip())
    artifacts_dir = Path(input("Artifacts dir [artifacts]: ").strip() or "artifacts")
    chroma_dir = Path(input("Chroma dir [chroma_db]: ").strip() or "chroma_db")
    collection = input("Collection [docling_chunks]: ").strip() or "docling_chunks"

    docs = pipeline.find_documents(input_path)
    if not docs:
        print(f"No supported documents found at {input_path}")
        return None

    converter = DocumentConverter()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        out_dir = pipeline.process_document(doc, artifacts_dir, converter)
        print(f"Processed: {doc} -> {out_dir}")

    indexed = index_chunks.index_chunks(artifacts_dir, chroma_dir, collection)
    print(f"Indexed {indexed} chunks in collection '{collection}'")
    return str(chroma_dir), collection


def main() -> None:
    chroma_dir = "chroma_db"
    collection = "docling_chunks"
    retriever = DoclingRetriever(chroma_dir=chroma_dir, collection_name=collection)
    tasks = ResearchTasks(retriever)

    while True:
        print("\n=== Docling Research CLI ===")
        print("1) Index documents")
        print("2) Search")
        print("3) Ask question")
        print("4) Summarize document")
        print("5) Compare documents")
        print("6) Literature review")
        print("7) Extract references")
        print("0) Exit")

        choice = input("Select option: ").strip()
        if choice == "0":
            print("Goodbye.")
            break
        if choice == "1":
            updated = _index_documents()
            if updated is not None:
                chroma_dir, collection = updated
            retriever = DoclingRetriever(chroma_dir=chroma_dir, collection_name=collection)
            tasks = ResearchTasks(retriever)
        elif choice == "2":
            q = input("Query: ").strip()
            k = int(input("Top-k [5]: ").strip() or "5")
            results = retriever.search(q, k=k)
            _print_evidence(results)
        elif choice == "3":
            q = input("Question: ").strip()
            k = int(input("Top-k [5]: ").strip() or "5")
            resp = retriever.ask(q, k=k)
            print("\nAnswer:\n" + resp.get("answer", ""))
            print("\nSources:")
            _print_evidence(resp.get("sources", []))
        elif choice == "4":
            ref = input("Document id or query: ").strip()
            result = tasks.summarize_document(ref)
            print("\nSummary:\n" + result["result_text"])
            print("\nEvidence:")
            _print_evidence(result["evidence"])
        elif choice == "5":
            topic = input("Topic/query: ").strip()
            k = int(input("Top-k [8]: ").strip() or "8")
            result = tasks.compare_documents(topic, k=k)
            print("\nComparison:\n" + result["result_text"])
            print("\nEvidence:")
            _print_evidence(result["evidence"])
        elif choice == "6":
            topic = input("Topic: ").strip()
            k = int(input("Top-k [10]: ").strip() or "10")
            result = tasks.literature_review(topic, k=k)
            print("\nLiterature Review:\n" + result["result_text"])
            print("\nEvidence:")
            _print_evidence(result["evidence"])
        elif choice == "7":
            ref = input("Document id or query: ").strip()
            result = tasks.extract_references(ref)
            print("\nReferences:\n" + result["result_text"])
            print("\nEvidence:")
            _print_evidence(result["evidence"])
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
