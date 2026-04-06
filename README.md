# docling_research_pipeline

Docling-native research pipeline with ingestion, indexing, retrieval, task workflows, and interactive CLI.

## Architecture

- `pipeline.py` — Phase 1 ingestion using **Docling**.
  - Per document outputs:
    - `document.json` (**source of truth**)
    - `document.md` (display only)
    - `chunks.jsonl` (structure-aware chunks)
    - `metadata.json`
- `index_chunks.py` — Chroma indexing for `chunks.jsonl`.
- `retriever.py` — Chroma-backed `search()` + RAG `ask()`.
- `tasks.py` — Higher-level research tasks built on `retriever.py`:
  - `summarize_document(doc_id_or_query)`
  - `compare_documents(topic_or_query, k=8)`
  - `literature_review(topic, k=10)`
  - `extract_references(doc_id_or_query)`
- `cli.py` — interactive menu for indexing, retrieval, and tasks.
- `annotations/` remains a modular boundary for future workflow expansion.

## Metadata

Chunk metadata is preserved end-to-end and includes:

- `doc_id`
- `chunk_id`
- `chunk_type` (`text` or `table`)
- `section_title`
- `page_numbers`
- `source_path`

## Quickstart

### 1) Ingest with Docling

```bash
python pipeline.py <input_path> --output-dir artifacts
```

### 2) Index chunks into Chroma

```bash
python index_chunks.py --artifacts-dir artifacts --chroma-dir chroma_db --collection docling_chunks
```

### 3) Run interactive CLI

```bash
python cli.py
```

Menu:

1. Index documents
2. Search
3. Ask question
4. Summarize document
5. Compare documents
6. Literature review
7. Extract references
0. Exit


## Retrieval improvements (Phase 4)

`retriever.py` now includes retrieval-quality enhancements while preserving API compatibility:

- Public API unchanged:
  - `search(query, k=5, metadata_filter=None)`
  - `ask(query, k=5, metadata_filter=None, ...)`
- Metadata filtering support for:
  - `doc_id`
  - `chunk_type`
  - `section_title`
  - `source_path`
- Hybrid-search hook (`_hybrid_search`) that combines vector scores with lexical BM25-like scoring.
- Reranking hook (`_rerank`) for future cross-encoder or model rerankers.
- Result filtering hook (`_filter_results`) for post-retrieval metadata constraints.
- Table-aware behavior that boosts table chunks for table-oriented queries.
- Multilingual embedding readiness hook (`_prepare_query_for_embedding`) for future language-aware routing/model selection.
- Improved `ask()` evidence output includes `evidence_summary` for concise source tracing.


## Annotation layer (Phase 5)

Annotation support is implemented under `annotations/` with a reusable data/storage model for scientific PDFs (no rendering UI yet):

- `annotations/models.py`
  - `Annotation` schema with:
    - `annotation_id`
    - `doc_id`
    - `chunk_id` (optional)
    - `source_path`
    - `page_numbers`
    - `selected_text`
    - `note`
    - `created_at`
- `annotations/store.py`
  - `AnnotationStore` for JSONL-backed persistence (`artifacts/annotations/annotations.jsonl`)
  - add/create/list/find query helpers
- `annotations/hooks.py`
  - `attach_annotations_to_results(...)` to enrich retrieval evidence
  - `annotation_context_for_prompt(...)` to build annotation-aware prompt context

Example:

```python
from annotations import AnnotationStore, attach_annotations_to_results
from retriever import DoclingRetriever

store = AnnotationStore()
store.create_and_add(
    doc_id="paper-123",
    chunk_id="chunk-abc",
    source_path="/data/paper-123.pdf",
    page_numbers=[4],
    selected_text="Key finding sentence...",
    note="Important for methods comparison",
)

retriever = DoclingRetriever()
results = retriever.search("methods", k=5)
results_with_annotations = attach_annotations_to_results(results, store)
```

## tasks.py usage

```python
from retriever import DoclingRetriever
from tasks import ResearchTasks

retriever = DoclingRetriever(chroma_dir="chroma_db", collection_name="docling_chunks")
workflow = ResearchTasks(retriever)

summary = workflow.summarize_document("my_doc_id")
comparison = workflow.compare_documents("evaluation methodology", k=8)
review = workflow.literature_review("document intelligence", k=10)
refs = workflow.extract_references("my_doc_id")
```

Each task returns:

- `result_text` (final generated output)
- `evidence` (retrieved chunks/sources)

## Example interaction flow

1. Use option **1) Index documents** and point at your source folder.
2. Use **2) Search** to inspect retrieved chunks.
3. Use **3) Ask question** for direct RAG Q&A.
4. Use **4-7** for task-oriented outputs with source evidence.

## Notes

- Retrieval/task behavior remains Docling-native through `document.json`-derived chunk artifacts.
- `document.md` is display-only and should not be treated as canonical.
- `doc_id` values are generated as stable slug+hash identifiers to avoid filename-collision overwrites.
- Future work can plug in hybrid search, reranking, metadata filters, and annotation-aware retrieval without changing public task interfaces.
