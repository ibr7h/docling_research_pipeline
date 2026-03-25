# docling_research_pipeline

Phase 1 implements a Docling-first ingestion flow and a Chroma indexing utility.

## Phase 1 scope

- Use **Docling** as the core conversion engine.
- Persist per-document artifacts under an output directory:
  - `document.json` (**source of truth**)
  - `document.md` (display only)
  - `chunks.jsonl` (structure-aware chunks)
  - `metadata.json`
- Structure-aware chunking supports `text` and `table` chunks.
- Every chunk record includes:
  - `doc_id`
  - `chunk_id`
  - `chunk_type`
  - `section_title`
  - `page_numbers`
  - `source_path`
- Chroma indexing is implemented in `index_chunks.py`.
- Future module boundaries are reserved for:
  - `retriever.py`
  - `tasks.py`
  - `annotations/`

## Files

- `pipeline.py` — ingest/convert documents with Docling and emit Phase 1 artifacts.
- `index_chunks.py` — read `chunks.jsonl` and upsert chunks into a Chroma collection.
- `retriever.py` — Chroma-backed retrieval with `search()` and RAG `ask()` methods.
- `tasks.py`, `annotations/` — placeholders for later phases.

## Quickstart

### 1) Ingest documents with Docling

```bash
python pipeline.py <input_path> --output-dir artifacts
```

`<input_path>` can be either a single file or a directory.

### 2) Index chunks into Chroma

```bash
python index_chunks.py --artifacts-dir artifacts --chroma-dir chroma_db --collection docling_chunks
```

## Notes

- `document.json` should be treated as canonical for downstream processing.
- `document.md` is intended for display/debugging only.
- Later phases for tasks and annotations workflows are intentionally not implemented yet.


### 3) Retrieve or ask (RAG)

```python
from retriever import DoclingRetriever

r = DoclingRetriever(chroma_dir="chroma_db", collection_name="docling_chunks")
results = r.search("What does the contract say about termination?", k=5)
answer = r.ask("Summarize termination conditions")
```
