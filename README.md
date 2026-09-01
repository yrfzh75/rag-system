# Bilingual RAG MVP

This repository is the starting point for a bilingual retrieval-augmented generation service.
The first milestone intentionally contains only a runnable API and a health check. Ingestion,
retrieval, generation, and evaluation will be added as separate, testable vertical slices.

## Ingestion design: MVP and production

### MVP

- Ingestion is a manually triggered batch command: `rag-ingest ./documents`.
- Repeated runs are idempotent: an existing document's chunks are replaced rather than duplicated.
- The milestone prioritizes validating parsing, OCR, chunking, embedding, metadata, and Qdrant
  storage.
- Automatic scheduling, event processing, and deletion reconciliation are intentionally deferred.

### Production

- Use event-driven, incremental ingestion so only added, modified, or deleted documents are
  processed instead of repeatedly ingesting the entire corpus.
- Under normal load, target **95% of changed documents becoming searchable within 15 minutes**.
  This target measures the delay for an individual change; it does not require millions of files
  to be rescanned within 15 minutes.
- Process urgent compliance and security changes immediately or through a higher-priority queue.
- Use a durable queue and multiple workers to handle changes in parallel and scale with corpus size.
- Run a separate nightly or weekly full reconciliation to detect missed events, changed files, and
  deleted source documents. Large reconciliations and backfills may take several hours and should
  have separate completion targets.
- Prevent overlapping jobs and monitor queue delay, ingestion failures, processing latency, and
  end-to-end document freshness.

## Prerequisites

- Python 3.11 or newer

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn rag_mvp.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation, or check:

```bash
curl http://127.0.0.1:8000/health
```

## Run the test cases

From the repository root, activate the virtual environment and run the automated tests:

```bash
cd /Users/yrfzh/Coding/rag-system
source .venv/bin/activate
python -m pytest -q
```

A successful run currently ends with `11 passed`. Run the static checks separately:

```bash
python -m ruff check .
```

## Ingestion milestone

The offline ingestion pipeline currently supports:

- recursive discovery of PDF, DOCX, TXT, and Markdown files;
- native PDF text extraction with reading-order sorting;
- page-level OCR fallback for text-poor scanned PDF pages;
- bilingual CN/EN-aware chunking that does not depend on whitespace;
- stable document and chunk identifiers for repeatable ingestion;
- local multilingual embeddings; and
- idempotent replacement of each document in a persistent Qdrant collection.

Put internal documents under `documents/`, then install/synchronize dependencies and run:

```bash
uv sync --extra dev
cp .env.example .env
uv run rag-ingest ./documents
```

If the project was installed with `pip install -e '.[dev]'` instead of `uv`, use the
`rag-ingest` command directly as shown below.

The first run downloads the configured multilingual embedding model into `data/models/`. The command emits one
structured JSON event per document followed by an ingestion summary. If any document fails, the
remaining documents are still processed and the command exits nonzero after printing the report.

Scanned PDFs require the Tesseract executable plus the `eng` and `chi_sim` language data. Native
PDFs do not use Tesseract. To ingest only native-text documents while OCR is unavailable, set:

```dotenv
INGEST_OCR_ENABLED=false
```

### Validate ingestion in Terminal

This repository has a project-local Tesseract installation under `.tools/`. In every new Terminal
session, activate Python and add that installation to the current shell environment:

```bash
cd /Users/yrfzh/Coding/rag-system
source .venv/bin/activate
export PATH="$PWD/.tools/mamba-root/envs/ocr/bin:$PATH"
export TESSDATA_PREFIX="$PWD/.tools/mamba-root/envs/ocr/share/tessdata"
```

Confirm that Tesseract and both OCR languages are available:

```bash
which tesseract
tesseract --list-langs | grep -E 'eng|chi_sim'
```

The first command should point inside `.tools/mamba-root/envs/ocr/bin/`, and the language check
should print both `chi_sim` and `eng`.

Run the complete ingestion flow, including OCR for scanned PDFs:

```bash
INGEST_OCR_ENABLED=true \
INGEST_OCR_LANGUAGES=eng+chi_sim \
rag-ingest ./documents
```

With the five included sample documents, the final JSON event should report:

```json
{
  "event": "ingestion.completed",
  "discovered_files": 5,
  "ingested_files": 5,
  "failed_files": 0,
  "chunks_written": 8,
  "native_pages": 5,
  "ocr_pages": 1,
  "errors": []
}
```

Run the same command a second time to verify idempotent replacement. It should succeed with the
same counts rather than creating duplicate chunks.

If the scanned PDF fails with `Tesseract executable is not installed`, repeat the `PATH` and
`TESSDATA_PREFIX` exports above in the current Terminal session. The exports do not persist when a
new Terminal window is opened.

Qdrant runs in embedded local mode and persists its collection under `data/qdrant/`; no separate
database server is required for this milestone. Only one process should open this local database at
a time. A Qdrant server can replace it later for multi-process deployment and load testing.

### Ingestion output metadata

Every stored chunk includes its stable document/chunk IDs, source path and filename, content hash,
file type, page number when available, detected language, extraction method (`native` or `ocr`),
chunk index, and original chunk text. This metadata supports citations, incremental replacement,
language analysis, and ingestion diagnosis in later milestones.

## Planned milestones

1. Document ingestion and metadata-preserving chunking
2. Vector retrieval with a small evaluation dataset
3. BM25 hybrid retrieval and optional reranking
4. Grounded answer generation, citations, and refusal behavior
5. Bilingual, safety, latency, cost, and reliability evaluation
