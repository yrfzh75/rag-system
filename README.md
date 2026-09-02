# Bilingual RAG MVP

This repository is the starting point for a bilingual retrieval-augmented generation service.
It includes ingestion, configurable retrieval, local grounded generation, safety controls,
observability, caching, and reproducible evaluation as separate, testable vertical slices.

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

A successful run currently ends with all tests passing. Run the static checks separately:

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

## Test vector retrieval

The retrieval-only API embeds a user query with the same multilingual model used for ingestion and
returns ranked Qdrant chunks with scores and source metadata. It does not call an LLM or generate an
answer, which allows retrieval quality to be tested independently.

Start the API from the repository root:

```bash
source .venv/bin/activate
uvicorn rag_mvp.main:app --reload
```

In a second Terminal window, submit an English query:

```bash
curl -s http://127.0.0.1:8000/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many weeks of parental leave are available?","top_k":3}' \
  | python -m json.tool
```

The response should return `employee_handbook.md` as the highest-ranked source and include the
passage stating that eligible employees receive sixteen weeks of paid parental leave. A Chinese
query can be tested with:

```bash
curl -s http://127.0.0.1:8000/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"员工有多少周育儿假？","top_k":3}' \
  | python -m json.tool
```

Optional request fields are `top_k` (1–20, default 5) and `score_threshold` (-1.0–1.0, default
0.3). The response contains retrieved evidence only: chunk text, similarity score, source file,
page number, language, and extraction method.

### Retrieval modes and evaluation

Retrieval behavior is configuration-driven and does not require a code change:

```dotenv
RETRIEVAL_MODE=hybrid
RETRIEVAL_CANDIDATE_K=10
RETRIEVAL_RERANKER_ENABLED=false
```

- `vector_only` uses multilingual dense embeddings and Qdrant cosine search.
- `hybrid` combines dense results with an in-memory bilingual BM25 index using reciprocal-rank
  fusion.
- Enabling the reranker applies a deterministic bilingual lexical-overlap refinement to the fused
  candidates. It requires no additional model download.

Run the reproducible three-way comparison with:

```bash
source .venv/bin/activate
rag-evaluate \
  --dataset evals/retrieval_cases.json \
  --output evals/reports/retrieval_baseline \
  --top-k 3
```

The evaluator compares vector-only, hybrid, and hybrid+rerank using Hit@k, MRR, rank-aware context
precision, out-of-scope refusal accuracy, and p50/p95 retrieval latency. It writes JSON details, a
CSV summary, and a Markdown report. The included 12-case dataset covers English, Chinese, OCR,
compliance, architecture, and out-of-scope queries.

Metric definitions:

- **Hit@k:** fraction of answerable queries with an expected source in the first `k` results.
- **MRR:** mean reciprocal rank of the first expected source.
- **Context precision:** average precision at the ranks containing expected sources, averaged over
  answerable queries; this rewards placing relevant context before irrelevant context.
- **Refusal accuracy:** fraction of labeled out-of-scope queries that return no context above the
  confidence threshold.
- **p50/p95 latency:** median and 95th-percentile local retrieval time, excluding generation.

The current local result recommends hybrid without reranking: both hybrid configurations improved
Hit@3 and context precision from `0.90` to `1.00` and retained `1.00` refusal accuracy, while plain
hybrid was slightly faster in this run. Reranking remains configurable for evaluation on a larger
corpus. See `evals/reports/retrieval_baseline/report.md` for the comparison and limitations.

## Test grounded question answering

The `POST /qa` endpoint connects vector retrieval to a free local model served by Ollama. It
answers only from retrieved context, cites numbered passages, refuses before calling the LLM when
no context passes the retrieval threshold, and reports model, token usage, and
retrieval/generation timings.

Install Ollama, start it, and download the bilingual Qwen3 model once:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Configure `.env` for local generation with no per-call API charge:

```dotenv
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
LLM_MAX_OUTPUT_TOKENS=256
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=120
```

Ollama must be running while the RAG API handles `/qa` requests. Local generation avoids provider
charges, but uses the Mac's CPU, GPU, memory, and electricity and may be slower or less accurate
than a hosted model. The service uses the dedicated non-thinking Instruct model and also sends
`think: false` to prevent reasoning traces from appearing in answers.

Start the API:

```bash
source .venv/bin/activate
uvicorn rag_mvp.main:app --reload
```

In a second Terminal, ask a grounded question:

```bash
curl -s http://127.0.0.1:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many weeks of parental leave are available?","top_k":3}' \
  | python3 -m json.tool
```

The answer should state sixteen weeks and cite `[1]`; the first source should be
`employee_handbook.md`. Automated tests use fake retrieval and Ollama clients, so
`python -m pytest -q` does not require Ollama to be running.

### Why the MVP uses a free local model

The MVP uses the pinned `qwen3:4b-instruct-2507-q4_K_M` model through Ollama because:

- **Zero API cost:** local inference has **$0 API cost per 1,000 calls**, which enables repeated
  development and evaluation without consuming a hosted-model budget.
- **Privacy:** retrieved internal document text remains on the developer's machine instead of being
  sent to an external model provider.
- **Bilingual support:** Qwen3 supports the English and Chinese corpus used by this project.
- **Practical local size:** the quantized 4B model is small enough for MVP development on a modern
  Mac while still producing grounded answers in roughly one second for the included sample corpus.
- **Reproducibility:** the full model tag is pinned so the shorter `qwen3:4b` alias cannot silently
  change to a different or thinking-only build.

The trade-offs are local CPU/GPU and memory usage, machine-dependent latency, and potentially lower
answer quality than larger hosted models. The service records model name, latency, and token usage
so this choice can be evaluated quantitatively and reconsidered for production.

## Service hardening and multi-turn QA

The `/qa` request accepts an optional `session_id`. The service keeps the last three redacted turns
in process and uses the most recent turn to improve follow-up retrieval. This is intentionally an
MVP store: production should move encrypted conversation state to a shared store with retention and
deletion policies.

Runtime controls include:

- deterministic English/Chinese prompt-injection refusal before retrieval or generation;
- email, phone, and US SSN redaction from model input, answer, citations, and conversation history;
- a bounded five-minute process-local answer cache; requests containing PII and redacted answers
  are never cached;
- a post-generation grounding gate that validates citation IDs and context support, refusing output
  below `QA_GROUNDING_MIN_SUPPORT`;
- structured JSONL events containing request/trace/span IDs, retrieval/model choices, tokens,
  grounding score, latency, cache, refusal, and PII signals without raw prompts or answers; and
- configuration-only retrieval, reranker, model, cache, history, and logging changes through `.env`.

Example multi-turn requests:

```bash
curl -s http://127.0.0.1:8000/qa -H 'Content-Type: application/json' \
  -d '{"query":"How many weeks of parental leave are available?","session_id":"demo-1"}'
curl -s http://127.0.0.1:8000/qa -H 'Content-Type: application/json' \
  -d '{"query":"Who is eligible for it?","session_id":"demo-1"}'
```

### One-command validation

With the virtual environment active and Ollama running:

```bash
./scripts/evaluate.sh
```

This starts an isolated cache-disabled validation API, runs unit tests, Ruff, the three retrieval
configurations, QA quality evaluation, and a five-distinct-request/5-concurrent cold-cache load
check. It writes the operations CSV and shuts down the temporary API automatically.

The QA evaluator uses the labeled bilingual dataset in `evals/qa_cases.json` and reports:

- **Faithfulness:** fraction of non-stopword answer tokens present in cited passages; target
  `>= 0.85`. This deterministic metric is reproducible but should later be supplemented by human or
  claim-level judging for valid paraphrases.
- **Answer compliance:** correct refusal label, all expected facts, and citations; target `>= 0.90`
  (stricter than the original 80% requirement).
- **Style consistency:** same language as the question, numbered citations, and no thinking trace;
  target `>= 0.85`.
- **Refusal appropriateness:** correct decision for answerable, out-of-scope, and injection cases;
  target `>= 0.90`.
- **Performance:** at least 90% of calls at or below 10 seconds with five concurrent requests.

Local Ollama has `$0` provider/API cost per 1,000 calls. Token counts are still recorded so a
hosted replacement can calculate `(input tokens × input rate) + (output tokens × output rate)` for
1,000 calls without changing the service contract.

Generate an operational report independently with:

```bash
rag-ops-report --logs artifacts/qa_events.jsonl --output artifacts/operations.csv
```

It contains p50/p95 latency, total token usage, cache-hit rate, refusal rate, and imports the
answer-compliance rate from the latest QA evaluation (or marks it `not_measured`). See
`docs/log_field_dictionary.md`,
`docs/sample_logs.jsonl`, and `evals/reports/issue_diagnosis.md` for the log contract, safe samples,
and two before/after diagnoses with at least 10% improvement.

The recorded final run in `evals/reports/qa_latest/` passes the requested thresholds:
faithfulness `0.870`, answer compliance `1.00`, style consistency `1.00`, refusal appropriateness
`1.00`, and 100% of the six QA cases within 10 seconds. The five-distinct-request cold-cache load
run used five concurrent clients, had zero cache hits, and passed with p95 `6.76s`. These results
are evidence for this sample corpus
and machine, not a production capacity guarantee; rerun the script after any corpus, prompt, model,
or infrastructure change.

## Milestones

1. Document ingestion and metadata-preserving chunking
2. Vector retrieval with a small evaluation dataset
3. BM25 hybrid retrieval and optional reranking
4. Grounded answer generation, citations, and refusal behavior
5. Bilingual, safety, latency, cost, and reliability evaluation (implemented; run locally to record
   machine-specific QA and load results)
