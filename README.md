# Bilingual RAG MVP

A multi-turn retrieval-augmented generation service for a bilingual Chinese/English internal
knowledge base. It ingests PDF, DOCX, TXT, and Markdown documents, retrieves evidence with vector
or hybrid search, and produces cited answers with a free local Ollama model.

## High-level technology stack

```text
Documents
   -> Python ingestion pipeline
      -> PyMuPDF / python-docx: document parsing
      -> Tesseract: scanned-PDF OCR
      -> Multilingual MiniLM: embeddings
      -> Qdrant: vector and metadata storage
   -> FastAPI service
      -> vector search or vector + BM25 hybrid search
      -> optional lexical reranking
      -> PII, prompt-injection, and grounding controls
      -> TTL/LRU caching and structured JSONL tracing
      -> Ollama + Qwen3: grounded bilingual generation
   -> JSON response with answer, citations, tokens, and timings
```

- **Language and API:** Python 3.11+, FastAPI, and Uvicorn.
- **Document processing:** PyMuPDF for PDFs, python-docx for Word documents, plain-text readers,
  and Tesseract for scanned Chinese/English PDFs.
- **Retrieval:** `paraphrase-multilingual-MiniLM-L12-v2` embeddings, embedded Qdrant vector
  storage, bilingual BM25, reciprocal-rank fusion, and an optional lexical reranker.
- **Generation:** Ollama with the pinned local `qwen3:4b-instruct-2507-q4_K_M` model.
- **Runtime controls:** bounded in-process conversation history, TTL/LRU cache, prompt-injection
  detection, PII redaction, grounded-answer validation, refusals, and structured tracing.
- **Quality tooling:** Pytest, Ruff, labeled retrieval/QA datasets, concurrent load testing, and
  JSON, CSV, and Markdown evaluation reports.

The MVP runs locally without a paid model API or separate Qdrant server. Production would normally
replace embedded Qdrant and process-local state with shared services.

## 1. Run and test the complete flow locally

These instructions assume you forked or cloned the repository and are running commands from its
root directory. No machine-specific paths are required.

### Prerequisites

- Git.
- Python 3.11 or newer; Python 3.12 is recommended for the commands below.
- Ollama plus the pinned Qwen3 model for local answer generation.
- Tesseract plus `eng` and `chi_sim` language data for the complete scanned-PDF flow.
- Approximately 3 GB of free disk space for Python dependencies, the embedding model, and the
  quantized generation model.

Embedded Qdrant is included through the Python dependency; you do not need to install or run a
separate database for this MVP. Python packages do not provide the Ollama or Tesseract executables;
they must be installed at the operating-system level.

### macOS fresh-clone quick start

The commands below cover the complete path used to verify a clean clone. Run one block at a time so
any missing system dependency is clear.

#### 1. Install Homebrew if `brew` is unavailable

Check first:

```bash
brew --version
```

If that prints `command not found`, install Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Make the newly installed command available in the current Terminal on either Apple Silicon or an
Intel Mac:

```bash
if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  eval "$(/usr/local/bin/brew shellenv)"
fi
brew --version
```

Homebrew is only an installation method; it is not used by the RAG application at runtime.

#### 2. Install Python and OCR dependencies

```bash
brew install python@3.12
brew install tesseract tesseract-lang
```

Verify the exact dependencies before cloning:

```bash
python3.12 --version
tesseract --version
tesseract --list-langs | grep -E 'eng|chi_sim'
```

The final command must print both `eng` and `chi_sim`. The `pytesseract` Python package is only a
wrapper and cannot perform OCR when the `tesseract` executable is missing.

#### 3. Clone and install the project

```bash
git clone https://github.com/yrfzh75/rag-system.git
cd rag-system
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
```

The `python --version` output must be Python 3.11 or newer. A virtual environment keeps the Python
version used to create it; activating an old Python 3.9 environment does not upgrade it.

On Windows PowerShell, create and activate the environment with an installed Python 3.11+ version:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 4. Install and start the local model

Install Ollama from [ollama.com](https://ollama.com), then download the pinned model:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

If Ollama is not already running as a background application, start it in a separate Terminal:

```bash
ollama serve
```

Confirm that it is available:

```bash
ollama list
curl -s http://127.0.0.1:11434/api/tags
```

#### 5. Run a preflight check

From the repository root:

```bash
python --version
python -c 'import rag_mvp; print("Python package: OK")'
tesseract --list-langs | grep -E 'eng|chi_sim'
ollama list | grep 'qwen3:4b-instruct-2507-q4_K_M'
```

Do not continue until Python is 3.11+, both OCR languages are listed, and the Qwen3 model appears.

#### 6. Ingest all sample documents, including the scanned PDF

```bash
INGEST_OCR_ENABLED=true \
INGEST_OCR_LANGUAGES=eng+chi_sim \
rag-ingest ./documents
```

For the included corpus, the final event should report:

```text
"ingested_files": 5
"failed_files": 0
"chunks_written": 8
"native_pages": 5
"ocr_pages": 1
```

Run the same command a second time. Identical counts confirm idempotent replacement rather than
duplicate chunks.

#### 7. Run the complete automated validation

```bash
./scripts/evaluate.sh
```

The final load-test object should contain `"cold_cache_verified": true` and `"passes": true`.

### Native-text-only ingestion

If you intentionally do not want to install Tesseract, you can validate the four native-text
documents with:

```bash
INGEST_OCR_ENABLED=false .venv/bin/rag-ingest ./documents
```

The scanned PDF will be reported as a failure because it contains no embedded text. This is
expected, but the complete evaluation will not pass its OCR retrieval case until that PDF has been
ingested with OCR enabled.

The first run downloads the multilingual embedding model to `data/models/`. Qdrant persists the
index under `data/qdrant/`. Because embedded Qdrant permits one owning process, stop the API before
running ingestion or use Qdrant server in a multi-process deployment.

### What the automated validation runs

The validation command is:

```bash
./scripts/evaluate.sh
```

The script automatically uses `.venv/bin/python`, starts a temporary cache-disabled API on port
8010, and then:

1. runs all unit tests;
2. runs Ruff static checks;
3. compares vector-only, hybrid, and hybrid+rerank retrieval;
4. evaluates bilingual QA quality and refusal behavior;
5. sends five distinct requests concurrently with a cold cache;
6. creates an operations CSV; and
7. stops the temporary API.

A successful run ends with `passes: true`. Important thresholds are faithfulness `>= 0.85`, answer
compliance `>= 0.90`, style consistency `>= 0.85`, refusal appropriateness `>= 0.90`, and at least
90% of concurrent QA requests within 10 seconds.

Results are written to:

- `evals/reports/retrieval_latest/`
- `evals/reports/qa_latest/`
- `artifacts/operations.csv`
- `artifacts/qa_validation_events.jsonl`

### Common setup problems

| Message or symptom | Cause | Fix |
|---|---|---|
| `repository ... does not exist` | `git clone` received `username/repository` instead of a URL, or the repository is private. | Use the complete HTTPS URL above; authenticate with GitHub first for a private repository. |
| `requires a different Python: 3.9.6 not in >=3.11` | `.venv` was created with macOS system Python 3.9. | Deactivate it, remove `.venv`, and recreate it with `python3.12 -m venv .venv`. |
| `brew: command not found` | Homebrew is not installed or is not in the current shell's `PATH`. | Install Homebrew and run the `shellenv` block above. |
| `tesseract: command not found` | Installing `pytesseract` did not install the system executable. | Run `brew install tesseract tesseract-lang`. |
| OCR reports missing `chi_sim` | Simplified Chinese language data is absent. | Install `tesseract-lang`, then recheck `tesseract --list-langs`. |
| Scanned PDF says `No text could be extracted` | OCR was disabled for a PDF with no native text. | Set `INGEST_OCR_ENABLED=true` after installing Tesseract. |
| Ollama connection fails | Ollama is not running or the model is missing. | Run `ollama serve` and `ollama pull qwen3:4b-instruct-2507-q4_K_M`. |
| The prompt shows `((.venv) )` | The environment label was added twice to the shell prompt. | Cosmetic only; confirm the active interpreter with `python --version`. |

### Test the API manually

Start the service:

```bash
source .venv/bin/activate
uvicorn rag_mvp.main:app --reload
```

Check health:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Test retrieval without calling the LLM:

```bash
curl -s http://127.0.0.1:8000/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many weeks of parental leave are available?","top_k":3}' \
  | python3 -m json.tool
```

Test grounded English and Chinese answers:

```bash
curl -s http://127.0.0.1:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many weeks of parental leave are available?","top_k":3}' \
  | python3 -m json.tool

curl -s http://127.0.0.1:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"query":"员工有多少周育儿假？","top_k":3}' \
  | python3 -m json.tool
```

Test a multi-turn conversation by reusing `session_id`:

```bash
curl -s http://127.0.0.1:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"query":"How many weeks of parental leave are available?","session_id":"demo-1"}'

curl -s http://127.0.0.1:8000/qa \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who is eligible for it?","session_id":"demo-1"}'
```

API documentation is available at `http://127.0.0.1:8000/docs`.

## 2. Flow design and key decisions

The system has two main flows: an offline ingestion flow and an online query flow. Keeping them
separate lets documents be processed once while user queries remain fast.

### Flow A: document ingestion

```text
Documents
   -> file discovery
   -> PDF/DOCX/TXT/Markdown parsing
   -> page-level OCR fallback when native text is insufficient
   -> bilingual-aware chunking
   -> multilingual embeddings
   -> Qdrant chunks, vectors, and metadata
```

Key design decisions:

- **Native extraction before OCR:** native text is faster and more accurate; OCR is used only for
  text-poor PDF pages to control latency and extraction noise.
- **Chinese/English-aware chunking:** Chinese text cannot rely on whitespace boundaries, so the
  chunker uses punctuation-aware boundaries and overlap to preserve context in both languages.
- **One multilingual embedding model:** the same model embeds Chinese and English documents and
  queries, enabling cross-language retrieval without maintaining two indexes.
- **Stable document and chunk IDs:** identifiers are derived deterministically, so rerunning
  ingestion replaces a document's chunks instead of creating duplicates.
- **Metadata preservation:** source path, filename, page, language, extraction method, content hash,
  and chunk position support citations, debugging, and future incremental synchronization.
- **Embedded Qdrant for the MVP:** it requires no extra service and makes local setup simple. A
  Qdrant server is the production path for multiple API/worker processes and larger collections.

MVP ingestion is a manually triggered batch command because the immediate goal is validating
parsing, OCR, chunking, embedding, and storage. In production, use source-system events, a durable
queue, and parallel workers. Target 95% of ordinary document changes becoming searchable within 15
minutes; urgent compliance updates should use a priority path. A separate nightly or weekly
reconciliation detects missed updates and deletions without rescanning millions of files every 15
minutes.

### Flow B: retrieval and grounded QA

```text
User query + optional session ID
   -> prompt-injection check and PII redaction
   -> recent redacted conversation context
   -> vector-only or vector+BM25 hybrid retrieval
   -> optional reranking and confidence threshold
   -> early refusal when context is missing
   -> local Ollama generation from numbered passages
   -> citation and context-support validation
   -> answer and citation PII redaction
   -> response cache, structured event, and operations metrics
```

Key design decisions:

- **Retrieval can be tested independently:** `/retrieval/search` isolates retrieval quality from
  model variability, cost, and generation latency.
- **Configurable strategies:** `RETRIEVAL_MODE` selects vector-only or hybrid search, while
  `RETRIEVAL_RERANKER_ENABLED` changes reranking without a code deployment.
- **Hybrid retrieval:** multilingual dense search handles semantic similarity; bilingual BM25
  improves exact terminology and identifiers. Reciprocal-rank fusion combines their rankings.
- **Early refusal:** out-of-scope or low-confidence queries stop before generation, reducing
  hallucination risk and unnecessary model work.
- **Grounding after generation:** every answer needs valid numbered citations and must meet
  `QA_GROUNDING_MIN_SUPPORT`; otherwise the service returns `ungrounded_generation` guidance.
- **Free local generation:** pinned `qwen3:4b-instruct-2507-q4_K_M` gives `$0` provider cost per
  1,000 calls, keeps internal context local, supports Chinese and English, and runs on a modern Mac.
  Its trade-offs are local compute usage and lower quality than larger hosted models.
- **Bounded local state:** the MVP retains three redacted conversation turns and uses a TTL/LRU
  cache. Production should use encrypted shared storage with explicit retention and deletion rules.
- **Privacy-aware observability:** logs contain hashes, request/trace/span IDs, configuration,
  tokens, decisions, grounding scores, and timings—not raw prompts, answers, or retrieved passages.
- **Thread-safe initialization:** the retrieval and QA singletons are initialized under locks so
  concurrent requests do not open multiple embedded-Qdrant clients.

Important configuration is documented in `.env.example`, including retrieval mode, reranking,
model, cache, history, log destination, and grounding threshold.

## 3. Milestones

1. **Initial service foundation:** Python package, FastAPI application, health endpoint, tests, and
   local configuration.
2. **Bilingual ingestion:** native parsing, scanned-PDF OCR fallback, bilingual chunking,
   multilingual embeddings, metadata, and idempotent Qdrant storage.
3. **Independent retrieval:** vector search endpoint and labeled bilingual retrieval dataset.
4. **Hybrid retrieval and reranking:** BM25 fusion, configurable reranking, and three-way
   quantitative comparison.
5. **Grounded generation:** local Ollama integration, citations, refusals, token usage, and
   retrieval/generation latency.
6. **Service hardening:** multi-turn context, prompt-injection checks, PII redaction, caching,
   grounding validation, structured tracing, and thread-safe concurrency.
7. **Reproducible evaluation:** one-command QA/retrieval/load validation, operations reporting,
   before/after comparison, log dictionary, and issue-diagnosis evidence.

## 4. Evidence that the MVP meets the requirements

The numbers below come from the committed local reports. They demonstrate this sample corpus and
machine, not production capacity. Rerun `./scripts/evaluate.sh` after any model, prompt, corpus,
configuration, or infrastructure change.

### Global constraints

| Constraint | Implementation and evidence | Status |
|---|---|---|
| 90% of QA requests complete within 10 seconds | Five distinct cold-cache requests at concurrency 5: 100% within 10 seconds; p95 6.76 seconds. | Pass |
| At least five concurrent requests on one instance | Cache-disabled five-client load test; thread-safe dependency initialization prevents duplicate Qdrant clients. | Pass |
| Token-cost estimate per 1,000 calls | Local Ollama provider cost is `$0`; token usage is recorded. A hosted replacement can apply provider input/output rates to the same counters. | Pass |
| Model selection and trade-offs | README documents bilingual capability, privacy, `$0` API cost, local latency/compute, size, reproducibility, and hosted-model quality trade-offs. | Pass |
| RAG faithfulness >= 0.85 | Deterministic non-stopword answer-token support in cited passages: `0.870`. | Pass |
| Context precision >= 0.70 | Hybrid and hybrid+rerank both measured `1.00`. | Pass |
| Answer compliance >= 80% | Hardened evaluation measured `1.00`; it also exceeds the advanced 90% target. | Pass |
| Style consistency >= 80% | Same-language response, citations, and no thinking trace measured `1.00`. | Pass |
| Refusal appropriateness >= 80% | Labeled answerable, out-of-scope, and injection cases measured `1.00`. | Pass |
| Structured logging and tracing | JSONL includes request/trace/span IDs, retrieval settings, model, refusal, cache, PII, grounding, tokens, and stage timings. | Pass |
| Minimal injection defense and basic PII handling | CN/EN override patterns refuse before retrieval; email, phone, and US SSN patterns redact queries, answers, citations, history, and logs. | Pass for MVP |
| Answers grounded to retrieved context | Prompt constraints plus post-generation citation-ID and lexical-support checks; unsupported output is refused. | Pass for MVP |

### Functional requirements

| Requirement | Evidence | Status |
|---|---|---|
| Vector-only and hybrid retrieval | `RETRIEVAL_MODE=vector_only` or `hybrid`. | Pass |
| Reranker controlled without code changes | `RETRIEVAL_RERANKER_ENABLED=true/false`. | Pass |
| Low-confidence, out-of-scope, and safety refusals with guidance | `no_relevant_context`, `safety_policy`, and `ungrounded_generation` responses provide next-step guidance. | Pass |
| PII redaction in outputs and logs | Redaction covers generated answers and returned citation text; operational logs exclude raw content. | Pass for MVP |
| Operations report | CSV contains p50/p95 latency, token usage, cache-hit rate, refusal rate, and answer-compliance rate. | Pass |

### Non-functional and quantitative requirements

| Requirement | Evidence | Status |
|---|---|---|
| Compare vector-only, hybrid, and hybrid+rerank | The 12-case report records Hit@3, MRR, context precision, refusal accuracy, and p50/p95 latency for all three. | Pass |
| Evolvability | Retriever and generator interfaces plus environment configuration isolate strategy, model, cache, logging, and metric changes. | Pass |
| Answer compliance >= 90% | `1.00`. | Pass |
| Refusal appropriateness >= 90% | `1.00`. | Pass |
| Style consistency >= 0.85 | `1.00`. | Pass |
| Two diagnosed issues with >=10% post-fix improvement | Retrieval context precision improved `0.90 -> 1.00` (11.1% relative); pre-generation injection refusal improved `0.00 -> 1.00`. | Pass |

### Evaluation definitions and artifacts

- **Hit@k:** fraction of answerable queries with an expected source among the first `k` results.
- **MRR:** mean reciprocal rank of the first expected source.
- **Context precision:** average precision at ranks containing expected sources across answerable
  queries.
- **Faithfulness:** fraction of non-stopword answer tokens also present in cited passages. This is
  reproducible but should be supplemented with claim-level human or semantic-entailment evaluation
  before production.
- **Answer compliance:** correct refusal decision or, for answered cases, required facts plus a
  citation.
- **Style consistency:** answer language matches the question, citations are numbered, and no
  thinking trace is exposed.
- **Refusal appropriateness:** refusal decision matches the answerable/out-of-scope/safety label.

Committed evidence:

- `evals/reports/retrieval_latest/report.md`
- `evals/reports/qa_latest/report.md`
- `evals/reports/qa_comparison.md`
- `evals/reports/issue_diagnosis.md`
- `evals/reports/issue_diagnosis_metrics.json`
- `docs/log_field_dictionary.md`
- `docs/sample_logs.jsonl`

MVP limitations are deliberate: embedded Qdrant is single-process, conversation/cache state is
in-memory, PII detection is regex-based, the grounding check is lexical rather than semantic, and
the evaluation datasets are small. These boundaries should be replaced or expanded before a
production launch.

## Potential improvements

- **Scale storage and shared state:** replace embedded Qdrant with Qdrant server and move the
  in-process cache and conversation history to an encrypted shared store such as Redis. This
  enables multiple API instances, persistence, backups, and horizontal scaling.
- **Automate incremental ingestion:** process document create, update, and delete events through a
  durable queue instead of manually rescanning the directory. Add priority handling for urgent
  compliance changes and periodic reconciliation for missed events.
- **Strengthen grounding and security:** replace lexical grounding with claim-level semantic
  entailment, expand PII detection beyond regular expressions, and detect indirect prompt
  injection inside retrieved documents.
- **Expand evaluation coverage:** add more bilingual, multi-turn, OCR, adversarial, and
  domain-specific cases. Run sustained cold-cache load tests to measure throughput, failure rate,
  memory use, and model saturation on production-like hardware.
- **Improve production operations:** add containerized deployment, CI/CD, readiness checks,
  retries, circuit breakers, and OpenTelemetry export to a metrics and tracing platform. These
  controls make model or database failures visible and allow graceful degradation.
