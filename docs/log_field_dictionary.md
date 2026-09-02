# QA log field dictionary

Each completed or refused request emits one `qa.request_completed` JSON object to stdout and to
`QA_LOG_PATH`. Raw queries and answers are deliberately excluded; `query_sha256` permits grouping
identical requests without leaking their contents.

| Field | Type | Meaning |
|---|---|---|
| `event` | string | Stable event name, currently `qa.request_completed`. |
| `request_id` | UUID | Unique identifier for one API request. |
| `trace_id` | string | End-to-end trace identifier returned in the API response. |
| `retrieval_span_id` | string | Identifier for the retrieval stage. |
| `generation_span_id` | string/null | Identifier for generation; null when it was skipped. |
| `session_id` | string/null | Client-provided multi-turn conversation identifier. |
| `query_sha256` | string | One-way hash of the original query; never the raw query. |
| `retrieval_mode` | string | Active `vector_only` or `hybrid` strategy. |
| `reranker_enabled` | boolean | Whether configured reranking was active. |
| `model` | string/null | Generation model; null when refused before generation. |
| `refused` | boolean | Whether the service declined to answer. |
| `refusal_reason` | string/null | `no_relevant_context` or `safety_policy`. |
| `cache_hit` | boolean | Whether an answer came from the TTL cache. |
| `pii_redacted` | boolean | Whether PII was removed from the query or answer. |
| `grounding_score` | number/null | Context-support score; null before generation. |
| `source_count` | integer | Number of citations returned. |
| `input_tokens`, `output_tokens`, `total_tokens` | integer | Model token usage. |
| `retrieval_ms`, `generation_ms`, `total_ms` | number | Stage and end-to-end latency. |

Do not add raw prompts, retrieved chunks, answers, email addresses, phone numbers, or government
identifiers to operational logs.
