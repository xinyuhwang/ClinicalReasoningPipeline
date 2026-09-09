# Clinical Reasoning Pipeline — Design Doc

**Status:** Draft / reference implementation
**Owner:** (fill in)
**Related code:** `clinical_pipeline.py`

## 1. Problem

A clinical question buried in a free-text conversation needs to be answered
with a structured, evidence-backed result: what does the patient's own
history say, and what does external evidence (guidelines, literature) say,
combined into one synthesized recommendation.

Doing this as a single LLM call is fast to prototype but breaks down in
production: no separation of concerns, no retry semantics, no way to tell
*which* step failed, and no defense against a step silently returning
garbage that gets treated as ground truth. This doc describes a pipeline
that decomposes the problem into independently-callable tools, wired
together by an orchestrator with explicit failure handling.

## 2. Goals

- Decompose "answer a clinical question" into single-responsibility tools
  that can be tested, retried, and swapped independently.
- Run independent work concurrently instead of serially.
- Treat every external call (EHR, search index, LLM) as unreliable by
  default: it can be slow, it can error, it can return the wrong shape.
- Make failures legible — a request_id ties every log line across
  concurrent branches to one pipeline run.
- Fail predictably: partial data should produce either a degraded-but-labeled
  result or a clean failure, never a silently wrong answer.

## 3. Non-goals

- Real integrations (EHR API, literature index, guideline DB) — tool bodies
  are mocked and meant to be swapped in without touching the orchestrator.
- Clinical validation of the actual recommendations — this is an
  engineering scaffold, not a decision-support product. Every structured
  result carries an explicit "requires clinician review" caveat.
- Auth, PII handling, audit-log persistence — noted in Section 8 as
  required before any real deployment.

## 4. Architecture

```
Conversation
     │
     ▼
extract_patient_context()  ──────► Clinical Question
     │                                    │
     ▼                                    ▼
┌─────────────────────┐      ┌──────────────────────────┐
│   Patient facts      │      │      Evidence search      │
│                      │      │                            │
│ retrieve_patient_    │      │  search_guidelines()  ─┐  │
│ history()            │      │  search_literature()  ─┼─►│ (run concurrently)
└──────────┬───────────┘      └────────────┬──────────────┘
           │                                │
           └───────────────┬────────────────┘
                            ▼
                    generate_summary()
                            │
                            ▼
                  Structured result (JSON)
```

Two levels of concurrency:

1. **Top level** — the "patient facts" branch and the "evidence search"
   branch run in parallel; neither depends on the other's output.
2. **Within evidence search** — `search_guidelines()` and
   `search_literature()` also run in parallel, since they're independent
   queries against different sources.

The synthesizer (`generate_summary`) is the single join point and only
runs once both branches have returned.

## 5. Components

### 5.1 `extract_patient_context(conversation) -> dict`
Parses free text into structured facts and a normalized clinical question.
Production version: an LLM call constrained to a JSON schema, or a
fine-tuned extraction model.

**Output schema:**
```json
{
  "chief_complaint": "str",
  "clinical_question": "str",
  "age": "int",
  "sex": "str",
  "symptoms": ["str", "..."]
}
```

### 5.2 `retrieve_patient_history(patient_id) -> dict`
Looks up structured history from the EHR.

**Output schema:**
```json
{
  "patient_id": "str",
  "conditions": ["str"],
  "medications": ["str"],
  "allergies": ["str"]
}
```

### 5.3 `search_guidelines(clinical_question) -> dict`
Queries a clinical guideline database.

### 5.4 `search_literature(clinical_question) -> dict`
Queries a biomedical literature index.

Both search tools share one output schema:
```json
{ "source": "guidelines | literature", "results": [ { ... } ] }
```
`results` must always be a list — a common malformed-response failure mode
in the mock is returning a string or dict instead.

### 5.5 `generate_summary(patient_facts, evidence) -> dict`
The synthesizer. In production, a final constrained LLM call (structured
output / tool-use) rather than free text, so downstream consumers get a
guaranteed shape.

**Output schema:**
```json
{
  "summary": "str",
  "recommendations": ["str"],
  "evidence_cited": ["str"],
  "confidence": "float [0,1]",
  "caveats": ["str"]
}
```

## 6. Resilience strategy

Every tool is wrapped by a single decorator, `with_resilience`, so the
failure-handling logic lives in one place rather than being copy-pasted
into each tool.

| Concern | Mechanism |
|---|---|
| **Timeout** | Call runs in a worker thread; `future.result(timeout=N)` enforces a hard per-attempt wall-clock budget. A hang can't block the pipeline. |
| **Retry** | Exponential backoff: `backoff_base * 2^(attempt-1)` + jitter, up to `max_retries` additional attempts. |
| **Malformed response handling** | An optional `validator(result)` runs after every successful call. It checks required keys, types, and value ranges (e.g. `confidence` in `[0,1]`). A validation failure raises `MalformedResponseError`, which is retried exactly like a transient network error — most malformed output from an upstream LLM/API is transient, not permanent. |
| **Logging** | Every attempt (success, timeout, error, malformed, retry-scheduled) is logged with the tool name, attempt number, and elapsed time, tagged with a `request_id` shared across the whole pipeline run — including both concurrent branches. |

Design choice: timeout, retry count, and backoff are per-tool, not global.
`extract_patient_context` (2.5s) and `search_literature` (3.5s) have
different budgets because they have different expected latencies — a
single global timeout would either be too tight for slow tools or too
loose for fast ones.

## 7. Failure semantics

The orchestrator distinguishes three outcomes:

- **`success`** — both branches returned valid data and synthesis
  completed.
- **`failed`** — a required branch (patient facts or evidence) exhausted
  all retries. The pipeline aborts before calling `generate_summary`,
  since synthesizing from an incomplete input would produce an
  unlabeled, misleadingly confident result. `PipelineResult.errors`
  records which branch and which tool failed.
- **`partial`** — reserved for a future refinement (see Section 9):
  currently unused, but the schema supports it for cases where, e.g.,
  literature search fails but guidelines succeed, and synthesis proceeds
  with a caveat instead of aborting.

Every `PipelineResult` is a structured object (not raw text), so a caller
can branch on `status` programmatically instead of parsing prose.

## 8. Observability

- `request_id`: an 8-character UUID generated once per `run_pipeline()`
  call, threaded through every tool invocation and log line. This is the
  minimum needed to reconstruct a single request's timeline when two
  branches are logging from different threads concurrently.
- Log format includes timestamp, level, and `request_id`-tagged message.
  In production this should emit structured (JSON) logs to a log
  aggregator rather than formatted text to stdout, and `duration_s` /
  per-tool latency should be exported as metrics (histogram per tool
  name) rather than only logged.
- Not yet implemented, required before production: audit logging of
  which patient record was accessed, by which request, at what time
  (compliance requirement for anything touching real patient data).

## 9. Known limitations / follow-ups

1. **All-or-nothing synthesis.** If evidence search fails entirely but
   patient facts succeed, the whole request fails rather than degrading
   to a patient-facts-only summary with a caveat. Worth revisiting once
   there's a product decision on what a "degraded" answer should look
   like to an end user.
2. **No caching.** Repeated identical guideline/literature queries hit
   the backend every time. A cache keyed on the normalized clinical
   question would cut latency and cost.
3. **No circuit breaker.** Retries currently happen per-call with no
   awareness of whether a downstream service is broadly down. At scale,
   a circuit breaker would stop hammering a failing dependency across
   many concurrent requests.
4. **No PII/auth handling.** `patient_id` is passed as a plain string
   with no access control. Any real deployment needs auth on the EHR
   lookup and encryption in transit/at rest for anything derived from
   patient data.
5. **Mocked tool bodies.** The five tool functions simulate latency and
   failure for demonstration. Swapping in real integrations should not
   require changes to the orchestrator or the resilience decorator.

## 10. Alternatives considered

- **Single LLM call with a long prompt.** Rejected: no retry granularity
  (a malformed JSON blob means retrying the entire expensive call), no
  parallelism, and failure of one "step" is indistinguishable from
  failure of the whole thing.
- **Sequential pipeline (no concurrency).** Simpler, but patient-facts
  lookup and evidence search are genuinely independent — running them
  sequentially adds latency for no correctness benefit.
- **Global timeout instead of per-tool.** Simpler to configure, but a
  guideline search and a history lookup don't have the same expected
  latency profile; a shared budget either over-constrains fast tools or
  under-protects slow ones.
