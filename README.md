# ClinicalReasoningPipeline

A reference implementation of a clinical-reasoning workflow built as a **tool
pipeline with an explicit resilience layer**, rather than a single LLM call.

A free-text clinical conversation goes in; a structured, evidence-cited,
programmatically-inspectable result comes out — or a clean, labeled failure.

```text
Conversation
     │
     ▼
extract_patient_context()  ──────►  Clinical Question
     │                                     │
     ▼                                     ▼
┌──────────────────┐          ┌──────────────────────────┐
│  Patient facts   │          │     Evidence search      │
│                  │          │  search_guidelines()  ─┐ │   ← both branches
│ retrieve_        │          │  search_literature()  ─┤ │     run concurrently
│ patient_history()│          └────────────┬───────────┘ │
└────────┬─────────┘                       │             │
         └───────────────┬─────────────────┘─────────────┘
                         ▼
                 generate_summary()        ← single join point
                         ▼
              PipelineResult (JSON)
```

Full rationale, alternatives considered, and production follow-ups live in
[design_doc.md](design_doc.md). This README covers what the code actually does
today, how to run it, and where it currently diverges from the doc.

> **Not a medical device.** Tool bodies are mocked, recommendations are
> canned, and nothing here is clinically validated. Every result carries a
> "requires clinician review" caveat by design.

## Quick start

No dependencies — stdlib only, Python 3.8+ (developed and tested on 3.13).

```bash
git clone https://github.com/xinyuhwang/ClinicalReasoningPipeline.git
cd ClinicalReasoningPipeline

python3 clinical_pipeline.py          # run the demo pipeline
python3 -m unittest discover -v       # run the test suite
```

The demo takes roughly 6–12 seconds. That variance is intentional: the mock
tools inject latency, connection errors, and malformed responses so the
resilience layer has something real to handle. A typical run:

```text
15:57:44.140 | INFO    | [req=cf6bdffe] === pipeline start ===
15:57:44.141 | WARNING | [req=cf6bdffe] extract_patient_context ERROR on attempt 1: ConnectionError(...)
15:57:44.141 | INFO    | [req=cf6bdffe] extract_patient_context retrying in 0.53s...
15:57:44.791 | INFO    | [req=cf6bdffe] extract_patient_context OK on attempt 2 (0.12s)
15:57:44.872 | INFO    | [req=cf6bdffe] search_literature OK on attempt 1 (0.08s)
15:57:49.034 | WARNING | [req=cf6bdffe] search_guidelines TIMEOUT on attempt 1 (>3.0s)
15:57:49.641 | WARNING | [req=cf6bdffe] search_guidelines MALFORMED OUTPUT on attempt 2: missing required keys: ['results']
15:57:50.676 | INFO    | [req=cf6bdffe] search_guidelines OK on attempt 3 (0.15s)
15:57:50.737 | INFO    | [req=cf6bdffe] === pipeline end: status=success (6.60s) ===
```

Every line is tagged with the same 8-character `request_id`, which is what
makes a run readable when two branches are logging from different threads.

## Using it

```python
from clinical_pipeline import run_pipeline

result = run_pipeline(
    "Patient reports increasing shortness of breath over the past two weeks, "
    "worse with exertion, plus intermittent chest tightness. Known history of "
    "hypertension and type 2 diabetes.",
    patient_id="P-1024",
)

if result.status == "success":
    print(result.structured_result["recommendations"])
else:
    print(result.errors)      # which branch failed, and which tool

print(result.to_json())       # full result, JSON-serializable
```

`run_pipeline` never raises for tool-level failures — it always returns a
`PipelineResult` and you branch on `status`.

### `PipelineResult`

| Field | Type | Notes |
|---|---|---|
| `request_id` | `str` | 8-char UUID fragment, on every log line for the run |
| `status` | `str` | `"success"` \| `"failed"` (`"partial"` is reserved, see below) |
| `clinical_question` | `str \| None` | Normalized question derived from the conversation |
| `patient_facts` | `dict \| None` | Extracted context, plus EHR history under `history` |
| `evidence` | `dict \| None` | `{"guidelines": {...}, "literature": {...}}` |
| `structured_result` | `dict \| None` | The synthesized answer; `None` unless `status == "success"` |
| `errors` | `list[str]` | Which branch and which tool failed |
| `duration_s` | `float` | Wall-clock time for the run |

Branch data is retained even on failure — if evidence search dies but the
patient-facts branch succeeded, `patient_facts` is still populated.

## Tools

Five single-responsibility functions, each independently testable, retryable,
and swappable. All five are mocked; swapping in real integrations should not
require touching the orchestrator or the resilience decorator.

| Tool | Timeout | Retries | Validates |
|---|---|---|---|
| `extract_patient_context(conversation)` | 2.5s | 2 | `chief_complaint`, `clinical_question`, `age`, `sex`, `symptoms` |
| `retrieve_patient_history(patient_id)` | 2.0s | 2 | `conditions`, `medications`, `allergies` |
| `search_guidelines(question)` | 3.0s | 2 | `source`, `results` (must be a list) |
| `search_literature(question)` | 3.5s | 2 | `source`, `results` (must be a list) |
| `generate_summary(facts, evidence)` | 4.0s | 2 | 5 keys + `confidence` in `[0,1]` |

Budgets are per-tool, not global, because a history lookup and a literature
search have genuinely different latency profiles.

## Resilience layer

One decorator, `with_resilience`, gives every tool the same four guarantees so
the logic isn't copy-pasted into five function bodies:

- **Timeout** — each attempt runs in a worker thread with a wall-clock budget.
  (See finding #1 below: this currently labels slow calls without actually
  bounding them.)
- **Retry** — exponential backoff, `backoff_base * 2^(attempt-1)` plus up to
  0.15s jitter, for `max_retries` additional attempts.
- **Malformed-response handling** — an optional `validator(result)` runs after
  every successful call and raises `MalformedResponseError`, which is retried
  exactly like a transient network error. Half-formed JSON from a flaky
  upstream LLM usually *is* transient.
- **Logging** — every attempt is logged with tool name, attempt number,
  elapsed time, and the shared `request_id`.

Exhausting all attempts raises `PipelineError`, chained (`__cause__`) to the
underlying `ToolTimeoutError`, `MalformedResponseError`, or transport error.

### Failure semantics

- **`success`** — both branches returned valid data and synthesis completed.
- **`failed`** — a required branch exhausted its retries. The pipeline aborts
  *before* calling `generate_summary`, because synthesizing from incomplete
  input produces an unlabeled, misleadingly confident answer.
- **`partial`** — defined in the schema but never produced today. Intended for
  the case where, say, literature search fails but guidelines succeed and
  synthesis proceeds with a caveat. See design doc §9.1.

## Tests

```bash
python3 -m unittest discover -v          # all 69 tests, ~17s
python3 -m unittest test_clinical_pipeline.TestWithResilience -v
```

[test_clinical_pipeline.py](test_clinical_pipeline.py) is stdlib `unittest`
(no pytest needed) and mirrors the design doc's structure:

| Class | Covers |
|---|---|
| `TestRequireKeys`, `TestValidators` | Doc §5 — every output schema, per tool |
| `TestWithResilience` | Doc §6 — retry counts, backoff curve, timeout, validator wiring, `request_id` threading, thread independence |
| `TestMockTools` | Doc §5 — tool outputs conform to their own validators; malformed and transient-error paths |
| `TestBranches` | Doc §4 — branch composition and *measured* concurrency |
| `TestOrchestrator` | Doc §7 — success, per-branch failure, synthesis failure, abort-before-synthesis, JSON round-trip |
| `TestDesignDocConformance` | Places the code and the doc disagree |

Tests neutralize the mock flakiness knobs (`FAILURE_RATE`, `SLOW_RATE`,
`MALFORMED_RATE`) so they're deterministic, and drive each failure mode
explicitly instead of waiting for the dice to land on it.

The five tests in `TestDesignDocConformance` are marked
`@unittest.expectedFailure`. Each asserts the behavior the design doc
*promises* and fails against the current code, so the suite reports
`OK (expected failures=5)` while keeping the gaps visible. Fixing one flips it
to "unexpected success" and turns the suite red, prompting removal of the
marker.

## Findings

What the tests turned up, ordered by impact.

**1. The timeout does not bound wall-clock time.**
[clinical_pipeline.py:131-133](clinical_pipeline.py#L131-L133) —
`with ThreadPoolExecutor(...)` calls `shutdown(wait=True)` on exit, so after
`future.result(timeout=N)` raises, leaving the `with` block blocks until the
hung worker actually finishes. The timeout relabels the error; it never
reclaims the time. Design doc §6 claims "a hang can't block the pipeline" —
it can. A tool that hangs for 60s costs 60s per attempt, and with
`max_retries=2` that's 180s against a nominal 4s budget. Visible in the demo
log above: `search_guidelines TIMEOUT (>3.0s)` is emitted 4.0s after the
preceding line. Fix: hold the executor open past the timeout (don't use a
`with` block, or `shutdown(wait=False)`), and prefer cancellable I/O with its
own client-level timeout over thread-based interruption.

**2. `extract_patient_context` runs twice per pipeline run.**
[clinical_pipeline.py:429](clinical_pipeline.py#L429) derives the clinical
question, then [clinical_pipeline.py:393](clinical_pipeline.py#L393) extracts
it all over again inside `_get_patient_facts`. The doc's architecture diagram
shows one extraction feeding both branches. In production this is the most
expensive step (an LLM call), so the duplication doubles cost and latency on
that branch and doubles the chance it fails. Fix: pass `seed_ctx` into
`_get_patient_facts` instead of re-extracting.

**3. The reported clinical question may not be the one the branch used.**
A direct consequence of #2: `PipelineResult.clinical_question` and
`patient_facts["clinical_question"]` come from two separate calls. A real,
non-deterministic LLM can return different text each time, so the question
surfaced to the caller isn't guaranteed to be the one the evidence search ran
against. Resolved by the same fix.

**4. `random.seed(1)` does not make the demo deterministic.**
[clinical_pipeline.py:203](clinical_pipeline.py#L203) is commented
"deterministic demo run", but the tools draw from the process-wide `random`
module while running on concurrent threads, so interleaving decides which
branch gets which draw. Reseeding and re-running yields both `success` (~5.7s)
and `failed` (~11.3s) from the same seed — measured 3 distinct outcomes across
8 reseeded runs. Fix: give each branch its own `random.Random(seed)` instance,
or drop the comment's promise.

**5. `_validate_history` doesn't check `patient_id`.**
[clinical_pipeline.py:252](clinical_pipeline.py#L252) validates
conditions/medications/allergies but not `patient_id`, which doc §5.2 lists in
the schema. An EHR response missing its record identifier passes validation,
so the pipeline can't confirm whose history it merged into `patient_facts` —
worth tightening before this touches real records.

**6. A non-numeric `confidence` raises `TypeError`, not `MalformedResponseError`.**
[clinical_pipeline.py:327](clinical_pipeline.py#L327) compares against
`0.0`/`1.0` with no type check, so `confidence: "high"` raises `TypeError`.
The decorator's catch-all still retries it, so behavior is correct — but it's
logged as a generic `ERROR` instead of `MALFORMED OUTPUT`, losing exactly the
signal doc §6 says to preserve.

### What holds up

Verified by the passing tests: retry counts and the exponential backoff curve
are correct; no backoff is wasted after the final attempt; malformed responses
retry like transient errors; the validator is skipped when a call raises;
`_request_id` is consumed by the decorator and never leaks into tool
signatures; both concurrency claims are real (measured, not asserted);
synthesis is genuinely skipped when a branch fails; successful branch data
survives a failed run; concurrent pipeline runs don't interfere; and results
are fully JSON-serializable.

## Known limitations

Carried from design doc §9, all still open: no caching of repeated guideline or
literature queries; no circuit breaker (retries have no awareness of a broadly
failing dependency); no auth, PII handling, or audit logging on `patient_id`;
all-or-nothing synthesis rather than a labeled degraded answer; and mocked tool
bodies throughout.

Before anything here touches real patient data, §8's audit-logging requirement
— which record was accessed, by which request, when — needs to be implemented.

## License

[Apache 2.0](LICENSE)
