"""
clinical_pipeline.py
====================

A controlled, production-style orchestration of a clinical-reasoning workflow:

    Conversation
         |
    Clinical Question
         |
   +-----------+
   |           |
   v           v
Patient facts  Evidence search
   |           |
   +-----+-----+
         |
     Synthesizer
         |
   Structured result

Design notes
------------
- Every "tool" is a plain function that could later be swapped for a real
  API call, DB query, or retrieval-augmented LLM call. They're implemented
  here as mocks with simulated latency and simulated failure modes so the
  resilience layer (timeout / retry / malformed-response handling /
  logging) has something real to exercise.
- `with_resilience` is a decorator, not a copy-pasted try/except in every
  function. It gives every tool: a hard timeout, exponential-backoff
  retries, and (optionally) output-schema validation that treats a
  malformed response the same as a transient failure -> retry.
- The orchestrator (`run_pipeline`) encodes the DAG in the diagram above.
  The two independent branches (patient facts, evidence search) run
  concurrently; within the evidence branch, guideline search and
  literature search also run concurrently.
- A `request_id` is generated once per pipeline run and threaded through
  every log line, which is the minimum you need to debug a production
  system with concurrent branches.

This is a reference implementation / scaffold. Swap the mock bodies of
the five tool functions for real integrations (EHR API, a guideline
database, a literature search index, an LLM call) without touching the
orchestrator or the resilience layer.
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field, asdict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
# One request_id ties together every log line from a single pipeline run,
# even though two branches execute on different threads.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clinical_pipeline")


def log(request_id: str, msg: str, level: int = logging.INFO) -> None:
    logger.log(level, f"[req={request_id}] {msg}")


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class ToolTimeoutError(Exception):
    """Raised when a tool exceeds its allotted time budget."""


class MalformedResponseError(Exception):
    """Raised when a tool returns output that fails schema validation."""


class PipelineError(Exception):
    """Raised when the pipeline cannot produce a result after all resilience
    strategies (retry, fallback) are exhausted."""


# --------------------------------------------------------------------------
# Resilience layer: timeout + retry + malformed-response handling
# --------------------------------------------------------------------------

def with_resilience(
    timeout_s: float = 4.0,
    max_retries: int = 2,
    backoff_base: float = 0.4,
    validator: Optional[Callable[[Any], None]] = None,
):
    """Decorator adding timeout, retry-with-backoff, and schema validation
    to a tool function.

    - timeout_s: hard wall-clock budget per attempt. Enforced with a
      worker thread so a hung call can't block the pipeline forever.
    - max_retries: number of *additional* attempts after the first.
    - backoff_base: base seconds for exponential backoff (attempt N waits
      backoff_base * 2**(N-1), plus small jitter).
    - validator: optional callable(result) -> None that raises
      MalformedResponseError if `result` doesn't match the expected shape.
      A malformed response is treated exactly like a transient failure
      and triggers a retry, since it usually *is* transient (a flaky
      upstream LLM/API returning half-formed JSON) rather than a
      permanent one.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            tool_name = fn.__name__
            request_id = kwargs.pop("_request_id", "n/a")
            last_exc: Optional[Exception] = None

            for attempt in range(1, max_retries + 2):  # first try + retries
                start = time.time()
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(fn, *args, **kwargs)
                        result = future.result(timeout=timeout_s)

                    if validator is not None:
                        validator(result)  # raises MalformedResponseError

                    elapsed = time.time() - start
                    log(request_id, f"{tool_name} OK on attempt {attempt} ({elapsed:.2f}s)")
                    return result

                except FuturesTimeoutError:
                    elapsed = time.time() - start
                    last_exc = ToolTimeoutError(
                        f"{tool_name} exceeded {timeout_s}s timeout"
                    )
                    log(
                        request_id,
                        f"{tool_name} TIMEOUT on attempt {attempt} (>{timeout_s}s)",
                        logging.WARNING,
                    )

                except MalformedResponseError as e:
                    last_exc = e
                    log(
                        request_id,
                        f"{tool_name} MALFORMED OUTPUT on attempt {attempt}: {e}",
                        logging.WARNING,
                    )

                except Exception as e:
                    last_exc = e
                    log(
                        request_id,
                        f"{tool_name} ERROR on attempt {attempt}: {e!r}",
                        logging.WARNING,
                    )

                if attempt <= max_retries:
                    sleep_s = backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.15)
                    log(request_id, f"{tool_name} retrying in {sleep_s:.2f}s...")
                    time.sleep(sleep_s)

            log(
                request_id,
                f"{tool_name} FAILED after {max_retries + 1} attempts: {last_exc}",
                logging.ERROR,
            )
            raise PipelineError(f"{tool_name} failed: {last_exc}") from last_exc

        return wrapper

    return decorator


def _require_keys(result: Any, keys: List[str], tool_name: str) -> None:
    """Generic validator factory body: confirm `result` is a dict with all
    required keys and non-null values."""
    if not isinstance(result, dict):
        raise MalformedResponseError(
            f"{tool_name} expected dict, got {type(result).__name__}"
        )
    missing = [k for k in keys if k not in result or result[k] is None]
    if missing:
        raise MalformedResponseError(f"{tool_name} missing required keys: {missing}")


# --------------------------------------------------------------------------
# Mock "flakiness" knobs — remove these once real integrations are wired in.
# They exist purely so the resilience layer above has something to catch.
# --------------------------------------------------------------------------

random.seed(1)  # deterministic demo run; delete for real randomness

FAILURE_RATE = 0.20       # chance a call raises a transient error
SLOW_RATE = 0.12          # chance a call is slow enough to trip the timeout
MALFORMED_RATE = 0.15     # chance a call returns bad output shape


def _maybe_misbehave(tool_name: str, timeout_budget: float) -> None:
    r = random.random()
    if r < SLOW_RATE:
        time.sleep(timeout_budget + 1.0)  # guarantees a timeout upstream
    elif r < SLOW_RATE + FAILURE_RATE:
        raise ConnectionError(f"{tool_name}: upstream connection reset")
    time.sleep(random.uniform(0.05, 0.3))  # normal latency


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def _validate_patient_context(result: Any) -> None:
    _require_keys(
        result,
        ["chief_complaint", "clinical_question", "age", "sex", "symptoms"],
        "extract_patient_context",
    )


@with_resilience(timeout_s=2.5, max_retries=2, validator=_validate_patient_context)
def extract_patient_context(conversation: str) -> Dict[str, Any]:
    """Parse the raw conversation into structured patient facts and a
    normalized clinical question. In production this would be an
    LLM call with a strict output schema (or a fine-tuned NER model)."""
    _maybe_misbehave("extract_patient_context", timeout_budget=2.5)

    if random.random() < MALFORMED_RATE:
        return {"chief_complaint": "chest pain"}  # missing required keys

    return {
        "chief_complaint": "shortness of breath and chest tightness",
        "clinical_question": "What is the recommended workup for new-onset "
        "exertional dyspnea in a 58-year-old with hypertension?",
        "age": 58,
        "sex": "female",
        "symptoms": ["dyspnea on exertion", "chest tightness", "fatigue"],
    }


def _validate_history(result: Any) -> None:
    _require_keys(result, ["conditions", "medications", "allergies"], "retrieve_patient_history")


@with_resilience(timeout_s=2.0, max_retries=2, validator=_validate_history)
def retrieve_patient_history(patient_id: str) -> Dict[str, Any]:
    """Look up structured history from the EHR (mocked)."""
    _maybe_misbehave("retrieve_patient_history", timeout_budget=2.0)

    if random.random() < MALFORMED_RATE:
        return {"conditions": ["hypertension"]}  # missing keys

    return {
        "patient_id": patient_id,
        "conditions": ["hypertension", "type 2 diabetes"],
        "medications": ["lisinopril", "metformin"],
        "allergies": ["penicillin"],
    }


def _validate_search_result(result: Any) -> None:
    _require_keys(result, ["source", "results"], "search_result")
    if not isinstance(result["results"], list):
        raise MalformedResponseError("'results' must be a list")


@with_resilience(timeout_s=3.0, max_retries=2, validator=_validate_search_result)
def search_guidelines(clinical_question: str) -> Dict[str, Any]:
    """Search a clinical guideline database (mocked)."""
    _maybe_misbehave("search_guidelines", timeout_budget=3.0)

    if random.random() < MALFORMED_RATE:
        return {"source": "guidelines"}  # missing 'results'

    return {
        "source": "guidelines",
        "results": [
            {
                "title": "Evaluation of Dyspnea in Adults",
                "organization": "ACP",
                "year": 2023,
                "recommendation": "Obtain ECG, chest x-ray, BNP, and consider "
                "echocardiography before pursuing advanced cardiac testing.",
            },
        ],
    }


@with_resilience(timeout_s=3.5, max_retries=2, validator=_validate_search_result)
def search_literature(clinical_question: str) -> Dict[str, Any]:
    """Search biomedical literature (mocked)."""
    _maybe_misbehave("search_literature", timeout_budget=3.5)

    if random.random() < MALFORMED_RATE:
        return {"source": "literature", "results": "not-a-list"}  # wrong type

    return {
        "source": "literature",
        "results": [
            {
                "title": "Diagnostic yield of BNP testing in undifferentiated dyspnea",
                "journal": "JAMA Internal Medicine",
                "year": 2021,
                "summary": "BNP-guided evaluation reduced unnecessary imaging "
                "without missing clinically significant heart failure cases.",
            },
        ],
    }


def _validate_summary(result: Any) -> None:
    _require_keys(
        result,
        ["summary", "recommendations", "evidence_cited", "confidence", "caveats"],
        "generate_summary",
    )
    if not (0.0 <= result["confidence"] <= 1.0):
        raise MalformedResponseError("confidence must be between 0 and 1")


@with_resilience(timeout_s=4.0, max_retries=2, validator=_validate_summary)
def generate_summary(
    patient_facts: Dict[str, Any], evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """Synthesize patient facts + evidence into a structured, cited result.
    In production this is the final LLM call, constrained to a JSON schema
    (e.g. via tool-use / structured outputs), not free text."""
    _maybe_misbehave("generate_summary", timeout_budget=4.0)

    if random.random() < MALFORMED_RATE:
        return {"summary": "incomplete"}  # missing required keys

    guideline_titles = [r["title"] for r in evidence.get("guidelines", {}).get("results", [])]
    literature_titles = [r["title"] for r in evidence.get("literature", {}).get("results", [])]

    return {
        "summary": (
            f"For a {patient_facts.get('age')}-year-old {patient_facts.get('sex')} with "
            f"{patient_facts.get('chief_complaint')} and a history of "
            f"{', '.join(patient_facts.get('history', {}).get('conditions', []))}, "
            "guideline-directed workup and a supporting literature base "
            "suggest a staged, symptom-driven cardiac evaluation."
        ),
        "recommendations": [
            "Obtain ECG, chest x-ray, and BNP as first-line studies.",
            "Reserve echocardiography and stress testing for abnormal "
            "initial results or high pre-test probability.",
            "Review medication list for agents that could contribute to "
            "dyspnea (e.g., beta-blocker dosing).",
        ],
        "evidence_cited": guideline_titles + literature_titles,
        "confidence": 0.78,
        "caveats": [
            "This is a decision-support draft, not a diagnosis.",
            "Requires clinician review before acting on any recommendation.",
        ],
    }


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

@dataclass
class PipelineResult:
    request_id: str
    status: str  # "success" | "partial" | "failed"
    clinical_question: Optional[str] = None
    patient_facts: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    structured_result: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _get_patient_facts(conversation: str, patient_id: str, request_id: str) -> Dict[str, Any]:
    """Branch: extract_patient_context -> retrieve_patient_history (sequential,
    since history lookup doesn't need the context, but conceptually belongs
    to the same 'patient facts' branch)."""
    ctx = extract_patient_context(conversation, _request_id=request_id)
    history = retrieve_patient_history(patient_id, _request_id=request_id)
    return {**ctx, "history": history}


def _get_evidence(clinical_question: str, request_id: str) -> Dict[str, Any]:
    """Branch: search_guidelines and search_literature run concurrently."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        g_future = pool.submit(search_guidelines, clinical_question, _request_id=request_id)
        l_future = pool.submit(search_literature, clinical_question, _request_id=request_id)
        guidelines = g_future.result()
        literature = l_future.result()
    return {"guidelines": guidelines, "literature": literature}


def run_pipeline(conversation: str, patient_id: str = "P-1024") -> PipelineResult:
    """Orchestrate the full DAG:

        conversation -> clinical question
                     -> [patient facts]      (concurrent)
                     -> [evidence search]     (concurrent)
                     -> synthesizer -> structured result
    """
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    log(request_id, "=== pipeline start ===")

    errors: List[str] = []
    patient_facts: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    structured: Optional[Dict[str, Any]] = None
    clinical_question: Optional[str] = None

    try:
        # Step 1: derive the clinical question up front so both downstream
        # branches (which run concurrently) have what they need.
        seed_ctx = extract_patient_context(conversation, _request_id=request_id)
        clinical_question = seed_ctx["clinical_question"]
        log(request_id, f"clinical question: {clinical_question}")

        # Step 2: run the two independent branches concurrently.
        with ThreadPoolExecutor(max_workers=2) as pool:
            facts_future = pool.submit(
                _get_patient_facts, conversation, patient_id, request_id
            )
            evidence_future = pool.submit(_get_evidence, clinical_question, request_id)

            try:
                patient_facts = facts_future.result()
            except Exception as e:
                errors.append(f"patient_facts branch failed: {e}")
                log(request_id, f"patient_facts branch failed: {e}", logging.ERROR)

            try:
                evidence = evidence_future.result()
            except Exception as e:
                errors.append(f"evidence branch failed: {e}")
                log(request_id, f"evidence branch failed: {e}", logging.ERROR)

        if patient_facts is None or evidence is None:
            # Can't synthesize without both inputs -> partial failure.
            status = "failed"
            log(request_id, "aborting: missing required inputs for synthesis", logging.ERROR)
        else:
            # Step 3: synthesize.
            structured = generate_summary(patient_facts, evidence, _request_id=request_id)
            status = "success"

    except Exception as e:
        errors.append(str(e))
        status = "failed"
        log(request_id, f"pipeline aborted: {e}", logging.ERROR)

    duration = time.time() - start
    log(request_id, f"=== pipeline end: status={status} ({duration:.2f}s) ===")

    return PipelineResult(
        request_id=request_id,
        status=status,
        clinical_question=clinical_question,
        patient_facts=patient_facts,
        evidence=evidence,
        structured_result=structured,
        errors=errors,
        duration_s=round(duration, 3),
    )


# --------------------------------------------------------------------------
# Demo entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    sample_conversation = (
        "Patient reports increasing shortness of breath over the past two "
        "weeks, worse with exertion, plus intermittent chest tightness. "
        "Known history of hypertension and type 2 diabetes."
    )

    result = run_pipeline(sample_conversation, patient_id="P-1024")

    print("\n--- PIPELINE RESULT (JSON) ---")
    print(result.to_json())
