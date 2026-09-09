"""
test_clinical_pipeline.py
=========================

Test suite for `clinical_pipeline.py`, organized around the claims made in
`design_doc.md`. Stdlib `unittest` only — the module under test has no
third-party dependencies and neither does this suite.

    python3 -m unittest test_clinical_pipeline -v

Layout
------
- TestRequireKeys / TestValidators .... Section 5 output schemas
- TestWithResilience ................. Section 6 resilience strategy
- TestMockTools ...................... Section 5 tool bodies
- TestBranches ....................... Section 4 concurrency
- TestOrchestrator ................... Section 7 failure semantics
- TestDesignDocConformance ........... places the code and the doc disagree

The tests in TestDesignDocConformance marked `@unittest.expectedFailure`
assert the behavior the design doc promises. They currently fail, which is
the point: each one pins a real divergence between doc and code. When a
divergence is fixed the test flips to "unexpected success" and the suite
turns red, prompting removal of the marker.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import clinical_pipeline as cp


# --------------------------------------------------------------------------
# Test harness helpers
# --------------------------------------------------------------------------

def setUpModule() -> None:
    """The module under test logs to stdout on import. Silence it so test
    output stays readable; individual tests re-enable it when asserting."""
    logging.disable(logging.CRITICAL)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


class QuietMocksMixin:
    """Zeroes the mock flakiness knobs so tool bodies behave deterministically.

    The knobs are module-level globals read at call time (`clinical_pipeline`
    lines 205-207), so overwriting them is enough — no need to touch the
    tool bodies.
    """

    def setUp(self) -> None:
        super().setUp()
        saved = (cp.FAILURE_RATE, cp.SLOW_RATE, cp.MALFORMED_RATE)
        cp.FAILURE_RATE = cp.SLOW_RATE = cp.MALFORMED_RATE = 0.0

        def restore() -> None:
            cp.FAILURE_RATE, cp.SLOW_RATE, cp.MALFORMED_RATE = saved

        self.addCleanup(restore)

    def force_malformed(self) -> None:
        """Make every tool take its malformed-output branch."""
        cp.MALFORMED_RATE = 1.0

    def no_backoff_sleep(self) -> mock.MagicMock:
        """Skip real retry backoff so failure paths run fast."""
        patcher = mock.patch("clinical_pipeline.time.sleep")
        sleeper = patcher.start()
        self.addCleanup(patcher.stop)
        return sleeper


# --------------------------------------------------------------------------
# Section 5 — output schema validation
# --------------------------------------------------------------------------

class TestRequireKeys(unittest.TestCase):
    """`_require_keys` is the shared body behind every tool validator."""

    def test_accepts_dict_with_all_keys(self):
        cp._require_keys({"a": 1, "b": 2}, ["a", "b"], "tool")  # no raise

    def test_ignores_extra_keys(self):
        cp._require_keys({"a": 1, "extra": 2}, ["a"], "tool")  # no raise

    def test_rejects_non_dict(self):
        for bad in ([], "string", 42, None, ("a", "b")):
            with self.subTest(value=bad):
                with self.assertRaises(cp.MalformedResponseError) as ctx:
                    cp._require_keys(bad, ["a"], "tool")
                self.assertIn(type(bad).__name__, str(ctx.exception))

    def test_rejects_missing_key(self):
        with self.assertRaises(cp.MalformedResponseError) as ctx:
            cp._require_keys({"a": 1}, ["a", "b"], "tool")
        self.assertIn("b", str(ctx.exception))

    def test_rejects_none_valued_key(self):
        """A key present but null counts as missing — a null from an upstream
        LLM is just as unusable as an absent field."""
        with self.assertRaises(cp.MalformedResponseError):
            cp._require_keys({"a": 1, "b": None}, ["a", "b"], "tool")

    def test_error_message_names_the_tool(self):
        with self.assertRaises(cp.MalformedResponseError) as ctx:
            cp._require_keys({}, ["a"], "search_guidelines")
        self.assertIn("search_guidelines", str(ctx.exception))

    def test_reports_all_missing_keys_at_once(self):
        with self.assertRaises(cp.MalformedResponseError) as ctx:
            cp._require_keys({}, ["a", "b", "c"], "tool")
        msg = str(ctx.exception)
        for key in ("a", "b", "c"):
            self.assertIn(key, msg)


class TestValidators(unittest.TestCase):
    """One test per schema in design doc Section 5."""

    # 5.1 extract_patient_context
    def test_patient_context_accepts_full_schema(self):
        cp._validate_patient_context({
            "chief_complaint": "cc",
            "clinical_question": "q",
            "age": 58,
            "sex": "female",
            "symptoms": ["dyspnea"],
        })

    def test_patient_context_rejects_partial(self):
        with self.assertRaises(cp.MalformedResponseError):
            cp._validate_patient_context({"chief_complaint": "chest pain"})

    # 5.2 retrieve_patient_history
    def test_history_accepts_full_schema(self):
        cp._validate_history({
            "patient_id": "P-1",
            "conditions": [],
            "medications": [],
            "allergies": [],
        })

    def test_history_rejects_partial(self):
        with self.assertRaises(cp.MalformedResponseError):
            cp._validate_history({"conditions": ["hypertension"]})

    # 5.3 / 5.4 search tools share one schema
    def test_search_result_accepts_full_schema(self):
        cp._validate_search_result({"source": "guidelines", "results": []})

    def test_search_result_rejects_missing_results(self):
        with self.assertRaises(cp.MalformedResponseError):
            cp._validate_search_result({"source": "guidelines"})

    def test_search_result_rejects_non_list_results(self):
        """Doc 5.4: "`results` must always be a list"."""
        for bad in ("not-a-list", {"a": 1}, 5):
            with self.subTest(value=bad):
                with self.assertRaises(cp.MalformedResponseError) as ctx:
                    cp._validate_search_result({"source": "s", "results": bad})
                self.assertIn("list", str(ctx.exception))

    # 5.5 generate_summary
    def _summary(self, **overrides):
        base = {
            "summary": "s",
            "recommendations": [],
            "evidence_cited": [],
            "confidence": 0.5,
            "caveats": [],
        }
        base.update(overrides)
        return base

    def test_summary_accepts_full_schema(self):
        cp._validate_summary(self._summary())

    def test_summary_rejects_partial(self):
        with self.assertRaises(cp.MalformedResponseError):
            cp._validate_summary({"summary": "incomplete"})

    def test_summary_accepts_confidence_at_bounds(self):
        for value in (0.0, 1.0):
            with self.subTest(confidence=value):
                cp._validate_summary(self._summary(confidence=value))

    def test_summary_rejects_confidence_out_of_range(self):
        """Doc 5.5: confidence is a float in [0, 1]."""
        for value in (-0.1, 1.1, 42):
            with self.subTest(confidence=value):
                with self.assertRaises(cp.MalformedResponseError):
                    cp._validate_summary(self._summary(confidence=value))


# --------------------------------------------------------------------------
# Section 6 — resilience strategy
# --------------------------------------------------------------------------

class TestWithResilience(unittest.TestCase):
    """`with_resilience` is the single place failure handling lives."""

    def test_returns_result_without_retrying_on_success(self):
        calls = []

        @cp.with_resilience(timeout_s=1.0, max_retries=2, backoff_base=0.0)
        def tool():
            calls.append(1)
            return {"ok": True}

        self.assertEqual(tool(), {"ok": True})
        self.assertEqual(len(calls), 1)

    def test_forwards_args_and_kwargs(self):
        @cp.with_resilience(timeout_s=1.0, max_retries=0, backoff_base=0.0)
        def tool(a, b, c=None):
            return (a, b, c)

        self.assertEqual(tool(1, b=2, c=3), (1, 2, 3))

    def test_preserves_function_metadata(self):
        @cp.with_resilience()
        def documented_tool():
            """docstring"""

        self.assertEqual(documented_tool.__name__, "documented_tool")
        self.assertEqual(documented_tool.__doc__, "docstring")

    def test_request_id_is_consumed_not_forwarded(self):
        """The orchestrator threads `_request_id` through every call; the
        tool body must never see it."""
        seen = {}

        @cp.with_resilience(timeout_s=1.0, max_retries=0, backoff_base=0.0)
        def tool(**kwargs):
            seen.update(kwargs)
            return {}

        tool(real_arg=1, _request_id="abc12345")
        self.assertEqual(seen, {"real_arg": 1})

    def test_retries_transient_error_then_succeeds(self):
        calls = []

        @cp.with_resilience(timeout_s=1.0, max_retries=2, backoff_base=0.0)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("upstream connection reset")
            return {"ok": True}

        self.assertEqual(flaky(), {"ok": True})
        self.assertEqual(len(calls), 3)

    def test_raises_pipeline_error_after_exhausting_retries(self):
        calls = []

        @cp.with_resilience(timeout_s=1.0, max_retries=2, backoff_base=0.0)
        def always_fails():
            calls.append(1)
            raise ConnectionError("boom")

        with self.assertRaises(cp.PipelineError) as ctx:
            always_fails()

        self.assertEqual(len(calls), 3, "first attempt + max_retries")
        self.assertIn("always_fails", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, ConnectionError)

    def test_max_retries_zero_means_one_attempt(self):
        calls = []

        @cp.with_resilience(timeout_s=1.0, max_retries=0, backoff_base=0.0)
        def once():
            calls.append(1)
            raise ValueError("nope")

        with self.assertRaises(cp.PipelineError):
            once()
        self.assertEqual(len(calls), 1)

    def test_malformed_response_is_retried_like_a_transient_error(self):
        """Doc 6: "A validation failure raises MalformedResponseError, which
        is retried exactly like a transient network error"."""
        calls = []

        def validator(result):
            if result.get("shape") == "bad":
                raise cp.MalformedResponseError("bad shape")

        @cp.with_resilience(
            timeout_s=1.0, max_retries=2, backoff_base=0.0, validator=validator
        )
        def flaky_shape():
            calls.append(1)
            return {"shape": "bad" if len(calls) < 2 else "good"}

        self.assertEqual(flaky_shape(), {"shape": "good"})
        self.assertEqual(len(calls), 2)

    def test_persistently_malformed_response_fails_the_call(self):
        def validator(result):
            raise cp.MalformedResponseError("always bad")

        @cp.with_resilience(
            timeout_s=1.0, max_retries=1, backoff_base=0.0, validator=validator
        )
        def bad():
            return {}

        with self.assertRaises(cp.PipelineError) as ctx:
            bad()
        self.assertIsInstance(ctx.exception.__cause__, cp.MalformedResponseError)

    def test_validator_not_called_when_the_tool_raises(self):
        validator = mock.Mock()

        @cp.with_resilience(
            timeout_s=1.0, max_retries=0, backoff_base=0.0, validator=validator
        )
        def raises():
            raise ConnectionError("boom")

        with self.assertRaises(cp.PipelineError):
            raises()
        validator.assert_not_called()

    def test_validator_receives_the_tool_result(self):
        validator = mock.Mock()

        @cp.with_resilience(
            timeout_s=1.0, max_retries=0, backoff_base=0.0, validator=validator
        )
        def tool():
            return {"payload": 1}

        tool()
        validator.assert_called_once_with({"payload": 1})

    def test_timeout_surfaces_as_pipeline_error_wrapping_tool_timeout(self):
        @cp.with_resilience(timeout_s=0.05, max_retries=0, backoff_base=0.0)
        def slow():
            time.sleep(0.4)
            return {}

        with self.assertRaises(cp.PipelineError) as ctx:
            slow()
        self.assertIsInstance(ctx.exception.__cause__, cp.ToolTimeoutError)
        self.assertIn("0.05s timeout", str(ctx.exception))

    def test_backoff_grows_exponentially_with_jitter(self):
        """Doc 6: attempt N waits `backoff_base * 2**(N-1)` plus jitter."""
        with mock.patch("clinical_pipeline.time.sleep") as sleeper:

            @cp.with_resilience(timeout_s=1.0, max_retries=3, backoff_base=1.0)
            def always_fails():
                raise ConnectionError("boom")

            with self.assertRaises(cp.PipelineError):
                always_fails()

        waits = [call.args[0] for call in sleeper.call_args_list]
        self.assertEqual(len(waits), 3, "one backoff between each pair of attempts")
        for i, wait in enumerate(waits):
            expected = 1.0 * (2 ** i)
            self.assertGreaterEqual(wait, expected)
            self.assertLessEqual(wait, expected + 0.15, "jitter is capped at 0.15s")

    def test_no_backoff_after_the_final_attempt(self):
        """The last failure should raise immediately, not sleep first."""
        with mock.patch("clinical_pipeline.time.sleep") as sleeper:

            @cp.with_resilience(timeout_s=1.0, max_retries=1, backoff_base=1.0)
            def always_fails():
                raise ConnectionError("boom")

            with self.assertRaises(cp.PipelineError):
                always_fails()

        self.assertEqual(sleeper.call_count, 1, "2 attempts -> 1 backoff")

    def test_every_attempt_is_logged_with_the_request_id(self):
        """Doc 8: a request_id ties every log line to one pipeline run."""
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, logging.CRITICAL)

        @cp.with_resilience(timeout_s=1.0, max_retries=1, backoff_base=0.0)
        def flaky():
            raise ConnectionError("boom")

        with self.assertLogs("clinical_pipeline", level="DEBUG") as captured:
            with self.assertRaises(cp.PipelineError):
                flaky(_request_id="req12345")

        self.assertTrue(captured.output)
        for line in captured.output:
            self.assertIn("[req=req12345]", line)
        joined = "\n".join(captured.output)
        self.assertIn("attempt 1", joined)
        self.assertIn("attempt 2", joined)
        self.assertIn("FAILED after 2 attempts", joined)

    def test_logs_fall_back_to_na_when_no_request_id_is_given(self):
        logging.disable(logging.NOTSET)
        self.addCleanup(logging.disable, logging.CRITICAL)

        @cp.with_resilience(timeout_s=1.0, max_retries=0, backoff_base=0.0)
        def tool():
            return {}

        with self.assertLogs("clinical_pipeline", level="DEBUG") as captured:
            tool()
        self.assertIn("[req=n/a]", captured.output[0])

    def test_concurrent_calls_are_independent(self):
        """Two branches call decorated tools from different threads; the
        decorator holds no cross-call state."""
        @cp.with_resilience(timeout_s=2.0, max_retries=0, backoff_base=0.0)
        def echo(value):
            time.sleep(0.05)
            return {"value": value}

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda i: echo(i), range(8)))

        self.assertEqual([r["value"] for r in results], list(range(8)))


# --------------------------------------------------------------------------
# Section 5 — mock tool bodies
# --------------------------------------------------------------------------

class TestMockTools(QuietMocksMixin, unittest.TestCase):

    def test_extract_patient_context_matches_doc_schema(self):
        result = cp.extract_patient_context("conversation text")
        cp._validate_patient_context(result)
        self.assertIsInstance(result["age"], int)
        self.assertIsInstance(result["symptoms"], list)
        self.assertTrue(result["clinical_question"])

    def test_retrieve_patient_history_echoes_the_patient_id(self):
        result = cp.retrieve_patient_history("P-9999")
        cp._validate_history(result)
        self.assertEqual(result["patient_id"], "P-9999")
        self.assertIn("penicillin", result["allergies"])

    def test_search_guidelines_matches_doc_schema(self):
        result = cp.search_guidelines("question")
        cp._validate_search_result(result)
        self.assertEqual(result["source"], "guidelines")
        self.assertTrue(all("title" in r for r in result["results"]))

    def test_search_literature_matches_doc_schema(self):
        result = cp.search_literature("question")
        cp._validate_search_result(result)
        self.assertEqual(result["source"], "literature")
        self.assertTrue(all("title" in r for r in result["results"]))

    def test_generate_summary_cites_both_evidence_sources(self):
        facts = cp.extract_patient_context("conversation")
        facts["history"] = cp.retrieve_patient_history("P-1")
        evidence = {
            "guidelines": cp.search_guidelines("q"),
            "literature": cp.search_literature("q"),
        }

        result = cp.generate_summary(facts, evidence)

        cp._validate_summary(result)
        self.assertEqual(
            result["evidence_cited"],
            ["Evaluation of Dyspnea in Adults",
             "Diagnostic yield of BNP testing in undifferentiated dyspnea"],
        )
        self.assertIn("58-year-old female", result["summary"])
        self.assertIn("hypertension, type 2 diabetes", result["summary"])

    def test_generate_summary_always_carries_a_clinician_review_caveat(self):
        """Doc 3 (non-goals): "Every structured result carries an explicit
        'requires clinician review' caveat"."""
        result = cp.generate_summary({}, {})
        joined = " ".join(result["caveats"]).lower()
        self.assertIn("clinician review", joined)
        self.assertIn("not a diagnosis", joined)

    def test_generate_summary_tolerates_missing_evidence_keys(self):
        result = cp.generate_summary({}, {})
        self.assertEqual(result["evidence_cited"], [])

    def test_malformed_output_exhausts_retries_and_raises(self):
        """With MALFORMED_RATE pinned to 1.0 every attempt returns a bad
        shape, so each tool should fail cleanly rather than return garbage."""
        self.force_malformed()
        self.no_backoff_sleep()

        cases = [
            (cp.extract_patient_context, ("conversation",)),
            (cp.retrieve_patient_history, ("P-1",)),
            (cp.search_guidelines, ("q",)),
            (cp.search_literature, ("q",)),
            (cp.generate_summary, ({}, {})),
        ]
        for tool, args in cases:
            with self.subTest(tool=tool.__name__):
                with self.assertRaises(cp.PipelineError) as ctx:
                    tool(*args)
                self.assertIsInstance(
                    ctx.exception.__cause__, cp.MalformedResponseError
                )

    def test_transient_connection_error_is_retried_and_recovers(self):
        """FAILURE_RATE drives `_maybe_misbehave` to raise ConnectionError;
        the decorator should absorb it as long as a later attempt succeeds."""
        self.no_backoff_sleep()
        attempts = []
        real = cp._maybe_misbehave

        def fail_once(tool_name, timeout_budget):
            attempts.append(tool_name)
            if len(attempts) == 1:
                raise ConnectionError(f"{tool_name}: upstream connection reset")

        with mock.patch.object(cp, "_maybe_misbehave", fail_once):
            result = cp.search_guidelines("q")

        self.assertEqual(len(attempts), 2)
        cp._validate_search_result(result)
        self.assertIs(cp._maybe_misbehave, real)


# --------------------------------------------------------------------------
# Section 4 — branch composition and concurrency
# --------------------------------------------------------------------------

class TestBranches(QuietMocksMixin, unittest.TestCase):

    def test_patient_facts_branch_merges_context_and_history(self):
        facts = cp._get_patient_facts("conversation", "P-77", "req12345")

        self.assertEqual(facts["age"], 58)
        self.assertIn("history", facts)
        self.assertEqual(facts["history"]["patient_id"], "P-77")

    def test_patient_facts_branch_propagates_tool_failure(self):
        with mock.patch.object(
            cp, "retrieve_patient_history",
            side_effect=cp.PipelineError("retrieve_patient_history failed"),
        ):
            with self.assertRaises(cp.PipelineError):
                cp._get_patient_facts("conversation", "P-1", "req12345")

    def test_evidence_branch_returns_both_sources(self):
        evidence = cp._get_evidence("question", "req12345")

        self.assertEqual(set(evidence), {"guidelines", "literature"})
        self.assertEqual(evidence["guidelines"]["source"], "guidelines")
        self.assertEqual(evidence["literature"]["source"], "literature")

    def test_evidence_searches_run_concurrently(self):
        """Doc 4: "search_guidelines() and search_literature() also run in
        parallel"."""
        delay = 0.4

        def slow_search(source):
            def _search(question, _request_id=None):
                time.sleep(delay)
                return {"source": source, "results": []}
            return _search

        with mock.patch.object(cp, "search_guidelines", slow_search("guidelines")), \
             mock.patch.object(cp, "search_literature", slow_search("literature")):
            start = time.time()
            cp._get_evidence("question", "req12345")
            elapsed = time.time() - start

        self.assertLess(
            elapsed, delay * 1.8,
            f"expected overlap (~{delay}s), got {elapsed:.2f}s — searches ran serially",
        )

    def test_evidence_branch_propagates_search_failure(self):
        with mock.patch.object(
            cp, "search_literature",
            side_effect=cp.PipelineError("search_literature failed"),
        ):
            with self.assertRaises(cp.PipelineError):
                cp._get_evidence("question", "req12345")


# --------------------------------------------------------------------------
# Section 7 — orchestrator failure semantics
# --------------------------------------------------------------------------

class TestOrchestrator(QuietMocksMixin, unittest.TestCase):

    def test_happy_path_returns_success_with_every_field_populated(self):
        result = cp.run_pipeline("conversation", patient_id="P-1024")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.errors, [])
        self.assertIsNotNone(result.clinical_question)
        self.assertIsNotNone(result.patient_facts)
        self.assertIsNotNone(result.evidence)
        self.assertIsNotNone(result.structured_result)
        self.assertGreater(result.duration_s, 0)

    def test_request_id_is_an_8_char_uuid_fragment(self):
        """Doc 8: "an 8-character UUID generated once per run_pipeline() call"."""
        result = cp.run_pipeline("conversation")
        self.assertRegex(result.request_id, r"^[0-9a-f]{8}$")

    def test_each_run_gets_a_distinct_request_id(self):
        ids = {cp.run_pipeline("conversation").request_id for _ in range(3)}
        self.assertEqual(len(ids), 3)

    def test_patient_id_defaults_and_reaches_the_ehr_lookup(self):
        result = cp.run_pipeline("conversation")
        self.assertEqual(result.patient_facts["history"]["patient_id"], "P-1024")

    def test_result_serializes_to_json(self):
        """Doc 7: callers branch on a structured object, not prose."""
        result = cp.run_pipeline("conversation")
        payload = json.loads(result.to_json())

        self.assertEqual(
            set(payload),
            {"request_id", "status", "clinical_question", "patient_facts",
             "evidence", "structured_result", "errors", "duration_s"},
        )
        self.assertEqual(payload["status"], "success")

    def test_synthesis_is_skipped_when_the_patient_branch_fails(self):
        """Doc 7: "The pipeline aborts before calling generate_summary"."""
        synth = mock.Mock()
        with mock.patch.object(
            cp, "retrieve_patient_history",
            side_effect=cp.PipelineError("retrieve_patient_history failed: boom"),
        ), mock.patch.object(cp, "generate_summary", synth):
            result = cp.run_pipeline("conversation")

        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.structured_result)
        synth.assert_not_called()
        self.assertTrue(any("patient_facts branch failed" in e for e in result.errors))

    def test_evidence_branch_failure_names_the_failing_branch(self):
        with mock.patch.object(
            cp, "search_guidelines",
            side_effect=cp.PipelineError("search_guidelines failed: boom"),
        ):
            result = cp.run_pipeline("conversation")

        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.structured_result)
        self.assertTrue(any("evidence branch failed" in e for e in result.errors))
        self.assertTrue(any("search_guidelines" in e for e in result.errors))

    def test_both_branches_failing_records_both_errors(self):
        with mock.patch.object(
            cp, "retrieve_patient_history",
            side_effect=cp.PipelineError("history down"),
        ), mock.patch.object(
            cp, "search_guidelines", side_effect=cp.PipelineError("guidelines down"),
        ):
            result = cp.run_pipeline("conversation")

        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.errors), 2)

    def test_successful_branch_data_is_retained_when_the_other_fails(self):
        """Even on abort the caller can see what did come back."""
        with mock.patch.object(
            cp, "search_guidelines", side_effect=cp.PipelineError("guidelines down"),
        ):
            result = cp.run_pipeline("conversation")

        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.patient_facts, "patient branch succeeded")
        self.assertIsNone(result.evidence)

    def test_synthesis_failure_fails_the_run_but_keeps_branch_data(self):
        with mock.patch.object(
            cp, "generate_summary",
            side_effect=cp.PipelineError("generate_summary failed: boom"),
        ):
            result = cp.run_pipeline("conversation")

        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.structured_result)
        self.assertIsNotNone(result.patient_facts)
        self.assertIsNotNone(result.evidence)
        self.assertTrue(any("generate_summary" in e for e in result.errors))

    def test_question_extraction_failure_aborts_before_any_branch_runs(self):
        history = mock.Mock()
        with mock.patch.object(
            cp, "extract_patient_context",
            side_effect=cp.PipelineError("extract_patient_context failed: boom"),
        ), mock.patch.object(cp, "retrieve_patient_history", history):
            result = cp.run_pipeline("conversation")

        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.clinical_question)
        history.assert_not_called()

    def test_status_is_never_partial_yet(self):
        """Doc 7: "partial" is "reserved for a future refinement: currently
        unused". This test guards the doc's claim, not a desirable end state
        — see doc Section 9.1."""
        outcomes = set()
        outcomes.add(cp.run_pipeline("conversation").status)
        with mock.patch.object(
            cp, "search_guidelines", side_effect=cp.PipelineError("down"),
        ):
            outcomes.add(cp.run_pipeline("conversation").status)

        self.assertEqual(outcomes, {"success", "failed"})

    def test_branches_run_concurrently(self):
        """Doc 4: the patient-facts and evidence branches run in parallel."""
        delay = 0.4

        def slow_branch(*args, **kwargs):
            time.sleep(delay)
            return {"clinical_question": "q", "history": {}}

        with mock.patch.object(cp, "extract_patient_context",
                               return_value={"clinical_question": "q"}), \
             mock.patch.object(cp, "_get_patient_facts", slow_branch), \
             mock.patch.object(cp, "_get_evidence", slow_branch), \
             mock.patch.object(cp, "generate_summary", return_value={}):
            start = time.time()
            cp.run_pipeline("conversation")
            elapsed = time.time() - start

        self.assertLess(
            elapsed, delay * 1.8,
            f"expected overlap (~{delay}s), got {elapsed:.2f}s — branches ran serially",
        )

    def test_concurrent_pipeline_runs_do_not_interfere(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(
                lambda pid: cp.run_pipeline("conversation", patient_id=pid),
                ["P-1", "P-2", "P-3"],
            ))

        self.assertEqual(len({r.request_id for r in results}), 3)
        for pid, result in zip(["P-1", "P-2", "P-3"], results):
            self.assertEqual(result.status, "success")
            self.assertEqual(result.patient_facts["history"]["patient_id"], pid)


# --------------------------------------------------------------------------
# Divergences between design_doc.md and clinical_pipeline.py
# --------------------------------------------------------------------------

class TestDesignDocConformance(QuietMocksMixin, unittest.TestCase):
    """Each `expectedFailure` below asserts what the doc promises and fails
    against the current code. They are the actionable output of this suite."""

    @unittest.expectedFailure
    def test_timeout_bounds_wall_clock_time(self):
        """Doc 6: "Call runs in a worker thread; future.result(timeout=N)
        enforces a hard per-attempt wall-clock budget. A hang can't block
        the pipeline."

        It doesn't. `with ThreadPoolExecutor(...)` calls `shutdown(wait=True)`
        on exit, so leaving the block after a timeout blocks until the hung
        worker finishes. The timeout only relabels the error; it never
        reclaims the time. A tool that hangs for 60s costs 60s per attempt,
        and with `max_retries=2` that is 180s despite a 4s "budget".
        """
        body_duration = 1.0

        @cp.with_resilience(timeout_s=0.1, max_retries=0, backoff_base=0.0)
        def hangs():
            time.sleep(body_duration)
            return {}

        start = time.time()
        with self.assertRaises(cp.PipelineError):
            hangs()
        elapsed = time.time() - start

        self.assertLess(
            elapsed, body_duration / 2,
            f"timeout_s=0.1 but the call took {elapsed:.2f}s",
        )

    @unittest.expectedFailure
    def test_context_extraction_runs_once_per_pipeline_run(self):
        """Doc 4: the architecture diagram shows `extract_patient_context()`
        running once, feeding both downstream branches.

        The code runs it twice: once in `run_pipeline` to derive the clinical
        question, then again inside `_get_patient_facts`. In production this
        is the most expensive step (an LLM call), so it doubles cost and
        latency on that branch and doubles the chance the branch fails.
        """
        calls = []
        real = cp.extract_patient_context

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        with mock.patch.object(cp, "extract_patient_context", counting):
            cp.run_pipeline("conversation")

        self.assertEqual(len(calls), 1, f"extracted {len(calls)}x")

    @unittest.expectedFailure
    def test_reported_clinical_question_matches_the_one_in_patient_facts(self):
        """Follows from the duplicate extraction above. Because the question
        is extracted twice, `PipelineResult.clinical_question` and
        `patient_facts["clinical_question"]` come from two separate calls. A
        real (non-deterministic) LLM can return different text each time, so
        the reported question may not be the one the branch actually used.
        """
        counter = iter(range(100))

        def varying(conversation, _request_id=None):
            return {
                "chief_complaint": "cc",
                "clinical_question": f"question-{next(counter)}",
                "age": 58,
                "sex": "female",
                "symptoms": [],
            }

        with mock.patch.object(cp, "extract_patient_context", varying):
            result = cp.run_pipeline("conversation")

        self.assertEqual(
            result.clinical_question,
            result.patient_facts["clinical_question"],
        )

    @unittest.expectedFailure
    def test_history_validator_enforces_patient_id(self):
        """Doc 5.2 lists `patient_id` in the output schema for
        `retrieve_patient_history`, but `_validate_history` only checks
        conditions/medications/allergies. An EHR response missing its record
        identifier passes validation, so the pipeline cannot tell whose
        history it just merged into `patient_facts`.
        """
        with self.assertRaises(cp.MalformedResponseError):
            cp._validate_history(
                {"conditions": [], "medications": [], "allergies": []}
            )

    @unittest.expectedFailure
    def test_non_numeric_confidence_is_reported_as_malformed(self):
        """Doc 6: the validator "checks required keys, types, and value
        ranges" and a failure "raises MalformedResponseError".

        `_validate_summary` compares `confidence` against 0.0/1.0 without a
        type check, so a string confidence raises TypeError instead. The
        decorator's catch-all still retries it, but the failure is logged as
        a generic ERROR rather than MALFORMED OUTPUT, which is exactly the
        signal Section 6 says to preserve.
        """
        bad = {
            "summary": "s",
            "recommendations": [],
            "evidence_cited": [],
            "confidence": "high",
            "caveats": [],
        }
        with self.assertRaises(cp.MalformedResponseError):
            cp._validate_summary(bad)

    def test_tools_share_one_global_random_stream(self):
        """`random.seed(1)` is commented "deterministic demo run", but the
        tools draw from the process-wide `random` module while running on
        concurrent threads. Interleaving decides which branch gets which
        draw, so the seed does not pin the outcome: reseeding and re-running
        the demo yields both `success` (~6s) and `failed` (~11s).

        This test pins the mechanism (a single shared stream, no per-branch
        RNG) rather than the race, so it is not itself flaky.
        """
        drawn_by = []
        real_random = random.random

        def tracking_random():
            drawn_by.append(threading.get_ident())
            return real_random()

        with mock.patch.object(random, "random", tracking_random):
            cp._get_evidence("question", "req12345")

        self.assertGreater(
            len(set(drawn_by)), 1,
            "expected both search threads to pull from the shared RNG",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
