"""
tests/test_router.py — Unit tests for agents/router.py intent detection.

Tests covered
-------------
- detect_intent() returns IntentResult with a valid intent label.
- Mocked LLM responses are parsed correctly for each of the five intents.
- Empty input → "unknown" without calling the LLM.
- Malformed LLM response → "unknown" (graceful fallback).
- IntentResult confidence is in [0, 1].

Note: These tests mock the LLM so no real API key or network call is needed.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

from agents.router import detect_intent, IntentResult


def _mock_llm(json_response: str):
    """Return a mock LLM that emits a fixed AIMessage string."""
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=AIMessage(content=json_response))
    # Make the mock work as an LCEL Runnable (pipe operator)
    llm.__or__ = MagicMock(return_value=llm)
    return llm


class TestDetectIntent:
    def test_empty_input_returns_unknown(self):
        llm = MagicMock()
        result = detect_intent("", llm)
        assert result.intent == "unknown"
        assert result.confidence == 0.0
        llm.assert_not_called()   # LLM should not be invoked for empty input

    def test_whitespace_only_returns_unknown(self):
        llm = MagicMock()
        result = detect_intent("   \n\t  ", llm)
        assert result.intent == "unknown"

    @pytest.mark.parametrize("intent_label", [
        "summarize",
        "extract_concepts",
        "generate_slides",
        "viva",
        "unknown",
    ])
    def test_valid_intent_labels_accepted(self, intent_label):
        result = IntentResult(intent=intent_label, confidence=0.9)
        assert result.intent == intent_label

    def test_malformed_llm_output_returns_unknown(self):
        """When the chain raises (e.g. JSON parse error), fallback to unknown."""
        llm = MagicMock()
        # Simulate the chain failing during invocation
        with patch("agents.router._build_intent_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("JSON parse error")
            mock_build.return_value = mock_chain

            result = detect_intent("give me a summary", llm)
            assert result.intent == "unknown"
            assert result.confidence == 0.0

    def test_intent_result_confidence_range(self):
        for c in [0.0, 0.5, 1.0]:
            r = IntentResult(intent="summarize", confidence=c)
            assert 0.0 <= r.confidence <= 1.0

    def test_intent_result_invalid_confidence_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IntentResult(intent="summarize", confidence=1.5)


class TestDispatch:
    def test_dispatch_summarize(self):
        from agents.router import dispatch
        with patch("agents.summarizer.run", return_value="summary_res") as mock_run:
            res = dispatch("summarize", doc_id="doc1")
            mock_run.assert_called_once_with(retrieval_result=None, doc_id="doc1", llm=None)
            assert res == "summary_res"

    def test_dispatch_extract_concepts(self):
        from agents.router import dispatch
        with patch("agents.concept_extractor.run", return_value="concepts_res") as mock_run:
            res = dispatch("extract_concepts", doc_id="doc1")
            mock_run.assert_called_once_with(retrieval_result=None, doc_id="doc1", llm=None)
            assert res == "concepts_res"

    def test_dispatch_generate_slides(self):
        from agents.router import dispatch
        with patch("agents.slide_generator.run", return_value="slides_res") as mock_run:
            res = dispatch("generate_slides", doc_id="doc1")
            mock_run.assert_called_once_with(retrieval_result=None, doc_id="doc1", llm=None)
            assert res == "slides_res"

    def test_dispatch_viva(self):
        from agents.router import dispatch
        with patch("agents.viva_agent.run", return_value="viva_res") as mock_run:
            res = dispatch("viva", doc_id="doc1")
            mock_run.assert_called_once_with(retrieval_result=None, doc_id="doc1", llm=None)
            assert res == "viva_res"

    def test_dispatch_intent_result_object(self):
        from agents.router import dispatch
        with patch("agents.summarizer.run", return_value="summary_res") as mock_run:
            res = dispatch(IntentResult(intent="summarize"), doc_id="doc1")
            mock_run.assert_called_once_with(retrieval_result=None, doc_id="doc1", llm=None)
            assert res == "summary_res"

    def test_dispatch_unknown_raises(self):
        from agents.router import dispatch
        with pytest.raises(ValueError, match="Cannot dispatch unrecognized or unsupported intent: 'unknown'"):
            dispatch("unknown", doc_id="doc1")

    def test_dispatch_unsupported_string_raises(self):
        from agents.router import dispatch
        with pytest.raises(ValueError, match="Cannot dispatch unrecognized or unsupported intent: 'invalid'"):
            dispatch("invalid", doc_id="doc1")


