import sys
import types
from pathlib import Path

import pytest


def _ensure_yaml_stub() -> None:
    if "yaml" in sys.modules:
        return
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(stream):  # pragma: no cover - simple stub
        return {}

    yaml_stub.safe_load = _safe_load  # type: ignore[attr-defined]
    sys.modules["yaml"] = yaml_stub


_ensure_yaml_stub()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.dataset import _compile_target_pattern


def assert_match(pattern, text):
    assert pattern is not None, "Expected a compiled pattern"
    assert pattern.search(text), f"{pattern.pattern!r} did not match {text!r}"


def assert_no_match(pattern, text):
    assert pattern is not None, "Expected a compiled pattern"
    assert not pattern.search(text), f"{pattern.pattern!r} unexpectedly matched {text!r}"


def test_plural_and_possessive_forms_included():
    pattern = _compile_target_pattern("Immigrant")
    assert_match(pattern, "Immigrants will attend")
    assert_match(pattern, "The immigrant's story")
    assert_no_match(pattern, "Immigration policy is complex")


def test_compound_and_closed_compound_rules():
    pattern = _compile_target_pattern("fire")
    assert_match(pattern, "The fire station is nearby")
    assert_no_match(pattern, "A firetruck passed by")


def test_multi_word_phrase_pluralization_and_hyphenation():
    pattern = _compile_target_pattern("Law and order")
    assert_match(pattern, "They promised laws and order")
    assert_match(pattern, "The platform emphasized law-and-order policies")


def test_curly_apostrophes_are_supported():
    pattern = _compile_target_pattern("Egg")
    assert_match(pattern, "The eggs’ shells are fragile")


def test_numeric_ordinals_are_included():
    pattern = _compile_target_pattern("January 6")
    assert_match(pattern, "The hearing is on January 6th.")


def test_homonyms_and_case_insensitivity():
    pattern = _compile_target_pattern("ICE")
    assert_match(pattern, "She added ice water to the glass")


def test_adjacent_context_and_alternatives():
    pattern = _compile_target_pattern("Elon / Musk")
    assert_match(pattern, "Elon University has a program")
    assert_match(pattern, "Musk said it plainly")
    assert_no_match(pattern, "An elongated shape appeared")
