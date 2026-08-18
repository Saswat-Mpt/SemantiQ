import pytest
from src.phase1_data import normalize_question_text
from src.phase4_representations import clean_text


def test_normalize_question_text():
    raw_1 = "What is the capital of France???"
    raw_2 = "What   is the capital of France"
    assert normalize_question_text(raw_1) == normalize_question_text(raw_2)


def test_normalize_empty_handling():
    assert normalize_question_text("") == ""
    assert normalize_question_text("???!!!") == ""


def test_clean_text_preserves_words():
    raw = "  How to   learn PYTHON?  "
    cleaned = clean_text(raw)
    assert cleaned == "how to learn python?"
