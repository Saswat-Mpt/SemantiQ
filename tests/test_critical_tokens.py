import pytest
from src.critical_tokens import evaluate_critical_tokens


def test_numeric_mismatch():
    q1 = "How can I score 50% in 12th board exams?"
    q2 = "How can I score 60% in 12th board exams?"
    diag = evaluate_critical_tokens(q1, q2)
    assert diag["numeric_mismatch"] is True
    assert "50%" in diag["q1_numbers"] or "50" in diag["q1_numbers"]
    assert "60%" in diag["q2_numbers"] or "60" in diag["q2_numbers"]


def test_year_mismatch():
    q1 = "What was the cutoff for IIT in 2017?"
    q2 = "What was the cutoff for IIT in 2018?"
    diag = evaluate_critical_tokens(q1, q2)
    assert diag["numeric_mismatch"] is True


def test_negation_mismatch():
    q1 = "Why should I learn Python?"
    q2 = "Why shouldn't I learn Python?"
    diag = evaluate_critical_tokens(q1, q2)
    assert diag["negation_mismatch"] is True


def test_entity_mismatch():
    q1 = "What is life like at IIT Delhi?"
    q2 = "What is life like at IIT Bombay?"
    diag = evaluate_critical_tokens(q1, q2)
    assert diag["entity_mismatch"] is True


def test_identical_questions():
    q1 = "How do I become a software engineer?"
    q2 = "How do I become a software engineer?"
    diag = evaluate_critical_tokens(q1, q2)
    assert diag["numeric_mismatch"] is False
    assert diag["entity_mismatch"] is False
    assert diag["negation_mismatch"] is False
