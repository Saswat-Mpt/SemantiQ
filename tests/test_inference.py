import pytest
from src.inference import SemantIQ


@pytest.fixture(scope="module")
def engine():
    return SemantIQ()


def test_predict_identical_pair(engine):
    q = "What is the capital of Australia?"
    res = engine.predict_pair(q, q)
    assert res["decision"] == "DUPLICATE"
    assert res["score"] > 0.80
    assert res["confidence"] == "HIGH"


def test_predict_distinct_pair(engine):
    q1 = "How do rockets escape Earth gravity?"
    q2 = "What are the health benefits of green tea?"
    res = engine.predict_pair(q1, q2)
    assert res["decision"] == "DISTINCT"
    assert res["score"] < 0.50


def test_empty_input_validation(engine):
    with pytest.raises(ValueError):
        engine.predict_pair("", "Valid question")

    with pytest.raises(ValueError):
        engine.predict_pair("Valid question", "   ")


def test_contradiction_warning_flag(engine):
    q1 = "Why should I learn Python in 2017?"
    q2 = "Why shouldn't I learn Python in 2018?"
    res = engine.predict_pair(q1, q2)
    assert "critical_tokens" in res
    assert res["critical_tokens"]["negation_mismatch"] is True
