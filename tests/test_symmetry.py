import pytest
from src.inference import SemantIQ


@pytest.fixture(scope="module")
def engine():
    return SemantIQ()


def test_pair_symmetry(engine):
    """
    Verifies that model duplicate probability is symmetric:
    f(q1, q2) approx f(q2, q1)
    """
    pairs = [
        ("How do I learn Python?", "What is the best way to study Python?"),
        ("What is machine learning?", "Can someone explain machine learning?"),
        ("How to lose weight fast?", "Best ways to reduce body fat quickly?"),
    ]

    for q1, q2 in pairs:
        res_forward = engine.predict_pair(q1, q2)
        res_reverse = engine.predict_pair(q2, q1)

        diff = abs(res_forward["score"] - res_reverse["score"])
        assert diff < 0.05, f"Asymmetry exceeded tolerance for '{q1}' vs '{q2}': {diff}"
