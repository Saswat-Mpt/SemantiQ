import pytest
from src.phase3_features import extract_pair_features


def test_feature_extraction_schema():
    q1 = "How can I study mathematics?"
    q2 = "What is the best way to learn math?"
    features = extract_pair_features(q1, q2)

    expected_keys = [
        "q1_char_length", "q2_char_length", "q1_word_count", "q2_word_count",
        "length_difference", "length_ratio", "common_word_count", "common_word_ratio",
        "jaccard_similarity", "fuzzy_ratio", "token_sort_ratio", "token_set_ratio",
        "char_bigram_jaccard", "char_trigram_jaccard", "word_bigram_overlap", "word_trigram_overlap",
    ]

    for key in expected_keys:
        assert key in features
        assert isinstance(features[key], (int, float))
        assert not (features[key] != features[key])  # No NaN


def test_jaccard_bounds():
    q1 = "identical question text"
    q2 = "identical question text"
    features = extract_pair_features(q1, q2)
    assert features["jaccard_similarity"] == 1.0

    q3 = "completely disjoint words"
    q4 = "unrelated sentence"
    features_disjoint = extract_pair_features(q3, q4)
    assert features_disjoint["jaccard_similarity"] == 0.0
