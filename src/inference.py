from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from src.phase3_features import extract_pair_features
from src.phase4_representations import (
    clean_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE3_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase3"
)

PHASE4_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase4"
)

PHASE5_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase5"
)

PHASE6_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6"
)


STATISTICAL_FEATURES = [
    "q1_char_length",
    "q2_char_length",
    "q1_word_count",
    "q2_word_count",
    "length_difference",
    "length_ratio",
    "common_word_count",
    "common_word_ratio",
]

STRING_FEATURES = [
    "jaccard_similarity",
    "fuzzy_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "char_bigram_jaccard",
    "char_trigram_jaccard",
    "word_bigram_overlap",
    "word_trigram_overlap",
]

TFIDF_FEATURES = [
    "word_tfidf_cosine",
    "char_tfidf_cosine",
]

SEMANTIC_FEATURES = [
    "minilm_cosine",
]

FINAL_FEATURES = (
    STATISTICAL_FEATURES
    + STRING_FEATURES
    + TFIDF_FEATURES
    + SEMANTIC_FEATURES
)


class SemantIQ:

    def __init__(
        self,
    ) -> None:

        print(
            "Loading SemantIQ..."
        )

        self.model = joblib.load(
            PHASE5_DIR
            / "xgboost_experiment_E.joblib"
        )

        with (
            PHASE6_DIR
            / "selected_threshold.json"
        ).open(
            "r",
            encoding="utf-8",
        ) as file:

            threshold_data = json.load(
                file
            )

        self.threshold = float(
            threshold_data[
                "selected_threshold"
            ]
        )

        self.word_vectorizer = (
            joblib.load(
                PHASE4_DIR
                / "word_tfidf_vectorizer.joblib"
            )
        )

        self.char_vectorizer = (
            joblib.load(
                PHASE4_DIR
                / "char_tfidf_vectorizer.joblib"
            )
        )

        self.embedding_model = (
            SentenceTransformer(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )
        )

        print(
            f"SemantIQ loaded. Frozen T* = {self.threshold:.4f} (Precision >= 90% mode)"
        )

    def _build_features(
        self,
        question1: str,
        question2: str,
    ) -> pd.DataFrame:

        # ----------------------------------------------------
        # Phase 3 features
        # ----------------------------------------------------

        classical = extract_pair_features(
            question1,
            question2,
        )

        # ----------------------------------------------------
        # Clean text
        # ----------------------------------------------------

        q1 = clean_text(
            question1
        )

        q2 = clean_text(
            question2
        )

        # ----------------------------------------------------
        # Word TF-IDF
        # ----------------------------------------------------

        q1_word = (
            self.word_vectorizer
            .transform([q1])
        )

        q2_word = (
            self.word_vectorizer
            .transform([q2])
        )

        word_cosine = float(
            q1_word.multiply(
                q2_word
            ).sum()
        )

        # ----------------------------------------------------
        # Character TF-IDF
        # ----------------------------------------------------

        q1_char = (
            self.char_vectorizer
            .transform([q1])
        )

        q2_char = (
            self.char_vectorizer
            .transform([q2])
        )

        char_cosine = float(
            q1_char.multiply(
                q2_char
            ).sum()
        )

        # ----------------------------------------------------
        # MiniLM
        # ----------------------------------------------------

        embeddings = (
            self.embedding_model.encode(
                [q1, q2],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        )

        minilm_cosine = float(
            np.dot(
                embeddings[0],
                embeddings[1],
            )
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        features = {
            **classical,
            "word_tfidf_cosine":
                word_cosine,
            "char_tfidf_cosine":
                char_cosine,
            "minilm_cosine":
                minilm_cosine,
        }

        result = pd.DataFrame(
            [
                [
                    features[name]
                    for name in FINAL_FEATURES
                ]
            ],
            columns=FINAL_FEATURES,
        )

        return result

    def predict_pair(
        self,
        question1: str,
        question2: str,
    ) -> dict:

        if not question1.strip():
            raise ValueError(
                "question1 cannot be empty."
            )

        if not question2.strip():
            raise ValueError(
                "question2 cannot be empty."
            )

        features = self._build_features(
            question1,
            question2,
        )

        probability = float(
            self.model.predict_proba(
                features
            )[0, 1]
        )

        is_duplicate = (
            probability
            >= self.threshold
        )

        return {
            "question1": question1,
            "question2": question2,
            "duplicate_probability":
                round(
                    probability,
                    4,
                ),
            "threshold":
                round(
                    self.threshold,
                    4,
                ),
            "is_duplicate":
                bool(is_duplicate),
            "decision":
                "DUPLICATE" if is_duplicate else "DISTINCT",
            "semantic_similarity":
                round(
                    float(
                        features[
                            "minilm_cosine"
                        ].iloc[0]
                    ),
                    4,
                ),
            "features": {
                k: round(float(v), 4)
                for k, v in features.iloc[0].to_dict().items()
            },
        }
