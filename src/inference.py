from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.critical_tokens import evaluate_critical_tokens
from src.phase3_features import extract_pair_features
from src.phase4_representations import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASE4_DIR = PROJECT_ROOT / "artifacts" / "phase4"
PHASE5_DIR = PROJECT_ROOT / "artifacts" / "phase5"
PHASE6_DIR = PROJECT_ROOT / "artifacts" / "phase6"


class SemantIQ:
    """
    Production-oriented inference engine for SemantIQ deduplication.
    
    Provides:
      - 19-feature fusion vector extraction
      - Calibrated 3-tier decision policy (DUPLICATE / NEEDS_REVIEW / DISTINCT)
      - Critical token contradiction & mismatch diagnostics
      - Decision evidence and execution latency tracking
    """

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or PROJECT_ROOT

        # Load Champion XGBoost Model (Experiment E)
        model_path = root / "artifacts" / "phase5" / "xgboost_experiment_E.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        self.model = joblib.load(model_path)

        # Load Decision Policy & Thresholds
        with (root / "artifacts" / "phase6" / "selected_threshold.json").open("r", encoding="utf-8") as f:
            threshold_data = json.load(f)
        self.high_precision_threshold = float(threshold_data["selected_threshold"])
        self.default_threshold = 0.50

        # Load Pretrained & Fitted Encoders
        self.word_vectorizer = joblib.load(root / "artifacts" / "phase4" / "word_tfidf_vectorizer.joblib")
        self.char_vectorizer = joblib.load(root / "artifacts" / "phase4" / "char_tfidf_vectorizer.joblib")
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        with (root / "artifacts" / "phase5" / "feature_groups.json").open("r", encoding="utf-8") as f:
            groups = json.load(f)
        self.feature_columns = groups["experiments"]["E"]

    def extract_features(self, q1: str, q2: str) -> dict[str, float]:
        """Extracts complete 19-feature vector for arbitrary pair."""
        # 1. Phase 3 Classical Features (16 features)
        classical = extract_pair_features(q1, q2)

        # 2. Phase 4 Representation Features (3 features)
        q1_clean = clean_text(q1)
        q2_clean = clean_text(q2)

        # Word TF-IDF cosine
        w_m1 = self.word_vectorizer.transform([q1_clean])
        w_m2 = self.word_vectorizer.transform([q2_clean])
        word_tfidf_cos = float(np.asarray(w_m1.multiply(w_m2).sum(axis=1)).ravel()[0])

        # Char TF-IDF cosine
        c_m1 = self.char_vectorizer.transform([q1_clean])
        c_m2 = self.char_vectorizer.transform([q2_clean])
        char_tfidf_cos = float(np.asarray(c_m1.multiply(c_m2).sum(axis=1)).ravel()[0])

        # MiniLM cosine
        embeddings = self.embedding_model.encode(
            [q1_clean, q2_clean],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        minilm_cos = float(np.dot(embeddings[0], embeddings[1]))

        features = {
            **classical,
            "word_tfidf_cosine": word_tfidf_cos,
            "char_tfidf_cosine": char_tfidf_cos,
            "minilm_cosine": minilm_cos,
        }
        return features

    def predict_pair(self, question1: str, question2: str) -> dict[str, Any]:
        """
        Executes end-to-end inference on a single question pair.
        Returns 3-tier calibrated decision, duplicate score, and explanatory evidence.
        """
        start_time = time.perf_counter()

        if not question1 or not str(question1).strip():
            raise ValueError("question1 cannot be empty.")
        if not question2 or not str(question2).strip():
            raise ValueError("question2 cannot be empty.")

        q1 = str(question1).strip()
        q2 = str(question2).strip()

        # Extract features
        features_dict = self.extract_features(q1, q2)
        features_df = pd.DataFrame([[features_dict[name] for name in self.feature_columns]], columns=self.feature_columns)

        # Predict model score
        model_score = float(self.model.predict_proba(features_df)[0, 1])

        # Critical token contradictions check
        diag = evaluate_critical_tokens(q1, q2)

        # 3-Tier Calibrated Decision Policy
        #   - score >= T* (0.8034): DUPLICATE (High confidence, >= 90% validation target precision)
        #   - 0.50 <= score < T*:   NEEDS_REVIEW (Uncertain region / potential subtle distinction)
        #   - score < 0.50:         DISTINCT (High confidence non-duplicate)
        if model_score >= self.high_precision_threshold:
            decision = "DUPLICATE"
            confidence = "HIGH"
        elif model_score >= self.default_threshold:
            decision = "NEEDS_REVIEW"
            confidence = "MODERATE"
        else:
            decision = "DISTINCT"
            confidence = "HIGH" if model_score <= 0.20 else "MODERATE"

        # Diagnostic alert: Flag high semantic score with critical token mismatch
        contradiction_warning = bool(
            model_score >= self.default_threshold and 
            (diag["numeric_mismatch"] or diag["entity_mismatch"] or diag["negation_mismatch"])
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "question1": q1,
            "question2": q2,
            "score": round(model_score, 4),
            "decision": decision,
            "confidence": confidence,
            "thresholds": {
                "high_precision_T_star": round(self.high_precision_threshold, 4),
                "default_T": round(self.default_threshold, 4),
            },
            "contradiction_warning": contradiction_warning,
            "critical_tokens": diag,
            "evidence": {
                "semantic_similarity_minilm": round(features_dict["minilm_cosine"], 4),
                "word_tfidf_similarity": round(features_dict["word_tfidf_cosine"], 4),
                "char_tfidf_similarity": round(features_dict["char_tfidf_cosine"], 4),
                "fuzzy_token_set_ratio": round(features_dict["token_set_ratio"], 4),
                "jaccard_similarity": round(features_dict["jaccard_similarity"], 4),
                "word_trigram_overlap": round(features_dict["word_trigram_overlap"], 4),
            },
            "all_features": {k: round(float(v), 4) for k, v in features_dict.items()},
            "latency_ms": round(latency_ms, 2),
        }

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Batch prediction optimized for fast sequential execution."""
        return [self.predict_pair(q1, q2) for q1, q2 in pairs]
