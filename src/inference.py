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
      - 19-feature fusion vector extraction (Champion Model E)
      - Cost-aware 3-tier decision policy (DUPLICATE / NEEDS_REVIEW / DISTINCT)
      - Critical token contradiction & mismatch diagnostics
      - In-memory query embedding cache for fast repeated query lookups
      - Vectorized high-throughput batch prediction
      - Latency tracking & decision evidence
    """

    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or PROJECT_ROOT

        # Load Champion XGBoost Model (Experiment E - 19 features)
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

        # In-memory query embedding cache
        self._embedding_cache: dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Cached single-question embedding retrieval."""
        cleaned = clean_text(text)
        if cleaned in self._embedding_cache:
            return self._embedding_cache[cleaned]
        
        emb = self.embedding_model.encode(
            [cleaned],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        
        if len(self._embedding_cache) < 10000:
            self._embedding_cache[cleaned] = emb
        return emb

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

        # MiniLM cosine (cached)
        emb1 = self._get_embedding(q1_clean)
        emb2 = self._get_embedding(q2_clean)
        minilm_cos = float(np.dot(emb1, emb2))

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
        Returns 3-tier cost-aware decision, raw model score, and explanatory evidence.
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

        # Predict raw model score (P in [0, 1])
        model_score = float(self.model.predict_proba(features_df)[0, 1])

        # Critical token contradictions check
        diag = evaluate_critical_tokens(q1, q2)

        # 3-Tier Cost-Aware Decision Policy
        #   - score >= T* (0.8034): DUPLICATE (High confidence, >= 90% validation target precision)
        #   - 0.50 <= score < T*:   NEEDS_REVIEW (Uncertain region / potential subtle distinction)
        #   - score < 0.50:         DISTINCT (High confidence non-duplicate)
        if model_score >= self.high_precision_threshold:
            decision = "DUPLICATE"
            decision_band = "HIGH_CONFIDENCE_MERGE"
            confidence = "HIGH"
        elif model_score >= self.default_threshold:
            decision = "NEEDS_REVIEW"
            decision_band = "HUMAN_REVIEW_REQUIRED"
            confidence = "MODERATE"
        else:
            decision = "DISTINCT"
            decision_band = "HIGH_CONFIDENCE_DISTINCT"
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
            "decision_band": decision_band,
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
        """
        True vectorized batch prediction for high-throughput pipelines.
        Batch transforms TF-IDF and batch encodes MiniLM embeddings.
        """
        if not pairs:
            return []

        start_time = time.perf_counter()
        n = len(pairs)

        q1_list = [str(p[0]).strip() for p in pairs]
        q2_list = [str(p[1]).strip() for p in pairs]

        q1_clean = [clean_text(q) for q in q1_list]
        q2_clean = [clean_text(q) for q in q2_list]

        # 1. Batch TF-IDF
        w1 = self.word_vectorizer.transform(q1_clean)
        w2 = self.word_vectorizer.transform(q2_clean)
        word_cosines = np.asarray(w1.multiply(w2).sum(axis=1)).ravel()

        c1 = self.char_vectorizer.transform(q1_clean)
        c2 = self.char_vectorizer.transform(q2_clean)
        char_cosines = np.asarray(c1.multiply(c2).sum(axis=1)).ravel()

        # 2. Batch MiniLM (encode unique questions to minimize forward passes)
        unique_texts = list(set(q1_clean + q2_clean))
        uncached = [t for t in unique_texts if t not in self._embedding_cache]
        
        if uncached:
            embeddings = self.embedding_model.encode(
                uncached,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
            )
            for text, emb in zip(uncached, embeddings):
                if len(self._embedding_cache) < 10000:
                    self._embedding_cache[text] = emb

        minilm_cosines = np.empty(n, dtype=np.float32)
        for i in range(n):
            emb1 = self._embedding_cache[q1_clean[i]]
            emb2 = self._embedding_cache[q2_clean[i]]
            minilm_cosines[i] = float(np.dot(emb1, emb2))

        # 3. Batch Classical Features & Combine
        rows = []
        for i in range(n):
            classical = extract_pair_features(q1_list[i], q2_list[i])
            feat_dict = {
                **classical,
                "word_tfidf_cosine": float(word_cosines[i]),
                "char_tfidf_cosine": float(char_cosines[i]),
                "minilm_cosine": float(minilm_cosines[i]),
            }
            rows.append([feat_dict[col] for col in self.feature_columns])

        features_df = pd.DataFrame(rows, columns=self.feature_columns)
        scores = self.model.predict_proba(features_df)[:, 1]

        batch_latency_ms = (time.perf_counter() - start_time) * 1000.0
        per_item_latency = batch_latency_ms / n

        results = []
        for i in range(n):
            score = float(scores[i])
            diag = evaluate_critical_tokens(q1_list[i], q2_list[i])
            
            if score >= self.high_precision_threshold:
                decision = "DUPLICATE"
                decision_band = "HIGH_CONFIDENCE_MERGE"
                confidence = "HIGH"
            elif score >= self.default_threshold:
                decision = "NEEDS_REVIEW"
                decision_band = "HUMAN_REVIEW_REQUIRED"
                confidence = "MODERATE"
            else:
                decision = "DISTINCT"
                decision_band = "HIGH_CONFIDENCE_DISTINCT"
                confidence = "HIGH" if score <= 0.20 else "MODERATE"

            contradiction_warning = bool(
                score >= self.default_threshold and 
                (diag["numeric_mismatch"] or diag["entity_mismatch"] or diag["negation_mismatch"])
            )

            results.append({
                "question1": q1_list[i],
                "question2": q2_list[i],
                "score": round(score, 4),
                "decision": decision,
                "decision_band": decision_band,
                "confidence": confidence,
                "thresholds": {
                    "high_precision_T_star": round(self.high_precision_threshold, 4),
                    "default_T": round(self.default_threshold, 4),
                },
                "contradiction_warning": contradiction_warning,
                "critical_tokens": diag,
                "evidence": {
                    "semantic_similarity_minilm": round(float(minilm_cosines[i]), 4),
                    "word_tfidf_similarity": round(float(word_cosines[i]), 4),
                    "char_tfidf_similarity": round(float(char_cosines[i]), 4),
                },
                "latency_ms": round(per_item_latency, 2),
            })
        return results
