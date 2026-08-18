from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.phase3_features import extract_pair_features as extract_phase3_pair_features
from src.phase4_representations import clean_text


# ============================================================
# Hard Case Categorization & Analysis (Validation Partition)
# ============================================================

def analyze_validation_hard_cases(
    val_df: pd.DataFrame,
    val_features: pd.DataFrame,
    val_scores: np.ndarray,
    optimal_threshold: float,
    artifact_dir: Path,
    report_dir: Path,
) -> dict:
    """
    Analyzes error modes and hard cases on the VALIDATION split only.
    Test split is kept untouched.
    """
    df = val_df.copy()
    df["model_score"] = val_scores
    df["pred_default"] = (val_scores >= 0.50).astype(int)
    df["pred_cost_aware"] = (val_scores >= optimal_threshold).astype(int)
    
    # Attach key features for analysis
    for col in [
        "jaccard_similarity",
        "fuzzy_ratio",
        "word_tfidf_cosine",
        "char_tfidf_cosine",
        "minilm_cosine",
    ]:
        if col in val_features.columns:
            df[col] = val_features[col].to_numpy()

    # Category 1: High lexical overlap, but Non-Duplicate (Tricky negatives)
    high_lexical_non_dup = df[
        (df["is_duplicate"] == 0) & (df["fuzzy_ratio"] >= 0.70)
    ].sort_values(by="fuzzy_ratio", ascending=False).head(15)

    # Category 2: Low lexical overlap, but Duplicate (Semantic paraphrases)
    low_lexical_dup = df[
        (df["is_duplicate"] == 1) & (df["fuzzy_ratio"] <= 0.45)
    ].sort_values(by="minilm_cosine", ascending=False).head(15)

    # Category 3: False Positives under cost-aware T* (High confidence mistakes)
    fp_cost_aware = df[
        (df["is_duplicate"] == 0) & (df["pred_cost_aware"] == 1)
    ].sort_values(by="model_score", ascending=False).head(10)

    # Category 4: False Negatives under cost-aware T* (Missed duplicates)
    fn_cost_aware = df[
        (df["is_duplicate"] == 1) & (df["pred_cost_aware"] == 0)
    ].sort_values(by="model_score", ascending=True).head(10)

    def serialize_cases(subset_df: pd.DataFrame, category_name: str) -> list[dict]:
        cases = []
        for _, row in subset_df.iterrows():
            cases.append({
                "category": category_name,
                "question1": str(row["question1"]),
                "question2": str(row["question2"]),
                "ground_truth": int(row["is_duplicate"]),
                "model_score": round(float(row["model_score"]), 4),
                "decision_cost_aware": int(row["pred_cost_aware"]),
                "decision_default": int(row["pred_default"]),
                "fuzzy_ratio": round(float(row["fuzzy_ratio"]), 4),
                "jaccard_similarity": round(float(row["jaccard_similarity"]), 4),
                "minilm_cosine": round(float(row["minilm_cosine"]), 4),
                "word_tfidf_cosine": round(float(row["word_tfidf_cosine"]), 4),
            })
        return cases

    all_cases = {
        "high_lexical_non_duplicates": serialize_cases(high_lexical_non_dup, "High Lexical Overlap, Non-Duplicate"),
        "low_lexical_duplicates": serialize_cases(low_lexical_dup, "Low Lexical Overlap, Duplicate (Paraphrase)"),
        "high_confidence_false_positives": serialize_cases(fp_cost_aware, "Cost-Aware False Positive"),
        "false_negatives": serialize_cases(fn_cost_aware, "Cost-Aware False Negative"),
    }

    summary = {
        "total_validation_pairs": len(df),
        "high_lexical_non_duplicates_count": len(all_cases["high_lexical_non_duplicates"]),
        "low_lexical_duplicates_count": len(all_cases["low_lexical_duplicates"]),
        "cost_aware_fp_count": int(((df["is_duplicate"] == 0) & (df["pred_cost_aware"] == 1)).sum()),
        "cost_aware_fn_count": int(((df["is_duplicate"] == 1) & (df["pred_cost_aware"] == 0)).sum()),
        "cases": all_cases,
    }

    with (artifact_dir / "validation_hard_cases.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (report_dir / "phase7_hard_cases_report.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# End-to-End Live Inference Engine
# ============================================================

class SemantIQInferenceEngine:
    """
    Production-ready live inference engine for SemantIQ deduplication system.
    Extracts all 19 features on the fly and provides decision evidence.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        
        # Load artifacts
        p4_dir = project_root / "artifacts" / "phase4"
        p5_dir = project_root / "artifacts" / "phase5"
        p6_dir = project_root / "artifacts" / "phase6"

        print("Initializing SemantIQ Inference Engine...")
        self.word_vectorizer = joblib.load(p4_dir / "word_tfidf_vectorizer.joblib")
        self.char_vectorizer = joblib.load(p4_dir / "char_tfidf_vectorizer.joblib")
        
        # Pretrained embedding model
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Champion XGBoost model
        self.model = joblib.load(p5_dir / "xgboost_experiment_E.joblib")
        
        with open(p5_dir / "feature_groups.json", "r", encoding="utf-8") as f:
            groups = json.load(f)
        self.feature_columns = groups["experiments"]["E"]

        with open(p6_dir / "decision_policy.json", "r", encoding="utf-8") as f:
            policy = json.load(f)
        self.optimal_threshold = policy["optimal_threshold_T_star"]
        self.default_threshold = policy["default_threshold"]

        print(f"Engine Ready. Frozen T* = {self.optimal_threshold:.4f} (Precision >= 90% mode)")

    def extract_features(self, q1: str, q2: str) -> dict[str, float]:
        """Extracts complete 19-feature vector for arbitrary pair."""
        # 1. Phase 3 Classical Features (16 features)
        p3_features = extract_phase3_pair_features(q1, q2)

        # 2. Phase 4 Representation Features (3 features)
        cleaned_q1 = clean_text(q1)
        cleaned_q2 = clean_text(q2)

        # Word TF-IDF cosine
        w_m1 = self.word_vectorizer.transform([cleaned_q1])
        w_m2 = self.word_vectorizer.transform([cleaned_q2])
        word_tfidf_cos = float(np.asarray(w_m1.multiply(w_m2).sum(axis=1)).ravel()[0])

        # Char TF-IDF cosine
        c_m1 = self.char_vectorizer.transform([cleaned_q1])
        c_m2 = self.char_vectorizer.transform([cleaned_q2])
        char_tfidf_cos = float(np.asarray(c_m1.multiply(c_m2).sum(axis=1)).ravel()[0])

        # MiniLM cosine
        emb = self.embedding_model.encode(
            [cleaned_q1, cleaned_q2],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        minilm_cos = float(np.dot(emb[0], emb[1]))

        features = {
            **p3_features,
            "word_tfidf_cosine": word_tfidf_cos,
            "char_tfidf_cosine": char_tfidf_cos,
            "minilm_cosine": minilm_cos,
        }
        return features

    def predict(self, q1: str, q2: str) -> dict:
        """Inference for a single question pair."""
        feat_dict = self.extract_features(q1, q2)
        feat_df = pd.DataFrame([feat_dict])[self.feature_columns]

        model_score = float(self.model.predict_proba(feat_df)[0, 1])

        cost_aware_decision = "DUPLICATE" if model_score >= self.optimal_threshold else "DISTINCT"
        default_decision = "DUPLICATE" if model_score >= self.default_threshold else "DISTINCT"

        return {
            "question1": q1,
            "question2": q2,
            "model_score": round(model_score, 4),
            "cost_aware_decision": cost_aware_decision,
            "cost_aware_threshold_T_star": self.optimal_threshold,
            "default_decision": default_decision,
            "default_threshold": self.default_threshold,
            "key_evidence": {
                "semantic_similarity_minilm": round(feat_dict["minilm_cosine"], 4),
                "word_tfidf_similarity": round(feat_dict["word_tfidf_cosine"], 4),
                "char_tfidf_similarity": round(feat_dict["char_tfidf_cosine"], 4),
                "fuzzy_token_set_ratio": round(feat_dict["token_set_ratio"], 4),
                "jaccard_similarity": round(feat_dict["jaccard_similarity"], 4),
                "word_trigram_overlap": round(feat_dict["word_trigram_overlap"], 4),
            },
            "all_features": {k: round(v, 4) for k, v in feat_dict.items()},
        }
