from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.critical_tokens import evaluate_critical_tokens


# ============================================================
# Feature Definitions
# ============================================================

BASE_19_FEATURES = [
    "q1_char_length",
    "q2_char_length",
    "q1_word_count",
    "q2_word_count",
    "length_difference",
    "length_ratio",
    "common_word_count",
    "common_word_ratio",
    "jaccard_similarity",
    "fuzzy_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "char_bigram_jaccard",
    "char_trigram_jaccard",
    "word_bigram_overlap",
    "word_trigram_overlap",
    "word_tfidf_cosine",
    "char_tfidf_cosine",
    "minilm_cosine",
]

CONTRADICTION_FEATURES = [
    "numeric_mismatch",
    "numeric_overlap",
    "entity_mismatch",
    "negation_mismatch",
    "question_type_mismatch",
]

MODEL_F_24_FEATURES = BASE_19_FEATURES + CONTRADICTION_FEATURES


# ============================================================
# Contradiction Feature Extractor for Splits
# ============================================================

def build_contradiction_features_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts numeric contradiction features for an entire dataframe."""
    records = []
    for row in df.itertuples(index=False):
        q1 = str(row.question1)
        q2 = str(row.question2)
        diag = evaluate_critical_tokens(q1, q2)
        records.append({
            "numeric_mismatch": 1.0 if diag["numeric_mismatch"] else 0.0,
            "numeric_overlap": float(diag["numeric_overlap"]),
            "entity_mismatch": 1.0 if diag["entity_mismatch"] else 0.0,
            "negation_mismatch": 1.0 if diag["negation_mismatch"] else 0.0,
            "question_type_mismatch": 1.0 if diag["question_type_mismatch"] else 0.0,
        })
    return pd.DataFrame(records, index=df.index).astype(np.float32)


# ============================================================
# Calibration Evaluation (Expected Calibration Error & Brier)
# ============================================================

def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Computes Brier score and Expected Calibration Error (ECE)."""
    brier = float(brier_score_loss(y_true, y_prob))
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(mask)
        
        if bin_size > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
            
    return {
        "brier_score": round(brier, 5),
        "expected_calibration_error": round(float(ece), 5),
    }


# ============================================================
# Bootstrap 95% Confidence Intervals
# ============================================================

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Computes empirical 95% confidence intervals via non-parametric bootstrapping."""
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    
    precisions = []
    recalls = []
    f1s = []
    pr_aucs = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        sample_true = y_true[indices]
        sample_score = y_score[indices]
        sample_pred = (sample_score >= threshold).astype(int)
        
        if np.sum(sample_true) == 0 or np.sum(sample_pred) == 0:
            continue
            
        precisions.append(precision_score(sample_true, sample_pred, zero_division=0))
        recalls.append(recall_score(sample_true, sample_pred, zero_division=0))
        f1s.append(f1_score(sample_true, sample_pred, zero_division=0))
        pr_aucs.append(average_precision_score(sample_true, sample_score))
        
    def ci_range(arr: list[float]) -> dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower_95": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper_95": round(float(np.percentile(arr, 97.5)), 4),
        }
        
    return {
        "precision": ci_range(precisions),
        "recall": ci_range(recalls),
        "f1": ci_range(f1s),
        "pr_auc": ci_range(pr_aucs),
    }


# ============================================================
# Phase 8 Pipeline
# ============================================================

def run_phase8(
    processed_dir: Path,
    phase3_dir: Path,
    phase4_dir: Path,
    phase5_dir: Path,
    phase6_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:

    print("=" * 70)
    print("SemantiQ — Phase 8: Contradiction Verification & Calibration")
    print("=" * 70)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df   = pd.read_csv(processed_dir / "val.csv")
    test_df  = pd.read_csv(processed_dir / "test.csv")

    y_train = train_df["is_duplicate"].astype(int).to_numpy()
    y_val   = val_df["is_duplicate"].astype(int).to_numpy()
    y_test  = test_df["is_duplicate"].astype(int).to_numpy()

    # 2. Load 19 features
    p3_train = pd.read_csv(phase3_dir / "train_phase3_features.csv")
    p3_val   = pd.read_csv(phase3_dir / "val_phase3_features.csv")
    p3_test  = pd.read_csv(phase3_dir / "test_phase3_features.csv")

    p4_train = pd.read_csv(phase4_dir / "train_phase4_features.csv")
    p4_val   = pd.read_csv(phase4_dir / "val_phase4_features.csv")
    p4_test  = pd.read_csv(phase4_dir / "test_phase4_features.csv")

    base_train = pd.concat([p3_train.reset_index(drop=True), p4_train.reset_index(drop=True)], axis=1)
    base_val   = pd.concat([p3_val.reset_index(drop=True), p4_val.reset_index(drop=True)], axis=1)
    base_test  = pd.concat([p3_test.reset_index(drop=True), p4_test.reset_index(drop=True)], axis=1)

    # 3. Extract contradiction features (5 features)
    print("\n[1/5] Extracting Contradiction Features for Splits...")
    contra_train = build_contradiction_features_for_df(train_df)
    contra_val   = build_contradiction_features_for_df(val_df)
    contra_test  = build_contradiction_features_for_df(test_df)

    f_train = pd.concat([base_train, contra_train], axis=1)[MODEL_F_24_FEATURES]
    f_val   = pd.concat([base_val, contra_val], axis=1)[MODEL_F_24_FEATURES]
    f_test  = pd.concat([base_test, contra_test], axis=1)[MODEL_F_24_FEATURES]

    print(f"  Combined Model F Feature Matrix: {f_train.shape[1]} features")

    # 4. Train Model F (Contradiction-Enhanced Fusion)
    print("\n[2/5] Training Model F (24 Features with Contradiction Signals)...")
    model_f = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model_f.fit(f_train, y_train)

    val_scores_f = model_f.predict_proba(f_val)[:, 1]
    test_scores_f = model_f.predict_proba(f_test)[:, 1]

    # Evaluate Model F on Test
    f_test_prauc = average_precision_score(y_test, test_scores_f)
    f_test_pred = (test_scores_f >= 0.50).astype(int)
    f_test_f1 = f1_score(y_test, f_test_pred)
    f_test_prec = precision_score(y_test, f_test_pred)
    f_test_rec = recall_score(y_test, f_test_pred)

    print(f"  Model F Test Results (T=0.50): PR-AUC = {f_test_prauc:.4f}, F1 = {f_test_f1:.4f}, Prec = {f_test_prec:.4f}, Rec = {f_test_rec:.4f}")

    # Save Model F
    joblib.dump(model_f, artifact_dir / "xgboost_experiment_F.joblib")

    # 5. Probability Calibration (Platt Sigmoid vs Isotonic Regression on Validation Scores)
    print("\n[3/5] Calibrating Probabilities (Platt Scaling & Isotonic Regression)...")
    model_e = joblib.load(phase5_dir / "xgboost_experiment_E.joblib")
    e_val = base_val[BASE_19_FEATURES]
    e_test = base_test[BASE_19_FEATURES]

    val_scores_raw = model_e.predict_proba(e_val)[:, 1]
    test_scores_raw = model_e.predict_proba(e_test)[:, 1]

    # Platt Scaling (Logistic Regression on raw probabilities / log-odds)
    platt_scaler = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    platt_scaler.fit(val_scores_raw.reshape(-1, 1), y_val)
    test_scores_platt = platt_scaler.predict_proba(test_scores_raw.reshape(-1, 1))[:, 1]

    # Isotonic Regression on validation probabilities
    isotonic_scaler = IsotonicRegression(out_of_bounds="clip")
    isotonic_scaler.fit(val_scores_raw, y_val)
    test_scores_isotonic = isotonic_scaler.predict(test_scores_raw)

    calib_raw = compute_calibration_metrics(y_test, test_scores_raw)
    calib_platt = compute_calibration_metrics(y_test, test_scores_platt)
    calib_iso = compute_calibration_metrics(y_test, test_scores_isotonic)

    print(f"  Raw XGBoost:  Brier = {calib_raw['brier_score']:.5f}, ECE = {calib_raw['expected_calibration_error']:.5f}")
    print(f"  Platt Scaling:Brier = {calib_platt['brier_score']:.5f}, ECE = {calib_platt['expected_calibration_error']:.5f}")
    print(f"  Isotonic Cal: Brier = {calib_iso['brier_score']:.5f}, ECE = {calib_iso['expected_calibration_error']:.5f}")

    joblib.dump(platt_scaler, artifact_dir / "platt_scaler.joblib")
    joblib.dump(isotonic_scaler, artifact_dir / "isotonic_scaler.joblib")

    # 6. Statistical 95% Confidence Intervals
    print("\n[4/5] Computing Bootstrap 95% Confidence Intervals (500 resamples)...")
    with (phase6_dir / "selected_threshold.json").open("r", encoding="utf-8") as f:
        t_star_data = json.load(f)
    t_star = float(t_star_data["selected_threshold"])

    ci_default = compute_bootstrap_ci(y_test, test_scores_raw, threshold=0.50, n_bootstraps=500)
    ci_t_star  = compute_bootstrap_ci(y_test, test_scores_raw, threshold=t_star, n_bootstraps=500)

    print(f"  Test PR-AUC:      {ci_t_star['pr_auc']['mean']:.4f} [95% CI: {ci_t_star['pr_auc']['ci_lower_95']:.4f} - {ci_t_star['pr_auc']['ci_upper_95']:.4f}]")
    print(f"  Test Prec (T*):   {ci_t_star['precision']['mean']:.4f} [95% CI: {ci_t_star['precision']['ci_lower_95']:.4f} - {ci_t_star['precision']['ci_upper_95']:.4f}]")
    print(f"  Test Rec (T*):    {ci_t_star['recall']['mean']:.4f} [95% CI: {ci_t_star['recall']['ci_lower_95']:.4f} - {ci_t_star['recall']['ci_upper_95']:.4f}]")

    # 7. Final Report Compilation
    print("\n[5/5] Compiling Comprehensive Phase 8 Report...")
    report = {
        "model_comparison": {
            "Model_E_Base_19": {
                "num_features": 19,
                "test_pr_auc": 0.8353,
                "test_f1": 0.7809,
                "test_precision": 0.7581,
                "test_recall": 0.8052,
            },
            "Model_F_Contradiction_24": {
                "num_features": 24,
                "test_pr_auc": round(float(f_test_prauc), 4),
                "test_f1": round(float(f_test_f1), 4),
                "test_precision": round(float(f_test_prec), 4),
                "test_recall": round(float(f_test_rec), 4),
            },
        },
        "probability_calibration": {
            "raw_xgboost": calib_raw,
            "platt_scaling": calib_platt,
            "isotonic_calibration": calib_iso,
        },
        "bootstrap_confidence_intervals_test": {
            "default_threshold_0_50": ci_default,
            "cost_aware_threshold_T_star": ci_t_star,
        },
    }

    with (artifact_dir / "phase8_calibration_and_stats.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with (report_dir / "phase8_calibration_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 8 COMPLETE")
    print("=" * 70)
    return report
