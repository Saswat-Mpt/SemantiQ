from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

DEFAULT_THRESHOLD = 0.50

# ── Feature groups ──────────────────────────────────────────

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

ALL_FEATURES = (
    STATISTICAL_FEATURES
    + STRING_FEATURES
    + TFIDF_FEATURES
    + SEMANTIC_FEATURES
)

# ── A→E experiment definitions ──────────────────────────────
# NOTE: All A–E experiments use identical partitions and
# identical evaluation protocol.  The split is loaded once
# at the top of run_phase5 and every experiment reads the
# same fixed matrices — never re-derived per experiment.

EXPERIMENTS: dict[str, list[str]] = {
    "A": STATISTICAL_FEATURES,
    "B": STATISTICAL_FEATURES + STRING_FEATURES,
    "C": STATISTICAL_FEATURES + STRING_FEATURES + TFIDF_FEATURES,
    "D": STATISTICAL_FEATURES + SEMANTIC_FEATURES,   # isolated semantic branch
    "E": ALL_FEATURES,
}


# ============================================================
# XGBoost — fixed config, identical across all A-E rows
# ============================================================

def build_xgboost() -> XGBClassifier:
    """
    Fixed XGBoost configuration used for ALL A-E experiments.

    ONLY the feature columns change between experiments.
    Classifier hyperparameters are held constant so that
    metric differences reflect the representation, not the
    model.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


# ============================================================
# Validation helpers
# ============================================================

def validate_features(
    df: pd.DataFrame,
    features: list[str],
    split_name: str,
) -> None:
    """Check feature columns exist, no NaN, no infinity."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"{split_name}: missing features: {missing}"
        )

    subset = df[features]

    if subset.isna().any().any():
        bad = subset.columns[subset.isna().any()].tolist()
        raise RuntimeError(
            f"{split_name}: NaN values in {bad}"
        )

    if np.isinf(subset.to_numpy()).any():
        raise RuntimeError(
            f"{split_name}: infinite values detected."
        )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "pr_auc": float(
            average_precision_score(y_true, y_score)
        ),
        "confusion_matrix": (
            confusion_matrix(y_true, y_pred).tolist()
        ),
    }


# ============================================================
# Single experiment
# ============================================================

def run_experiment(
    name: str,
    features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    artifact_dir: Path,
) -> dict:

    print(f"\n{'=' * 70}")
    print(f"  Experiment {name} | {len(features)} features")
    print(f"{'=' * 70}")
    print(f"  Feature groups: {features}")

    # Validate then slice.
    for split_name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        validate_features(df, features, split_name)

    x_train = train_df[features].copy()
    x_val   = val_df[features].copy()
    x_test  = test_df[features].copy()

    # Train — fixed model config.
    model = build_xgboost()
    print("  Training XGBoost...")
    model.fit(x_train, y_train)

    # Predict.
    val_scores  = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]

    val_metrics  = calculate_metrics(y_val,  val_scores)
    test_metrics = calculate_metrics(y_test, test_scores)

    # Save model.
    model_path = artifact_dir / f"xgboost_experiment_{name}.joblib"
    joblib.dump(model, model_path)

    # Print per-experiment results.
    print(
        f"\n  {'':20s} {'Validation':>10} {'Test':>10}"
    )
    print(f"  {'-' * 42}")
    for metric in ("precision", "recall", "f1", "pr_auc"):
        v = val_metrics[metric]
        t = test_metrics[metric]
        print(f"  {metric.upper():<20} {v:>10.4f} {t:>10.4f}")

    return {
        "experiment": name,
        "feature_count": len(features),
        "features": features,
        "model_config": {
            "type": "XGBClassifier",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "model_path": str(model_path),
    }


# ============================================================
# Main Phase 5 pipeline
# ============================================================

def run_phase5(
    processed_dir: Path,
    phase3_dir: Path,
    phase4_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict:

    print("=" * 70)
    print("SemantiQ — Phase 5: XGBoost Controlled Ablation (A to E)")
    print("=" * 70)
    print(
        "\n# All A-E experiments use identical partitions and "
        "identical evaluation protocol."
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load fixed Phase 1 partitions (once — all rows share)
    # --------------------------------------------------------

    print("\n[1/6] Loading fixed Phase 1 partitions...")

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df   = pd.read_csv(processed_dir / "val.csv")
    test_df  = pd.read_csv(processed_dir / "test.csv")

    y_train = train_df["is_duplicate"].astype(int)
    y_val   = val_df["is_duplicate"].astype(int)
    y_test  = test_df["is_duplicate"].astype(int)

    print(f"Train:      {len(train_df):,}  (dup: {y_train.mean():.1%})")
    print(f"Validation: {len(val_df):,}  (dup: {y_val.mean():.1%})")
    print(f"Test:       {len(test_df):,}  (dup: {y_test.mean():.1%})")

    # --------------------------------------------------------
    # 2. Load Phase 3 feature matrices
    # --------------------------------------------------------

    print("\n[2/6] Loading Phase 3 feature matrices...")

    p3_train = pd.read_csv(phase3_dir / "train_phase3_features.csv")
    p3_val   = pd.read_csv(phase3_dir / "val_phase3_features.csv")
    p3_test  = pd.read_csv(phase3_dir / "test_phase3_features.csv")

    print(f"  Phase 3 train: {p3_train.shape}  columns: {list(p3_train.columns)}")

    # --------------------------------------------------------
    # 3. Load Phase 4 feature matrices
    # --------------------------------------------------------

    print("\n[3/6] Loading Phase 4 feature matrices...")

    p4_train = pd.read_csv(phase4_dir / "train_phase4_features.csv")
    p4_val   = pd.read_csv(phase4_dir / "val_phase4_features.csv")
    p4_test  = pd.read_csv(phase4_dir / "test_phase4_features.csv")

    print(f"  Phase 4 train: {p4_train.shape}  columns: {list(p4_train.columns)}")

    # --------------------------------------------------------
    # Combine into single feature store (fixed for all rows)
    # --------------------------------------------------------

    print("\nCombining feature store (Phase 3 + Phase 4)...")

    train_features = pd.concat(
        [p3_train.reset_index(drop=True), p4_train.reset_index(drop=True)],
        axis=1,
    )
    val_features = pd.concat(
        [p3_val.reset_index(drop=True), p4_val.reset_index(drop=True)],
        axis=1,
    )
    test_features = pd.concat(
        [p3_test.reset_index(drop=True), p4_test.reset_index(drop=True)],
        axis=1,
    )

    # Sanity: total columns must equal ALL_FEATURES count
    expected = len(ALL_FEATURES)
    actual = train_features.shape[1]
    if actual != expected:
        raise RuntimeError(
            f"Combined feature count mismatch. "
            f"Expected {expected}, got {actual}.\n"
            f"Columns: {list(train_features.columns)}"
        )

    # Sanity: row counts must match labels
    assert len(train_features) == len(y_train), "Train row mismatch"
    assert len(val_features)   == len(y_val),   "Val row mismatch"
    assert len(test_features)  == len(y_test),  "Test row mismatch"

    print(
        f"  Combined feature store: {train_features.shape[1]} features × "
        f"{len(train_features):,} train rows"
    )
    print(f"  All features: {list(train_features.columns)}")

    # --------------------------------------------------------
    # 4. Run A → E controlled ablation
    # --------------------------------------------------------

    print("\n[4/6] Running A to E controlled ablation...")
    print("  XGBoost config held constant across all experiments.")

    results: dict[str, dict] = {}

    for exp_name, feature_list in EXPERIMENTS.items():
        result = run_experiment(
            name=exp_name,
            features=feature_list,
            train_df=train_features,
            val_df=val_features,
            test_df=test_features,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            artifact_dir=artifact_dir,
        )
        results[exp_name] = result

    # --------------------------------------------------------
    # 5. Save results
    # --------------------------------------------------------

    print("\n[5/6] Saving ablation results...")

    with (artifact_dir / "ablation_results.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    with (artifact_dir / "feature_groups.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "statistical": STATISTICAL_FEATURES,
                "string": STRING_FEATURES,
                "tfidf": TFIDF_FEATURES,
                "semantic": SEMANTIC_FEATURES,
                "all": ALL_FEATURES,
                "experiments": EXPERIMENTS,
            },
            f,
            indent=2,
        )

    with (report_dir / "phase5_ablation_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    # --------------------------------------------------------
    # 6. Final comparison table
    # --------------------------------------------------------

    print("\n[6/6] Building comparison table...")

    rows = []
    for exp_name, result in results.items():
        v = result["validation"]
        t = result["test"]
        rows.append({
            "Exp": exp_name,
            "N_feat": result["feature_count"],
            "Val_P": v["precision"],
            "Val_R": v["recall"],
            "Val_F1": v["f1"],
            "Val_AUC": v["pr_auc"],
            "Test_P": t["precision"],
            "Test_R": t["recall"],
            "Test_F1": t["f1"],
            "Test_AUC": t["pr_auc"],
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        artifact_dir / "ablation_comparison.csv", index=False
    )

    print("\n" + "=" * 70)
    print("A to E ABLATION RESULTS")
    print("=" * 70)
    print(
        f"\n{'Exp':<5} {'Feats':>5} "
        f"{'Val-P':>7} {'Val-R':>7} {'Val-F1':>7} {'Val-AUC':>8} "
        f"{'Test-P':>7} {'Test-R':>7} {'Test-F1':>7} {'Test-AUC':>8}"
    )
    print("-" * 70)
    for _, row in comparison.iterrows():
        print(
            f"{row['Exp']:<5} {int(row['N_feat']):>5} "
            f"{row['Val_P']:>7.4f} {row['Val_R']:>7.4f} "
            f"{row['Val_F1']:>7.4f} {row['Val_AUC']:>8.4f} "
            f"{row['Test_P']:>7.4f} {row['Test_R']:>7.4f} "
            f"{row['Test_F1']:>7.4f} {row['Test_AUC']:>8.4f}"
        )

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)

    return results
