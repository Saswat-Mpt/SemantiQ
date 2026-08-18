from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

DEFAULT_THRESHOLD = 0.50

PRECISION_TARGET = 0.90

BASELINE_FEATURES = [
    "q1_char_length",
    "q2_char_length",
    "q1_word_count",
    "q2_word_count",
    "length_difference",
    "length_ratio",
    "common_word_count",
    "common_word_ratio",
]


# ============================================================
# Text utilities
# ============================================================

def clean_text(text: object) -> str:
    """
    Lightweight normalization for baseline features.

    This is NOT the leakage-group normalization.
    It is the preprocessing used for feature extraction.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(text: str) -> list[str]:
    """Simple whitespace/token-based tokenizer."""

    return text.split()


# ============================================================
# Baseline feature engineering
# ============================================================

def build_baseline_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the small statistical feature set for the
    Logistic Regression baseline.
    """

    result = pd.DataFrame(
        index=df.index
    )

    q1 = (
        df["question1"]
        .fillna("")
        .map(clean_text)
    )

    q2 = (
        df["question2"]
        .fillna("")
        .map(clean_text)
    )

    q1_tokens = q1.map(tokenize)
    q2_tokens = q2.map(tokenize)

    q1_lengths = q1.str.len()
    q2_lengths = q2.str.len()

    q1_word_counts = q1_tokens.str.len()
    q2_word_counts = q2_tokens.str.len()

    common_word_counts = []

    for tokens1, tokens2 in zip(
        q1_tokens,
        q2_tokens,
    ):
        words1 = set(tokens1)
        words2 = set(tokens2)

        common_word_counts.append(
            len(words1 & words2)
        )

    common_word_counts = pd.Series(
        common_word_counts,
        index=df.index,
        dtype=float,
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    result["q1_char_length"] = q1_lengths
    result["q2_char_length"] = q2_lengths

    result["q1_word_count"] = q1_word_counts
    result["q2_word_count"] = q2_word_counts

    result["length_difference"] = (
        (q1_lengths - q2_lengths)
        .abs()
    )

    # Avoid division by zero.
    result["length_ratio"] = (
        np.minimum(
            q1_lengths,
            q2_lengths,
        )
        /
        np.maximum(
            q1_lengths,
            q2_lengths,
        ).replace(0, 1)
    )

    result["common_word_count"] = (
        common_word_counts
    )

    # Jaccard denominator is deliberately NOT used here.
    # Jaccard belongs to Phase 3.
    denominator = (
        q1_word_counts
        + q2_word_counts
        - common_word_counts
    )

    result["common_word_ratio"] = (
        common_word_counts
        /
        denominator.replace(0, 1)
    )

    result = result[
        BASELINE_FEATURES
    ]

    return result.astype(float)


# ============================================================
# Naive heuristic
# ============================================================

def heuristic_predictions(
    common_word_ratio: pd.Series,
    threshold: float,
) -> np.ndarray:
    """Convert common-word ratio into binary predictions."""

    return (
        common_word_ratio.to_numpy()
        >= threshold
    ).astype(int)


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                y_pred,
            ).tolist()
        ),
    }

    if y_score is not None:
        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                y_score,
            )
        )

    return metrics


def select_heuristic_threshold(
    y_true: pd.Series,
    common_word_ratio: pd.Series,
    precision_target: float = PRECISION_TARGET,
) -> tuple[float, dict]:
    """
    Find the LOWEST threshold that reaches the required
    validation precision.

    This threshold is selected using validation data only.
    """

    candidate_thresholds = np.unique(
        np.round(
            common_word_ratio.to_numpy(),
            6,
        )
    )

    candidate_thresholds = np.concatenate(
        [
            np.array([0.0]),
            candidate_thresholds,
            np.array([1.0]),
        ]
    )

    best_threshold = None
    best_metrics = None

    for threshold in candidate_thresholds:

        predictions = heuristic_predictions(
            common_word_ratio,
            threshold,
        )

        metrics = calculate_metrics(
            y_true,
            predictions,
            common_word_ratio.to_numpy(),
        )

        if (
            metrics["precision"]
            >= precision_target
        ):
            best_threshold = float(
                threshold
            )
            best_metrics = metrics
            break

    if best_threshold is None:
        raise RuntimeError(
            f"The naive heuristic could not reach "
            f"{precision_target:.0%} precision on validation data."
        )

    return (
        best_threshold,
        best_metrics,
    )


# ============================================================
# Logistic Regression
# ============================================================

def build_logistic_regression() -> Pipeline:
    """
    Build the baseline Logistic Regression pipeline.

    Standardization is useful because the handcrafted features
    operate on different numeric scales.
    """

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:

    model = build_logistic_regression()

    model.fit(
        x_train,
        y_train,
    )

    return model


# ============================================================
# Main pipeline
# ============================================================

def run_phase2(
    processed_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict:

    print("=" * 70)
    print("SemantiQ — Phase 2: Baselines")
    print("=" * 70)

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load fixed Phase 1 splits
    # --------------------------------------------------------

    print("\n[1/7] Loading fixed Phase 1 splits...")

    train_df = pd.read_csv(
        processed_dir / "train.csv"
    )

    val_df = pd.read_csv(
        processed_dir / "val.csv"
    )

    test_df = pd.read_csv(
        processed_dir / "test.csv"
    )

    print(
        f"Train:      {len(train_df):,}"
    )

    print(
        f"Validation: {len(val_df):,}"
    )

    print(
        f"Test:       {len(test_df):,}"
    )

    # --------------------------------------------------------
    # 2. Build baseline features
    # --------------------------------------------------------

    print(
        "\n[2/7] Building baseline statistical features..."
    )

    x_train = build_baseline_features(
        train_df
    )

    x_val = build_baseline_features(
        val_df
    )

    x_test = build_baseline_features(
        test_df
    )

    y_train = train_df[
        "is_duplicate"
    ].astype(int)

    y_val = val_df[
        "is_duplicate"
    ].astype(int)

    y_test = test_df[
        "is_duplicate"
    ].astype(int)

    print(
        f"Features: {list(x_train.columns)}"
    )

    # --------------------------------------------------------
    # 3. Naive heuristic — threshold selected on validation
    # --------------------------------------------------------

    print(
        "\n[3/7] Selecting naive heuristic threshold on validation..."
    )

    heuristic_threshold, heuristic_val_metrics = (
        select_heuristic_threshold(
            y_true=y_val,
            common_word_ratio=x_val[
                "common_word_ratio"
            ],
            precision_target=PRECISION_TARGET,
        )
    )

    print(
        f"Selected threshold (validation): "
        f"{heuristic_threshold:.6f}"
    )

    print(
        f"Validation precision : "
        f"{heuristic_val_metrics['precision']:.4f}"
    )

    print(
        f"Validation recall    : "
        f"{heuristic_val_metrics['recall']:.4f}"
    )

    print(
        f"Validation F1        : "
        f"{heuristic_val_metrics['f1']:.4f}"
    )

    # Final heuristic test evaluation — called exactly once.
    heuristic_test_pred = heuristic_predictions(
        x_test["common_word_ratio"],
        heuristic_threshold,
    )

    heuristic_test_metrics = calculate_metrics(
        y_test,
        heuristic_test_pred,
        x_test["common_word_ratio"].to_numpy(),
    )

    # --------------------------------------------------------
    # 4. Save heuristic result
    # --------------------------------------------------------

    print(
        "\n[4/7] Saving naive baseline artifact..."
    )

    heuristic_report = {
        "method": "common_word_ratio_threshold",
        "precision_target": PRECISION_TARGET,
        "selected_threshold": heuristic_threshold,
        "validation_metrics": heuristic_val_metrics,
        "test_metrics": heuristic_test_metrics,
    }

    with (
        artifact_dir / "naive_baseline.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(heuristic_report, f, indent=2)

    # --------------------------------------------------------
    # 5. Logistic Regression — trained on train, default 0.50
    # --------------------------------------------------------

    print(
        "\n[5/7] Training Logistic Regression baseline..."
    )

    model = train_logistic_regression(
        x_train,
        y_train,
    )

    val_scores = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]

    val_pred = (val_scores >= DEFAULT_THRESHOLD).astype(int)
    test_pred = (test_scores >= DEFAULT_THRESHOLD).astype(int)

    logistic_val_metrics = calculate_metrics(
        y_val, val_pred, val_scores,
    )

    logistic_test_metrics = calculate_metrics(
        y_test, test_pred, test_scores,
    )

    # --------------------------------------------------------
    # 6. Save Logistic Regression
    # --------------------------------------------------------

    print(
        "\n[6/7] Saving Logistic Regression artifact..."
    )

    joblib.dump(
        model,
        artifact_dir / "logistic_regression.joblib",
    )

    logistic_report = {
        "method": "logistic_regression",
        "threshold": DEFAULT_THRESHOLD,
        "features": BASELINE_FEATURES,
        "validation_metrics": logistic_val_metrics,
        "test_metrics": logistic_test_metrics,
    }

    with (
        artifact_dir / "logistic_regression_metrics.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(logistic_report, f, indent=2)

    # --------------------------------------------------------
    # 7. Comparison + reports
    # --------------------------------------------------------

    print(
        "\n[7/7] Building baseline comparison report..."
    )

    comparison = {
        "naive_heuristic": {
            "validation": heuristic_val_metrics,
            "test": heuristic_test_metrics,
        },
        "logistic_regression": {
            "validation": logistic_val_metrics,
            "test": logistic_test_metrics,
        },
    }

    with (
        artifact_dir / "baseline_comparison.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with (
        report_dir / "phase2_baseline_report.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)

    print(f"\nNaive Heuristic (threshold={heuristic_threshold:.6f})")
    print(f"  {'':12s}  {'Validation':>10}  {'Test':>10}")
    print(f"  {'Precision':12s}  {heuristic_val_metrics['precision']:>10.4f}  {heuristic_test_metrics['precision']:>10.4f}")
    print(f"  {'Recall':12s}  {heuristic_val_metrics['recall']:>10.4f}  {heuristic_test_metrics['recall']:>10.4f}")
    print(f"  {'F1':12s}  {heuristic_val_metrics['f1']:>10.4f}  {heuristic_test_metrics['f1']:>10.4f}")
    print(f"  {'PR-AUC':12s}  {heuristic_val_metrics['pr_auc']:>10.4f}  {heuristic_test_metrics['pr_auc']:>10.4f}")

    print(f"\nLogistic Regression (threshold={DEFAULT_THRESHOLD})")
    print(f"  {'':12s}  {'Validation':>10}  {'Test':>10}")
    print(f"  {'Precision':12s}  {logistic_val_metrics['precision']:>10.4f}  {logistic_test_metrics['precision']:>10.4f}")
    print(f"  {'Recall':12s}  {logistic_val_metrics['recall']:>10.4f}  {logistic_test_metrics['recall']:>10.4f}")
    print(f"  {'F1':12s}  {logistic_val_metrics['f1']:>10.4f}  {logistic_test_metrics['f1']:>10.4f}")
    print(f"  {'PR-AUC':12s}  {logistic_val_metrics['pr_auc']:>10.4f}  {logistic_test_metrics['pr_auc']:>10.4f}")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    return comparison
