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


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

PRECISION_TARGET = 0.90

BASELINE_THRESHOLD = 0.50

FINAL_EXPERIMENT = "E"


# ============================================================
# Feature groups
# ============================================================

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


# ============================================================
# Data loading
# ============================================================

def load_fixed_partitions(
    processed_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load the exact Phase 1 partitions."""

    train_df = pd.read_csv(
        processed_dir / "train.csv"
    )

    val_df = pd.read_csv(
        processed_dir / "val.csv"
    )

    test_df = pd.read_csv(
        processed_dir / "test.csv"
    )

    return (
        train_df,
        val_df,
        test_df,
    )


def load_feature_matrices(
    phase3_dir: Path,
    phase4_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load and combine the already-created Phase 3 and
    Phase 4 feature matrices.

    No features are recomputed.
    """

    phase3_train = pd.read_csv(
        phase3_dir
        / "train_phase3_features.csv"
    )

    phase3_val = pd.read_csv(
        phase3_dir
        / "val_phase3_features.csv"
    )

    phase3_test = pd.read_csv(
        phase3_dir
        / "test_phase3_features.csv"
    )

    phase4_train = pd.read_csv(
        phase4_dir
        / "train_phase4_features.csv"
    )

    phase4_val = pd.read_csv(
        phase4_dir
        / "val_phase4_features.csv"
    )

    phase4_test = pd.read_csv(
        phase4_dir
        / "test_phase4_features.csv"
    )

    train_features = pd.concat(
        [
            phase3_train.reset_index(
                drop=True
            ),
            phase4_train.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    val_features = pd.concat(
        [
            phase3_val.reset_index(
                drop=True
            ),
            phase4_val.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    test_features = pd.concat(
        [
            phase3_test.reset_index(
                drop=True
            ),
            phase4_test.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    return (
        train_features,
        val_features,
        test_features,
    )


# ============================================================
# Validation
# ============================================================

def validate_features(
    features: pd.DataFrame,
    split_name: str,
) -> None:

    missing = [
        feature
        for feature in FINAL_FEATURES
        if feature not in features.columns
    ]

    if missing:
        raise RuntimeError(
            f"{split_name}: missing features: "
            f"{missing}"
        )

    selected = features[
        FINAL_FEATURES
    ]

    if selected.isna().any().any():
        raise RuntimeError(
            f"{split_name}: NaN values detected."
        )

    if np.isinf(
        selected.to_numpy()
    ).any():
        raise RuntimeError(
            f"{split_name}: infinite values detected."
        )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:

    predictions = (
        scores >= threshold
    ).astype(int)

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "positive_predictions": int(
            predictions.sum()
        ),
        "total_predictions": int(
            len(predictions)
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                predictions,
            ).tolist()
        ),
    }


# ============================================================
# Threshold search
# ============================================================

def search_precision_constrained_threshold(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    precision_target: float,
) -> tuple[float, dict, pd.DataFrame]:
    """
    Find the threshold that maximizes recall while satisfying
    the required precision target.

    Selection happens ONLY on validation data.

    Tie-breaking:
        1. Highest recall
        2. Highest precision
        3. Lowest threshold
    """

    scores = np.asarray(
        scores,
        dtype=float,
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    # --------------------------------------------------------
    # Candidate thresholds
    # --------------------------------------------------------

    unique_scores = np.unique(
        scores
    )

    candidate_thresholds = np.sort(
        unique_scores
    )[::-1]

    records = []

    best = None

    for threshold in candidate_thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        true_positive = int(
            np.sum(
                (predictions == 1)
                & (y_true == 1)
            )
        )

        false_positive = int(
            np.sum(
                (predictions == 1)
                & (y_true == 0)
            )
        )

        false_negative = int(
            np.sum(
                (predictions == 0)
                & (y_true == 1)
            )
        )

        predicted_positive = (
            true_positive
            + false_positive
        )

        precision = (
            true_positive
            / predicted_positive
            if predicted_positive > 0
            else 0.0
        )

        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            )
            if (
                true_positive
                + false_negative
            ) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (
                precision + recall
            ) > 0
            else 0.0
        )

        record = {
            "threshold": float(
                threshold
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "predicted_positive": predicted_positive,
        }

        records.append(
            record
        )

        # ----------------------------------------------------
        # Precision constraint
        # ----------------------------------------------------

        if precision < precision_target:
            continue

        # ----------------------------------------------------
        # Select best feasible threshold
        # ----------------------------------------------------

        if best is None:

            best = record

        else:

            current_key = (
                record["recall"],
                record["precision"],
                -record["threshold"],
            )

            best_key = (
                best["recall"],
                best["precision"],
                -best["threshold"],
            )

            if current_key > best_key:
                best = record

    threshold_table = pd.DataFrame(
        records
    )

    if best is None:
        raise RuntimeError(
            "\nNo validation threshold satisfies "
            f"the required precision target of "
            f"{precision_target:.2%}.\n"
            "The model cannot currently satisfy the "
            "Phase 6 operating constraint."
        )

    return (
        float(best["threshold"]),
        best,
        threshold_table,
    )


# ============================================================
# Main Phase 6
# ============================================================

def run_phase6(
    processed_dir: Path,
    phase3_dir: Path,
    phase4_dir: Path,
    phase5_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict:

    print("=" * 70)
    print(
        "SemantiQ — Phase 6: "
        "Precision-Constrained Decision System"
    )
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
    # 1. Load partitions
    # --------------------------------------------------------

    print(
        "\n[1/7] Loading fixed Phase 1 partitions..."
    )

    (
        train_df,
        val_df,
        test_df,
    ) = load_fixed_partitions(
        processed_dir
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
    # 2. Load features
    # --------------------------------------------------------

    print(
        "\n[2/7] Loading Phase 3 + Phase 4 features..."
    )

    (
        train_features,
        val_features,
        test_features,
    ) = load_feature_matrices(
        phase3_dir,
        phase4_dir,
    )

    validate_features(
        train_features,
        "train",
    )

    validate_features(
        val_features,
        "validation",
    )

    validate_features(
        test_features,
        "test",
    )

    # --------------------------------------------------------
    # 3. Load final E model
    # --------------------------------------------------------

    print(
        "\n[3/7] Loading Experiment E model..."
    )

    model_path = (
        phase5_dir
        / "xgboost_experiment_E.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Final Experiment E model not found:\n"
            f"{model_path}"
        )

    model = joblib.load(
        model_path
    )

    print(
        f"Loaded: {model_path}"
    )

    # --------------------------------------------------------
    # 4. Generate validation scores
    # --------------------------------------------------------

    print(
        "\n[4/7] Generating validation probabilities..."
    )

    x_val = val_features[
        FINAL_FEATURES
    ]

    x_test = test_features[
        FINAL_FEATURES
    ]

    y_val = val_df[
        "is_duplicate"
    ].astype(int)

    y_test = test_df[
        "is_duplicate"
    ].astype(int)

    validation_scores = (
        model.predict_proba(
            x_val
        )[:, 1]
    )

    print(
        "Validation probability generation complete."
    )

    # --------------------------------------------------------
    # 5. Select threshold on validation ONLY
    # --------------------------------------------------------

    print(
        "\n[5/7] Searching for precision-constrained threshold..."
    )

    (
        selected_threshold,
        selected_record,
        threshold_table,
    ) = search_precision_constrained_threshold(
        y_true=y_val,
        scores=validation_scores,
        precision_target=PRECISION_TARGET,
    )

    print(
        "\nSelected threshold:"
    )

    print(
        f"  T* = {selected_threshold:.8f}"
    )

    print(
        "\nValidation performance at T*:"
    )

    print(
        f"  Precision : "
        f"{selected_record['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{selected_record['recall']:.4f}"
    )

    print(
        f"  F1        : "
        f"{selected_record['f1']:.4f}"
    )

    print(
        f"  TP        : "
        f"{selected_record['true_positive']}"
    )

    print(
        f"  FP        : "
        f"{selected_record['false_positive']}"
    )

    print(
        f"  FN        : "
        f"{selected_record['false_negative']}"
    )

    # --------------------------------------------------------
    # 6. Freeze threshold and evaluate test
    # --------------------------------------------------------

    print(
        "\n[6/7] FREEZING threshold and evaluating test..."
    )

    # IMPORTANT:
    # No threshold search occurs here.

    test_scores = (
        model.predict_proba(
            x_test
        )[:, 1]
    )

    test_metrics = calculate_metrics(
        y_true=y_test,
        scores=test_scores,
        threshold=selected_threshold,
    )

    baseline_test_metrics = calculate_metrics(
        y_true=y_test,
        scores=test_scores,
        threshold=BASELINE_THRESHOLD,
    )

    print(
        "\nFinal test performance at frozen T*:"
    )

    print(
        f"  Threshold : "
        f"{selected_threshold:.8f}"
    )

    print(
        f"  Precision : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"  Recall    : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"  F1        : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"  PR-AUC    : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    # --------------------------------------------------------
    # 7. Save everything
    # --------------------------------------------------------

    print(
        "\n[7/7] Saving Phase 6 artifacts..."
    )

    threshold_table.to_csv(
        artifact_dir
        / "validation_threshold_search.csv",
        index=False,
    )

    # Save validation scores.
    pd.DataFrame(
        {
            "y_true": y_val,
            "score": validation_scores,
        }
    ).to_csv(
        artifact_dir
        / "validation_scores.csv",
        index=False,
    )

    # Save test scores.
    pd.DataFrame(
        {
            "y_true": y_test,
            "score": test_scores,
            "prediction": (
                test_scores
                >= selected_threshold
            ).astype(int),
        }
    ).to_csv(
        artifact_dir
        / "test_predictions.csv",
        index=False,
    )

    report = {
        "phase": "phase6",
        "final_model": "Experiment E",
        "precision_target": PRECISION_TARGET,
        "threshold_selection": {
            "data": "validation_only",
            "selected_threshold": selected_threshold,
            "validation_precision": selected_record[
                "precision"
            ],
            "validation_recall": selected_record[
                "recall"
            ],
            "validation_f1": selected_record[
                "f1"
            ],
            "true_positive": selected_record[
                "true_positive"
            ],
            "false_positive": selected_record[
                "false_positive"
            ],
            "false_negative": selected_record[
                "false_negative"
            ],
        },
        "baseline_threshold_0_50": {
            "test": baseline_test_metrics,
        },
        "frozen_threshold_test": {
            "test": test_metrics,
        },
        "features": FINAL_FEATURES,
        "data_partitions": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
    }

    with (
        artifact_dir
        / "selected_threshold.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report[
                "threshold_selection"
            ],
            file,
            indent=2,
        )

    with (
        artifact_dir
        / "phase6_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    with (
        report_dir
        / "phase6_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 6 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nFinal operating point:"
    )

    print(
        f"  Precision target : "
        f"{PRECISION_TARGET:.0%}"
    )

    print(
        f"  Frozen threshold : "
        f"{selected_threshold:.8f}"
    )

    print(
        f"  Test precision   : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"  Test recall      : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"  Test F1          : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"  Test PR-AUC      : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    return report
