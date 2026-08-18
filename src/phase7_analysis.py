from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

FINAL_FEATURES = [
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


# ============================================================
# Loading
# ============================================================

def load_test_data(
    processed_dir: Path,
) -> pd.DataFrame:

    return pd.read_csv(
        processed_dir / "test.csv"
    )


def load_test_features(
    phase3_dir: Path,
    phase4_dir: Path,
) -> pd.DataFrame:

    phase3 = pd.read_csv(
        phase3_dir
        / "test_phase3_features.csv"
    )

    phase4 = pd.read_csv(
        phase4_dir
        / "test_phase4_features.csv"
    )

    features = pd.concat(
        [
            phase3.reset_index(drop=True),
            phase4.reset_index(drop=True),
        ],
        axis=1,
    )

    return features[
        FINAL_FEATURES
    ]


def load_phase6_predictions(
    phase6_dir: Path,
) -> pd.DataFrame:

    path = (
        phase6_dir
        / "test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Phase 6 predictions not found:\n{path}"
        )

    return pd.read_csv(path)


# ============================================================
# Error classification
# ============================================================

def classify_prediction_errors(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result["error_type"] = "correct"

    result.loc[
        (
            result["y_true"] == 0
        )
        & (
            result["prediction"] == 1
        ),
        "error_type",
    ] = "false_positive"

    result.loc[
        (
            result["y_true"] == 1
        )
        & (
            result["prediction"] == 0
        ),
        "error_type",
    ] = "false_negative"

    result.loc[
        (
            result["y_true"] == 1
        )
        & (
            result["prediction"] == 1
        ),
        "error_type",
    ] = "true_positive"

    result.loc[
        (
            result["y_true"] == 0
        )
        & (
            result["prediction"] == 0
        ),
        "error_type",
    ] = "true_negative"

    return result


# ============================================================
# Hard-case discovery
# ============================================================

def find_high_lexical_low_semantic(
    data: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:

    result = data.copy()

    # High lexical similarity:
    # use the maximum of word/character TF-IDF.
    result["lexical_score"] = (
        result[
            [
                "word_tfidf_cosine",
                "char_tfidf_cosine",
                "jaccard_similarity",
            ]
        ]
        .mean(axis=1)
    )

    # Semantic score.
    result["semantic_score"] = (
        result["minilm_cosine"]
    )

    # High lexical + low semantic.
    result["hard_case_score"] = (
        result["lexical_score"]
        - result["semantic_score"]
    )

    return (
        result.sort_values(
            "hard_case_score",
            ascending=False,
        )
        .head(n)
    )


def find_low_lexical_high_semantic(
    data: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:

    result = data.copy()

    result["lexical_score"] = (
        result[
            [
                "word_tfidf_cosine",
                "char_tfidf_cosine",
                "jaccard_similarity",
            ]
        ]
        .mean(axis=1)
    )

    result["semantic_score"] = (
        result["minilm_cosine"]
    )

    result["hard_case_score"] = (
        result["semantic_score"]
        - result["lexical_score"]
    )

    return (
        result.sort_values(
            "hard_case_score",
            ascending=False,
        )
        .head(n)
    )


# ============================================================
# Error analysis
# ============================================================

def analyze_false_positives(
    data: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:

    fp = data[
        data["error_type"]
        == "false_positive"
    ].copy()

    if fp.empty:
        return fp

    fp["error_score"] = (
        fp["score"]
    )

    return (
        fp.sort_values(
            "error_score",
            ascending=False,
        )
        .head(n)
    )


def analyze_false_negatives(
    data: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:

    fn = data[
        data["error_type"]
        == "false_negative"
    ].copy()

    if fn.empty:
        return fn

    fn["error_score"] = (
        fn["score"]
    )

    return (
        fn.sort_values(
            "error_score",
            ascending=False,
        )
        .head(n)
    )


# ============================================================
# Final summary
# ============================================================

def build_error_summary(
    data: pd.DataFrame,
) -> dict:

    counts = (
        data["error_type"]
        .value_counts()
        .to_dict()
    )

    total = len(data)

    summary = {
        "total_examples": int(total),
        "true_positive": int(
            counts.get(
                "true_positive",
                0,
            )
        ),
        "true_negative": int(
            counts.get(
                "true_negative",
                0,
            )
        ),
        "false_positive": int(
            counts.get(
                "false_positive",
                0,
            )
        ),
        "false_negative": int(
            counts.get(
                "false_negative",
                0,
            )
        ),
    }

    if total > 0:
        summary["error_rate"] = float(
            (
                summary["false_positive"]
                + summary["false_negative"]
            )
            / total
        )

    return summary


# ============================================================
# Feature importance
# ============================================================

def extract_feature_importance(
    model_path: Path,
    output_path: Path,
) -> pd.DataFrame:

    import joblib

    model = joblib.load(
        model_path
    )

    if not hasattr(
        model,
        "feature_importances_",
    ):
        raise RuntimeError(
            "Loaded model does not expose "
            "feature_importances_."
        )

    importances = (
        model.feature_importances_
    )

    if len(importances) != len(
        FINAL_FEATURES
    ):
        raise RuntimeError(
            "Feature importance count does not "
            "match final feature count."
        )

    result = pd.DataFrame(
        {
            "feature": FINAL_FEATURES,
            "importance": importances,
        }
    )

    result = (
        result.sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result.to_csv(
        output_path,
        index=False,
    )

    return result


# ============================================================
# Main
# ============================================================

def run_phase7(
    processed_dir: Path,
    phase3_dir: Path,
    phase4_dir: Path,
    phase5_dir: Path,
    phase6_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict:

    print("=" * 70)
    print(
        "SemantiQ — Phase 7: "
        "Final Evaluation + Error Analysis"
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
    # 1. Load
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading final test data..."
    )

    test_df = load_test_data(
        processed_dir
    )

    features = load_test_features(
        phase3_dir,
        phase4_dir,
    )

    predictions = load_phase6_predictions(
        phase6_dir
    )

    if len(test_df) != len(
        features
    ):
        raise RuntimeError(
            "Test data and feature matrix lengths differ."
        )

    if len(test_df) != len(
        predictions
    ):
        raise RuntimeError(
            "Test data and Phase 6 predictions "
            "have different lengths."
        )

    # --------------------------------------------------------
    # 2. Create analysis dataframe
    # --------------------------------------------------------

    print(
        "\n[2/6] Building final analysis dataframe..."
    )

    analysis = pd.DataFrame(
        {
            "question1": (
                test_df["question1"]
            ),
            "question2": (
                test_df["question2"]
            ),
            "y_true": (
                predictions["y_true"]
                .astype(int)
            ),
            "score": (
                predictions["score"]
            ),
            "prediction": (
                predictions["prediction"]
                .astype(int)
            ),
        }
    )

    analysis = pd.concat(
        [
            analysis.reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )

    analysis = classify_prediction_errors(
        analysis
    )

    # --------------------------------------------------------
    # 3. Error analysis
    # --------------------------------------------------------

    print(
        "\n[3/6] Performing error analysis..."
    )

    error_summary = (
        build_error_summary(
            analysis
        )
    )

    print(
        "\nError summary:"
    )

    for key, value in (
        error_summary.items()
    ):
        print(
            f"  {key}: {value}"
        )

    false_positives = (
        analyze_false_positives(
            analysis
        )
    )

    false_negatives = (
        analyze_false_negatives(
            analysis
        )
    )

    high_lexical_low_semantic = (
        find_high_lexical_low_semantic(
            analysis
        )
    )

    low_lexical_high_semantic = (
        find_low_lexical_high_semantic(
            analysis
        )
    )

    # --------------------------------------------------------
    # 4. Save hard cases
    # --------------------------------------------------------

    print(
        "\n[4/6] Saving hard-case datasets..."
    )

    false_positives.to_csv(
        artifact_dir
        / "false_positives.csv",
        index=False,
    )

    false_negatives.to_csv(
        artifact_dir
        / "false_negatives.csv",
        index=False,
    )

    high_lexical_low_semantic.to_csv(
        artifact_dir
        / "high_lexical_low_semantic.csv",
        index=False,
    )

    low_lexical_high_semantic.to_csv(
        artifact_dir
        / "low_lexical_high_semantic.csv",
        index=False,
    )

    analysis.to_csv(
        artifact_dir
        / "final_test_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Feature importance
    # --------------------------------------------------------

    print(
        "\n[5/6] Extracting final feature importance..."
    )

    model_path = (
        phase5_dir
        / "xgboost_experiment_E.joblib"
    )

    feature_importance = (
        extract_feature_importance(
            model_path,
            artifact_dir
            / "feature_importance.csv",
        )
    )

    print(
        "\nTop features:"
    )

    print(
        feature_importance.head(
            10
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 6. Save report
    # --------------------------------------------------------

    print(
        "\n[6/6] Saving final report..."
    )

    report = {
        "phase": "phase7",
        "final_model": (
            "Experiment E"
        ),
        "feature_count": len(
            FINAL_FEATURES
        ),
        "features": FINAL_FEATURES,
        "error_summary": error_summary,
        "artifacts": {
            "false_positives":
                "false_positives.csv",
            "false_negatives":
                "false_negatives.csv",
            "high_lexical_low_semantic":
                "high_lexical_low_semantic.csv",
            "low_lexical_high_semantic":
                "low_lexical_high_semantic.csv",
            "feature_importance":
                "feature_importance.csv",
            "final_test_analysis":
                "final_test_analysis.csv",
        },
    }

    with (
        artifact_dir
        / "phase7_results.json"
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
        / "phase7_final_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 7 COMPLETE"
    )

    print(
        "=" * 70
    )

    return report
