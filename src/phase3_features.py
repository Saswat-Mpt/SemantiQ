from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rapidfuzz import fuzz

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

DEFAULT_THRESHOLD = 0.50

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

PHASE3_FEATURES = [
    # Statistical
    "q1_char_length",
    "q2_char_length",
    "q1_word_count",
    "q2_word_count",
    "length_difference",
    "length_ratio",
    "common_word_count",
    "common_word_ratio",

    # Set similarity
    "jaccard_similarity",

    # Fuzzy similarity
    "fuzzy_ratio",
    "token_sort_ratio",
    "token_set_ratio",

    # Character similarity
    "char_bigram_jaccard",
    "char_trigram_jaccard",

    # Word n-gram overlap
    "word_bigram_overlap",
    "word_trigram_overlap",
]


# ============================================================
# Text preprocessing
# ============================================================

def clean_text(text: object) -> str:
    """
    Lightweight preprocessing for feature extraction.

    This is intentionally different from the Phase 1
    leakage-group normalization.

    We preserve meaningful words and only normalize:
    - case
    - whitespace
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
    """Simple whitespace tokenization."""

    return text.split()


# ============================================================
# Generic similarity utilities
# ============================================================

def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Safely compute a ratio."""

    if denominator == 0:
        return 0.0

    return numerator / denominator


def jaccard_similarity(
    tokens1: list[str],
    tokens2: list[str],
) -> float:
    """
    Jaccard similarity between two token sets.

        |A ∩ B|
    ----------------
        |A ∪ B|
    """

    set1 = set(tokens1)
    set2 = set(tokens2)

    union = set1 | set2

    if not union:
        return 0.0

    return len(set1 & set2) / len(union)


def ngram_set(
    tokens: list[str],
    n: int,
) -> set[tuple[str, ...]]:
    """Create word n-grams."""

    if len(tokens) < n:
        return set()

    return {
        tuple(tokens[i:i + n])
        for i in range(
            len(tokens) - n + 1
        )
    }


def ngram_overlap(
    tokens1: list[str],
    tokens2: list[str],
    n: int,
) -> float:
    """
    Symmetric overlap:

        |A ∩ B|
    ----------------
        min(|A|, |B|)

    This answers:
    "What fraction of the smaller question's n-grams
     occur in the other question?"
    """

    grams1 = ngram_set(tokens1, n)
    grams2 = ngram_set(tokens2, n)

    if not grams1 or not grams2:
        return 0.0

    denominator = min(
        len(grams1),
        len(grams2),
    )

    return len(
        grams1 & grams2
    ) / denominator


def character_ngrams(
    text: str,
    n: int,
) -> set[str]:
    """
    Character n-grams.

    Example:
        text = "hello", n = 3
        produces: {"hel", "ell", "llo"}
    """

    if len(text) < n:
        return set()

    return {
        text[i:i + n]
        for i in range(
            len(text) - n + 1
        )
    }


def character_jaccard(
    text1: str,
    text2: str,
    n: int,
) -> float:
    """Jaccard similarity over character n-grams."""

    grams1 = character_ngrams(text1, n)
    grams2 = character_ngrams(text2, n)

    union = grams1 | grams2

    if not union:
        return 0.0

    return len(grams1 & grams2) / len(union)


# ============================================================
# Pair-level feature extraction
# ============================================================

def extract_pair_features(
    question1: str,
    question2: str,
) -> dict[str, float]:
    """
    Extract all Phase 3 features for one question pair.
    """

    q1 = clean_text(question1)
    q2 = clean_text(question2)

    tokens1 = tokenize(q1)
    tokens2 = tokenize(q2)

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    q1_char_length = len(q1)
    q2_char_length = len(q2)

    q1_word_count = len(tokens1)
    q2_word_count = len(tokens2)

    length_difference = abs(
        q1_char_length - q2_char_length
    )

    length_ratio = safe_ratio(
        min(q1_char_length, q2_char_length),
        max(q1_char_length, q2_char_length),
    )

    set1 = set(tokens1)
    set2 = set(tokens2)

    common_word_count = len(set1 & set2)

    common_word_ratio = safe_ratio(
        common_word_count,
        len(set1 | set2),
    )

    # --------------------------------------------------------
    # Jaccard
    # --------------------------------------------------------

    jaccard = jaccard_similarity(tokens1, tokens2)

    # --------------------------------------------------------
    # Fuzzy similarity
    # --------------------------------------------------------

    fuzzy_ratio = fuzz.ratio(q1, q2) / 100.0

    token_sort_ratio = fuzz.token_sort_ratio(q1, q2) / 100.0

    token_set_ratio = fuzz.token_set_ratio(q1, q2) / 100.0

    # --------------------------------------------------------
    # Character similarity
    # --------------------------------------------------------

    char_bigram_jaccard = character_jaccard(q1, q2, n=2)

    char_trigram_jaccard = character_jaccard(q1, q2, n=3)

    # --------------------------------------------------------
    # Word n-gram overlap
    # --------------------------------------------------------

    word_bigram_overlap = ngram_overlap(tokens1, tokens2, n=2)

    word_trigram_overlap = ngram_overlap(tokens1, tokens2, n=3)

    return {
        "q1_char_length": float(q1_char_length),
        "q2_char_length": float(q2_char_length),
        "q1_word_count": float(q1_word_count),
        "q2_word_count": float(q2_word_count),
        "length_difference": float(length_difference),
        "length_ratio": float(length_ratio),
        "common_word_count": float(common_word_count),
        "common_word_ratio": float(common_word_ratio),
        "jaccard_similarity": float(jaccard),
        "fuzzy_ratio": float(fuzzy_ratio),
        "token_sort_ratio": float(token_sort_ratio),
        "token_set_ratio": float(token_set_ratio),
        "char_bigram_jaccard": float(char_bigram_jaccard),
        "char_trigram_jaccard": float(char_trigram_jaccard),
        "word_bigram_overlap": float(word_bigram_overlap),
        "word_trigram_overlap": float(word_trigram_overlap),
    }


# ============================================================
# Dataset-level feature extraction
# ============================================================

def build_phase3_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build Phase 3 feature matrix.

    The same deterministic transformation is applied to
    train, validation and test — no fitting step needed
    because all features are parameter-free.
    """

    records = []
    total = len(df)

    for index, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        features = extract_pair_features(
            row.question1,
            row.question2,
        )

        records.append(features)

        if total >= 10000 and index % 10000 == 0:
            print(
                f"  Processed {index:,}/{total:,} pairs..."
            )

    features_df = pd.DataFrame(
        records,
        index=df.index,
    )

    features_df = features_df[PHASE3_FEATURES]

    return features_df.astype(np.float32)


# ============================================================
# Model
# ============================================================

def build_logistic_regression() -> Pipeline:
    """
    Enhanced Logistic Regression.

    StandardScaler is used because the feature magnitudes
    differ substantially across the 16 features.
    """

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# Evaluation
# ============================================================

def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:

    return {
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
# Feature summary
# ============================================================

def build_feature_summary(
    features: pd.DataFrame,
) -> dict:
    """Create a compact summary of the feature matrix."""

    summary = {
        "num_rows": int(len(features)),
        "num_features": int(features.shape[1]),
        "features": list(features.columns),
        "missing_values": {
            col: int(features[col].isna().sum())
            for col in features.columns
        },
        "infinite_values": {
            col: int(np.isinf(features[col]).sum())
            for col in features.columns
        },
        "statistics": {},
    }

    for col in features.columns:
        summary["statistics"][col] = {
            "mean": float(features[col].mean()),
            "std": float(features[col].std()),
            "min": float(features[col].min()),
            "max": float(features[col].max()),
        }

    return summary


# ============================================================
# Main Phase 3 pipeline
# ============================================================

def run_phase3(
    processed_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict:

    print("=" * 70)
    print("SemantiQ — Phase 3: Classical Similarity Features")
    print("=" * 70)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load fixed Phase 1 partitions
    # --------------------------------------------------------

    print("\n[1/7] Loading fixed Phase 1 partitions...")

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df   = pd.read_csv(processed_dir / "val.csv")
    test_df  = pd.read_csv(processed_dir / "test.csv")

    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")

    # --------------------------------------------------------
    # 2. Build features
    # --------------------------------------------------------

    print("\n[2/7] Building Phase 3 features...")

    print("\nTrain features:")
    x_train = build_phase3_features(train_df)

    print("\nValidation features:")
    x_val = build_phase3_features(val_df)

    print("\nTest features:")
    x_test = build_phase3_features(test_df)

    # --------------------------------------------------------
    # 3. Validate feature matrices
    # --------------------------------------------------------

    print("\n[3/7] Validating feature matrices...")

    if list(x_train.columns) != list(x_val.columns):
        raise RuntimeError(
            "Train and validation feature columns differ."
        )

    if list(x_train.columns) != list(x_test.columns):
        raise RuntimeError(
            "Train and test feature columns differ."
        )

    for name, mat in [
        ("train", x_train),
        ("validation", x_val),
        ("test", x_test),
    ]:
        if mat.isna().any().any():
            raise RuntimeError(
                f"NaN values detected in {name} features."
            )
        if np.isinf(mat.to_numpy()).any():
            raise RuntimeError(
                f"Infinite values detected in {name} features."
            )

    print(f"Feature count : {x_train.shape[1]}")
    print(f"Feature names : {list(x_train.columns)}")
    print("Feature validation: PASSED")

    # --------------------------------------------------------
    # 4. Save feature metadata
    # --------------------------------------------------------

    print("\n[4/7] Saving feature metadata...")

    with (artifact_dir / "feature_columns.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(PHASE3_FEATURES, f, indent=2)

    feature_summary = build_feature_summary(x_train)

    with (artifact_dir / "phase3_feature_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(feature_summary, f, indent=2)

    # Save feature matrices — loaded by Phase 5 ablation.
    x_train.to_csv(artifact_dir / "train_phase3_features.csv", index=False)
    x_val.to_csv(artifact_dir / "val_phase3_features.csv",   index=False)
    x_test.to_csv(artifact_dir / "test_phase3_features.csv",  index=False)
    print("  Feature CSVs saved (train / val / test).")


    # --------------------------------------------------------
    # 5. Train enhanced Logistic Regression
    # --------------------------------------------------------

    print("\n[5/7] Training enhanced Logistic Regression...")

    y_train = train_df["is_duplicate"].astype(int)
    y_val   = val_df["is_duplicate"].astype(int)
    y_test  = test_df["is_duplicate"].astype(int)

    model = build_logistic_regression()
    model.fit(x_train, y_train)

    val_scores  = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]

    val_pred  = (val_scores  >= DEFAULT_THRESHOLD).astype(int)
    test_pred = (test_scores >= DEFAULT_THRESHOLD).astype(int)

    val_metrics  = calculate_metrics(y_val,  val_pred,  val_scores)
    test_metrics = calculate_metrics(y_test, test_pred, test_scores)

    # --------------------------------------------------------
    # 6. Save model and results
    # --------------------------------------------------------

    print("\n[6/7] Saving model and results...")

    joblib.dump(
        model,
        artifact_dir / "enhanced_logistic_regression.joblib",
    )

    metrics = {
        "model": "logistic_regression",
        "feature_set": "phase3_classical_similarity",
        "num_features": len(PHASE3_FEATURES),
        "threshold": DEFAULT_THRESHOLD,
        "features": PHASE3_FEATURES,
        "validation": val_metrics,
        "test": test_metrics,
    }

    with (artifact_dir / "phase3_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    with (report_dir / "phase3_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)

    # --------------------------------------------------------
    # 7. Console summary — Phase 2 vs Phase 3
    # --------------------------------------------------------

    print("\n[7/7] Building comparison summary...")

    # Load Phase 2 LR test metrics for comparison.
    p2_metrics_path = (
        artifact_dir.parent / "baselines" / "logistic_regression_metrics.json"
    )

    p2_f1    = "—"
    p2_prauc = "—"

    if p2_metrics_path.exists():
        with p2_metrics_path.open(encoding="utf-8") as f:
            p2 = json.load(f)
        p2_f1    = f"{p2['test_metrics']['f1']:.4f}"
        p2_prauc = f"{p2['test_metrics']['pr_auc']:.4f}"

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<30} {'Features':>8} {'F1 (Test)':>10} {'PR-AUC (Test)':>14}")
    print("-" * 65)
    print(f"{'Phase 2 LR (statistical)':<30} {'8':>8} {p2_f1:>10} {p2_prauc:>14}")
    print(
        f"{'Phase 3 LR (+ lexical/string)':<30} "
        f"{'16':>8} "
        f"{test_metrics['f1']:>10.4f} "
        f"{test_metrics['pr_auc']:>14.4f}"
    )

    print(f"\nPhase 3 — Validation")
    print(f"  Precision : {val_metrics['precision']:.4f}")
    print(f"  Recall    : {val_metrics['recall']:.4f}")
    print(f"  F1        : {val_metrics['f1']:.4f}")
    print(f"  PR-AUC    : {val_metrics['pr_auc']:.4f}")

    print(f"\nPhase 3 — Test")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"  PR-AUC    : {test_metrics['pr_auc']:.4f}")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)

    return metrics
