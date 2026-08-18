from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

# Word-level TF-IDF — plan spec: word analyzer, (1,2) n-grams
WORD_NGRAM_RANGE = (1, 2)
WORD_MIN_DF = 2
WORD_MAX_FEATURES = 100_000

# Character-level TF-IDF — plan spec: char_wb, (3,5) n-grams
CHAR_ANALYZER = "char_wb"
CHAR_NGRAM_RANGE = (3, 5)
CHAR_MIN_DF = 2
CHAR_MAX_FEATURES = 100_000

# Pretrained MiniLM — inference only, no fine-tuning
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 128

PHASE4_FEATURES = [
    "word_tfidf_cosine",
    "char_tfidf_cosine",
    "minilm_cosine",
]


# ============================================================
# Text preprocessing
# ============================================================

def clean_text(text: object) -> str:
    """
    Lightweight normalization for representation learning.

    We do NOT stem, lemmatize, or remove stopwords.
    Preserving word form is important for TF-IDF and MiniLM.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = " ".join(text.split())
    return text.strip()


# ============================================================
# TF-IDF (fit on train only, transform val/test)
# ============================================================

def fit_word_tfidf(
    train_questions: list[str],
) -> TfidfVectorizer:
    """
    Fit word-level TF-IDF on training questions ONLY.

    Vocabulary and IDF values are derived entirely from train.
    Validation and test are only transformed — never fit.
    """
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        min_df=WORD_MIN_DF,
        max_features=WORD_MAX_FEATURES,
        lowercase=False,      # already lowercased in clean_text
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.fit(train_questions)
    return vectorizer


def fit_char_tfidf(
    train_questions: list[str],
) -> TfidfVectorizer:
    """
    Fit character-level TF-IDF on training questions ONLY.

    Uses char_wb analyzer (word-boundary padding) and
    (3, 5) n-gram range as specified in the plan.
    """
    vectorizer = TfidfVectorizer(
        analyzer=CHAR_ANALYZER,
        ngram_range=CHAR_NGRAM_RANGE,
        min_df=CHAR_MIN_DF,
        max_features=CHAR_MAX_FEATURES,
        lowercase=False,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.fit(train_questions)
    return vectorizer


def pairwise_tfidf_cosine(
    vectorizer: TfidfVectorizer,
    q1_texts: list[str],
    q2_texts: list[str],
) -> np.ndarray:
    """
    Transform q1 and q2 with an already-fitted vectorizer,
    then compute element-wise cosine similarity for each pair.

    TF-IDF vectors from sklearn are L2-normalized by default
    (norm='l2'), so cosine similarity = dot product.
    """
    mat1 = vectorizer.transform(q1_texts)   # sparse
    mat2 = vectorizer.transform(q2_texts)   # sparse

    # Element-wise dot product across rows → cosine similarity.
    similarities = np.asarray(
        mat1.multiply(mat2).sum(axis=1)
    ).ravel()

    return similarities.astype(np.float32)


# ============================================================
# MiniLM — qid-cache optimization
#
# A single question can appear in many pairs (qid1 of one row
# can be qid2 of another). Encoding every question for every
# pair would waste time.  Instead:
#
#   1. Collect unique (qid, cleaned_text) per split.
#   2. Encode each unique question ONCE.
#   3. Build qid → embedding lookup.
#   4. Compute pair cosine similarity via lookup.
#
# This is an implementation optimization — the output is
# identical to pair-level encoding but faster on CPU.
# ============================================================

def load_embedding_model() -> SentenceTransformer:
    """Load pretrained MiniLM for inference only."""
    print(f"  Loading pretrained model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def build_qid_embedding_cache(
    df: pd.DataFrame,
    model: SentenceTransformer,
    split_name: str,
) -> dict[int, np.ndarray]:
    """
    Encode every unique question in this split exactly once.

    Returns a dict: qid → normalized 384-d embedding.
    """
    # Build unique qid → cleaned text mapping.
    qid_to_text: dict[int, str] = {}

    for _, row in df.iterrows():
        if row["qid1"] not in qid_to_text:
            qid_to_text[int(row["qid1"])] = clean_text(
                row["question1"]
            )
        if row["qid2"] not in qid_to_text:
            qid_to_text[int(row["qid2"])] = clean_text(
                row["question2"]
            )

    unique_qids = list(qid_to_text.keys())
    unique_texts = [qid_to_text[q] for q in unique_qids]

    print(
        f"  [{split_name}] Encoding {len(unique_texts):,} "
        f"unique questions (from {len(df):,} pairs)..."
    )

    embeddings = model.encode(
        unique_texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine = dot
    )

    return {
        qid: embeddings[i].astype(np.float32)
        for i, qid in enumerate(unique_qids)
    }


def pairwise_minilm_cosine(
    df: pd.DataFrame,
    model: SentenceTransformer,
    split_name: str,
) -> np.ndarray:
    """
    Compute MiniLM cosine similarity for each pair in df.

    Uses the qid-cache to avoid redundant encodings.
    Because embeddings are L2-normalized, cosine similarity
    is equivalent to the dot product.
    """
    cache = build_qid_embedding_cache(df, model, split_name)

    similarities = np.empty(len(df), dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False)):
        emb1 = cache[int(row.qid1)]
        emb2 = cache[int(row.qid2)]
        similarities[i] = float(np.dot(emb1, emb2))

    return similarities


# ============================================================
# Phase 4 feature matrix builder
# ============================================================

def build_phase4_features(
    df: pd.DataFrame,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    embedding_model: SentenceTransformer,
    split_name: str,
) -> pd.DataFrame:
    """
    Build the three Phase 4 representation features.

    TF-IDF vectorizers are already fitted on train.
    MiniLM is inference-only.
    """

    # Prepare cleaned text lists for TF-IDF transforms.
    q1_texts = (
        df["question1"].fillna("").map(clean_text).tolist()
    )
    q2_texts = (
        df["question2"].fillna("").map(clean_text).tolist()
    )

    print(f"\n  [{split_name}] Word TF-IDF cosine similarity...")
    word_sim = pairwise_tfidf_cosine(
        word_vectorizer, q1_texts, q2_texts
    )

    print(f"  [{split_name}] Character TF-IDF cosine similarity...")
    char_sim = pairwise_tfidf_cosine(
        char_vectorizer, q1_texts, q2_texts
    )

    print(f"  [{split_name}] MiniLM semantic similarity (qid-cached)...")
    minilm_sim = pairwise_minilm_cosine(df, embedding_model, split_name)

    features = pd.DataFrame(
        {
            "word_tfidf_cosine": word_sim,
            "char_tfidf_cosine": char_sim,
            "minilm_cosine": minilm_sim,
        },
        index=df.index,
    )

    return features.astype(np.float32)


# ============================================================
# Validation
# ============================================================

def validate_features(
    features: pd.DataFrame,
    split_name: str,
) -> None:
    """Confirm schema, no NaNs, no infinities."""
    if list(features.columns) != PHASE4_FEATURES:
        raise RuntimeError(
            f"{split_name}: unexpected columns "
            f"{list(features.columns)}"
        )
    if features.isna().any().any():
        raise RuntimeError(f"{split_name}: NaN values detected.")
    if np.isinf(features.to_numpy()).any():
        raise RuntimeError(
            f"{split_name}: infinite values detected."
        )


# ============================================================
# Diagnostics
# ============================================================

def representation_diagnostics(
    df: pd.DataFrame,
    features: pd.DataFrame,
) -> dict:
    """
    Compare mean similarity for duplicate vs non-duplicate pairs.

    Diagnostic only — not used for model tuning or threshold.
    """
    labels = df["is_duplicate"].astype(int).to_numpy()
    result = {}

    for col in features.columns:
        vals = features[col].to_numpy()
        dup_mean = float(vals[labels == 1].mean())
        non_dup_mean = float(vals[labels == 0].mean())
        result[col] = {
            "duplicate_mean": dup_mean,
            "non_duplicate_mean": non_dup_mean,
            "mean_gap": round(dup_mean - non_dup_mean, 6),
        }

    return result


def feature_statistics(features: pd.DataFrame) -> dict:
    result = {}
    for col in features.columns:
        v = features[col]
        result[col] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "min": float(v.min()),
            "max": float(v.max()),
            "median": float(v.median()),
        }
    return result


# ============================================================
# Diagnostic Logistic Regression (Phase 4 features only)
# ============================================================

def train_diagnostic_lr(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Diagnostic model — representation features only.

    This is NOT a formal experiment.
    The controlled A→E XGBoost ablation is Phase 5.
    """
    model = Pipeline(
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
    model.fit(x_train, y_train)

    def _metrics(y_true, y_score):
        y_pred = (y_score >= DEFAULT_THRESHOLD).astype(int)
        return {
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

    val_scores  = model.predict_proba(x_val)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]

    return {
        "model": "logistic_regression",
        "feature_set": "phase4_representation_only",
        "note": (
            "Diagnostic only — not an official ablation result. "
            "Formal A-E XGBoost experiment is Phase 5."
        ),
        "threshold": DEFAULT_THRESHOLD,
        "validation": _metrics(y_val, val_scores),
        "test":       _metrics(y_test, test_scores),
    }


# ============================================================
# Main Phase 4 pipeline
# ============================================================

def run_phase4(
    processed_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict:

    print("=" * 70)
    print("SemantiQ — Phase 4: TF-IDF + Semantic Representations")
    print("=" * 70)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load fixed Phase 1 partitions
    # --------------------------------------------------------

    print("\n[1/8] Loading fixed Phase 1 partitions...")

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df   = pd.read_csv(processed_dir / "val.csv")
    test_df  = pd.read_csv(processed_dir / "test.csv")

    print(f"Train:      {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")

    # --------------------------------------------------------
    # 2. Build training corpus for TF-IDF fitting
    # --------------------------------------------------------

    print("\n[2/8] Preparing training corpus for TF-IDF...")

    train_q1 = (
        train_df["question1"].fillna("").map(clean_text).tolist()
    )
    train_q2 = (
        train_df["question2"].fillna("").map(clean_text).tolist()
    )

    # Fit on all unique training question texts (q1 + q2).
    training_corpus = train_q1 + train_q2

    print(f"Training corpus size: {len(training_corpus):,} texts")

    # --------------------------------------------------------
    # 3. Fit TF-IDF — train only, transform val/test later
    # --------------------------------------------------------

    print("\n[3/8] Fitting TF-IDF vectorizers on train only...")

    print("  Fitting Word TF-IDF...")
    word_vectorizer = fit_word_tfidf(training_corpus)
    print(
        f"  Word TF-IDF vocabulary: "
        f"{len(word_vectorizer.vocabulary_):,} terms"
    )

    print("  Fitting Character TF-IDF (char_wb, 3-5 grams)...")
    char_vectorizer = fit_char_tfidf(training_corpus)
    print(
        f"  Char TF-IDF vocabulary: "
        f"{len(char_vectorizer.vocabulary_):,} terms"
    )

    # Persist fitted vectorizers — loaded by Phase 5 ablation.
    joblib.dump(
        word_vectorizer,
        artifact_dir / "word_tfidf_vectorizer.joblib",
    )
    joblib.dump(
        char_vectorizer,
        artifact_dir / "char_tfidf_vectorizer.joblib",
    )
    print("  Vectorizers saved.")

    # --------------------------------------------------------
    # 4. Load MiniLM (pretrained, inference only)
    # --------------------------------------------------------

    print("\n[4/8] Loading pretrained MiniLM...")
    embedding_model = load_embedding_model()
    print("  Model loaded.")

    # --------------------------------------------------------
    # 5. Build Phase 4 features for all splits
    # --------------------------------------------------------

    print("\n[5/8] Building Phase 4 representation features...")

    x_train = build_phase4_features(
        train_df, word_vectorizer, char_vectorizer,
        embedding_model, "train"
    )

    x_val = build_phase4_features(
        val_df, word_vectorizer, char_vectorizer,
        embedding_model, "validation"
    )

    x_test = build_phase4_features(
        test_df, word_vectorizer, char_vectorizer,
        embedding_model, "test"
    )

    # --------------------------------------------------------
    # 6. Validate and persist feature matrices
    # --------------------------------------------------------

    print("\n[6/8] Validating Phase 4 feature matrices...")

    validate_features(x_train, "train")
    validate_features(x_val, "validation")
    validate_features(x_test, "test")

    print(f"  Feature count : {len(PHASE4_FEATURES)}")
    print(f"  Features      : {PHASE4_FEATURES}")
    print("  Validation    : PASSED")

    # Save — Phase 5 loads these directly.
    x_train.to_csv(artifact_dir / "train_phase4_features.csv", index=False)
    x_val.to_csv(artifact_dir / "val_phase4_features.csv",   index=False)
    x_test.to_csv(artifact_dir / "test_phase4_features.csv",  index=False)

    with (artifact_dir / "phase4_feature_columns.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(PHASE4_FEATURES, f, indent=2)

    # --------------------------------------------------------
    # 7. Representation diagnostics (train split)
    # --------------------------------------------------------

    print("\n[7/8] Running representation diagnostics...")

    diagnostics = representation_diagnostics(train_df, x_train)
    feat_stats  = feature_statistics(x_train)

    # --------------------------------------------------------
    # 8. Diagnostic Logistic Regression
    # --------------------------------------------------------

    print("\n[8/8] Training diagnostic Logistic Regression...")

    y_train = train_df["is_duplicate"].astype(int)
    y_val   = val_df["is_duplicate"].astype(int)
    y_test  = test_df["is_duplicate"].astype(int)

    diag_metrics = train_diagnostic_lr(
        x_train, y_train, x_val, y_val, x_test, y_test
    )

    # --------------------------------------------------------
    # Save full report
    # --------------------------------------------------------

    report = {
        "phase": "phase4",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "tfidf_config": {
            "word": {
                "analyzer": "word",
                "ngram_range": list(WORD_NGRAM_RANGE),
                "min_df": WORD_MIN_DF,
                "max_features": WORD_MAX_FEATURES,
                "vocabulary_size": len(word_vectorizer.vocabulary_),
            },
            "character": {
                "analyzer": CHAR_ANALYZER,
                "ngram_range": list(CHAR_NGRAM_RANGE),
                "min_df": CHAR_MIN_DF,
                "max_features": CHAR_MAX_FEATURES,
                "vocabulary_size": len(char_vectorizer.vocabulary_),
            },
        },
        "features": PHASE4_FEATURES,
        "train_feature_statistics": feat_stats,
        "train_representation_diagnostics": diagnostics,
        "diagnostic_model": diag_metrics,
    }

    with (artifact_dir / "phase4_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    with (report_dir / "phase4_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY")
    print("=" * 70)

    print("\nRepresentation diagnostics (train split):")
    print(
        f"\n  {'Feature':<22} {'Dup mean':>10} "
        f"{'Non-dup mean':>14} {'Gap':>8}"
    )
    print("  " + "-" * 58)
    for feat, vals in diagnostics.items():
        print(
            f"  {feat:<22} "
            f"{vals['duplicate_mean']:>10.4f} "
            f"{vals['non_duplicate_mean']:>14.4f} "
            f"{vals['mean_gap']:>8.4f}"
        )

    print("\nDiagnostic Logistic Regression (representation features only):")
    print(
        f"  {'':22} {'Validation':>10} {'Test':>10}"
    )
    print("  " + "-" * 44)
    for metric in ("precision", "recall", "f1", "pr_auc"):
        v = diag_metrics["validation"][metric]
        t = diag_metrics["test"][metric]
        print(f"  {metric.upper():<22} {v:>10.4f} {t:>10.4f}")

    print("\nProgression so far (Test PR-AUC):")
    print("  Phase 2 LR (statistical)         : 0.5167")
    print("  Phase 3 LR (+ lexical/string)    : 0.6087")
    print(
        f"  Phase 4 LR (representation only) : "
        f"{diag_metrics['test']['pr_auc']:.4f}  [diagnostic]"
    )
    print("  Phase 5: XGBoost A→E combined    : TBD")

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)

    return report
