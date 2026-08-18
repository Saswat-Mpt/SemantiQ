from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

EXPECTED_COLUMNS = [
    "id",
    "qid1",
    "qid2",
    "question1",
    "question2",
    "is_duplicate",
]

REQUIRED_COLUMNS = [
    "qid1",
    "qid2",
    "question1",
    "question2",
    "is_duplicate",
]


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Train/validation/test ratios must sum to 1.0."
            )

        if min(
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        ) <= 0:
            raise ValueError(
                "All split ratios must be greater than zero."
            )


# ============================================================
# Text normalization
# ============================================================

def normalize_question_text(text: object) -> str:
    """
    Normalize question text for EXACT-TEXT duplicate grouping.

    Important:
    This is intentionally conservative.

    We do NOT:
    - remove stopwords
    - stem
    - lemmatize
    - perform semantic normalization

    The purpose is only to identify questions that are
    essentially identical at the text level.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    # Remove punctuation/symbols while preserving
    # alphanumeric characters and spaces.
    text = re.sub(r"[^\w\s]", "", text)

    # Final whitespace cleanup.
    text = text.strip()

    return text


# ============================================================
# Data loading, cleaning, and validation
# ============================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """Load and validate the raw Quora question-pair dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{path}\n\n"
            "Expected the Quora dataset at data/raw/train.csv."
        )

    df = pd.read_csv(path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns:\n"
            f"{missing}\n\n"
            f"Found columns:\n{list(df.columns)}"
        )

    return df


def clean_raw_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Drop rows where question1 or question2 is null OR normalizes
    to an empty string (e.g. pure punctuation like '??', '...').

    These rows cannot form meaningful features and are discarded
    before any further validation or splitting.

    Returns the cleaned DataFrame, the number of null-dropped rows,
    and the number of empty-after-normalization-dropped rows.
    """
    # Step 1: drop null question rows.
    null_mask = df["question1"].isna() | df["question2"].isna()
    n_dropped_null = int(null_mask.sum())
    cleaned = df.loc[~null_mask].copy().reset_index(drop=True)

    # Step 2: drop rows where either question normalizes to "".
    norm1 = cleaned["question1"].map(normalize_question_text)
    norm2 = cleaned["question2"].map(normalize_question_text)
    empty_mask = norm1.eq("") | norm2.eq("")
    n_dropped_empty = int(empty_mask.sum())
    cleaned = cleaned.loc[~empty_mask].copy().reset_index(drop=True)

    return cleaned, n_dropped_null, n_dropped_empty


def validate_raw_data(df: pd.DataFrame) -> None:
    """Run structural and label validation checks."""

    if df.empty:
        raise ValueError("The dataset is empty.")

    # Required columns.
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # QIDs should not be missing.
    for column in ["qid1", "qid2"]:
        missing_count = df[column].isna().sum()

        if missing_count > 0:
            raise ValueError(
                f"{column} contains {missing_count} missing values."
            )

    # Questions should not be missing (after cleaning step).
    for column in ["question1", "question2"]:
        missing_count = df[column].isna().sum()

        if missing_count > 0:
            raise ValueError(
                f"{column} contains {missing_count} missing values."
            )

    # Labels should be binary.
    labels = set(df["is_duplicate"].dropna().unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            "is_duplicate must contain only 0/1 labels. "
            f"Found: {sorted(labels)}"
        )

    # No missing labels.
    missing_labels = df["is_duplicate"].isna().sum()

    if missing_labels > 0:
        raise ValueError(
            f"is_duplicate contains {missing_labels} missing values."
        )


# ============================================================
# Duplicate-text grouping
# ============================================================

def build_question_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unique question table.

    Every qid gets a normalized text representation.

    Questions having identical normalized text are assigned
    to the same text_group_id.
    """

    q1 = df[
        ["qid1", "question1"]
    ].rename(
        columns={
            "qid1": "qid",
            "question1": "question",
        }
    )

    q2 = df[
        ["qid2", "question2"]
    ].rename(
        columns={
            "qid2": "qid",
            "question2": "question",
        }
    )

    questions = pd.concat(
        [q1, q2],
        ignore_index=True,
    )

    # A qid should map to one question.
    conflicting = (
        questions.groupby("qid")["question"]
        .nunique(dropna=False)
    )

    conflicting = conflicting[
        conflicting > 1
    ]

    if not conflicting.empty:
        sample = conflicting.head(10)

        raise ValueError(
            "Some qids map to multiple different question texts.\n"
            f"Example conflicting qids:\n{sample}"
        )

    questions = (
        questions
        .drop_duplicates(subset=["qid"])
        .reset_index(drop=True)
    )

    questions["normalized_text"] = (
        questions["question"]
        .map(normalize_question_text)
    )

    empty_count = (
        questions["normalized_text"]
        .eq("")
        .sum()
    )

    if empty_count > 0:
        raise ValueError(
            f"{empty_count} unique questions became empty after "
            "normalization."
        )

    # Each normalized text becomes a single leakage group.
    questions["text_group_id"] = (
        pd.factorize(
            questions["normalized_text"],
            sort=True,
        )[0]
    )

    return questions


def attach_text_groups(
    df: pd.DataFrame,
    questions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach normalized-text group IDs to each question pair."""

    qid_to_group = questions.set_index(
        "qid"
    )["text_group_id"]

    result = df.copy()

    result["group1"] = result["qid1"].map(qid_to_group)
    result["group2"] = result["qid2"].map(qid_to_group)

    if result["group1"].isna().any():
        raise ValueError(
            "Some qid1 values could not be mapped to a text group."
        )

    if result["group2"].isna().any():
        raise ValueError(
            "Some qid2 values could not be mapped to a text group."
        )

    return result


# ============================================================
# Group-level splitting
# ============================================================

def split_groups(
    group_ids: pd.Series,
    config: SplitConfig,
) -> tuple[set[int], set[int], set[int]]:
    """
    Split normalized-text groups into train/validation/test.

    The split is performed on GROUPS, never individual pairs.
    """

    config.validate()

    unique_groups = pd.Series(
        group_ids.unique(),
        name="text_group_id",
    )

    # Shuffle deterministically.
    shuffled = unique_groups.sample(
        frac=1.0,
        random_state=config.random_state,
    ).tolist()

    n_groups = len(shuffled)

    train_end = int(
        n_groups * config.train_ratio
    )

    val_end = train_end + int(
        n_groups * config.val_ratio
    )

    train_groups = set(
        shuffled[:train_end]
    )

    val_groups = set(
        shuffled[train_end:val_end]
    )

    test_groups = set(
        shuffled[val_end:]
    )

    # Sanity checks.
    if not train_groups:
        raise ValueError("Training group split is empty.")

    if not val_groups:
        raise ValueError("Validation group split is empty.")

    if not test_groups:
        raise ValueError("Test group split is empty.")

    if (
        train_groups
        & val_groups
        or train_groups
        & test_groups
        or val_groups
        & test_groups
    ):
        raise RuntimeError(
            "Group leakage detected between partitions."
        )

    return (
        train_groups,
        val_groups,
        test_groups,
    )


def create_pair_splits(
    df: pd.DataFrame,
    config: SplitConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create train/validation/test pair splits.

    A pair survives only if both questions belong to the
    same text-group partition.

    Cross-partition pairs are discarded.
    """

    all_groups = pd.concat(
        [
            df["group1"],
            df["group2"],
        ],
        ignore_index=True,
    )

    train_groups, val_groups, test_groups = split_groups(
        all_groups,
        config,
    )

    group1 = df["group1"]
    group2 = df["group2"]

    train_mask = (
        group1.isin(train_groups)
        & group2.isin(train_groups)
    )

    val_mask = (
        group1.isin(val_groups)
        & group2.isin(val_groups)
    )

    test_mask = (
        group1.isin(test_groups)
        & group2.isin(test_groups)
    )

    assigned = (
        train_mask.astype(int)
        + val_mask.astype(int)
        + test_mask.astype(int)
    )

    if (assigned > 1).any():
        raise RuntimeError(
            "A pair was assigned to more than one partition."
        )

    discarded_mask = assigned.eq(0)

    train_df = df.loc[
        train_mask
    ].copy()

    val_df = df.loc[
        val_mask
    ].copy()

    test_df = df.loc[
        test_mask
    ].copy()

    discarded_df = df.loc[
        discarded_mask
    ].copy()

    return (
        train_df,
        val_df,
        test_df,
        discarded_df,
    )


# ============================================================
# Leakage verification
# ============================================================

def verify_partition_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Verify qid and normalized-text group isolation."""

    partitions = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    # --------------------------------------------------------
    # QID isolation
    # --------------------------------------------------------

    qid_sets = {}

    for name, split in partitions.items():
        qids = set(
            split["qid1"]
        ) | set(
            split["qid2"]
        )

        qid_sets[name] = qids

    if qid_sets["train"] & qid_sets["validation"]:
        raise RuntimeError(
            "QID leakage detected: train ∩ validation."
        )

    if qid_sets["train"] & qid_sets["test"]:
        raise RuntimeError(
            "QID leakage detected: train ∩ test."
        )

    if qid_sets["validation"] & qid_sets["test"]:
        raise RuntimeError(
            "QID leakage detected: validation ∩ test."
        )

    # --------------------------------------------------------
    # Normalized-text group isolation
    # --------------------------------------------------------

    group_sets = {}

    for name, split in partitions.items():
        groups = set(
            split["group1"]
        ) | set(
            split["group2"]
        )

        group_sets[name] = groups

    if group_sets["train"] & group_sets["validation"]:
        raise RuntimeError(
            "Normalized-text group leakage detected: "
            "train ∩ validation."
        )

    if group_sets["train"] & group_sets["test"]:
        raise RuntimeError(
            "Normalized-text group leakage detected: "
            "train ∩ test."
        )

    if group_sets["validation"] & group_sets["test"]:
        raise RuntimeError(
            "Normalized-text group leakage detected: "
            "validation ∩ test."
        )


# ============================================================
# Reporting
# ============================================================

def split_statistics(
    split: pd.DataFrame,
) -> dict:
    """Return useful statistics for a split."""

    pairs = len(split)

    if pairs == 0:
        return {
            "pairs": 0,
            "duplicate_pairs": 0,
            "non_duplicate_pairs": 0,
            "duplicate_pct": None,
        }

    duplicate_pairs = int(
        split["is_duplicate"].sum()
    )

    non_duplicate_pairs = (
        pairs - duplicate_pairs
    )

    duplicate_pct = (
        duplicate_pairs / pairs
    ) * 100

    unique_qids = (
        set(split["qid1"])
        | set(split["qid2"])
    )

    unique_groups = (
        set(split["group1"])
        | set(split["group2"])
    )

    return {
        "pairs": pairs,
        "duplicate_pairs": duplicate_pairs,
        "non_duplicate_pairs": non_duplicate_pairs,
        "duplicate_pct": round(
            duplicate_pct,
            4,
        ),
        "unique_questions": len(unique_qids),
        "unique_text_groups": len(unique_groups),
    }


def build_metadata(
    raw_df: pd.DataFrame,
    n_dropped_nulls: int,
    n_dropped_empty: int,
    questions: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    discarded_df: pd.DataFrame,
    config: SplitConfig,
) -> dict:
    """Build reproducibility metadata."""

    n_total_dropped = n_dropped_nulls + n_dropped_empty

    return {
        "project": "SemantiQ",
        "phase": "phase_1_data_foundation",
        "random_state": config.random_state,
        "requested_split_ratios": {
            "train": config.train_ratio,
            "validation": config.val_ratio,
            "test": config.test_ratio,
        },
        "raw": {
            "rows": len(raw_df) + n_total_dropped,
            "columns": list(raw_df.columns),
            "null_question_rows_dropped": n_dropped_nulls,
            "empty_after_normalization_rows_dropped": n_dropped_empty,
            "total_rows_dropped": n_total_dropped,
            "rows_after_cleaning": len(raw_df),
        },
        "questions": {
            "unique_question_ids": int(
                questions["qid"].nunique()
            ),
            "unique_normalized_texts": int(
                questions["normalized_text"].nunique()
            ),
            "duplicate_text_groups": int(
                (
                    questions
                    .groupby("text_group_id")
                    .size()
                    .gt(1)
                    .sum()
                )
            ),
        },
        "splits": {
            "train": split_statistics(train_df),
            "validation": split_statistics(val_df),
            "test": split_statistics(test_df),
            "discarded": {
                "pairs": len(discarded_df),
            },
        },
    }


# ============================================================
# Saving
# ============================================================

def save_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    discarded_df: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
    report_dir: Path,
) -> None:
    """Save fixed split artifacts and metadata."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove internal group columns from saved pair files.
    save_columns = [
        column
        for column in train_df.columns
        if column not in ["group1", "group2"]
    ]

    train_df[save_columns].to_csv(
        output_dir / "train.csv",
        index=False,
    )

    val_df[save_columns].to_csv(
        output_dir / "val.csv",
        index=False,
    )

    test_df[save_columns].to_csv(
        output_dir / "test.csv",
        index=False,
    )

    discarded_df[save_columns].to_csv(
        output_dir / "discarded.csv",
        index=False,
    )

    metadata_path = (
        output_dir / "split_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    report_path = (
        report_dir / "phase1_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


# ============================================================
# Main Phase 1 pipeline
# ============================================================

def run_phase1(
    raw_path: Path,
    output_dir: Path,
    report_dir: Path,
    config: SplitConfig,
) -> dict:

    print("=" * 70)
    print("SemantiQ — Phase 1: Data Foundation")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    print("\n[1/8] Loading raw dataset...")

    df = load_raw_data(raw_path)

    print(
        f"Loaded {len(df):,} question pairs."
    )

    # --------------------------------------------------------
    # 1b. Clean — drop null question rows
    # --------------------------------------------------------

    df, n_dropped_null, n_dropped_empty = clean_raw_data(df)

    if n_dropped_null > 0:
        print(
            f"Dropped {n_dropped_null} rows with null question text."
        )

    if n_dropped_empty > 0:
        print(
            f"Dropped {n_dropped_empty} rows with empty-after-normalization questions."
        )

    if n_dropped_null > 0 or n_dropped_empty > 0:
        print(
            f"Remaining: {len(df):,} pairs."
        )

    # --------------------------------------------------------
    # 2. Validate
    # --------------------------------------------------------

    print("\n[2/8] Validating raw dataset...")

    validate_raw_data(df)

    print("Raw dataset validation: PASSED")

    # --------------------------------------------------------
    # 3. Build question table
    # --------------------------------------------------------

    print(
        "\n[3/8] Building unique question table..."
    )

    questions = build_question_table(df)

    print(
        f"Unique question IDs: "
        f"{questions['qid'].nunique():,}"
    )

    print(
        f"Unique normalized texts: "
        f"{questions['normalized_text'].nunique():,}"
    )

    duplicate_text_groups = (
        questions
        .groupby("text_group_id")
        .size()
        .gt(1)
        .sum()
    )

    print(
        f"Exact duplicate-text groups: "
        f"{duplicate_text_groups:,}"
    )

    # --------------------------------------------------------
    # 4. Attach groups
    # --------------------------------------------------------

    print(
        "\n[4/8] Attaching normalized-text groups..."
    )

    df_grouped = attach_text_groups(
        df,
        questions,
    )

    # --------------------------------------------------------
    # 5. Split
    # --------------------------------------------------------

    print(
        "\n[5/8] Creating leakage-safe group split..."
    )

    (
        train_df,
        val_df,
        test_df,
        discarded_df,
    ) = create_pair_splits(
        df_grouped,
        config,
    )

    print(
        f"Train pairs:      {len(train_df):,}"
    )

    print(
        f"Validation pairs: {len(val_df):,}"
    )

    print(
        f"Test pairs:       {len(test_df):,}"
    )

    print(
        f"Discarded pairs:  {len(discarded_df):,}"
    )

    # --------------------------------------------------------
    # 6. Verify
    # --------------------------------------------------------

    print(
        "\n[6/8] Verifying partition isolation..."
    )

    verify_partition_integrity(
        train_df,
        val_df,
        test_df,
    )

    print(
        "Leakage verification: PASSED"
    )

    # --------------------------------------------------------
    # 7. Metadata
    # --------------------------------------------------------

    print(
        "\n[7/8] Building split statistics..."
    )

    metadata = build_metadata(
        raw_df=df,
        n_dropped_nulls=n_dropped_null,
        n_dropped_empty=n_dropped_empty,
        questions=questions,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        discarded_df=discarded_df,
        config=config,
    )

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        stats = metadata["splits"][
            split_name
        ]

        print(
            f"{split_name.capitalize():12s}: "
            f"{stats['pairs']:,} pairs | "
            f"{stats['duplicate_pct']:.2f}% duplicate"
        )

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    print(
        "\n[8/8] Saving fixed split artifacts..."
    )

    save_outputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        discarded_df=discarded_df,
        metadata=metadata,
        output_dir=output_dir,
        report_dir=report_dir,
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {output_dir / 'train.csv'}"
    )
    print(
        f"  {output_dir / 'val.csv'}"
    )
    print(
        f"  {output_dir / 'test.csv'}"
    )
    print(
        f"  {output_dir / 'discarded.csv'}"
    )
    print(
        f"  {output_dir / 'split_metadata.json'}"
    )

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)

    return metadata
