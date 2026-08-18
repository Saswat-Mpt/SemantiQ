from __future__ import annotations

import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

from src.inference import SemantIQ
from src.critical_tokens import evaluate_critical_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA   = PROJECT_ROOT / "data" / "processed"
PHASE6_ARTIFACTS = PROJECT_ROOT / "artifacts" / "phase6"
PHASE7_ARTIFACTS = PROJECT_ROOT / "artifacts" / "phase7"
REPORTS          = PROJECT_ROOT / "reports"


def run_comprehensive_error_taxonomy(test_df: pd.DataFrame, preds_df: pd.DataFrame) -> dict:
    """
    Evaluates failure taxonomy and error rates across critical categories
    (Numeric, Entity, Negation, Question-type shift, Paraphrase).
    """
    df = pd.concat([test_df.reset_index(drop=True), preds_df[["score", "prediction", "y_true"]].reset_index(drop=True)], axis=1)
    
    # Classify error type
    df["is_fp"] = (df["y_true"] == 0) & (df["prediction"] == 1)
    df["is_fn"] = (df["y_true"] == 1) & (df["prediction"] == 0)
    df["is_tp"] = (df["y_true"] == 1) & (df["prediction"] == 1)
    df["is_tn"] = (df["y_true"] == 0) & (df["prediction"] == 0)

    # Diagnostic tags
    diag_records = []
    for _, row in df.iterrows():
        diag = evaluate_critical_tokens(str(row["question1"]), str(row["question2"]))
        diag_records.append(diag)
    
    diag_df = pd.DataFrame(diag_records)
    full_df = pd.concat([df, diag_df], axis=1)

    # Slices analysis
    taxonomy = {
        "overall": {
            "total_test_pairs": len(full_df),
            "total_duplicates": int(full_df["y_true"].sum()),
            "total_false_positives": int(full_df["is_fp"].sum()),
            "total_false_negatives": int(full_df["is_fn"].sum()),
        },
        "error_rates_by_category": {
            "numeric_mismatch": {
                "pair_count": int(full_df["numeric_mismatch"].sum()),
                "false_positive_rate": float(full_df[full_df["numeric_mismatch"]]["is_fp"].mean()) if full_df["numeric_mismatch"].sum() > 0 else 0.0,
                "precision": float(full_df[full_df["numeric_mismatch"] & (full_df["prediction"] == 1)]["y_true"].mean()) if (full_df["numeric_mismatch"] & (full_df["prediction"] == 1)).sum() > 0 else 1.0,
            },
            "entity_mismatch": {
                "pair_count": int(full_df["entity_mismatch"].sum()),
                "false_positive_rate": float(full_df[full_df["entity_mismatch"]]["is_fp"].mean()) if full_df["entity_mismatch"].sum() > 0 else 0.0,
                "precision": float(full_df[full_df["entity_mismatch"] & (full_df["prediction"] == 1)]["y_true"].mean()) if (full_df["entity_mismatch"] & (full_df["prediction"] == 1)).sum() > 0 else 1.0,
            },
            "negation_mismatch": {
                "pair_count": int(full_df["negation_mismatch"].sum()),
                "false_positive_rate": float(full_df[full_df["negation_mismatch"]]["is_fp"].mean()) if full_df["negation_mismatch"].sum() > 0 else 0.0,
            },
            "question_type_mismatch": {
                "pair_count": int(full_df["question_type_mismatch"].sum()),
                "false_positive_rate": float(full_df[full_df["question_type_mismatch"]]["is_fp"].mean()) if full_df["question_type_mismatch"].sum() > 0 else 0.0,
            },
        },
    }
    return taxonomy


def main() -> None:
    print("=" * 70)
    print("SemantiQ — Error Taxonomy & Diagnostics Runner")
    print("=" * 70)

    test_df = pd.read_csv(PROCESSED_DATA / "test.csv")
    preds_df = pd.read_csv(PHASE6_ARTIFACTS / "test_predictions.csv")

    taxonomy = run_comprehensive_error_taxonomy(test_df, preds_df)

    with (REPORTS / "error_taxonomy_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    with (PHASE7_ARTIFACTS / "error_taxonomy.json").open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    print("\nError Taxonomy Breakdown:")
    for cat, stats in taxonomy["error_rates_by_category"].items():
        print(f"  - {cat:<25}: {stats['pair_count']:>5} pairs | FP Rate: {stats['false_positive_rate']:.2%}")

    print("\nTaxonomy report saved to reports/error_taxonomy_analysis.json")


if __name__ == "__main__":
    main()
