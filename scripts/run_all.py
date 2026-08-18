from __future__ import annotations

import sys
import time
from pathlib import Path

from src.phase1_data import run_phase1
from src.phase2_baselines import run_phase2
from src.phase3_features import run_phase3
from src.phase4_representations import run_phase4
from src.phase5_ablation import run_phase5
from src.phase6_threshold import run_phase6
from src.phase7_analysis import run_phase7


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DATA   = PROJECT_ROOT / "data" / "processed"
BASELINES_DIR    = PROJECT_ROOT / "artifacts" / "baselines"
PHASE3_DIR       = PROJECT_ROOT / "artifacts" / "phase3"
PHASE4_DIR       = PROJECT_ROOT / "artifacts" / "phase4"
PHASE5_DIR       = PROJECT_ROOT / "artifacts" / "phase5"
PHASE6_DIR       = PROJECT_ROOT / "artifacts" / "phase6"
PHASE7_DIR       = PROJECT_ROOT / "artifacts" / "phase7"
REPORTS_DIR      = PROJECT_ROOT / "reports"


def main() -> None:
    print("=" * 75)
    print("SemantiQ — Full End-to-End Reproducibility Pipeline (Phases 1 to 7)")
    print("=" * 75)
    start_total = time.perf_counter()

    # 1. Phase 1: Data Ingestion & Leakage-Safe Partitioning
    print("\n>>> Running Phase 1: Data Partitioning...")
    if not RAW_DATA_PATH.exists():
        print(f"Error: Raw dataset not found at {RAW_DATA_PATH}")
        sys.exit(1)
    run_phase1(raw_data_path=RAW_DATA_PATH, processed_dir=PROCESSED_DATA, report_dir=REPORTS_DIR)

    # 2. Phase 2: Baselines
    print("\n>>> Running Phase 2: Statistical & Heuristic Baselines...")
    run_phase2(processed_dir=PROCESSED_DATA, artifact_dir=BASELINES_DIR, report_dir=REPORTS_DIR)

    # 3. Phase 3: Classical Similarity Features
    print("\n>>> Running Phase 3: Lexical / N-Gram Feature Extraction...")
    run_phase3(processed_dir=PROCESSED_DATA, artifact_dir=PHASE3_DIR, report_dir=REPORTS_DIR)

    # 4. Phase 4: TF-IDF & MiniLM Representations
    print("\n>>> Running Phase 4: TF-IDF and MiniLM Encodings...")
    run_phase4(processed_dir=PROCESSED_DATA, artifact_dir=PHASE4_DIR, report_dir=REPORTS_DIR)

    # 5. Phase 5: Controlled XGBoost Ablation
    print("\n>>> Running Phase 5: XGBoost Ablation (A to E)...")
    run_phase5(processed_dir=PROCESSED_DATA, phase3_dir=PHASE3_DIR, phase4_dir=PHASE4_DIR, artifact_dir=PHASE5_DIR, report_dir=REPORTS_DIR)

    # 6. Phase 6: Precision-Constrained Decision Policy
    print("\n>>> Running Phase 6: Precision Threshold Calibration...")
    run_phase6(processed_dir=PROCESSED_DATA, phase3_dir=PHASE3_DIR, phase4_dir=PHASE4_DIR, phase5_dir=PHASE5_DIR, artifact_dir=PHASE6_DIR, report_dir=REPORTS_DIR)

    # 7. Phase 7: Error Taxonomy & Final Analysis
    print("\n>>> Running Phase 7: Error Diagnostics & Taxonomy...")
    run_phase7(processed_dir=PROCESSED_DATA, phase3_dir=PHASE3_DIR, phase4_dir=PHASE4_DIR, phase5_dir=PHASE5_DIR, phase6_dir=PHASE6_DIR, artifact_dir=PHASE7_DIR, report_dir=REPORTS_DIR)

    elapsed_min = (time.perf_counter() - start_total) / 60.0
    print("\n" + "=" * 75)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_min:.2f} MINUTES")
    print("=" * 75)


if __name__ == "__main__":
    main()
