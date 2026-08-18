from pathlib import Path

from src.phase5_ablation import run_phase5


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA    = PROJECT_ROOT / "data"      / "processed"
PHASE3_ARTIFACTS  = PROJECT_ROOT / "artifacts" / "phase3"
PHASE4_ARTIFACTS  = PROJECT_ROOT / "artifacts" / "phase4"
PHASE5_ARTIFACTS  = PROJECT_ROOT / "artifacts" / "phase5"
REPORTS           = PROJECT_ROOT / "reports"


def main() -> None:

    run_phase5(
        processed_dir=PROCESSED_DATA,
        phase3_dir=PHASE3_ARTIFACTS,
        phase4_dir=PHASE4_ARTIFACTS,
        artifact_dir=PHASE5_ARTIFACTS,
        report_dir=REPORTS,
    )


if __name__ == "__main__":
    main()
