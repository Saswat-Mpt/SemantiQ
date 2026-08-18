from pathlib import Path

from src.phase7_analysis import run_phase7


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PHASE3_ARTIFACTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase3"
)

PHASE4_ARTIFACTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase4"
)

PHASE5_ARTIFACTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase5"
)

PHASE6_ARTIFACTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6"
)

PHASE7_ARTIFACTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase7"
)

REPORTS = (
    PROJECT_ROOT
    / "reports"
)


def main() -> None:

    run_phase7(
        processed_dir=PROCESSED_DATA,
        phase3_dir=PHASE3_ARTIFACTS,
        phase4_dir=PHASE4_ARTIFACTS,
        phase5_dir=PHASE5_ARTIFACTS,
        phase6_dir=PHASE6_ARTIFACTS,
        artifact_dir=PHASE7_ARTIFACTS,
        report_dir=REPORTS,
    )


if __name__ == "__main__":
    main()
