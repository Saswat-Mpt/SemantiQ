from pathlib import Path

from src.phase3_features import run_phase3


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ARTIFACTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase3"
)

REPORTS = (
    PROJECT_ROOT
    / "reports"
)


def main() -> None:

    run_phase3(
        processed_dir=PROCESSED_DATA,
        artifact_dir=ARTIFACTS,
        report_dir=REPORTS,
    )


if __name__ == "__main__":
    main()
