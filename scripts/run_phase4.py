from pathlib import Path

from src.phase4_representations import run_phase4


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

ARTIFACTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase4"
)

REPORTS = (
    PROJECT_ROOT
    / "reports"
)


def main() -> None:

    run_phase4(
        processed_dir=PROCESSED_DATA,
        artifact_dir=ARTIFACTS,
        report_dir=REPORTS,
    )


if __name__ == "__main__":
    main()
