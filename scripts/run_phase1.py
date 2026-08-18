from pathlib import Path

from src.phase1_data import (
    SplitConfig,
    run_phase1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)

PROCESSED_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORTS = (
    PROJECT_ROOT
    / "reports"
)


def main() -> None:

    config = SplitConfig(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    run_phase1(
        raw_path=RAW_DATA,
        output_dir=PROCESSED_DATA,
        report_dir=REPORTS,
        config=config,
    )


if __name__ == "__main__":
    main()
