from __future__ import annotations

import json
from pathlib import Path
from src.inference import SemantIQ
from src.adversarial_suite import evaluate_adversarial_suite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "phase8"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    print("=" * 70)
    print("SemantiQ — Adversarial Benchmark Suite Evaluation")
    print("=" * 70)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    engine = SemantIQ(project_root=PROJECT_ROOT)
    results = evaluate_adversarial_suite(engine)

    print(f"\nOverall Adversarial Accuracy: {results['overall_accuracy']:.2%} ({results['total_correct']}/{results['total_test_pairs']} pairs correct)\n")
    print("Category Breakdown:")
    for cat, stats in results["category_breakdown"].items():
        print(f"  - {cat:<25}: Accuracy: {stats['accuracy']:>6.1%} | Total: {stats['total']:>2} | FP: {stats['false_positives']} | FN: {stats['false_negatives']}")

    with (ARTIFACTS_DIR / "adversarial_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with (REPORTS_DIR / "adversarial_benchmark_report.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nAdversarial evaluation report saved.")


if __name__ == "__main__":
    main()
