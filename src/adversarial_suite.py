from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
from src.inference import SemantIQ


# ============================================================
# Curated Adversarial Benchmark Pairs
# ============================================================

ADVERSARIAL_BENCHMARK_PAIRS = [
    # --- Category 1: Entity Substitutions (Non-Duplicates, High Semantic Overlap) ---
    {
        "category": "entity_substitution",
        "question1": "What is the average placement package at IIT Delhi?",
        "question2": "What is the average placement package at IIT Bombay?",
        "expected_label": 0,
        "description": "Institution name substitution with identical structure",
    },
    {
        "category": "entity_substitution",
        "question1": "How do I prepare for Google software engineering interview?",
        "question2": "How do I prepare for Microsoft software engineering interview?",
        "expected_label": 0,
        "description": "Company name substitution with identical intent",
    },
    {
        "category": "entity_substitution",
        "question1": "What is the best way to travel from Paris to London?",
        "question2": "What is the best way to travel from Paris to Berlin?",
        "expected_label": 0,
        "description": "Destination substitution with identical grammar",
    },
    {
        "category": "entity_substitution",
        "question1": "Who is the CEO of Apple Inc?",
        "question2": "Who is the CEO of Tesla Inc?",
        "expected_label": 0,
        "description": "Entity swap in factual inquiry",
    },

    # --- Category 2: Numeric & Year Shifts (Non-Duplicates, High Lexical Overlap) ---
    {
        "category": "numeric_year_shift",
        "question1": "What was the JEE Advanced cutoff rank in 2017?",
        "question2": "What was the JEE Advanced cutoff rank in 2018?",
        "expected_label": 0,
        "description": "Year token shift in cutoff query",
    },
    {
        "category": "numeric_year_shift",
        "question1": "How can a student score 50% in 12th board exams?",
        "question2": "How can a student score 90% in 12th board exams?",
        "expected_label": 0,
        "description": "Target percentage mismatch with different target difficulty",
    },
    {
        "category": "numeric_year_shift",
        "question1": "What are the specs of iPhone 14 Pro?",
        "question2": "What are the specs of iPhone 15 Pro?",
        "expected_label": 0,
        "description": "Product version number mismatch",
    },
    {
        "category": "numeric_year_shift",
        "question1": "How long does it take to run 5 km for beginners?",
        "question2": "How long does it take to run 10 km for beginners?",
        "expected_label": 0,
        "description": "Distance metric mismatch",
    },

    # --- Category 3: Negation Flips (Non-Duplicates, Near-Identical Surface) ---
    {
        "category": "negation_flip",
        "question1": "Why should I learn Python programming language?",
        "question2": "Why shouldn't I learn Python programming language?",
        "expected_label": 0,
        "description": "Negation modifier changing core premise",
    },
    {
        "category": "negation_flip",
        "question1": "How can I invest in the stock market safely?",
        "question2": "How can I avoid investing in the stock market?",
        "expected_label": 0,
        "description": "Polarity and action inversion",
    },
    {
        "category": "negation_flip",
        "question1": "Is drinking green tea good for health?",
        "question2": "Is drinking green tea bad for health?",
        "expected_label": 0,
        "description": "Antonym / polarity swap",
    },

    # --- Category 4: Question-Type / Intent Shift (Same Topic, Different Question) ---
    {
        "category": "intent_shift",
        "question1": "What is machine learning?",
        "question2": "How can I get a job as a machine learning engineer?",
        "expected_label": 0,
        "description": "Definition vs Career path for identical topic",
    },
    {
        "category": "intent_shift",
        "question1": "What are the side effects of aspirin?",
        "question2": "Where can I buy aspirin online without prescription?",
        "expected_label": 0,
        "description": "Medical side-effect query vs purchasing query",
    },
    {
        "category": "intent_shift",
        "question1": "How does an airplane fly?",
        "question2": "How much does a commercial airplane cost?",
        "expected_label": 0,
        "description": "Physics mechanism vs financial cost query",
    },

    # --- Category 5: Tool / Language Substitutions ---
    {
        "category": "tool_substitution",
        "question1": "How to write a simple HTTP web server in C?",
        "question2": "How to write a simple HTTP web server in C++?",
        "expected_label": 0,
        "description": "C vs C++ language distinction",
    },
    {
        "category": "tool_substitution",
        "question1": "How do I build a REST API in Django?",
        "question2": "How do I build a REST API in FastAPI?",
        "expected_label": 0,
        "description": "Framework substitution for identical task",
    },

    # --- Category 6: Legitimate Paraphrases (True Duplicates, Low Lexical Overlap) ---
    {
        "category": "paraphrase_duplicate",
        "question1": "What is the most effective technique to shed excess body weight?",
        "question2": "How can someone slim down and burn fat rapidly?",
        "expected_label": 1,
        "description": "True duplicate paraphrase with zero content-word overlap",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "What is the salary of an entry level software developer in India?",
        "question2": "How much does a fresher programmer get paid in India?",
        "expected_label": 1,
        "description": "Synonym-heavy true duplicate",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "What causes thunderstorms and lightning during monsoon?",
        "question2": "Why do lightning and thunder occur in rainy season?",
        "expected_label": 1,
        "description": "Paraphrased scientific question",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "What should I do if I lost my passport in a foreign country?",
        "question2": "How to get emergency travel documents when your passport is stolen abroad?",
        "expected_label": 1,
        "description": "Complex situation paraphrase",
    },
]


def evaluate_adversarial_suite(engine: SemantIQ) -> dict[str, Any]:
    """
    Evaluates SemantIQ inference engine against the curated adversarial benchmark.
    """
    results = []
    category_stats: dict[str, dict[str, int]] = {}

    for item in ADVERSARIAL_BENCHMARK_PAIRS:
        q1 = item["question1"]
        q2 = item["question2"]
        expected = item["expected_label"]
        cat = item["category"]

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0, "false_positives": 0, "false_negatives": 0}

        pred = engine.predict_pair(q1, q2)
        score = pred["score"]
        # In strict high-precision mode, DUPLICATE requires score >= T*
        is_pred_dup = 1 if pred["decision"] == "DUPLICATE" else 0
        
        is_correct = (is_pred_dup == expected)
        category_stats[cat]["total"] += 1
        if is_correct:
            category_stats[cat]["correct"] += 1
        elif expected == 0 and is_pred_dup == 1:
            category_stats[cat]["false_positives"] += 1
        elif expected == 1 and is_pred_dup == 0:
            category_stats[cat]["false_negatives"] += 1

        results.append({
            "category": cat,
            "question1": q1,
            "question2": q2,
            "expected_label": expected,
            "model_score": score,
            "decision": pred["decision"],
            "confidence": pred["confidence"],
            "contradiction_warning": pred["contradiction_warning"],
            "is_correct": is_correct,
            "description": item["description"],
        })

    total_pairs = len(results)
    total_correct = sum(1 for r in results if r["is_correct"])
    overall_acc = total_correct / total_pairs if total_pairs > 0 else 0.0

    summary = {
        "overall_accuracy": round(overall_acc, 4),
        "total_test_pairs": total_pairs,
        "total_correct": total_correct,
        "category_breakdown": {
            cat: {
                "total": stats["total"],
                "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
                "false_positives": stats["false_positives"],
                "false_negatives": stats["false_negatives"],
            }
            for cat, stats in category_stats.items()
        },
        "detailed_results": results,
    }
    return summary
