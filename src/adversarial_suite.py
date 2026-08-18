from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd
from src.inference import SemantIQ


# ============================================================
# Curated Adversarial Benchmark (50 Comprehensive Pairs)
# ============================================================

ADVERSARIAL_BENCHMARK_PAIRS = [
    # ── Category 1: Entity Substitutions (Expected: 0 / Non-Duplicate) ──
    {
        "category": "entity_substitution",
        "question1": "What is the average placement package at IIT Delhi?",
        "question2": "What is the average placement package at IIT Bombay?",
        "expected_label": 0,
        "description": "Institution name substitution (IIT Delhi vs IIT Bombay)",
    },
    {
        "category": "entity_substitution",
        "question1": "How do I prepare for Google software engineering interview?",
        "question2": "How do I prepare for Microsoft software engineering interview?",
        "expected_label": 0,
        "description": "Tech company name substitution (Google vs Microsoft)",
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
        "question1": "Who is the current CEO of Apple Inc?",
        "question2": "Who is the current CEO of Tesla Inc?",
        "expected_label": 0,
        "description": "Entity swap in factual inquiry",
    },
    {
        "category": "entity_substitution",
        "question1": "What are the tourist attractions to visit in Tokyo?",
        "question2": "What are the tourist attractions to visit in Kyoto?",
        "expected_label": 0,
        "description": "City name substitution (Tokyo vs Kyoto)",
    },
    {
        "category": "entity_substitution",
        "question1": "How do I apply for a student visa to Canada?",
        "question2": "How do I apply for a student visa to Australia?",
        "expected_label": 0,
        "description": "Country name substitution in immigration query",
    },
    {
        "category": "entity_substitution",
        "question1": "What is the plot summary of Harry Potter and the Sorcerer's Stone?",
        "question2": "What is the plot summary of Harry Potter and the Chamber of Secrets?",
        "expected_label": 0,
        "description": "Book title subtitle substitution in franchise",
    },
    {
        "category": "entity_substitution",
        "question1": "How do I transfer money from Chase Bank to Wells Fargo?",
        "question2": "How do I transfer money from Chase Bank to Bank of America?",
        "expected_label": 0,
        "description": "Target banking institution substitution",
    },

    # ── Category 2: Numeric & Year Shifts (Expected: 0 / Non-Duplicate) ──
    {
        "category": "numeric_year_shift",
        "question1": "What was the JEE Advanced cutoff rank in 2017?",
        "question2": "What was the JEE Advanced cutoff rank in 2018?",
        "expected_label": 0,
        "description": "Year token shift (2017 vs 2018)",
    },
    {
        "category": "numeric_year_shift",
        "question1": "How can a student score 50% in 12th board exams?",
        "question2": "How can a student score 90% in 12th board exams?",
        "expected_label": 0,
        "description": "Target percentage mismatch (50% vs 90%)",
    },
    {
        "category": "numeric_year_shift",
        "question1": "What are the technical specs of iPhone 14 Pro?",
        "question2": "What are the technical specs of iPhone 15 Pro?",
        "expected_label": 0,
        "description": "Product version number mismatch (14 vs 15)",
    },
    {
        "category": "numeric_year_shift",
        "question1": "How long does it take to run 5 km for beginners?",
        "question2": "How long does it take to run 10 km for beginners?",
        "expected_label": 0,
        "description": "Distance metric mismatch (5 km vs 10 km)",
    },
    {
        "category": "numeric_year_shift",
        "question1": "What is the interest rate for a 15 year fixed mortgage?",
        "question2": "What is the interest rate for a 30 year fixed mortgage?",
        "expected_label": 0,
        "description": "Mortgage duration mismatch (15 vs 30 years)",
    },
    {
        "category": "numeric_year_shift",
        "question1": "How much salary can I expect with 2 years of experience in Java?",
        "question2": "How much salary can I expect with 10 years of experience in Java?",
        "expected_label": 0,
        "description": "Experience duration mismatch (2 vs 10 years)",
    },
    {
        "category": "numeric_year_shift",
        "question1": "What was the GDP growth rate of India in 2020?",
        "question2": "What was the GDP growth rate of India in 2023?",
        "expected_label": 0,
        "description": "Macroeconomic temporal shift (2020 vs 2023)",
    },
    {
        "category": "numeric_year_shift",
        "question1": "What is the battery capacity of Samsung Galaxy S23?",
        "question2": "What is the battery capacity of Samsung Galaxy S24?",
        "expected_label": 0,
        "description": "Device model number shift (S23 vs S24)",
    },

    # ── Category 3: Negation Flips (Expected: 0 / Non-Duplicate) ──
    {
        "category": "negation_flip",
        "question1": "Why should I learn Python programming language?",
        "question2": "Why shouldn't I learn Python programming language?",
        "expected_label": 0,
        "description": "Negation modifier changing core premise (should vs shouldn't)",
    },
    {
        "category": "negation_flip",
        "question1": "How can I invest in the stock market safely?",
        "question2": "How can I avoid investing in the stock market?",
        "expected_label": 0,
        "description": "Polarity and action inversion (invest vs avoid investing)",
    },
    {
        "category": "negation_flip",
        "question1": "Is drinking green tea good for health?",
        "question2": "Is drinking green tea bad for health?",
        "expected_label": 0,
        "description": "Antonym / polarity swap (good vs bad)",
    },
    {
        "category": "negation_flip",
        "question1": "Can I travel to Europe without a visa?",
        "question2": "Can I travel to Europe with a visa?",
        "expected_label": 0,
        "description": "Prepositional negation (with vs without)",
    },
    {
        "category": "negation_flip",
        "question1": "Why is democracy the best form of government?",
        "question2": "Why is democracy not the best form of government?",
        "expected_label": 0,
        "description": "Negative particle insertion (is vs is not)",
    },
    {
        "category": "negation_flip",
        "question1": "How do I enable cookies in Google Chrome?",
        "question2": "How do I disable cookies in Google Chrome?",
        "expected_label": 0,
        "description": "Functional antonym (enable vs disable)",
    },
    {
        "category": "negation_flip",
        "question1": "Should I accept a counter offer from my current employer?",
        "question2": "Should I reject a counter offer from my current employer?",
        "expected_label": 0,
        "description": "Action polarity inversion (accept vs reject)",
    },

    # ── Category 4: Question-Type / Intent Shift (Expected: 0 / Non-Duplicate) ──
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
    {
        "category": "intent_shift",
        "question1": "What is the history of the Eiffel Tower?",
        "question2": "How much are ticket prices for the Eiffel Tower?",
        "expected_label": 0,
        "description": "Historical inquiry vs ticketing price query",
    },
    {
        "category": "intent_shift",
        "question1": "How do I learn quantum physics from scratch?",
        "question2": "Who discovered quantum physics?",
        "expected_label": 0,
        "description": "Learning path vs biographical historical query",
    },
    {
        "category": "intent_shift",
        "question1": "What does a data analyst do on a daily basis?",
        "question2": "What are the best certifications for data analysts?",
        "expected_label": 0,
        "description": "Job description vs certification recommendation",
    },
    {
        "category": "intent_shift",
        "question1": "Why is the sky blue during daylight?",
        "question2": "How to photograph the blue sky with a DSLR camera?",
        "expected_label": 0,
        "description": "Scientific cause vs photography tutorial",
    },

    # ── Category 5: Tool / Language Substitutions (Expected: 0 / Non-Duplicate) ──
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
        "description": "Framework substitution (Django vs FastAPI)",
    },
    {
        "category": "tool_substitution",
        "question1": "What is the difference between MySQL and PostgreSQL?",
        "question2": "What is the difference between MySQL and MongoDB?",
        "expected_label": 0,
        "description": "Database comparison swap (PostgreSQL vs MongoDB)",
    },
    {
        "category": "tool_substitution",
        "question1": "How to train an image classification model in PyTorch?",
        "question2": "How to train an image classification model in TensorFlow?",
        "expected_label": 0,
        "description": "Deep learning library substitution (PyTorch vs TensorFlow)",
    },
    {
        "category": "tool_substitution",
        "question1": "How do I deploy a web application on AWS EC2?",
        "question2": "How do I deploy a web application on Google Cloud Run?",
        "expected_label": 0,
        "description": "Cloud service provider substitution (AWS vs GCP)",
    },
    {
        "category": "tool_substitution",
        "question1": "How to format code automatically in VS Code?",
        "question2": "How to format code automatically in PyCharm?",
        "expected_label": 0,
        "description": "IDE substitution (VS Code vs PyCharm)",
    },

    # ── Category 6: Legitimate Paraphrases (Expected: 1 / True Duplicate) ──
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
        "description": "Synonym-heavy true duplicate (salary/paid, entry level/fresher)",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "What causes thunderstorms and lightning during monsoon?",
        "question2": "Why do lightning and thunder occur in rainy season?",
        "expected_label": 1,
        "description": "Paraphrased scientific question with synonym swaps",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "What should I do if I lost my passport in a foreign country?",
        "question2": "How to get emergency travel documents when your passport is stolen abroad?",
        "expected_label": 1,
        "description": "Complex situation paraphrase with shared underlying query",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "How do I improve my spoken English fluency and confidence?",
        "question2": "What are the best methods to speak English more fluently?",
        "expected_label": 1,
        "description": "Language skill improvement query paraphrase",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "What is the best way to invest money for long term wealth creation?",
        "question2": "How should one invest funds to build wealth over the long haul?",
        "expected_label": 1,
        "description": "Financial planning paraphrase",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "Why is my laptop computer running so slowly all of a sudden?",
        "question2": "What causes a laptop to experience sudden severe lag and sluggishness?",
        "expected_label": 1,
        "description": "Tech support query paraphrase",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "How can someone overcome insomnia and fall asleep faster at night?",
        "question2": "What remedies help people with sleeping difficulties get to sleep quickly?",
        "expected_label": 1,
        "description": "Medical sleep problem paraphrase",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "What are effective ways to prepare for university examinations?",
        "question2": "How should college students study to score high marks in finals?",
        "expected_label": 1,
        "description": "Exam preparation paraphrase",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "Can you recommend good books for personal finance beginners?",
        "question2": "Which financial literacy books should a novice read first?",
        "expected_label": 1,
        "description": "Book recommendation query paraphrase",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "How to remove stubborn oil stains from cotton clothing?",
        "question2": "What is the best household remedy for grease stains on shirts?",
        "expected_label": 1,
        "description": "Household cleaning remedy paraphrase",
    },
    {
        "category": "paraphrase_duplicate",
        "question1": "Is it possible to learn computer programming without a degree in CS?",
        "question2": "Can someone become a software engineer without having studied computer science in college?",
        "expected_label": 1,
        "description": "Career self-study qualification paraphrase",
    },
]


def evaluate_adversarial_suite(engine: SemantIQ) -> dict[str, Any]:
    """
    Evaluates SemantIQ inference engine against the curated 50-pair adversarial benchmark.
    Reports raw counts, per-category accuracy, false positive counts, and false negative counts.
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
        
        # In strict cost-aware mode, DUPLICATE requires score >= T*
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
            "decision_band": pred.get("decision_band", pred.get("confidence", "MODERATE")),
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
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
                "false_positives": stats["false_positives"],
                "false_negatives": stats["false_negatives"],
            }
            for cat, stats in category_stats.items()
        },
        "detailed_results": results,
    }
    return summary
