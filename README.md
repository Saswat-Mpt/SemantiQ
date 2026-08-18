# SemantIQ: Semantic Duplicate Question Detection & Verification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Pytest-20%20Passing-brightgreen.svg)](https://docs.pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SemantIQ is an NLP system for detecting semantically duplicate question pairs and preventing false merges in search or question-answering systems. It combines classical lexical overlap features, sparse TF-IDF n-grams, and dense sentence embeddings with an explicit cost-aware threshold policy and rule-based critical token checks.

---

## 1. System Architecture

```text
                           Question Pair (q1, q2)
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         19-Feature Pipeline (Model E)       Critical-Token Checks
                    │                                 │
  ├── Statistical (8) [lengths, word counts]    ├── Number / year mismatch
  ├── Lexical (8)     [Jaccard, RapidFuzz]      ├── Entity / proper noun mismatch
  ├── TF-IDF (2)      [Word & char cosine]      ├── Negation mismatch
  └── MiniLM (1)      [Pretrained cosine]       └── Question starter mismatch
                    │                                 │
                    ▼                                 ▼
           XGBoost Classifier (Exp E)         Rule-Based Diagnostics
                    │                                 │
                    ▼                                 │
         XGBoost Model Score ∈ [0, 1]                 │
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                      3-Tier Cost-Aware Decision
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
             DUPLICATE          NEEDS_REVIEW          DISTINCT
          (Score ≥ 0.8034)  (0.50 ≤ Score < 0.8034) (Score < 0.50)
```

* **Deployed Model:** **Experiment E (19 features)**.
* **Experimental Extension (Phase 8):** Model F (24 features with embedded contradiction signals) achieves +0.0051 PR-AUC (0.8404); Model E is kept as the deployed model for simplicity.
* **Probability Calibration Study:** Evaluated raw XGBoost vs Platt scaling and isotonic regression. Raw XGBoost gave the lowest Brier score (0.1172) and lowest expected calibration error (1.20%), so raw XGBoost model scores are used directly without post-hoc transforms.

---

## 2. Controlled Feature Ablation (A → E)

All models were trained on the exact same Phase 1 training split (198,394 pairs) and evaluated on the exact same held-out test split (9,178 pairs) with fixed XGBoost hyperparameters:

| Experiment | Feature Family | Dimensions | Test Precision | Test Recall | Test F1 | Test PR-AUC | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | Statistical only | 8 | 0.5949 | 0.7037 | 0.6447 | 0.6296 | Baseline |
| **B** | A + String / Jaccard / Fuzzy | 16 | 0.6412 | 0.7118 | 0.6746 | 0.7037 | Lexical stack |
| **C** | B + Word & Char TF-IDF | 18 | 0.6612 | 0.7202 | 0.6894 | 0.7255 | Sparse features |
| **D** | A + Pretrained MiniLM | 9 | 0.7262 | 0.7912 | 0.7573 | 0.8115 | Dense feature |
| **E** | **Full Fusion (A + B + C + D)** | **19** | **0.7581** | **0.8052** | **0.7809** | **0.8353** | **Deployed model** |
| *F* | *E + Contradiction signals* | *24* | *0.7640* | *0.8072* | *0.7850* | *0.8404* | *Phase 8 experiment* |

### Findings:
1. **Dense semantics matter most:** Adding a single dense similarity score (`minilm_cosine`) to basic statistics (**D vs A**) increases test PR-AUC by **+0.1819**, outperforming the entire 18-feature classical stack (**C**) by **+0.0860**.
2. **Lexical features still help:** Combining classical features with MiniLM (**E vs D**) gives an additional **+0.0238 PR-AUC** gain.

---

## 3. Decision Policy & Test Evaluation

In duplicate detection, false merges corrupt search indices and knowledge bases. To control this, we select an operating threshold $T^*$ on the validation set targeting $\ge 90\%$ precision.

* **Threshold Selection Rule:** $T^* = \arg\max_T \text{Recall}(T) \quad \text{s.t.} \quad \text{Precision}(T) \ge 0.90 \quad (\text{Validation})$
* **Selected Threshold:** $T^* = 0.80343205$

### Held-Out Test Evaluation (with 95% bootstrap confidence intervals)

| Operating Policy | Test Precision (95% CI) | Test Recall (95% CI) | Test F1 | Test PR-AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Default ($T = 0.50$)** | 75.81% [74.5% – 77.1%] | 80.52% [79.2% – 81.8%] | 0.7809 | **0.8353 [0.8223 – 0.8474]** |
| **Cost-Aware ($T^* = 0.8034$)** | **88.08% [86.33% – 89.56%]** | **40.27% [38.51% – 41.99%]** | **0.5527** | **0.8353 [0.8223 – 0.8474]** |

> **Note on test precision:** A threshold selected to target 90% precision on validation achieved **88.08% precision on the untouched test set** while keeping $>40\%$ recall (1,389 duplicate pairs found), compared to naive string matching which found only 3 pairs (0.23% recall).

---

## 4. Curated Adversarial Stress Test

We tested the model against a curated set of 48 challenging question pairs:

| Category | Example Pair | Total ($N$) | Correct | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Entity Substitutions** | *"Placement package at IIT Delhi"* vs *"IIT Bombay"* | 8 | **8 / 8** | **0** | 0 |
| **Numeric & Year Shifts** | *"JEE cutoff in 2017"* vs *"2018"*, *"50% vs 90%"* | 8 | **8 / 8** | **0** | 0 |
| **Intent Shifts** | *"What is ML"* vs *"How to get an ML engineer job"* | 7 | **7 / 7** | **0** | 0 |
| **Tool Substitutions** | *"Web server in C"* vs *"Web server in C++"* | 6 | **6 / 6** | **0** | 0 |
| **Negation Inversions** | *"Why learn Python"* vs *"Why shouldn't I learn Python"* | 7 | **6 / 7** | **1** | 0 |
| **Paraphrase Duplicates** | *"Shed excess body weight"* vs *"Slim down rapidly"* | 12 | **1 / 12** | 0 | **11** |

### Known Limitations:
* **Strengths:** Across the entity, numeric, tool, and intent-substitution categories, no false merges were observed. Negation remains a harder case, with one false merge in the curated set (*"Is drinking green tea good for health?"* vs *"Is drinking green tea bad for health?"*).
* **Weakness:** Because the model relies on lexical overlap and bi-encoder cosine similarity, paraphrases with completely different vocabulary (zero word overlap) receive lower scores and get marked as `NEEDS_REVIEW` or `DISTINCT`. This is why the 3-tier review policy is used instead of auto-merging.

---

## 5. Setup & Usage

### Installation
```bash
git clone https://github.com/Saswat-Mpt/SemantiQ.git
cd SemantiQ
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Python API
```python
from src.inference import SemantIQ

detector = SemantIQ()

# Single pair
result = detector.predict_pair(
    "How can I learn Python for data science?",
    "What is the best way to study Python for data analysis?"
)
print(result["decision"])      # DUPLICATE / NEEDS_REVIEW / DISTINCT
print(result["score"])         # 0.7789 (XGBoost model score)
print(result["decision_band"]) # HUMAN_REVIEW_REQUIRED

# Batch prediction
batch_results = detector.predict_batch([
    ("What is machine learning?", "Can someone explain machine learning?"),
    ("Why is the sky blue?", "How do commercial airplanes fly?"),
])
```

### CLI
```bash
python scripts/predict.py
```

### REST API (FastAPI)
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
* Docs: `http://localhost:8000/docs`
* Health probes: `GET /live` and `GET /ready`
* Predict: `POST /api/v1/predict`
* Batch: `POST /api/v1/batch-predict`

### Run Tests
```bash
pytest tests/ -v
```

### Run Full Pipeline
```bash
python scripts/run_all.py
```

---

## 6. Project Structure

```text
SemantiQ/
├── LICENSE                      # MIT License
├── app/
│   └── api.py                   # FastAPI service
├── src/
│   ├── phase1_data.py           # Leakage-safe grouped partitioning
│   ├── phase2_baselines.py      # Statistical and naive baselines
│   ├── phase3_features.py       # Classical lexical/n-gram feature extraction
│   ├── phase4_representations.py# TF-IDF and MiniLM representations
│   ├── phase5_ablation.py       # Controlled XGBoost A->E ablation
│   ├── phase6_threshold.py      # Precision-constrained threshold calibration
│   ├── phase7_analysis.py       # Error taxonomy and failure analysis
│   ├── phase8_enhanced_verification.py # Model F experiment, calibration & CIs
│   ├── critical_tokens.py       # Contradiction & critical token checks
│   ├── adversarial_suite.py     # Curated adversarial stress test
│   └── inference.py             # Inference engine with embedding cache & batching
├── scripts/
│   ├── run_all.py               # End-to-end pipeline runner
│   ├── run_phase8.py            # Phase 8 experiment runner
│   ├── run_adversarial_eval.py  # Adversarial evaluation runner
│   ├── predict.py               # Interactive CLI
│   └── run_phase1.py ... run_phase7.py
├── tests/                       # 20 pytest unit, symmetry & API tests
│   ├── test_api.py
│   ├── test_critical_tokens.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_normalization.py
│   └── test_symmetry.py
├── artifacts/                   # Experiment manifests, threshold policies, and metrics
├── reports/                     # JSON experiment reports
├── Dockerfile                   # Docker container definition
├── docker-compose.yml           # Container orchestration
└── .github/workflows/ci.yml     # GitHub Actions workflow
```
