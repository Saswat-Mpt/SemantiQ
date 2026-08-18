# SemantIQ: Cost-Aware Semantic Deduplication & Verification System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Pytest-20%20Passing-brightgreen.svg)](https://docs.pytest.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SemantIQ** is a cost-aware semantic deduplication and intent-verification system designed for retrieval, search, question-answering, and knowledge-management pipelines where false merges can be costly. It combines classical surface lexical matching, TF-IDF weighted n-grams, and dense sentence transformers with critical-token contradiction verification and cost-aware decision thresholding.

---

## 1. System Architecture

```
                                  Raw Question Pair (q1, q2)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
      19-Feature Fusion Pipeline (Model E)                    Critical-Token Diagnostics
                    │                                                   │
  ├── Statistical (8)  [Lengths, word counts, ratios]      ├── Numeric/Year mismatch (e.g., 2017 vs 2018)
  ├── Lexical (8)      [Jaccard, RapidFuzz, n-grams]       ├── Entity mismatch (e.g., IITD vs IITG)
  ├── TF-IDF (2)       [Word & Char TF-IDF Cosine]         ├── Negation mismatch (e.g., why vs why not)
  └── MiniLM (1)       [Pretrained Dense Cosine]           └── Question-starter shift (e.g., how vs what)
                    │                                                   │
                    ▼                                                   ▼
       XGBoost Champion Classifier (Exp E)                Contradiction Verification Layer
                    │                                                   │
                    ▼                                                   │
             Raw Model Score P ∈ [0, 1]                                 │
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                              3-Tier Cost-Aware Decision Policy
                                              │
                   ┌──────────────────────────┼──────────────────────────┐
                   ▼                          ▼                          ▼
              DUPLICATE                  NEEDS_REVIEW                 DISTINCT
         (Score ≥ 0.8034)           (0.50 ≤ Score < 0.8034)        (Score < 0.50)
      High-Confidence Merge         Human Review Required        Independent Queries
```

---

## 2. Controlled Experimental Ablation (A → E)

All experiments were trained on the **identical Phase 1 partition** (198,394 train pairs) and evaluated on the **identical held-out test split** (9,178 pairs) with classifier hyperparameters held constant:

| Exp | Feature Family | Dim | Test Precision | Test Recall | Test F1 | Test PR-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | Statistical Only | 8 | 0.5949 | 0.7037 | 0.6447 | 0.6296 | Baseline |
| **B** | A + String / Jaccard / Fuzzy | 16 | 0.6412 | 0.7118 | 0.6746 | 0.7037 | Feature Layer |
| **C** | B + Word & Char TF-IDF | 18 | 0.6612 | 0.7202 | 0.6894 | 0.7255 | Sparse Representation |
| **D** | A + Pretrained MiniLM | 9 | 0.7262 | 0.7912 | 0.7573 | 0.8115 | Dense Representation |
| **E** | **Full Fusion (A + B + C + D)** | **19** | **0.7581** | **0.8052** | **0.7809** | **0.8353** | **Production Champion** |
| *F* | *E + Contradiction Signals* | *24* | *0.7640* | *0.8072* | *0.7850* | *0.8404* | *Phase 8 Experiment* |

### Key Experimental Insights:
1. **Semantic Dominance:** Adding a single dense semantic representation (`minilm_cosine`) to simple statistics (**D vs A**) increases test PR-AUC by **+0.1819**, outperforming the entire 18-feature classical stack (**C**) by **+0.0860**.
2. **Complementary Fusion:** Combining all classical features with MiniLM (**E vs D**) provides an additional **+0.0238 PR-AUC** gain.
3. **Production Deployment Choice:** While experimental **Model F** slightly improves PR-AUC (+0.0051), **Model E (19 features)** is selected as the primary production champion for architectural simplicity, maintaining contradiction diagnostics as a verification layer.

---

## 3. Cost-Aware Decision Policy & Statistical Rigor

In deduplication systems, false merges corrupt search indices and knowledge graphs. SemantIQ uses a validation-selected threshold $T^*$ targeting $\ge 90\%$ precision.

* **Threshold Selection Rule:** $T^* = \arg\max_T \text{Recall}(T) \quad \text{s.t.} \quad \text{Precision}(T) \ge 0.90 \quad (\text{Validation})$
* **Selected Operating Threshold:** $T^* = 0.80343205$

### Evaluation on Held-Out Test Set (with 95% Bootstrap Confidence Intervals)

| Operating Policy | Test Precision (95% CI) | Test Recall (95% CI) | Test F1 | Test PR-AUC (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Default ($T = 0.50$)** | 75.81% [74.5% – 77.1%] | 80.52% [79.2% – 81.8%] | 0.7809 | **0.8353 [0.8223 – 0.8474]** |
| **Cost-Aware ($T^* = 0.8034$)** | **88.08% [86.33% – 89.56%]** | **40.27% [38.51% – 41.99%]** | **0.5527** | **0.8353 [0.8223 – 0.8474]** |

> *Evaluation Transparency:* When $T^*$ is selected on validation to target $90\%$ precision, it achieves **88.08% precision on the untouched test set** (1,389 verified duplicates captured), vs the naive string baseline which collapsed to $0.23\%$ recall (3 pairs).

---

## 4. Probability Calibration Analysis

Evaluated on held-out test data:
* **Raw XGBoost Classifier:** Brier Score = `0.11720`, Expected Calibration Error (ECE) = **`0.01202` (1.20%)**.
* **Platt Scaling (Sigmoid):** Brier Score = `0.12004`, ECE = `0.03926`.
* **Isotonic Regression:** Brier Score = `0.11774`, ECE = `0.01549`.

*Raw XGBoost tree probabilities demonstrated the lowest Brier error and calibration error, so uncalibrated raw model probabilities are retained for production inference.*

---

## 5. Curated Adversarial Benchmark Breakdown

Evaluated against the 48-pair curated adversarial challenge suite (`src/adversarial_suite.py`):

| Challenge Category | Example Test Pair | Total | Correct | Accuracy | False Positives | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Entity Substitutions** | *"Placement package at IIT Delhi"* vs *"IIT Bombay"* | 8 | 8 | **100.0%** | **0** | 0 |
| **Numeric & Year Shifts** | *"JEE cutoff in 2017"* vs *"2018"*, *"50% vs 90%"* | 8 | 8 | **100.0%** | **0** | 0 |
| **Intent / Career Shifts** | *"What is ML"* vs *"How to get an ML engineer job"* | 7 | 7 | **100.0%** | **0** | 0 |
| **Tool / Language Shifts** | *"Web server in C"* vs *"Web server in C++"* | 6 | 6 | **100.0%** | **0** | 0 |
| **Negation Inversions** | *"Why learn Python"* vs *"Why shouldn't I learn Python"* | 7 | 6 | **85.7%** | **1** | 0 |
| **Paraphrase Duplicates** | *"Shed excess body weight"* vs *"Slim down rapidly"* | 12 | 1 | **8.3%** | 0 | **11** |

### Scientific Finding on Model Strengths & Weaknesses:
* **Strengths:** Under strict precision thresholding ($T^* = 0.8034$), SemantIQ provides near-perfect protection (**0% False Positive Rate**) against entity swaps, numeric differences, tool substitutions, and intent shifts.
* **Weaknesses:** Because high precision requires conservative thresholding, extreme zero-overlap paraphrases are flagged as `NEEDS_REVIEW` or `DISTINCT` rather than auto-merged. This demonstrates why the 3-tier policy is essential for human-in-the-loop review.

---

## 6. Quickstart & Usage

### Installation
```bash
git clone https://github.com/Saswat-Mpt/SemantiQ.git
cd SemantiQ
python -m venv .venv
# On Windows: .venv\Scripts\activate | On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Python API (with LRU Embedding Cache & Batch Vectorization)
```python
from src.inference import SemantIQ

detector = SemantIQ()

# Single pair prediction
result = detector.predict_pair(
    "How can I learn Python for data science?",
    "What is the best way to study Python for data analysis?"
)
print(result["decision"])              # DUPLICATE / NEEDS_REVIEW / DISTINCT
print(result["score"])                 # 0.7789
print(result["decision_band"])         # HUMAN_REVIEW_REQUIRED
print(result["contradiction_warning"]) # False

# High-throughput batch prediction
batch_results = detector.predict_batch([
    ("What is machine learning?", "Can someone explain machine learning?"),
    ("Why is the sky blue?", "How do commercial airplanes fly?"),
])
```

### Interactive CLI
```bash
python scripts/predict.py
```

### Production REST API (FastAPI)
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* Liveness / Readiness Probes: `GET http://localhost:8000/live` | `GET http://localhost:8000/ready`
* Versioned Prediction: `POST http://localhost:8000/api/v1/predict`
* Batch Prediction: `POST http://localhost:8000/api/v1/batch-predict`

### Docker Container Deployment
```bash
docker build -t semantiq:latest .
docker run -p 8000:8000 semantiq:latest
```

### Run 20-Test Automated Pytest Suite
```bash
pytest tests/ -v
```

### Run Master One-Click Reproducibility Runner
```bash
python scripts/run_all.py
```

---

## 7. Repository Structure

```text
SemantiQ/
├── app/
│   └── api.py                   # Production-hardened versioned FastAPI service
├── src/
│   ├── phase1_data.py           # Leakage-safe grouped partitioning
│   ├── phase2_baselines.py      # Statistical and naive baselines
│   ├── phase3_features.py       # Classical lexical/n-gram feature extraction
│   ├── phase4_representations.py# TF-IDF and MiniLM semantic representations
│   ├── phase5_ablation.py       # Controlled XGBoost A->E ablation
│   ├── phase6_threshold.py      # Precision-constrained decision calibration
│   ├── phase7_analysis.py       # Error taxonomy and failure analysis
│   ├── phase8_enhanced_verification.py # Model F, calibration & bootstrap CIs
│   ├── critical_tokens.py       # Contradiction & information-changing token engine
│   ├── adversarial_suite.py     # 48-Pair curated adversarial evaluation benchmark
│   └── inference.py             # Unified inference engine with LRU cache & batching
├── scripts/
│   ├── run_all.py               # Master one-click reproducibility runner
│   ├── run_phase8.py            # Phase 8 calibration & Model F runner
│   ├── run_adversarial_eval.py  # Adversarial evaluation runner
│   ├── predict.py               # Interactive CLI tool
│   └── run_phase1.py ... run_phase7.py
├── tests/                       # 20 Automated pytest unit, symmetry & API tests
│   ├── test_api.py
│   ├── test_critical_tokens.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_normalization.py
│   └── test_symmetry.py
├── artifacts/                   # Models, calibration scalers, manifests, and metrics
├── reports/                     # Comprehensive JSON experiment reports
├── Dockerfile                   # Production container definition
├── docker-compose.yml           # Container orchestration
└── .github/workflows/ci.yml     # Automated GitHub Actions CI workflow
```
