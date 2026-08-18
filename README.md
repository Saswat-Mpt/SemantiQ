# SemantIQ: Cost-Aware Semantic Deduplication & Verification System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Pytest-Passing-brightgreen.svg)](https://docs.pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SemantIQ** is a cost-aware semantic deduplication and intent-verification system designed for high-stakes question-answering, search, and knowledge retrieval pipelines. It bridges classical surface lexical matching, TF-IDF weighted n-grams, and dense transformer embeddings with critical-token contradiction analysis.

---

## 1. System Architecture

```
                                  Raw Question Pair (q1, q2)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         19-Feature Fusion Pipeline                           Critical-Token Engine
                    │                                                   │
  ├── Statistical (8)  [Lengths, word counts, ratios]      ├── Numeric/Year mismatch (e.g., 2017 vs 2018)
  ├── Lexical (8)      [Jaccard, RapidFuzz, n-grams]       ├── Entity mismatch (e.g., IITD vs IITG)
  ├── TF-IDF (2)       [Word & Char TF-IDF Cosine]         ├── Negation mismatch (e.g., why vs why not)
  └── MiniLM (1)       [Pretrained Dense Cosine]           └── Question-starter shift (e.g., how vs what)
                    │                                                   │
                    ▼                                                   │
        XGBoost Fusion Classifier (Exp E)                               │
                    │                                                   │
                    ▼                                                   │
          Duplicate Score P ∈ [0, 1]                                    │
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              │
                                              ▼
                             3-Tier Calibrated Decision Policy
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

| Exp | Feature Family | Dimension | Test Precision | Test Recall | Test F1 | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | Statistical Only | 8 | 0.5949 | 0.7037 | 0.6447 | 0.6296 |
| **B** | A + String / Jaccard / Fuzzy | 16 | 0.6412 | 0.7118 | 0.6746 | 0.7037 |
| **C** | B + Word & Char TF-IDF | 18 | 0.6612 | 0.7202 | 0.6894 | 0.7255 |
| **D** | A + Pretrained MiniLM | 9 | 0.7262 | 0.7912 | 0.7573 | 0.8115 |
| **E** | **Full Fusion (A + B + C + D)** | **19** | **0.7581** | **0.8052** | **0.7809** | **0.8353** |

### Key Experimental Insights:
1. **Semantic Dominance:** Adding a single dense semantic representation (`minilm_cosine`) to simple statistics (**D vs A**) increases test PR-AUC by **+0.1819**, outperforming the entire 18-feature classical stack (**C**) by **+0.0860**.
2. **Complementary Fusion:** Combining all classical features with MiniLM (**E vs D**) delivers an additional **+0.0238 PR-AUC** gain, proving surface string alignment and dense semantics provide mutually beneficial signals.

---

## 3. Cost-Aware Decision Policy

In deduplication systems, false merges corrupt search indices and knowledge graphs. SemantIQ uses a validation-selected threshold $T^*$ targeting $\ge 90\%$ precision.

* **Threshold Selection Rule:** $T^* = \arg\max_T \text{Recall}(T) \quad \text{s.t.} \quad \text{Precision}(T) \ge 0.90 \quad (\text{Validation})$
* **Selected Operating Threshold:** $T^* = 0.80343205$

| Operating Policy | Split | Precision | Recall | F1 | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Default ($T = 0.50$)** | Validation | 0.7764 | 0.8051 | 0.7905 | 0.8568 |
| **Target 90% ($T^* = 0.8034$)** | Validation | **0.9001** | **0.4031** | **0.5569** | **0.8568** |
| **Default ($T = 0.50$)** | Held-Out Test | 0.7581 | 0.8052 | 0.7809 | 0.8353 |
| **Target 90% ($T^* = 0.8034$)** | **Held-Out Test** | **0.8808** | **0.4027** | **0.5527** | **0.8353** |

> *Note on evaluation transparency:* When $T^*$ is selected on validation to target $90\%$ precision, it achieves **88.08% precision on the untouched test set** while retaining $>40\%$ recall (1,389 verified duplicates captured), vs the naive string baseline which collapsed to $0.23\%$ recall.

---

## 4. Feature Importance Breakdown (XGBoost Gain)

```text
minilm_cosine        ██████████████████████████████ 52.20%
token_set_ratio      █████ 8.07%
word_tfidf_cosine    ████ 7.23%
common_word_count    ███ 4.64%
jaccard_similarity   ██ 4.17%
common_word_ratio    ██ 4.10%
word_trigram_overlap █ 2.47%
word_bigram_overlap  █ 1.78%
char_tfidf_cosine    █ 1.71%
```

---

## 5. Quickstart & Usage

### Installation
```bash
git clone https://github.com/Saswat-Mpt/SemantiQ.git
cd SemantiQ
python -m venv .venv
# On Windows: .venv\Scripts\activate | On Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Python API
```python
from src.inference import SemantIQ

detector = SemantIQ()
result = detector.predict_pair(
    "How can I learn Python for data science?",
    "What is the best way to study Python for data analysis?"
)

print(result["decision"])              # DUPLICATE / NEEDS_REVIEW / DISTINCT
print(result["score"])                 # 0.7789
print(result["confidence"])            # MODERATE
print(result["contradiction_warning"]) # False
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
* Health Check: `GET http://localhost:8000/health`
* Prediction: `POST http://localhost:8000/predict`

### Run Test Suite
```bash
pytest tests/ -v
```

### Reproduce Entire Pipeline End-to-End
```bash
python scripts/run_all.py
```

---

## 6. Repository Structure

```text
SemantiQ/
├── app/
│   └── api.py                   # Production FastAPI service
├── src/
│   ├── phase1_data.py           # Leakage-safe grouped partitioning
│   ├── phase2_baselines.py      # Statistical and naive baselines
│   ├── phase3_features.py       # Classical lexical/n-gram feature extraction
│   ├── phase4_representations.py# TF-IDF and MiniLM semantic representations
│   ├── phase5_ablation.py       # Controlled XGBoost A->E ablation
│   ├── phase6_threshold.py      # Precision-constrained decision calibration
│   ├── phase7_analysis.py       # Error taxonomy and failure analysis
│   ├── critical_tokens.py       # Contradiction & information-changing token engine
│   └── inference.py             # Single canonical inference engine
├── scripts/
│   ├── run_all.py               # Master one-click reproducibility runner
│   ├── predict.py               # Interactive CLI tool
│   └── run_phase1.py ... run_phase7.py
├── tests/                       # 15+ Automated pytest unit & symmetry tests
├── artifacts/                   # Experiment manifests, threshold policies, and metrics
└── reports/                     # Comprehensive JSON experiment reports
```
