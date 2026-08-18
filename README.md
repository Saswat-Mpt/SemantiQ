# SemantIQ — Semantic Deduplication System

SemantIQ is a cost-aware semantic deduplication system for question-and-intent pairs that combines statistical properties, classical string algorithms, TF-IDF weighted n-grams, and dense sentence embeddings.

---

## 1. System Architecture & Representation Progression

SemantIQ uses a multi-representation fusion architecture feeding into a gradient-boosted decision engine (XGBoost) followed by a cost-aware precision-constrained decision policy ($T^*$).

```
Raw Pair (q1, q2)
       │
       ├── Statistical Features (8)       [lengths, counts, ratios]
       ├── Classical Lexical Features (8)  [Jaccard, RapidFuzz, character & word n-grams]
       ├── TF-IDF Representation (2)      [Word & Char TF-IDF Cosine]
       └── Dense Semantic Embedding (1)   [Pretrained all-MiniLM-L6-v2 Cosine]
       │
       ▼
19-Feature Fusion Vector
       │
       ▼
XGBoost Classifier (Experiment E)
       │
       ▼
Model Score P(Duplicate) ───[ T* = 0.8034 ]───► DUPLICATE (Precision >= 90%)
                                          └───► DISTINCT
```

---

## 2. Controlled Experimental Ablation (A → E)

All experiments were trained on the **identical Phase 1 partition** (198,394 train pairs) and evaluated on the **identical held-out test split** (9,178 pairs) with classifier hyperparameters held constant:

| Experiment | Feature Families | Feature Count | Test Precision | Test Recall | Test F1 | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | Statistical only | 8 | 0.5949 | 0.7037 | 0.6447 | 0.6296 |
| **B** | A + String / Jaccard / Fuzzy | 16 | 0.6412 | 0.7118 | 0.6746 | 0.7037 |
| **C** | B + Word & Char TF-IDF | 18 | 0.6612 | 0.7202 | 0.6894 | 0.7255 |
| **D** | A + Pretrained MiniLM | 9 | 0.7262 | 0.7912 | 0.7573 | 0.8115 |
| **E** | **Full Fusion (A + B + C + D)** | **19** | **0.7581** | **0.8052** | **0.7809** | **0.8353** |

### Key Scientific Findings:
1. **Semantic dominance:** Adding a single MiniLM semantic similarity feature to statistical features (**D vs A**) increases test PR-AUC by **+0.1819**, outperforming the entire 18-feature lexical stack (**C**) by **+0.0860**.
2. **Complementary fusion:** Combining all classical features with MiniLM (**E vs D**) gives an additional **+0.0238 PR-AUC** jump, proving that surface string alignment and dense semantics provide mutually beneficial signals.

---

## 3. Cost-Aware Decision Policy (Precision $\ge$ 90%)

In high-stakes deduplication workflows, merging non-duplicate questions causes irreversible context loss. Standard $T=0.50$ is uncalibrated for this cost structure.

* **Threshold Selection:** Swept exclusively on validation set to find the lowest threshold $T^*$ reaching $\ge 90\%$ precision.
* **Frozen Threshold:** $T^* = 0.80343205$

| Operating Point | Split | Precision | Recall | F1 | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Default ($T = 0.50$)** | Validation | 0.7764 | 0.8051 | 0.7905 | 0.8568 |
| **Cost-Aware ($T^* = 0.8034$)** | Validation | **0.9001** | **0.4031** | **0.5569** | **0.8568** |
| **Default ($T = 0.50$)** | Test | 0.7581 | 0.8052 | 0.7809 | 0.8353 |
| **Cost-Aware ($T^* = 0.8034$)** | Test | **0.8808** | **0.4027** | **0.5527** | **0.8353** |

*Under the strict $90\%$ precision constraint, the naive baseline collapsed to $0.23\%$ recall (3 pairs), whereas SemantIQ maintains $>40\%$ recall ($1,389$ true duplicate pairs captured with high confidence).*

---

## 4. Top Feature Importances (XGBoost Gain)

1. `minilm_cosine`: **52.20%**
2. `token_set_ratio`: **8.07%**
3. `word_tfidf_cosine`: **7.23%**
4. `common_word_count`: **4.64%**
5. `jaccard_similarity`: **4.17%**
6. `common_word_ratio`: **4.10%**
7. `word_trigram_overlap`: **2.47%**

---

## 5. Live Inference & Quickstart

### Installation
```bash
pip install -r requirements.txt
```

### Run Real-Time Prediction CLI
```bash
python scripts/predict.py
```

### Python API
```python
from src.inference import SemantIQ

detector = SemantIQ()
result = detector.predict_pair(
    "How can I learn Python?",
    "What is the best way to study Python?"
)

print(result["decision"])              # DUPLICATE / DISTINCT
print(result["duplicate_probability"])  # e.g., 0.7789
print(result["threshold"])              # 0.8034
```

---

## 6. Project Structure

```text
SemantiQ/
├── data/
│   ├── raw/                 # train.csv (Quora Question Pairs)
│   └── processed/           # Leakage-safe grouped splits (train/val/test)
├── src/
│   ├── phase1_data.py       # Grouped splitting & normalization
│   ├── phase2_baselines.py  # Heuristic & Logistic Regression baselines
│   ├── phase3_features.py   # Statistical & classical lexical feature extraction
│   ├── phase4_representations.py # Word/Char TF-IDF & MiniLM cosine encodings
│   ├── phase5_ablation.py   # XGBoost A->E ablation pipeline
│   ├── phase6_threshold.py  # Precision-constrained threshold selection
│   ├── phase7_analysis.py   # Error classification & feature importance
│   └── inference.py         # Production inference engine
├── scripts/
│   ├── run_phase1.py ... run_phase7.py
│   └── predict.py           # Live interactive CLI
├── artifacts/               # Saved models, vectorizers, feature stores, and metrics
└── reports/                 # Comprehensive JSON experiment reports
```
