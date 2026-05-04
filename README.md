# Review Intelligence System

End-to-end NLP system for analyzing e-commerce reviews. Combines sentiment classification, fake review detection, and a leakage-corrected evaluation framework. Built on the Amazon Reviews 2023 dataset across three product categories.

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-active%20development-yellow)]()

---

## Project Highlights

- **Multi-category sentiment classifier** trained on 99K Amazon reviews across Electronics, Books, and Beauty categories
- **DistilBERT fine-tuning** with class weighting, achieving 0.70 macro F1 on test set (vs 0.66 for TF-IDF baseline)
- **Hybrid fake review detector** using supervised classification and unsupervised anomaly detection
- **Critical methodological insight**: Identified and quantified label leakage in the fake detection pipeline — a 40% performance drop between naive and corrected versions
- **Production-relevant evaluation** with macro F1, per-class metrics, and Precision@TopK for actionable model assessment

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup and Reproduction](#setup-and-reproduction)
- [Detailed Documentation](#detailed-documentation)
- [Lessons Learned](#lessons-learned)
- [Future Work](#future-work)

---

## Project Overview

E-commerce platforms receive millions of reviews daily. This project builds an analytical pipeline that:

1. **Classifies sentiment** (negative / neutral / positive) of reviews
2. **Flags suspicious reviews** that may be fake or template-generated
3. **Provides interpretable evaluation** suitable for production decision-making

The system is designed as a foundation for review intelligence applications: customer feedback analysis, content moderation pipelines, and trust scoring systems.

### Architecture

```
                Raw Reviews (99K)
                       │
                       ▼
              ┌─────────────────┐
              │   Preprocessing  │
              │   (deduplication,│
              │   text cleaning) │
              └─────────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
    ┌──────────────┐      ┌──────────────┐
    │  Module A:   │      │  Module B:   │
    │  Sentiment   │      │  Fake Review │
    │  Classifier  │      │  Detection   │
    └──────────────┘      └──────────────┘
            │                     │
            ▼                     ▼
       Predictions           Fakeness Score
```

---

## Dataset

### Source
Amazon Reviews 2023, downloaded via Hugging Face: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

### Composition
- **99,000 reviews** total
- **3 categories** with equal sampling: Electronics (33K), Books (33K), Beauty (33K)
- **Date range:** 1997-09-10 to 2023-03-20
- **Original rating distribution:** 1-star (6.8%), 2-star (4.6%), 3-star (8.5%), 4-star (17.3%), 5-star (62.8%)

### After Preprocessing
- **89,021 reviews** retained (89.9% retention rate)
- **3-class sentiment labels:** negative (11.9%), neutral (9.1%), positive (79.0%)
- **Train/Val/Test splits:** 71,216 / 8,902 / 8,903 (stratified by sentiment + category)
- **4,666 text duplicates** isolated for fake review detection

---

## Methodology

### Phase 1: Exploratory Data Analysis

Comprehensive analysis revealed:

- **Severe class imbalance**: 5-star reviews dominated at 62.8%, motivating reformulation as 3-class sentiment task
- **4,758 text-level duplicates**: Identical short reviews ("Good", "Great", "Five Stars") repeated across users — a fake review signal
- **Cross-category length differences**: Books reviews 3x longer than Beauty (151 vs 45 words average)
- **Counterintuitive verified-purchase pattern**: Non-verified buyers gave higher average ratings than verified ones

Detailed findings: [`reports/EDA_FINDINGS.md`](reports/EDA_FINDINGS.md)

### Phase 2: Preprocessing

- Removed 95 exact duplicates and 4,666 text-level duplicates
- Filtered reviews with fewer than 5 words
- Light text cleaning (URL/HTML removal, whitespace normalization)
- Created 3-class sentiment labels from 1-5 ratings
- Stratified train/val/test split on combined (sentiment, category) key

A reusable preprocessing module was built: [`src/data/preprocessor.py`](src/data/preprocessor.py)

### Phase 3: Baseline Modeling

Three classical baselines established a performance floor:

| Model | Approach | Purpose |
|---|---|---|
| Dummy (always positive) | Predicts majority class | Sanity check |
| TF-IDF + Logistic Regression | Linear classifier on bag-of-words | Strong interpretable baseline |
| TF-IDF + LightGBM | Gradient boosting on text features | Non-linear comparison |

### Phase 4: DistilBERT Fine-Tuning

- Fine-tuned `distilbert-base-uncased` on Google Colab Free Tier (Tesla T4 GPU)
- Class weights applied via custom WeightedTrainer to address imbalance
- Mixed precision (fp16) training: 22.8 minutes for 3 epochs on 71K examples
- Best model selected via `load_best_model_at_end` (epoch 2)

Detailed findings: [`reports/DISTILBERT_FINDINGS.md`](reports/DISTILBERT_FINDINGS.md)

### Phase 5: Fake Review Detection (Module B)

Two implementations were built — and the comparison between them is itself the deliverable.

**Version 1 (Naive):** Used heuristic-derived metadata features alongside TF-IDF. Achieved 93.4% F1.

**Version 2 (Robust):** After identifying label leakage, rebuilt with:
- User-stratified train/test split (no user appears in both)
- Text-only features (no metadata that defined the labels)
- Stricter labeling threshold

Result: F1 dropped to 56.0% — **the 40% performance gap quantifies the leakage problem**.

Detailed findings: [`reports/FAKE_DETECTION_FINDINGS.md`](reports/FAKE_DETECTION_FINDINGS.md)

---

## Results

### Sentiment Classification — Test Set Performance

| Model | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 | Inference (ms) |
|---|---|---|---|---|---|---|
| Dummy (always positive) | 0.7904 | 0.2943 | 0.0000 | 0.0000 | 0.8829 | < 0.01 |
| TF-IDF + LogReg | 0.8523 | 0.6555 | 0.6695 | 0.3665 | 0.9306 | < 0.01 |
| TF-IDF + LightGBM | 0.8544 | 0.6158 | 0.6380 | 0.2847 | 0.9247 | 0.02 |
| **DistilBERT** | **0.8468** | **0.7010** | **0.7333** | **0.4394** | **0.9303** | **2.59 (GPU)** |

### Key Findings

**1. The most impactful improvement was on the neutral class.**
DistilBERT improved neutral recall from 34.12% (LogReg) to 60.44% — a **76% relative improvement**. This is exactly where contextual embeddings outperform bag-of-words: 3-star reviews contain mixed sentiment ("good but slow") that requires contextual disambiguation.

**2. Counterintuitive accuracy/F1 tradeoff.**
DistilBERT achieved slightly lower accuracy than LogReg (0.8468 vs 0.8523) while macro F1 rose 6.9%. The model became more balanced across classes — exactly why macro F1 is the right metric for imbalanced classification.

**3. Surprising LightGBM underperformance.**
LightGBM scored 4 F1 points below LogReg on TF-IDF features. The cause: 99.79% feature sparsity makes tree-based models struggle. This is a known but underappreciated tradeoff.

### Confusion Matrix Comparison

DistilBERT dramatically improved minority class detection:

| Error Type | LogReg | DistilBERT | Change |
|---|---|---|---|
| True negative → predicted positive | 19.58% | 3.78% | **-15.8 pts** |
| True neutral → predicted positive | 44.13% | 22.87% | **-21.3 pts** |
| True positive → predicted neutral | 3.82% | 9.17% | +5.4 pts (acceptable cost) |

### Per-Category Analysis

| Category | LogReg F1 | DistilBERT F1 | Improvement |
|---|---|---|---|
| Beauty | 0.6555 | 0.7168 | +0.0613 |
| Books | 0.6029 | 0.6501 | +0.0472 |
| Electronics | 0.6527 | 0.7000 | +0.0473 |

Books remained the hardest category — likely due to longer reviews exceeding the 256-token truncation boundary and a higher rate of nuanced critical reviews.

### Fake Review Detection — Test Set Performance (Robust Version)

| Model | F1 | ROC-AUC | PR-AUC | Precision@Top100 |
|---|---|---|---|---|
| Logistic Regression | 0.5463 | 0.8188 | 0.5163 | 0.7000 |
| **LightGBM** | **0.5600** | **0.8327** | **0.5464** | **0.7700** |
| Isolation Forest | 0.0581 | 0.3914 | 0.1667 | 0.3400 |
| Hybrid (60/40) | 0.4560 | 0.8193 | 0.5383 | 0.7600 |

### The Leakage Discovery

| Metric | V1 (Naive) | V2 (Robust) | Drop |
|---|---|---|---|
| LightGBM F1 | 0.9339 | 0.5600 | **-40.0%** |
| Hybrid F1 | 0.9352 | 0.4560 | -51.2% |

The drop between V1 and V2 represents the size of the label leakage problem. V1 trained on features derived from the labels themselves, learning to reproduce the heuristics rather than detecting actual fakes.

**The corrected V2's Precision@Top100 of 0.77 is the production-relevant metric**: a Trust & Safety team manually reviewing the top 100 most suspicious reviews would find 77 actual flagged reviews — actionable for analyst-in-the-loop deployment.

---

## Project Structure

```
review-intelligence-system/
├── data/
│   ├── raw/                          # Original dataset (gitignored)
│   ├── processed/                    # Train/val/test parquet files (gitignored)
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory analysis
│   ├── 02_preprocessing.ipynb        # Text cleaning, label engineering, splits
│   ├── 03_baseline_models.ipynb      # TF-IDF + LogReg + LightGBM + Dummy
│   ├── 04_distilbert_finetuning.ipynb # DistilBERT on Colab
│   ├── 05_fake_review_detection.ipynb # V1: naive (with leakage)
│   └── 05b_fake_review_detection_robust.ipynb # V2: corrected
├── src/
│   ├── data/
│   │   └── preprocessor.py           # Reusable preprocessing module
│   └── models/
├── reports/
│   ├── EDA_FINDINGS.md               # 900-line EDA + preprocessing + baseline analysis
│   ├── DISTILBERT_FINDINGS.md        # DistilBERT detailed analysis
│   ├── FAKE_DETECTION_FINDINGS.md    # Fake detection + leakage analysis
│   ├── baseline_results.json         # Programmatic baseline metrics
│   ├── distilbert_results.json       # Programmatic BERT metrics
│   ├── fake_detection_results.json   # V1 fake detection metrics
│   ├── fake_detection_results_robust.json  # V2 metrics
│   └── figures/                      # All visualizations
├── models/                           # Trained models (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup and Reproduction

### Prerequisites
- Python 3.10+
- Linux/macOS (Windows requires WSL for some packages)
- ~10GB disk space (for dataset and model weights)
- For BERT fine-tuning: GPU recommended (Google Colab Free Tier sufficient)

### Installation

```bash
# Clone repository
git clone https://github.com/3RAV0/review-intelligence-system.git
cd review-intelligence-system

# Create virtual environment
python3 -m venv venv_review
source venv_review/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Reproduction Order

The notebooks should be run in numerical order:

1. **`01_eda.ipynb`**: Downloads dataset and performs EDA (~15 minutes)
2. **`02_preprocessing.ipynb`**: Generates train/val/test parquet files
3. **`03_baseline_models.ipynb`**: Trains classical baselines (~10 minutes)
4. **`04_distilbert_finetuning.ipynb`**: Open in Colab, requires GPU (~25 minutes)
5. **`05_fake_review_detection.ipynb`**: V1 fake detector (naive)
6. **`05b_fake_review_detection_robust.ipynb`**: V2 fake detector (recommended)

### Dataset Setup

The notebooks download the dataset automatically via the Hugging Face `datasets` library on first run. Subsequent runs use the cached download.

For Colab notebook (`04_distilbert_finetuning.ipynb`), the parquet files from notebook 2 must first be uploaded to Google Drive at `/content/drive/MyDrive/review-intelligence-system/data/processed/`.

---

## Detailed Documentation

This project includes three deep-dive analysis documents totaling ~2,000 lines of methodological discussion:

| Document | Length | Content |
|---|---|---|
| [EDA_FINDINGS.md](reports/EDA_FINDINGS.md) | ~900 lines | Initial data analysis, preprocessing decisions, and baseline modeling results |
| [DISTILBERT_FINDINGS.md](reports/DISTILBERT_FINDINGS.md) | ~650 lines | DistilBERT architecture, training configuration, results, and cross-model comparison |
| [FAKE_DETECTION_FINDINGS.md](reports/FAKE_DETECTION_FINDINGS.md) | ~470 lines | Fake detection methodology, leakage discovery, and corrected evaluation |

Each document includes a section of "Interview Talking Points" with concrete sentences usable for explaining the project verbally.

---

## Lessons Learned

### Technical Lessons

**1. Macro F1 over accuracy on imbalanced data.**
The dummy classifier predicting "always positive" achieved 79% accuracy while having 0% F1 on minority classes. This sanity check validated the choice of macro F1 as the primary metric throughout.

**2. Sparse feature spaces favor linear models.**
With 99.79% sparsity on TF-IDF features, Logistic Regression outperformed LightGBM by 4 macro F1 points. Tree-based models struggle to find good splits in such sparse high-dimensional spaces.

**3. Class weighting is essential, not optional.**
On a 79/12/9 imbalanced dataset, class weighting (or its equivalent) is required to prevent majority-class collapse. Both classical models and DistilBERT used `balanced` weights.

**4. Contextual embeddings shine on nuanced classes.**
DistilBERT's biggest improvement was on the neutral class — exactly where 3-star reviews use mixed sentiment that bag-of-words cannot disambiguate.

### Methodological Lessons

**1. Validate against an obviously-wrong baseline.**
Including the always-positive dummy classifier was the single best decision in the project. Any model failing to dramatically beat it would have been suspicious.

**2. A 60-point performance gap is a leakage signal.**
When my naive fake detector scored 93% F1 while the unsupervised baseline scored 30%, the gap itself was the smoking gun. Real-world tasks rarely show such gaps with healthy data.

**3. User-stratified splits are non-negotiable for user-keyed data.**
If the same user can appear in both train and test, individual user patterns leak information. `GroupShuffleSplit` is the safety mechanism.

**4. Features used to define labels cannot be features in the model.**
This is the core lesson from the fake detection module's V1 → V2 evolution. If `verified_purchase` defines the label, it cannot be a model input.

### Project Management Lessons

**1. Build evaluation infrastructure first.**
The `evaluate_model` function in notebook 3 was reused for every subsequent model. Investing in evaluation code paid back many times over.

**2. Document as you go, not at the end.**
The three findings.md files were written incrementally during the project. Going back to reconstruct the reasoning later would have been much harder.

**3. Ship V2 with the lessons of V1.**
Rather than hiding the failed V1 fake detector, both versions are committed and the comparison is documented as the central learning. This transparency is more valuable than appearing infallible.

---

## Future Work

### Immediate Next Steps (1-2 day improvements)

- **Aspect-Based Sentiment Analysis (Module C)**: Extract per-aspect sentiment using pre-trained ABSA models (e.g., `yangheng/deberta-v3-base-absa-v1.1`)
- **Streamlit dashboard**: Interactive demo with real-time predictions
- **FastAPI deployment**: Production-ready prediction endpoints
- **Hugging Face Spaces**: Live demo deployment

### Medium-Term Improvements

- **Multilingual evaluation**: Test BERTurk on Turkish e-commerce reviews
- **Long-review handling**: Head-tail truncation strategy for Books category
- **Hyperparameter tuning**: Optuna search for DistilBERT
- **Cross-validation**: Replace single train/val/test split with stratified k-fold
- **Sentence-BERT embeddings**: Replace TF-IDF for fake detection anomaly approach

### Production-Readiness Gaps

For real deployment, the system would need:
- Ground-truth fake review annotations (replace weak labels)
- A/B testing framework against rule-based baselines
- Drift monitoring (fake patterns evolve)
- Adversarial robustness testing
- Bias auditing across user demographics
- Latency SLA definition (p95, p99)

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

---

## Contact

**Mehmet Bahçeci**

- GitHub: [@3RAV0](https://github.com/3RAV0)
- Email: bahceci.mehmet@outlook.com
- Project Link: [https://github.com/3RAV0/review-intelligence-system](https://github.com/3RAV0/review-intelligence-system)

---

## Acknowledgments

- **Dataset**: McAuley Lab at UCSD for the Amazon Reviews 2023 dataset
- **Models**: Hugging Face Transformers library and the DistilBERT team
- **Compute**: Google Colab Free Tier for GPU access during BERT fine-tuning
