# EDA Findings — Amazon Multi-Category Reviews

**Project:** Review Intelligence System  
**Notebook:** `notebooks/01_eda.ipynb`  
**Dataset:** McAuley-Lab/Amazon-Reviews-2023  
**Categories:** Electronics, Books, Beauty (33,000 reviews each)  
**Total Records:** 99,000 reviews  
**Date Range:** 1997-09-10 → 2023-03-20

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total reviews | 99,000 |
| Columns | 11 |
| Memory usage | 105.9 MB |
| Missing values | 0 |
| Categories | 3 (Electronics, Books, Beauty — 33K each) |

### Columns
`rating`, `title`, `text`, `images`, `asin`, `parent_asin`, `user_id`, `timestamp`, `helpful_vote`, `verified_purchase`, `category`

**Why this matters:** Dataset is clean (no missing values), well-balanced across categories, and large enough for both classical ML and deep learning approaches.

---

## 2. Target Variable — Rating Distribution

### Raw Distribution

| Rating | Count | Percentage |
|---|---|---|
| 1.0 | 6,728 | 6.80% |
| 2.0 | 4,529 | 4.57% |
| 3.0 | 8,458 | 8.54% |
| 4.0 | 17,090 | 17.26% |
| 5.0 | 62,195 | 62.82% |

**Mean rating:** 4.25  
**Median rating:** 5.0

### Interpretation

This is a **severe class imbalance** scenario. Almost two-thirds of reviews are 5-star, which is a typical pattern in e-commerce review datasets but creates challenges:

- A naive classifier predicting "5" for everything would achieve ~63% accuracy without learning anything meaningful
- Standard accuracy metric will be misleading — we must use F1-macro, balanced accuracy, or per-class metrics
- Minority classes (2-star, 1-star) need special attention

### Solution Strategy

**Decision:** Convert to 3-class sentiment classification:

| Original Rating | Sentiment Label | Count | Percentage |
|---|---|---|---|
| 1, 2 | Negative | 11,257 | 11.37% |
| 3 | Neutral | 8,458 | 8.54% |
| 4, 5 | Positive | 79,285 | 80.09% |

**Why 3-class instead of 5-class?**
- More realistic business framing (most companies care about positive/negative/neutral, not exact stars)
- Reduces severe imbalance from 5 classes to 3
- Better baseline performance achievable
- Aligns with industry-standard sentiment analysis tasks

**Mitigation techniques to apply during modeling:**
1. Class weighting (`class_weight='balanced'` in scikit-learn)
2. Stratified train/test split
3. Macro F1-score as primary metric
4. SMOTE or other oversampling techniques for the neutral class
5. Threshold tuning during evaluation

---

## 3. Text Length Analysis

### Character Length

| Statistic | Value |
|---|---|
| Mean | 510.97 |
| Std | 800.97 |
| Min | 0 |
| 25th percentile | 76 |
| Median | 209 |
| 75th percentile | 568 |
| Max | 23,991 |

### Word Count

| Statistic | Value |
|---|---|
| Mean | 91.44 |
| Std | 138.71 |
| Min | 0 |
| 25th percentile | 14 |
| Median | 39 |
| 75th percentile | 105 |
| Max | 4,091 |

### Cross-Category Differences

| Category | Avg Text Length (chars) | Avg Word Count |
|---|---|---|
| Beauty | 236.5 | 44.5 |
| Electronics | 426.8 | 78.7 |
| Books | 869.7 | 151.1 |

### Interpretation

The dataset shows a **right-skewed distribution** — most reviews are short, but there's a long tail of very detailed reviews (mostly in Books). This has direct implications for modeling:

- **BERT/DistilBERT compatibility:** Median word count (39) is well below the 512-token limit. Most reviews can be processed without truncation.
- **Books category is the outlier:** With avg 151 words, some reviews exceed 512 tokens. We need a truncation strategy.
- **Empty reviews exist:** Min word count is 0 — these need to be filtered out.

### Solution Strategy

1. **Filter:** Remove reviews with `word_count < 5` (likely uninformative)
2. **Tokenization config:** `max_length=256` (covers 75% of reviews fully) with truncation
3. **Books-specific strategy:** Consider "head + tail" truncation (first 256 tokens + last 256 tokens) for very long Books reviews
4. **Optional augmentation:** For very short reviews, concatenate `title + text` to add context

---

## 4. Duplicate Analysis

| Type | Count | Percentage |
|---|---|---|
| Exact duplicate rows | 95 | 0.10% |
| Identical text content | 4,758 | 4.81% |
| Same user + same product | 95 | 0.10% |

### Interpretation

**Exact duplicates (95):** Small number, easy to remove. Probably scraping artifacts.

**Text-only duplicates (4,758):** This is the **critical signal**. Many users write identical short reviews like:
- "Good"
- "Excellent"
- "Five stars"
- "Love it"

This pattern has **two implications**:

1. **Data leakage risk:** If "Good" appears in training and test sets, the model isn't really learning — it's memorizing. Must remove text duplicates before split.

2. **Fake review signal:** Some of these might be genuine short reviews, but template-like duplicates (same exact wording across different users/products) are a classic fake review pattern. **This is the foundation of our Fake Review Detection module (Module B).**

### Solution Strategy

**For sentiment classification (Module A):**
- Drop exact duplicates: `df.drop_duplicates(subset=hashable_cols)`
- Drop text duplicates: `df.drop_duplicates(subset=['text'])` — keeps first occurrence
- Result: ~94,000 unique reviews remaining

**For fake review detection (Module B):**
- **Keep duplicates** in a separate dataframe — they ARE the labeled fake review signal
- Use them as positive examples for fake/template detection
- Combine with anomaly detection on the unique reviews

---

## 5. Verified Purchase Analysis

| Group | Count | Percentage | Avg Rating |
|---|---|---|---|
| Verified = True | 68,003 | 68.69% | 4.230 |
| Verified = False | 30,997 | 31.31% | 4.285 |

### Interpretation

**Counterintuitive finding:** Non-verified buyers give **slightly higher** ratings (4.29 vs 4.23). One would expect verified buyers to be more accurate, but this gap suggests:

- Non-verified reviews may include promotional/fake reviews (boost average)
- Verified buyers are more critical (real product disappointment)
- Or could be selection bias (people who don't verify might be writing fake reviews)

**Mülakat anlatımı:** "I noticed an unexpected pattern: non-verified reviews had a 0.06 higher average rating. This counterintuitive result became a key feature in my fake review detector — verified status combined with rating-text mismatch became one of the strongest signals."

### Solution Strategy

`verified_purchase` becomes a **key feature** for:
- Fake review detection (Module B)
- Trust scoring system (potential extension)
- Cross-verification with rating-text alignment

---

## 6. Helpful Vote Analysis

| Statistic | Value |
|---|---|
| Mean | 1.84 |
| Median | 0 |
| Std | 16.29 |
| Max | 3,580 (outlier) |
| Reviews with 0 helpful votes | 66,180 (66.85%) |
| Reviews with >10 helpful votes | 3,133 |

### Interpretation

**Highly skewed distribution.** Most reviews receive zero helpful votes. The few that get many are extreme outliers (one review has 3,580 votes).

This signal is **weak alone** but useful when:
- **Normalized by review age:** Older reviews accumulate more votes naturally
- **Combined with rating polarity:** High helpful votes on extreme ratings (1 or 5) indicate strong community signal
- **Used as a feature, not target:** Don't predict helpful votes; use them as a feature

### Solution Strategy

- **Feature engineering:** Create `helpful_vote_rate = helpful_vote / days_since_review`
- **Outlier handling:** Cap at 99th percentile or log-transform for modeling
- **Use as auxiliary feature** in fake review detection

---

## 7. Temporal Distribution

| Year | Review Count |
|---|---|
| 1997-2013 | Sparse (< 5K total) |
| 2014 | 4,526 |
| 2015 | 6,570 |
| 2016 | 8,304 |
| 2017 | 8,198 |
| 2018 | 9,049 |
| 2019 | 11,714 |
| 2020 | 13,711 (peak) |
| 2021 | 12,935 |
| 2022 | 9,947 |
| 2023 | 1,592 (partial — through March) |

### Interpretation

**Concept drift implications:** Language patterns evolve over time. A review from 2010 ("It's good") is structurally different from a 2022 review (more emoji, different slang, shorter sentences). If we train on old data and test on new data without temporal awareness, model may underperform in production.

**Class distribution may shift:** Are 5-star reviews more common in 2020+ vs 2014? This isn't analyzed yet but is worth checking.

### Solution Strategy

**Two split strategies to compare:**

1. **Random stratified split** (default):
   - Pros: Standard, simple, balanced classes
   - Cons: Doesn't simulate production deployment

2. **Temporal split** (production-realistic):
   - Train: 1997-2020
   - Validation: 2021
   - Test: 2022-2023
   - Pros: Mimics real deployment (model trained on past, predicts future)
   - Cons: Class imbalance may differ across splits

**Decision:** Start with random stratified split for the baseline, then run temporal split as a separate experiment. Compare results — this comparison is excellent material for the README.

---

## 8. Empty and Short Text

| Issue | Count |
|---|---|
| Title empty strings | 0 |
| Text empty strings | 13 |
| Reviews with 0 words | 13 |
| Reviews with 1 word | 1,751 |
| Reviews with 2 words | 3,152 |

### Interpretation

A non-trivial number of reviews are extremely short (1-2 words like "Good", "Bad", "OK"). These provide minimal signal for the model:

- Cannot capture nuance or aspects
- Often duplicated (data leakage risk)
- Hard to learn from at the word level

### Solution Strategy

**Filter rule:** Remove reviews where `word_count < 5`. This:
- Removes ~5,000 uninformative reviews
- Reduces duplicate density
- Improves model signal-to-noise ratio
- Still keeps enough data (~94,000 reviews remaining)

**Alternative:** Concatenate `title + text` for short reviews to enrich context. (Decision: skip for now, can revisit if results are weak.)

---

## 9. Sample Reviews — Qualitative Insights

### What I observed reading samples:

**1-star (Beauty):** "Extremely expensive item. It's hard plastic, hard to roll hair into it..."  
→ Specific, detailed complaints. Multi-aspect (price, material, usability).

**2-star (Beauty):** "I wish it was tighter so it won't slide off my head..."  
→ Constructive criticism, single specific issue.

**3-star (Books):** Detailed but mixed. Mentions "misleading," "extensive lists," "extremely misleading numbers."  
→ Balanced, nuanced reviews are typical of mid-ratings.

**4-star (Books):** "Pretty Ornaments Knitted in Red & White..."  
→ Compliment + nuanced criticism ("subtitle is misleading").

**5-star (Books):** Short and enthusiastic.

### Interpretation

This confirms that **Aspect-Based Sentiment Analysis (ABSA, Module C)** will add real value. Same review can have multiple sentiments:
- Product quality: positive
- Price: negative
- Description accuracy: negative

Standard sentiment classifiers miss this nuance.

---

## 10. Critical Decisions Summary

This table consolidates all preprocessing decisions made based on EDA findings:

| Decision | Value | EDA Basis |
|---|---|---|
| Task formulation | 3-class sentiment (neg/neu/pos) | Severe 5-class imbalance |
| Min word count | 5 | 1,751 single-word reviews + duplicates |
| Max tokens | 256 (BERT) | 75th percentile is 105 words |
| Long-review strategy | Standard truncation, monitor Books | Books avg 151 words |
| Duplicate removal | Drop exact + text duplicates | 4,758 text duplicates risk leakage |
| Keep duplicates separately | For Module B | Fake review signal |
| Train/test split | 80/20 stratified by sentiment + category | Class + domain balance |
| Temporal split experiment | Yes, as secondary evaluation | Concept drift risk |
| Primary metric | Macro F1 | Class imbalance |
| Class weighting | Yes | Imbalance |
| Verified purchase | Use as feature | Counterintuitive rating gap |
| Helpful vote | Engineer rate feature | Weak alone, useful normalized |

---

## 11. Three Modules — Confirmed Direction

Based on EDA findings, our three modules are well-supported by the data:

### Module A: Sentiment Classifier
- **Target:** 3-class (negative/neutral/positive)
- **Models:** TF-IDF + LogReg (baseline) → LightGBM → DistilBERT (fine-tuned)
- **Challenge:** Class imbalance, neutral class detection (hardest)
- **Expected accuracy:** ~85-93% depending on model

### Module B: Fake Review Detector
- **Signals identified in EDA:**
  1. Text duplicates (4,758 exact text matches)
  2. Verified vs non-verified rating gap (counterintuitive)
  3. Helpful vote anomalies
  4. Short generic reviews ("Good", "Five Stars")
- **Approach:** Supervised (using duplicates as labels) + Unsupervised (Isolation Forest on embeddings)

### Module C: Aspect-Based Sentiment Analysis
- **Aspects to extract per category:**
  - Electronics: battery, shipping, price, quality, customer service
  - Books: writing, characters, plot, value-for-money, length
  - Beauty: scent, packaging, longevity, price, effectiveness
- **Models:** Pre-trained ABSA model (e.g., `yangheng/deberta-v3-base-absa-v1.1`) + BERTopic for topic discovery
- **Output:** Per-aspect sentiment dashboard

---

## 12. Next Steps (After EDA)

1. **Preprocessing notebook** (`02_preprocessing.ipynb`)
   - Apply all decisions above
   - Save cleaned dataset as parquet
   - Create train/val/test splits

2. **Reusable preprocessor module** (`src/data/preprocessor.py`)
   - Class-based pipeline
   - Configurable via `src/config.py`
   - Unit tests in `tests/test_preprocessor.py`

3. **Baseline modeling** (`03_baseline_models.ipynb`)
   - TF-IDF + Logistic Regression
   - TF-IDF + LightGBM
   - Evaluation framework setup

4. **Modern modeling** (`04_distilbert_finetuning.ipynb`)
   - Run on Colab Pro (GPU required)
   - Compare against baselines

5. **Module B and C notebooks**

---

## 13. Files Created

- `data/raw/amazon_multicat_100k.csv` (raw download, ~100 MB)
- `data/processed/amazon_eda_enriched.csv` (with text_length, word_count, datetime columns)
- `reports/figures/01_rating_distribution.png`
- `reports/figures/02_review_length.png`
- `reports/figures/03_yearly_trend.png`

---

## 14. Interview Talking Points

When discussing this project in an interview, key insights to mention:

1. **"I performed EDA before any modeling and discovered severe class imbalance — 63% of reviews are 5-star. This drove my decision to reformulate the task as 3-class sentiment instead of 5-class."**

2. **"I found 4,758 text-level duplicates which posed a data leakage risk for sentiment but became the foundation of my fake review detection module."**

3. **"Counterintuitively, non-verified buyers gave higher ratings than verified ones. This became a key engineered feature for fake review detection."**

4. **"Books reviews are 3x longer than Beauty reviews on average. I designed category-aware preprocessing to handle this, including a head-tail truncation strategy for long-form reviews."**

5. **"I evaluated my models with both random and temporal splits to assess generalization under concept drift."**
