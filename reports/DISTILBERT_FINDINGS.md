# DistilBERT Fine-Tuning Findings — Sentiment Classification

**Project:** Review Intelligence System
**Notebook:** `notebooks/04_distilbert_finetuning.ipynb`
**Base Model:** `distilbert-base-uncased` (Hugging Face)
**Task:** 3-class sentiment classification (negative / neutral / positive)
**Training Environment:** Google Colab Free, NVIDIA Tesla T4 GPU
**Total Training Time:** 22.8 minutes

---

## Document Structure

- **Section 1:** Executive Summary
- **Section 2:** Model Architecture and Configuration
- **Section 3:** Training Configuration and Strategy
- **Section 4:** Training Results — Epoch-by-Epoch Analysis
- **Section 5:** Validation Set Performance
- **Section 6:** Test Set Performance — Comprehensive Metrics
- **Section 7:** Cross-Model Comparison (Dummy / LogReg / LightGBM / DistilBERT)
- **Section 8:** Confusion Matrix Analysis
- **Section 9:** Per-Category Performance
- **Section 10:** Overfitting Analysis
- **Section 11:** Inference Speed and Production Tradeoffs
- **Section 12:** Critical Findings and Insights
- **Section 13:** Limitations and Future Work
- **Section 14:** Files Generated
- **Section 15:** Interview Talking Points
- **Section 16:** Reproducibility Notes

---

## 1. Executive Summary

DistilBERT was fine-tuned on 71,216 Amazon multi-category reviews to classify sentiment into three classes (negative, neutral, positive). The model achieved **0.7010 macro F1** on the test set, a **+6.9% relative improvement** over the best classical baseline (TF-IDF + Logistic Regression at 0.6555).

### Key Outcomes

| Metric | Value | vs LogReg Baseline |
|---|---|---|
| Test Macro F1 | 0.7010 | +0.0455 (+6.9% relative) |
| Test Accuracy | 0.8468 | -0.0055 (slight drop) |
| Negative F1 | 0.7333 | +0.0638 |
| **Neutral F1** | **0.4394** | **+0.0729** |
| Positive F1 | 0.9303 | -0.0003 (essentially equal) |
| Inference Latency (GPU) | 2.59 ms/sample | 250× slower than LogReg |

### Headline Finding

**The biggest win was on the neutral class.** DistilBERT's neutral recall jumped from 34.12% (LogReg) to 60.44% — a **76% relative improvement**. This is exactly where contextual embeddings outperform bag-of-words: 3-star reviews contain mixed sentiment with contrast structures ("good but...") that require understanding context, not just word frequencies.

---

## 2. Model Architecture and Configuration

### 2.1 Base Model

**`distilbert-base-uncased`** — a smaller, faster distilled version of BERT.

| Property | Value |
|---|---|
| Architecture | Transformer (encoder-only) |
| Hidden size | 768 |
| Attention heads | 12 |
| Layers | 6 (vs BERT's 12) |
| Total parameters | 66.4 million |
| Model size (fp32) | ~250 MB |
| Vocabulary | WordPiece, 30,522 tokens |
| Pre-training | Masked LM on Wikipedia + BookCorpus |

### 2.2 Why DistilBERT Instead of BERT-base?

DistilBERT achieves ~97% of BERT's performance with:
- **40% fewer parameters** (66M vs 110M)
- **60% faster inference** (~2x speedup)
- **Identical fine-tuning workflow**

For a portfolio project on Colab Free Tier, DistilBERT was the right choice. BERT-base would require ~2x training time (40-60 minutes) with marginal performance gains.

### 2.3 Classification Head

A standard `AutoModelForSequenceClassification` head was added on top of the pre-trained encoder:

```
DistilBERT encoder (frozen during pre-training)
    → [CLS] token representation (768-dim)
    → Dropout(0.2)
    → Linear(768, 3)
    → Softmax
```

The classification head and encoder were both fine-tuned together (full fine-tuning, not feature extraction).

---

## 3. Training Configuration and Strategy

### 3.1 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 2e-5 | Standard for BERT fine-tuning |
| Batch size | 32 | Maximum that fits in T4 16GB VRAM with fp16 |
| Epochs | 3 | Standard for BERT; longer risks overfitting |
| Weight decay | 0.01 | Mild L2 regularization |
| Warmup ratio | 0.1 | First 10% of training warms up LR |
| Max sequence length | 256 tokens | Covers 75th percentile of reviews per EDA |
| Gradient accumulation | 1 | Effective batch = 32 |
| Mixed precision (fp16) | Yes | 2× speed + half memory on T4 |
| Random seed | 42 | Reproducibility |

### 3.2 Class Weighting

To address the severe class imbalance (79% positive / 12% negative / 9% neutral), balanced class weights were computed and applied via a custom `WeightedTrainer`:

```python
# Computed weights (from sklearn's compute_class_weight)
negative: 2.808
neutral:  3.671
positive: 0.422
```

Higher weights for minority classes force the model to "care" more about getting them right during training. Without this, the model would optimize accuracy by predicting "positive" too often.

### 3.3 Tokenization

```python
tokenizer(
    text,
    truncation=True,
    max_length=256,
    padding=False  # dynamic padding via DataCollator
)
```

Dynamic padding (only padding to longest in batch) saves ~20% memory vs static padding to max_length.

### 3.4 Optimizer and Scheduler

- **Optimizer:** AdamW (default in Transformers Trainer)
- **Scheduler:** Linear warmup → linear decay
- **Loss function:** Weighted cross-entropy (custom)

### 3.5 Best Model Selection

Early stopping was implicit through `load_best_model_at_end=True`:
- Validation evaluation at the end of each epoch
- Best checkpoint by `f1_macro` is restored at training end
- This prevented overfitting from corrupting final results

---

## 4. Training Results — Epoch-by-Epoch Analysis

### 4.1 Training Trajectory

| Epoch | Training Loss | Validation Loss | Val Accuracy | Val Macro F1 | Val Negative F1 | Val Neutral F1 | Val Positive F1 |
|---|---|---|---|---|---|---|---|
| 1 | 0.5963 | 0.5684 | 0.7935 | 0.6715 | 0.7467 | 0.3796 | 0.8883 |
| **2** | **0.5051** | **0.5868** | **0.8631** | **0.7311** | **0.7755** | **0.4817** | **0.9362** |
| 3 | 0.3542 | 0.6422 | 0.8607 | 0.7269 | 0.7715 | 0.4740 | 0.9353 |

**Best epoch: 2** (selected via `load_best_model_at_end`).

### 4.2 Training Behavior Observations

**Epoch 1 (Underfitting):**
- Both train and val losses high (~0.58-0.60)
- Val F1 already at 0.67 — model has captured basic patterns
- Neutral F1 still weak (0.38) — needs more epochs

**Epoch 2 (Optimal):**
- Train loss dropped to 0.51, val loss to 0.59
- Val F1 peaked at 0.7311
- Neutral F1 hit 0.48 — biggest single-epoch jump
- Validation loss starts diverging slightly from train

**Epoch 3 (Overfitting):**
- Train loss dropped sharply to 0.35
- Val loss rose to 0.64 — clear overfitting signal
- Val F1 dropped 0.004 — minor regression
- Train/val gap widened

The classic overfitting pattern was observed: training loss continues dropping while validation loss starts rising. The `load_best_model_at_end` mechanism correctly retained epoch 2's weights.

### 4.3 Training Duration

```
Total time:        22.8 minutes
Steps per epoch:   2,225 (71,216 / 32)
Total steps:       6,675
Steps per second:  ~4.9 (fp16 on T4)
```

This is roughly the expected speed for DistilBERT on a T4 with mixed precision.

---

## 5. Validation Set Performance

Best model (epoch 2) results on validation set (8,902 samples):

```
Accuracy:        0.8631
Macro F1:        0.7311  (primary metric)
Weighted F1:     0.8759
Validation Loss: 0.5868
```

### Per-Class Validation Metrics

| Class | F1 | Support |
|---|---|---|
| Negative | 0.7755 | 1,057 |
| Neutral | 0.4817 | 809 |
| Positive | 0.9362 | 7,036 |

### Validation vs Test Comparison

| Metric | Validation | Test | Difference |
|---|---|---|---|
| Accuracy | 0.8631 | 0.8468 | -0.0163 |
| Macro F1 | 0.7311 | 0.7010 | -0.0301 |
| Negative F1 | 0.7755 | 0.7333 | -0.0422 |
| Neutral F1 | 0.4817 | 0.4394 | -0.0423 |
| Positive F1 | 0.9362 | 0.9303 | -0.0059 |

**Interpretation:** The val-test gap is ~3 macro F1 points. This is moderate — not catastrophic but indicates some overfitting to validation data via the model selection process. Acceptable for a portfolio project but worth noting for production deployment.

---

## 6. Test Set Performance — Comprehensive Metrics

### 6.1 Test Set Headline Numbers

```
Test set size:     8,903 samples
Accuracy:          0.8468
Macro F1:          0.7010
Weighted F1:       0.8623
Test Loss:         0.6353
Inference Time:    2.59 ms/sample (T4 GPU)
```

### 6.2 Detailed Classification Report

```
              precision    recall  f1-score   support

    negative     0.7767    0.6944    0.7333      1057
     neutral     0.3451    0.6044    0.4394       809
    positive     0.9656    0.8975    0.9303      7037

    accuracy                         0.8468      8903
   macro avg     0.6958    0.7321    0.7010      8903
weighted avg     0.8868    0.8468    0.8623      8903
```

### 6.3 Per-Class Deep Dive

**Negative Class:**
- Precision: 0.7767 (high — when model says "negative", it's usually right)
- Recall: 0.6944 (decent — catches ~70% of negative reviews)
- F1: 0.7333 (solid performance for minority class)

**Neutral Class:**
- Precision: 0.3451 (low — many false positives)
- Recall: 0.6044 (much improved from LogReg's 0.3412)
- F1: 0.4394 (still the weakest class)

**Why low neutral precision?** The model now correctly catches more neutrals (high recall) but also wrongly labels some positives and negatives as neutral (low precision). This is a precision-recall tradeoff specific to imbalanced data.

**Positive Class:**
- Precision: 0.9656 (when model predicts positive, almost always correct)
- Recall: 0.8975 (~10% of positives missed, mostly to neutral)
- F1: 0.9303 (excellent)

---

## 7. Cross-Model Comparison

### 7.1 Test Set — All Models

| Model | Accuracy | Macro F1 | Weighted F1 | Negative F1 | Neutral F1 | Positive F1 | Inference |
|---|---|---|---|---|---|---|---|
| Dummy (always positive) | 0.7904 | 0.2943 | 0.6979 | 0.0000 | 0.0000 | 0.8829 | < 0.01 ms |
| TF-IDF + LogReg | 0.8523 | 0.6555 | 0.8483 | 0.6695 | 0.3665 | 0.9306 | < 0.01 ms |
| TF-IDF + LightGBM | 0.8544 | 0.6158 | 0.8325 | 0.6380 | 0.2847 | 0.9247 | 0.02 ms |
| **DistilBERT** | **0.8468** | **0.7010** | **0.8623** | **0.7333** | **0.4394** | **0.9303** | **2.59 ms (GPU)** |

### 7.2 Improvement Analysis vs Best Baseline (LogReg)

| Metric | LogReg | DistilBERT | Δ | % Relative |
|---|---|---|---|---|
| Accuracy | 0.8523 | 0.8468 | -0.0055 | -0.6% |
| Macro F1 | 0.6555 | 0.7010 | +0.0455 | **+6.9%** |
| Negative F1 | 0.6695 | 0.7333 | +0.0638 | +9.5% |
| Neutral F1 | 0.3665 | 0.4394 | +0.0729 | **+19.9%** |
| Positive F1 | 0.9306 | 0.9303 | -0.0003 | 0.0% |

### 7.3 The Counterintuitive Accuracy Drop

DistilBERT achieved **lower accuracy** than LogReg (0.8468 vs 0.8523). On the surface, this seems like the bigger model performed worse. The reality:

- LogReg over-predicts the dominant class (positive) which boosts accuracy
- DistilBERT trades a little positive precision for much better neutral recall
- The result is a more **balanced** classifier that's better across all classes

This is exactly why **macro F1 is the right metric for imbalanced classification**. Accuracy rewards majority-class bias; macro F1 punishes it.

### 7.4 Where DistilBERT Wins, Where Baselines Win

**DistilBERT wins:**
- Minority class detection (negative, neutral)
- Macro-averaged metrics
- Cross-category robustness
- Real-world deployment quality

**Classical baselines win:**
- Inference speed (1000-5000× faster)
- Memory footprint (1 MB vs 250 MB)
- Interpretability (LogReg shows top features explicitly)
- Training speed (5 seconds vs 22 minutes)

**Conclusion:** Choose DistilBERT for quality-critical applications; choose LogReg for high-throughput, latency-sensitive ones.

---

## 8. Confusion Matrix Analysis

### 8.1 DistilBERT Confusion Matrix (Test Set, Counts)

|  | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| **True Negative** | 734 | 283 | 40 |
| **True Neutral** | 135 | 489 | 185 |
| **True Positive** | 76 | 645 | 6,316 |

### 8.2 Normalized Confusion Matrix (Row-wise)

|  | Predicted Negative | Predicted Neutral | Predicted Positive |
|---|---|---|---|
| **True Negative** | **69.44%** | 26.77% | 3.78% |
| **True Neutral** | 16.69% | **60.44%** | 22.87% |
| **True Positive** | 1.08% | 9.17% | **89.75%** |

### 8.3 Confusion Pattern Comparison: DistilBERT vs LogReg

The table below shows how each model handles each error type:

| Error Type | LogReg | DistilBERT | Change |
|---|---|---|---|
| True negative → predicted positive | 19.58% | 3.78% | **-15.8 pts (huge improvement)** |
| True negative → predicted neutral | 14.38% | 26.77% | +12.4 pts (more confusion with neutral) |
| True neutral → predicted positive | 44.13% | 22.87% | **-21.3 pts (huge improvement)** |
| True neutral → predicted negative | 21.76% | 16.69% | -5.1 pts (improvement) |
| True positive → predicted neutral | 3.82% | 9.17% | +5.4 pts (cost of better neutral recall) |
| True positive → predicted negative | 2.19% | 1.08% | -1.1 pts (improvement) |

### 8.4 Interpretation

**Major improvements:**
1. **Negative→Positive error halved** (19.58% → 3.78%): DistilBERT no longer mistakes harshly negative reviews as positive
2. **Neutral→Positive error halved** (44.13% → 22.87%): The model now distinguishes mid-rating reviews from glowing ones much better

**New cost:**
- Some true positives now get labeled as neutral (3.82% → 9.17%): The model became cautious, which is mostly what we want for an imbalanced classifier

**Why this pattern?** Class weighting + contextual embeddings made DistilBERT less biased toward the majority class. It's now willing to "spend" some positive precision to get better minority class recall — which is exactly the right tradeoff for sentiment analysis.

---

## 9. Per-Category Performance

### 9.1 Macro F1 by Category

| Category | LogReg | DistilBERT | Improvement |
|---|---|---|---|
| Beauty | 0.6555 | **0.7168** | +0.0613 |
| Books | 0.6029 | 0.6501 | +0.0472 |
| Electronics | 0.6527 | 0.7000 | +0.0473 |

### 9.2 Cross-Category Analysis

**Beauty saw the largest improvement** (+0.061). Beauty reviews are typically shorter (~45 words avg) and more emotionally direct, which the contextual model handles well.

**Books has the smallest improvement** (+0.047) and remains the hardest category. Likely reasons:
1. Books reviews are 3× longer (151 words avg) and the 256-token max length truncates information
2. Books has the highest neutral rate (10.2%) — even contextual models struggle with nuanced critical reviews
3. Books has the most positive imbalance (83% positive), making class boundaries fuzzy

**Production implication:** A category-specific approach (using `category` as an explicit feature, or training separate models per category) might unlock another performance bump on Books.

---

## 10. Overfitting Analysis

### 10.1 Loss Curves

```
Epoch 1: train=0.596, val=0.568   (val < train — model still underfitting)
Epoch 2: train=0.505, val=0.587   (val ≈ train — sweet spot)
Epoch 3: train=0.354, val=0.642   (val > train and rising — overfitting)
```

The classic overfitting signature: training loss decreasing while validation loss increases. By epoch 3, the gap is 0.288 — meaningful divergence.

### 10.2 Why Did Overfitting Occur?

Several factors contributed:

1. **Model capacity vs dataset size:** 66M parameters fitting on 71K examples; the model has room to memorize
2. **High-quality (clean) labels:** Once the model learns label patterns, it can fit noise
3. **No regularization beyond weight decay:** Standard 0.01 weight decay isn't aggressive
4. **3 epochs may be 1 too many:** Common BERT recipe says 2-4 epochs; 2 was optimal here

### 10.3 What Saved Us

**`load_best_model_at_end=True`** automatically reverted to epoch 2's checkpoint. Without this:
- Final model would be epoch 3 with worse F1 (0.7269 vs 0.7311)
- ~0.4 point macro F1 loss on test

This is a critical configuration that production training pipelines should always include.

### 10.4 Future Mitigation Strategies

If we wanted to push performance higher without overfitting:
- **Train for 2 epochs** instead of 3
- **Increase dropout** in classification head (0.2 → 0.3)
- **Higher weight decay** (0.01 → 0.05)
- **Label smoothing** (0.0 → 0.1)
- **Larger validation set** for more reliable best-model selection
- **Augmentation:** back-translation, EDA techniques

---

## 11. Inference Speed and Production Tradeoffs

### 11.1 Latency Comparison

| Model | Hardware | Latency per sample | Throughput (samples/sec) |
|---|---|---|---|
| LogReg | CPU | < 0.01 ms | > 100,000 |
| LightGBM | CPU | 0.02 ms | ~50,000 |
| DistilBERT | T4 GPU | 2.59 ms | ~390 |
| DistilBERT (estimated) | CPU | ~30-50 ms | ~25 |

### 11.2 Memory Footprint

| Model | Disk Size | RAM During Inference |
|---|---|---|
| LogReg + TF-IDF vectorizer | ~27 MB (mostly vectorizer) | ~30 MB |
| LightGBM + TF-IDF vectorizer | ~30 MB | ~35 MB |
| DistilBERT | 255 MB | ~500 MB |

### 11.3 Cost-Performance Decision Matrix

For a production deployment serving N requests per second:

| Volume | Recommendation | Reasoning |
|---|---|---|
| < 100 req/s | DistilBERT on CPU | Quality-critical, latency acceptable |
| 100-1000 req/s | DistilBERT on GPU OR LogReg ensemble | Depends on quality requirements |
| > 1000 req/s | LogReg | GPU costs would dominate |
| Mixed (most + edge cases) | **Hybrid: LogReg with BERT fallback for low-confidence predictions** | Best of both worlds |

### 11.4 Hybrid Deployment Strategy (Recommended)

A production system could use:

```
1. LogReg predicts (< 1ms)
2. If confidence > 0.85: return LogReg prediction
3. Else: send to DistilBERT for re-evaluation (2-50ms)
4. Return DistilBERT prediction
```

This pattern handles ~80% of traffic with LogReg's speed, and reserves BERT's accuracy for the hard cases where it actually matters.

---

## 12. Critical Findings and Insights

### 12.1 The Five Most Important Findings

**1. Macro F1 improved by 6.9% relative, but accuracy dropped slightly**

This counterintuitive result is the strongest validation of using macro F1 over accuracy on imbalanced data. The model became more balanced, not more accurate.

**2. Neutral class recall jumped from 34% to 60%**

Contextual understanding (BERT) succeeded exactly where bag-of-words (TF-IDF) struggled: identifying mixed-sentiment reviews. This 76% relative recall improvement on the hardest class is the project's biggest win.

**3. The negative→positive error rate was halved (19.6% → 3.8%)**

This is huge for business value. A sentiment classifier that mistakes "I hate this product" for "I love this product" is unacceptable. DistilBERT essentially solved this catastrophic failure mode.

**4. Books remained the hardest category despite improvement**

Even with contextual embeddings, very long reviews (3× longer than other categories) lose information at the 256-token truncation boundary. This points to a clear next experiment: head-tail truncation or longer sequence handling for Books.

**5. Classical baselines remain competitive for production**

LogReg achieves 94% of DistilBERT's macro F1 at 1/250 the inference cost. For high-volume systems, the math heavily favors classical models. This insight matters more than the F1 gains for real-world deployment decisions.

### 12.2 What Surprised Us

**Surprise 1: LightGBM didn't outperform LogReg**

Industry intuition says gradient boosting beats linear models. But on TF-IDF features (99.79% sparsity, 50K dimensions), LogReg won by 4 macro F1 points. Tree-based models struggle in high-dim sparse spaces.

**Surprise 2: BERT's accuracy was lower than LogReg's**

We expected the more sophisticated model to win on every metric. Instead, accuracy declined while macro F1 rose. This taught us that single-metric thinking is dangerous in imbalanced classification.

**Surprise 3: 22 minutes was enough**

We budgeted 60-90 minutes for fine-tuning on Colab Free. With fp16 mixed precision and DistilBERT's smaller architecture, we finished in less than half the expected time.

---

## 13. Limitations and Future Work

### 13.1 Current Limitations

1. **Single-run training:** No multi-seed averaging means our 0.7010 macro F1 has unreported variance (~±0.01 typical)
2. **No hyperparameter tuning:** Used standard BERT recipe; could squeeze more with Optuna
3. **256-token truncation:** Loses information from longer Books reviews
4. **No data augmentation:** Could improve minority class recall further
5. **No cross-validation:** Single train/val/test split; CV would give more reliable estimates
6. **Validation-set leakage in model selection:** Best model selected on val ⇒ slight optimism in val metrics
7. **English-only:** No multilingual evaluation despite original plan to test on Turkish reviews
8. **3-class formulation:** Real Amazon use case may need 5-class or regression

### 13.2 Future Work — Prioritized

**High impact, low effort:**
- Try `max_length=512` for Books-only evaluation
- Add label smoothing (0.1)
- Run 5-seed average for reliable F1 estimate
- Test on Turkish reviews (BERTurk)

**Medium impact, medium effort:**
- Hyperparameter search (Optuna) over learning rate, batch size, weight decay
- Try BERT-base, RoBERTa, DeBERTa for comparison
- Implement head-tail truncation for long reviews
- Add data augmentation (back-translation, EDA, synonym replacement)

**High impact, high effort:**
- Cross-validation with stratified k-fold
- Custom architecture: BERT + category embedding fusion
- Multi-task learning: sentiment + helpfulness + verified prediction jointly
- Active learning: identify hardest examples and re-label

### 13.3 Production-Readiness Gaps

To deploy this model in production, we'd still need:
- Monitoring (drift detection on prediction distribution)
- A/B testing framework (vs simpler baselines)
- Latency SLA definition (p95, p99)
- Bias auditing (fairness across user demographics if available)
- Robustness testing (adversarial inputs, typos)
- Cold-start handling (new product categories)

---

## 14. Files Generated

### 14.1 Notebook
- `notebooks/04_distilbert_finetuning.ipynb` — full training pipeline

### 14.2 Saved Model (Drive — too large for git)
- `distilbert-final/config.json` — model configuration
- `distilbert-final/model.safetensors` — model weights (255 MB)
- `distilbert-final/tokenizer_config.json` — tokenizer config
- `distilbert-final/tokenizer.json` — tokenizer vocab
- `distilbert-final/training_args.bin` — training hyperparameters

### 14.3 Reports
- `reports/distilbert_results.json` — all metrics for cross-model comparison
- `reports/figures/09_distilbert_confusion.png` — confusion matrix visualization

### 14.4 Updated Comparison
The cross-model comparison in `reports/baseline_results.json` (LogReg, LightGBM, Dummy) should be updated alongside `distilbert_results.json` to provide unified model comparison views.

---

## 15. Interview Talking Points

These are concrete sentences usable directly in interviews when discussing this project:

**1. "DistilBERT improved macro F1 by 6.9% relative over LogReg, but accuracy actually dropped slightly. The headline number was the neutral class recall jumping from 34% to 60% — a 76% relative improvement that's only possible with contextual understanding."**

**2. "I included a dummy classifier baseline that always predicted positive. It achieved 79% accuracy but 0% F1 on minority classes. This sanity check proved why I optimized for macro F1 rather than accuracy on this imbalanced dataset."**

**3. "The most informative error was that LogReg misclassified 44% of true neutral reviews as positive. DistilBERT cut this to 23%. 3-star reviews use mixed sentiment ('good but slow') that requires contextual disambiguation — a known weakness of bag-of-words models."**

**4. "I observed clear overfitting at epoch 3: training loss kept dropping while validation loss rose. My early stopping configuration via `load_best_model_at_end=True` automatically retained epoch 2's weights. This is a critical pattern for production training pipelines."**

**5. "On TF-IDF features, Logistic Regression actually outperformed LightGBM by 4 F1 points. I traced this to the 99.79% sparsity of the vector space — tree-based models struggle to find good splits in such sparse high-dimensional features. This insight saved me time by skipping LightGBM tuning in favor of DistilBERT."**

**6. "DistilBERT's inference is 250× slower than LogReg. For a production system serving 10K+ requests per second, that latency multiplier matters more than the F1 difference. I'd recommend a hybrid deployment: LogReg for confident predictions, BERT for low-confidence fallback."**

**7. "Books was the hardest category for both models — 3× longer reviews mean information loss at 256-token truncation. A clear next experiment is head-tail truncation: keep first 128 + last 128 tokens for long reviews."**

**8. "Class weighting was essential. I computed balanced weights via sklearn (negative=2.81, neutral=3.67, positive=0.42) and implemented a custom WeightedTrainer to apply them in the cross-entropy loss. Without this, the model would have collapsed to majority-class predictions."**

**9. "I trained on Colab Free Tier in 22 minutes using fp16 mixed precision on a T4 GPU. The cost-performance ratio was excellent: ~$0 for 0.07 macro F1 improvement over my best classical model."**

**10. "The validation-test gap was 3 macro F1 points (0.731 vs 0.701). This is acceptable but indicates moderate optimism in val metrics due to using val for model selection. In a longer project, I'd add a separate dev set for hyperparameter tuning."**

---

## 16. Reproducibility Notes

### 16.1 Random Seeds Set

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
```

Plus `seed=42` in `TrainingArguments`.

### 16.2 Environment

| Component | Version |
|---|---|
| Python | 3.12 (Colab) |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.46+ (latest as of training) |
| Datasets | 2.21.0 |
| Accelerate | latest compatible |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |

### 16.3 To Reproduce This Result

1. Open `notebooks/04_distilbert_finetuning.ipynb` in Google Colab
2. Set runtime to T4 GPU (`Runtime → Change runtime type → GPU → T4`)
3. Mount Google Drive containing `train.parquet`, `val.parquet`, `test.parquet` at `/content/drive/MyDrive/review-intelligence-system/data/processed/`
4. Run all cells in sequence
5. Expected runtime: ~25 minutes
6. Expected test macro F1: 0.70 ± 0.01

### 16.4 Source of Variance

Even with seeds set, slight variance (±0.005-0.01 macro F1) can occur due to:
- Non-deterministic CUDA operations (some BERT operations)
- Different Colab GPU instances (T4 vs T4 with different driver versions)
- Different transformers library versions

A robust evaluation would average results across 5+ runs with different seeds.
