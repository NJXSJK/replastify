# Replastify — Phase 2: Deep Result Analysis & Critical Review
**Run Date:** 2026-05-22 | **GPU:** Tesla T4 | **Total Training Time:** 14.4 min

---

## 1. The 60-Second Summary

Both models finished training and hit "good" accuracy numbers on paper. **But before declaring success, we need to be honest about what these numbers actually mean** — and what they don't. This document takes a critical, ground-level look at every result.

| Metric | EfficientNet-B0 | ResNet50 |
|---|---|---|
| Test Accuracy | 88.94% | 87.23% |
| **Macro F1** | **0.8641** | 0.8278 |
| Cohen's Kappa | 0.8636 | 0.8422 |
| Top-3 Accuracy | 97.45% | 97.87% |
| Worst-class F1 (PP) | 0.6061 | **0.4286 ❌** |

**Winner: EfficientNet-B0.** Deployed model going forward.

---

## 2. Which Metric Actually Matters for Replastify?

This is the most important question to answer before reading anything else. Different metrics tell different stories, and for a recycling guidance app, **not all stories are equally important**.

### The Case AGAINST Using Accuracy as the Primary Metric
Accuracy counts all 235 test images equally. But our test set is not balanced:
- LDPE: 53 images (22.6%)
- PS: 50 images (21.3%)
- HDPE: 51 images (21.7%)
- **PP: only 16 images (6.8%)**

If the model perfectly identifies every LDPE, PS, and HDPE image but completely fails on PP, it still scores ~65% accuracy and looks acceptable. **Accuracy masks minority class failure.** Reporting 88.94% accuracy without context is actively misleading for this dataset.

### The Primary Metric: Macro F1
**Macro F1 is the most honest single-number summary for Replastify.** It calculates F1 separately for each of the 6 plastic types and averages them with equal weight — PP (16 test samples) counts exactly as much as LDPE (53 samples). Our EfficientNet Macro F1 of **0.8641** means: averaged across all 6 plastic types equally, the model achieves 86.4% harmonic-mean of precision and recall.

### When to Use Each Metric

| Metric | What It Captures | Use When |
|---|---|---|
| **Macro F1** | Equal-weight performance across all classes | **Primary metric for Replastify** |
| **Cohen's Kappa** | Agreement above random chance | Cross-paper comparison, research reporting |
| **Weighted F1** | Performance weighted by class frequency | When class distribution mirrors real-world usage |
| **Accuracy** | Overall correct rate | Only when classes are balanced (they aren't here) |
| **Per-class F1** | Individual plastic type performance | Diagnosing which plastics to improve |
| **Top-3 Accuracy** | Whether correct answer is in top 3 | For UI/UX "suggestions" feature |

> **Conclusion:** Report **Macro F1** as the headline metric. Accuracy is a supporting number only.

---

## 3. Is the 88.94% Accuracy Real? A Critical Examination

This is the most important skeptical question to ask. Let's examine it from multiple angles.

### 3.1 — Statistical Reliability of the Test Set (The Biggest Concern)

The test set has only **235 images**. This is statistically very small for a 6-class classification problem.

**For PP specifically:** The test set has **only 16 PP images.** When EfficientNet gets F1=0.606 on PP, that means roughly 10 images correct and 6 wrong. If 2 of those wrong predictions were different, the F1 would jump to ~0.73 or drop to ~0.48. **A result based on 16 samples has extremely high variance and cannot be trusted as a stable estimate.**

A rule of thumb in ML is to have at least 100 test samples per class for reliable per-class metrics. We have 16 for PP. This means the PP F1 number could swing wildly on a different test split.

**For overall accuracy:** With 235 test samples, the 95% confidence interval for our 88.94% accuracy is approximately **±2.0% to ±3.0%**. The "true" underlying accuracy of this model could realistically be anywhere from ~86% to ~92%. The single number "88.94%" implies false precision.

### 3.2 — Dataset Scope (Distribution Gap)

The 2,295 images came from a curated academic/research dataset. We do not know exactly how they were collected, but common issues with plastic datasets include:
- Images taken in controlled lighting (lab or bright studio conditions)
- Plastics photographed clean, unwrapped, and isolated on plain backgrounds
- Limited variety of plastic shapes (mostly bottles, limited to bags, sheets, containers)

**Real-world photos** (what users will take with their phones) will have:
- Mixed/complex backgrounds (kitchen counter, garbage bin)
- Dirty, crushed, or torn plastics
- Partial objects (only half the bottle visible)
- Varying lighting (dim, outdoor, harsh flash)
- Multiple plastic items in one frame
- Recycling code labels (which might help, but only if legible)

**The honest expectation:** In production, this model will almost certainly perform 5–15% worse than the test set metrics suggest. This is called the **distribution gap** — the gap between the data distribution the model was trained and tested on, versus the real-world data distribution it will encounter.

### 3.3 — Why the Accuracy Looks "High" Despite Limitations

The numbers are genuinely high for this type of task. Why? Three legitimate reasons:

1. **Transfer Learning works extremely well.** EfficientNet pretrained on 1.2 million ImageNet images already knows textures, surfaces, and material properties. Plastics share visual properties with hundreds of ImageNet categories (transparent objects, colored containers, flexible sheets). The model isn't starting from zero — it starts from a very informed position.

2. **The test set is from the same distribution as training.** Because train/test came from the same curated dataset, the model has essentially "seen the style" of these images during training even if not the specific test images. This inflates results compared to true out-of-distribution testing.

3. **Six classes with strong visual discriminators.** Some pairs are genuinely easy: LDPE (translucent flexible film) looks nothing like PS (rigid, often clear or white foam). PVC is noticeably different from HDPE. The model doesn't need to solve a hard fine-grained classification problem for all pairs.

### 3.4 — The Unsettling Val-to-Test Drop

| Model | Val Accuracy | Test Accuracy | Drop |
|---|---|---|---|
| EfficientNet-B0 | 91.15% | 88.94% | **−2.2%** |
| ResNet50 | 91.59% | 87.23% | **−4.4%** |

This gap is significant. It means the checkpoint we saved as "best" was partially tuned (by early stopping) to the specific 226 validation images. With only 226 val images, early stopping can accidentally overfit to validation noise. **The real model performance is closer to the test number, not the val number.**

---

## 4. Class Imbalance — The Central Issue

This deserves its own section because it underlies almost every other concern.

### The Imbalance in Numbers
```
Training samples per class:
LDPE  ████████████████████████████████████████████   412  (22.5%)
HDPE  ███████████████████████████████████           403  (22.0%)
PS    ███████████████████████████████████           396  (21.6%)
PET   █████████████████████████████              324  (17.7%)
PVC   █████████████████               180   (9.8%)
PP    ███████████              119   (6.5%)
```

PP has **3.5× fewer training samples than LDPE.** Despite our WeightedRandomSampler forcing equal class sampling per batch, the fundamental limitation is: there are only 119 different PP images the model can learn from. Augmentation helps but cannot generate genuinely new visual information — it just varies the same 119 images.

### How Imbalance Corrupts Accuracy

If the model classifies every test image as LDPE (the majority class), accuracy would be:
`53/235 = 22.6%` — obviously bad.

But here's the insidious part: **a model biased toward majority classes can still look impressive on accuracy.** Consider: LDPE+HDPE+PS account for 157 of 235 test images (66.8%). Getting those three classes right almost perfectly while failing badly on PP (16 images) gives roughly: `(53+51+50)/235 ≈ 65.5%` baseline just from the easy classes. The model only needs to do a bit more work to reach 88%. This is why per-class F1 on PP tells the real story.

### The Test Set Imbalance Mirror

The test set reflects the same imbalance:
- PP: 16 samples (6.8% of test set)
- LDPE: 53 samples (22.6% of test set)

A model failing completely on PP loses at most 6.8% of accuracy points. This is why the accuracy number (88.94%) doesn't adequately punish PP failure, but Macro F1 (0.8641) does.

---

## 5. Training Dynamics — What the Numbers Reveal

### 5.1 EfficientNet-B0 — The Good and The Concerning

**Stage 1 (Feature Extraction) — Healthy behavior:**
- Epoch 1 immediately hit 84.96% val accuracy. This is the power of ImageNet pretraining — the frozen features are already plastic-relevant without any task-specific training.
- Train (93.1%) ≈ Val (88.9%) at end of Stage 1 → minimal overfitting when only the small head trains.

**Stage 2 (Fine-Tuning) — Overfitting is clearly present:**
```
Epoch  7: train=95.58%, val=91.15%  ← Best saved checkpoint (gap: 3.6%)
Epoch 14: train=97.87%, val=89.82%  ← Early stopped here (gap: 8.1%)
```

The train accuracy climbed from 88% → 98% while val accuracy did not follow. This is the textbook definition of overfitting. Early stopping correctly caught this, but the 7-epoch patience window means the model was still allowed to overfit considerably before stopping.

**The val loss divergence is especially revealing:**
```
Stage 2, Epoch 7:  val_loss = 0.3259  ← Best checkpoint
Stage 2, Epoch 14: val_loss = 0.4548  ← Where we stopped
```
Val loss increased by 40% after the peak. This means the model was becoming increasingly miscalibrated — more confidently wrong on certain images. The saved model (Epoch 7) is fine, but this trajectory warns that the model has limited capacity to improve further on this dataset without more data.

### 5.2 ResNet50 — The Red Flags

ResNet50's Stage 2 shows more alarming patterns:

```
Epoch  5: train=97.33%, val=90.71%,  val_loss=0.315
Epoch 10: train=98.47%, val=89.82%,  val_loss=0.525
Epoch 17: train=98.80%, val=84.51%,  val_loss=1.132  ← Spike
Epoch 19: train=98.96%, val=91.59%,  val_loss=0.675  ← Best saved
```

**Epoch 17 is a major red flag.** Val accuracy dropped to 84.51% while val loss spiked to 1.132 — this means the model made extremely confident wrong predictions on a batch of validation images. This is catastrophic miscalibration. The model "recovered" at Epoch 19 only because the cosine warm restart dropped the LR back down.

**The best checkpoint at Epoch 19 is fragile.** It achieved 91.59% val accuracy but with val_loss=0.675 — meaning even correct predictions had high uncertainty. The test accuracy drop (91.59% → 87.23%) confirms the val result was partially a statistical artifact.

**Why is ResNet50 overfitting harder?**
Unfreezing Layer4 alone adds **15.49M trainable parameters** trained on only 1,834 images. That's 8,416 parameters per training image — an extreme capacity-to-data ratio. EfficientNet's more efficient architecture unfreezes fewer effective parameters relative to its capacity.

---

## 6. Limitations — Honest Assessment

### Limitation 1: Dataset Size is Critically Small
2,295 total images across 6 classes is industry-standard "proof of concept" territory. Real-world production models for multi-class visual classification typically have tens of thousands of images per class. Everything here is valid research/academic work, but calling this "production-ready" without caveats would be dishonest.

### Limitation 2: Tiny Test Set Creates Unreliable Estimates
235 test images (16 for PP) gives very high-variance per-class metrics. The numbers we see are one sample from a distribution of possible results depending on how the split was done. A different random seed in `02_split_dataset.py` would give meaningfully different metric values.

### Limitation 3: No Out-of-Distribution Testing
We have zero evidence of how the model performs on:
- Real user photos taken with phones
- Compressed or low-resolution images
- Partially visible or occluded plastics
- Multiple plastic types in one image
- Plastics with printed labels, brands, or graphics covering the surface

### Limitation 4: Overfitting Present in Both Models
Both models show ~7–8% train-val accuracy gap in Stage 2. While early stopping prevented catastrophic overfitting, the models have some level of memorization. More diverse training data would reduce this.

### Limitation 5: PP Class Is Not Production-Ready
F1=0.606 for PP means the model misclassifies approximately 4 out of every 10 PP items it encounters. In a recycling application, this matters: PP (polypropylene) is commonly used in food containers, bottle caps, and straws — items that users are likely to scan.

### Limitation 6: Validation Set Used for Hyperparameter Selection = Overly Optimistic Val Metrics
Our early stopping patience, LR schedule, and architecture choices were all tuned (at least mentally) while watching validation metrics. This means the val set is not truly "held out" — it influenced decisions. This is an accepted practice but means val metrics are slightly optimistic.

### Limitation 7: No Calibration Testing
We have no measurement of how well the model's confidence (softmax probabilities) reflects its actual accuracy. A model that says "95% PET" should be right ~95% of the time. This is called **calibration**, and it matters a lot for user trust in the app. We haven't measured it.

---

## 7. What Could Be Better

### 7.1 More Data — The Highest ROI Fix
Collecting 200–300 more PP images alone would likely push PP F1 from 0.61 → 0.75+, and Macro F1 from 0.864 → 0.880+. Adding the same for PVC and PET would close the remaining gaps.

### 7.2 Reduce Early Stopping Patience
The current patience=7 allowed EfficientNet to overfit for 7 epochs after its peak. Reducing to patience=5 would save the best model closer to its actual peak and reduce overfitting.

### 7.3 Larger Test Set
Before claiming final results, the test set should be expanded. Ideally: min 30 images per class, ideally 50+. Currently PP has 16, which is statistically insufficient.

### 7.4 Test Time Augmentation (TTA)
At evaluation, averaging predictions over multiple augmented versions of the same image (original + horizontal flip + slight zoom) typically gives 1–2% free accuracy boost without any retraining.

### 7.5 Model Calibration / Temperature Scaling
After training, applying temperature scaling to calibrate the model's confidence outputs would make the app more trustworthy. Users should see "60% confident — PET likely but consider PVC" rather than always seeing artificially inflated 95%+ confidences.

### 7.6 K-Fold Cross-Validation for Robust Estimates
Instead of one train/val/test split, running 5-fold cross-validation on the full dataset would give much more reliable metric estimates with confidence intervals — especially important for PP with its tiny sample size.

### 7.7 Label Smoothing
Adding `label_smoothing=0.1` to `CrossEntropyLoss` would prevent the model from becoming overconfident on training samples, which directly reduces overfitting and typically improves generalization by 0.5–1%.

---

## 8. Per-Class Deep Dive — Why Each Class Performs as It Does

### LDPE — F1: 0.9623 (Excellent)
Physically: LDPE is flexible, translucent film/bags. It has a very distinctive visual signature (soft, crinkly, semi-transparent). The model learned this well. High performance on LDPE partly inflates overall accuracy because it's also the most common class (53 test samples).

### PVC — F1: 0.9778 (Excellent, Precision=1.000)
Physically: PVC is rigid, often clear or with a distinctive blue/green tint. The precision of 1.000 is remarkable — zero false PVC predictions. The model has found highly discriminative PVC-specific features. Also only 23 test samples, so this result has higher variance.

### HDPE — F1: 0.9038 (Good)
Physically: HDPE is opaque, often white/colored rigid plastic. Distinctive but can resemble PS. High recall (0.922) means the model catches most HDPE — missing 4 items out of 51.

### PS — F1: 0.9032 (Good, but Recall=0.840)
Recall of 0.840 means 8 out of 50 PS items were missed. Polystyrene (foam cups, rigid clear plastic) can look similar to HDPE or PET depending on the product form. The model's high PS precision (97.67%) means it rarely falsely labels other plastics as PS, but it does miss real PS items. This suggests PS features learned are conservative.

### PET — F1: 0.8315 (Acceptable, low Precision=0.787)
PET's low precision (78.7%) means the model raises too many PET false alarms — it predicts PET when the actual plastic is something else. This likely happens because PET (clear bottles) overlaps visually with PVC (clear sheets) and some clear HDPE forms. The model errs toward predicting PET for transparent plastics.

### PP — F1: 0.6061 (Problematic)
The root issue is data starvation. PP (polypropylene, used in yogurt containers, bottle caps, straws) has diverse physical forms that don't share one unified visual signature as strongly as LDPE does. With only 119 training images across many product types, the model cannot generalize well. This is not a model architecture failure — it is a training data failure.

---

## 9. Benchmark Achievement Summary

| Metric | Target (Good) | Actual (EffNet-B0) | Verdict |
|---|---|---|---|
| Test Accuracy | >85% | 88.94% | ✅ Passes |
| Macro F1 | >0.82 | 0.8641 | ✅ Passes (marginally) |
| Worst-class F1 | >0.70 | 0.6061 (PP) | ❌ Fails |
| Cohen's Kappa | >0.80 | 0.8636 | ✅ Passes |
| Top-3 Accuracy | >95% | 97.45% | ✅ Passes |

**4 out of 5 benchmarks met.** The single failure (PP F1) is a data limitation, not a fundamental model architecture problem.

---

## 10. Honest Verdict

**What is true:**
- For a ~2,300 image dataset using transfer learning, these results are genuinely good and align with published research on similar plastic/waste classification tasks.
- EfficientNet-B0 is clearly the better choice for deployment — better metrics, smaller size, more reliable generalization.
- The training pipeline works correctly. The GPU optimizations, early stopping, and weighted sampling all functioned as intended.

**What needs qualification:**
- These metrics were produced on a 235-image test set from the same source/distribution as training. Real-world accuracy will be lower, possibly by 5–15%.
- PP classification is unreliable (F1=0.606 on only 16 test samples). Do not present this as a confident result.
- Overfitting is present in both models. More data is the primary fix.
- "88.94% accuracy" should always be presented alongside "Macro F1 = 0.864" because accuracy alone is misleading with an imbalanced dataset.

**What this means for Phase 3:**
Deploy EfficientNet-B0 with these caveats built into the API:
1. Return the full confidence vector (all 6 probabilities), not just the top prediction.
2. Flag predictions with max confidence < 0.70 as "uncertain — please verify".
3. Log all low-confidence predictions for future dataset expansion.
4. Display "PP detection accuracy is limited; if in doubt, check the recycling code on the package" in the UI for PP predictions.
