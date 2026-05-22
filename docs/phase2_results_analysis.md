# Replastify — Phase 2 Result Analysis
**Run Date:** 2026-05-22 | **GPU:** Tesla T4 (15.6 GB VRAM) | **Dataset:** 1834 train / 226 val / 235 test

---

## 1. Executive Summary

Both models completed training in **14.4 minutes total** on a Kaggle T4 GPU. **EfficientNet-B0 is the clear winner** on all primary metrics despite having 5.5× fewer parameters than ResNet50, confirming our architecture hypothesis.

| Metric | EfficientNet-B0 | ResNet50 | Winner |
|---|---|---|---|
| **Test Accuracy** | **88.94%** | 87.23% | EffNet ✅ |
| **Macro F1** | **0.8641** | 0.8278 | EffNet ✅ |
| **Weighted F1** | **0.8909** | 0.8703 | EffNet ✅ |
| **Cohen's Kappa** | **0.8636** | 0.8422 | EffNet ✅ |
| Top-3 Accuracy | 0.9745 | **0.9787** | ResNet ✅ |
| Parameters | **4.3M** | 24.0M | EffNet ✅ |
| Model size | **~20 MB** | ~98 MB | EffNet ✅ |

> **Verdict:** EfficientNet-B0 achieves better accuracy with 5.5× fewer parameters. It is the model to deploy.

---

## 2. Training Dynamics — EfficientNet-B0

### Stage 1: Feature Extraction (10 epochs, LR 1e-3)
- **Only 7.6% of parameters were trained** (329K / 4.3M) — the frozen backbone acts as a fixed feature extractor.
- **Epoch 1 immediately achieved 84.96% val accuracy.** This is the power of ImageNet transfer learning — the pre-trained features are immediately useful for plastics.
- **Best val accuracy achieved at Epoch 5: 89.38%**
- Epochs 6–10 show the model plateauing (val acc oscillates around 88.5%). This is expected — the frozen backbone limits how much the head alone can improve.
- **Train/val gap is tight**: train=93.08% vs val=88.5% — minimal overfitting in Stage 1 because only the small head is being trained.

### Stage 2: Fine-Tuning (14 epochs, LR 1e-4 backbone / 1e-3 head)
- **Trainable params jumped to 80.4%** (3.49M) — last 3 blocks of the backbone are now adapting.
- **Epoch 1 of Stage 2 shows regression (87.61% val):** A small initial drop is normal. The newly unfrozen weights briefly "destabilize" the learned head mappings before settling.
- **Best result at Epoch 7 (of Stage 2): 91.15% val accuracy** — the warm restart in the cosine scheduler aligned perfectly, producing the peak performance.
- **Clear overfitting onset after Epoch 7:** Train accuracy climbed to 98.26% while val stuck at 89-91%. This is the classic signature of a small dataset (~2K images) being memorized.
- **Early stopping fired correctly at Epoch 14**, saving the best checkpoint from Epoch 7.
- **Val loss trend is concerning but normal:** Rose from 0.30 → 0.45 in the overfit zone. The saved model (Epoch 7) has val loss=0.326 which is acceptable.

### Key Observations
- The cosine warm restart at Epoch 6 (LR jumps back to 1e-4) triggered the best epoch (7). This confirms the scheduler is doing its job — the temporary LR bump helped escape a local minimum.
- Train acc reached 98.26% vs val 91.15% — a **7.1% train-val gap**. This is moderate overfitting. Acceptable for ~2K images, but improvement is possible.

---

## 3. Training Dynamics — ResNet50

### Stage 1: Feature Extraction (10 epochs, LR 1e-3)
- **Only 2.2% of params trained** (526K / 24M) — because the frozen ResNet backbone is massive (24M params) but we only trained the tiny head (526K).
- **Slower start**: Epoch 1 val=81.42% vs EfficientNet's 84.96%. ResNet's deeper architecture takes longer to adapt the head because the features are at a different scale/distribution.
- Best at Epoch 5: **88.05% val accuracy**
- Epochs 6–10 plateau similarly to EfficientNet.

### Stage 2: Fine-Tuning (20 epochs, ran all)
- **64.5% trainable params** (15.49M) — unfreezing Layer4 of ResNet50 adds a LOT of trainable params. This is dangerous for a ~2K dataset.
- **Heavy and progressive overfitting is clearly visible:**
  - Epoch 5: train=97.33%, val=90.71% (gap=6.6%)
  - Epoch 10: train=98.47%, val=89.82% (gap=8.7%)
  - Epoch 15: train=99.35%, val=89.82% (gap=9.5%)
  - Final: train=99.18%, val=89.38% — while val loss explodes to 0.55
- **Val loss becomes severely unstable:** Rose from 0.33 → 0.68 average in the second half, with a spike to **1.1323 at Epoch 17** — the worst recorded. This means the model was occasionally making very confident wrong predictions.
- **Despite this, it finds its best at Epoch 19 (91.59% val)** — a lucky warm restart alignment pulled it up right at the end. This is fragile; the high val loss (0.675) means it's not a reliable checkpoint.
- **Test set reality check:** Val accuracy was 91.59% but test accuracy was only 87.23% — a **4.36% drop**. This is the largest val-to-test gap of the two models, confirming the Epoch-19 checkpoint was partially lucky on the specific validation distribution.

---

## 4. Test Set Evaluation — Deep Per-Class Analysis

### EfficientNet-B0 — Per-Class Breakdown

| Class | Precision | Recall | F1 | Support | Diagnosis |
|---|---|---|---|---|---|
| **LDPE** | 0.9623 | 0.9623 | **0.9623** | 53 | 🟢 Excellent. Visually distinctive features learned well. |
| **PVC** | 1.0000 | 0.9565 | **0.9778** | 23 | 🟢 Perfect precision — zero false alarms. |
| **HDPE** | 0.8868 | 0.9216 | **0.9038** | 51 | 🟢 Strong. High recall means few HDPE items missed. |
| **PS** | 0.9767 | 0.8400 | **0.9032** | 50 | 🟡 Great precision but 16% of PS items missed. Possible PET confusion. |
| **PET** | 0.7872 | 0.8810 | **0.8315** | 42 | 🟡 Lower precision (21% false alarms). Likely confused with HDPE/PVC. |
| **PP** | 0.5882 | 0.6250 | **0.6061** | 16 | 🔴 Weakest class. Only 16 test samples and 119 training samples. |

**PP Analysis:** F1=0.606 is above our "minimum viable" threshold of 0.55 but below "good" (0.70). With only 119 training images (vs 412 for LDPE), this gap is mathematically expected. Adding ~150–200 more PP images would likely push this above 0.75.

**PS Analysis:** Recall=0.840 means 8 out of 50 PS items were misclassified. Polystyrene can look similar to HDPE (both opaque white/grey). The high precision (97.67%) means when the model does say "PS", it's almost always right.

**PVC Analysis:** Precision=1.000 is remarkable — zero false positive PVC predictions. The model has learned very strong PVC-specific features (likely the flexibility/clarity cues).

### ResNet50 — Per-Class Breakdown

| Class | Precision | Recall | F1 | Support | Diagnosis |
|---|---|---|---|---|---|
| **HDPE** | 0.9783 | 0.8824 | **0.9278** | 51 | 🟢 High precision but misses 12% of HDPE items. |
| **LDPE** | 0.8929 | 0.9434 | **0.9174** | 53 | 🟢 Good but slightly lower than EfficientNet. |
| **PVC** | 1.0000 | 0.9565 | **0.9778** | 23 | 🟢 Same as EfficientNet — PVC is easy to identify. |
| **PS** | 0.9167 | 0.8800 | **0.8980** | 50 | 🟡 Similar to EfficientNet. |
| **PET** | 0.7451 | 0.9048 | **0.8172** | 42 | 🟡 Higher recall than EfficientNet but even lower precision. |
| **PP** | 0.5000 | **0.3750** | **0.4286** | 16 | 🔴 **FAILING.** F1=0.4286 is below our 0.50 red flag threshold. ResNet misses 62.5% of PP items. |

**PP is ResNet50's critical failure.** Recall=0.375 means only 6 out of 16 PP test items were correctly identified. The remaining 10 were silently misclassified as other plastics. This directly explains why ResNet's Macro F1 (0.8278) is much lower than EfficientNet's (0.8641) — one failing class drags the macro average significantly.

---

## 5. Model Comparison — Head-to-Head

### Where EfficientNet Wins (Primary Metrics)
- **Macro F1: 0.8641 vs 0.8278** — 3.6% better. This is large for a classification task.
- **PP class F1: 0.606 vs 0.429** — EfficientNet is dramatically better at the hardest class. This is due to its Squeeze-and-Excitation attention mechanism that can focus on subtle texture differences.
- **Generalization gap (val → test): 91.15% → 88.94% = 2.2% drop.** Very acceptable.
- **Cohen's Kappa: 0.8636 vs 0.8422** — EfficientNet agrees with ground truth substantially more, adjusted for chance.

### Where ResNet Wins (Secondary Metrics)
- **Top-3 Accuracy: 0.9787 vs 0.9745** — ResNet keeps the correct answer in its top 3 guesses slightly more often. This is because ResNet's larger feature space (2048-dim vs 1280-dim) can sometimes "hedge" better even when the primary guess is wrong.
- **HDPE F1: 0.9278 vs 0.9038** — ResNet identifies HDPE more precisely (fewer false HDPE predictions).

### Generalization Analysis
| Model | Best Val Acc | Test Acc | Drop | Verdict |
|---|---|---|---|---|
| EfficientNet-B0 | 91.15% | 88.94% | **−2.2%** | ✅ Healthy |
| ResNet50 | 91.59% | 87.23% | **−4.4%** | ⚠️ Over-optimistic val |

ResNet50's val→test gap is 2× larger. This confirms the val accuracy at Epoch 19 was partially luck (the model happened to perform well on those specific 226 validation images while being heavily overfit overall).

---

## 6. Overfitting Assessment

| Model | Final Train Acc | Best Val Acc | Gap | Assessment |
|---|---|---|---|---|
| EfficientNet-B0 | 97.87% (Ep14) | 91.15% (Ep7) | 6.7% | ⚠️ Moderate overfitting |
| ResNet50 | 99.18% (Ep20) | 91.59% (Ep19) | 7.6% | ⚠️ Significant overfitting |

Both models overfit, but ResNet50 overfit more severely. The smoking gun is ResNet50's val loss: it kept rising (0.33 → 0.68 → 1.13) even as val accuracy occasionally improved — a sign the model was becoming increasingly **overconfident** on its correct answers while also making more confidently wrong predictions.

---

## 7. Benchmark Assessment vs. Phase 2 Targets

| Metric | Target (Good) | EfficientNet-B0 | ResNet50 | Status |
|---|---|---|---|---|
| Overall Accuracy | >85% | 88.94% ✅ | 87.23% ✅ | Both pass |
| Macro F1 | >0.82 | 0.8641 ✅ | 0.8278 ≈ | EffNet passes |
| Worst-class F1 | >0.70 | 0.6061 ❌ | 0.4286 ❌ | Both fail on PP |
| Cohen's Kappa | >0.80 | 0.8636 ✅ | 0.8422 ✅ | Both pass |
| Top-3 Accuracy | >95% | 97.45% ✅ | 97.87% ✅ | Both pass |

**The only failed benchmark is worst-class F1 on PP.** This is expected given PP has only 119 training images. It is not a model failure — it is a data limitation.

---

## 8. Identified Issues & Recommended Next Steps

### Issue 1 — PP class needs more data (Priority: High)
- **Root cause:** 119 training images is simply not enough for a visually ambiguous class.
- **Fix:** Collect 150–200 more PP images. A dataset expansion from 119 → 300 should push PP F1 from 0.61 → 0.75+.

### Issue 2 — PET precision is low (Priority: Medium)
- **Root cause:** PET (transparent bottles) is visually similar to both PVC (transparent sheets) and HDPE (translucent). The model generates 21% false PET predictions.
- **Fix:** Try label smoothing (`label_smoothing=0.1`) or Mixup augmentation to soften decision boundaries between PET, PVC, and HDPE.

### Issue 3 — ResNet50 val loss instability (Priority: Low — ResNet is not our deployed model)
- **Root cause:** With 15.49M trainable params on 1834 images, the 2nd stage LR warm restart (1e-4) is too aggressive for this many params.
- **Fix (if needed):** Reduce Stage 2 backbone LR to 5e-5 for ResNet50.

### Issue 4 — Val → Test accuracy drop (Priority: Low)
- EfficientNet drops 2.2% from val to test. This is acceptable but could be reduced by using the full train+val set for a final retraining pass before deployment.

---

## 9. Deployment Decision

**Deploy: EfficientNet-B0 (`best_efficientnet_b0.pth`)**

| Reason | Detail |
|---|---|
| Higher Macro F1 | 0.8641 vs 0.8278 — treats all plastic types more fairly |
| Better PP performance | F1=0.606 vs 0.429 — doesn't fail on minority class |
| Smaller model | 20 MB vs 98 MB — faster API response times |
| Better generalization | 2.2% val→test drop vs 4.4% for ResNet |
| Production safety | Lower val loss (0.326) vs ResNet's unstable (0.675) |

---

## 10. Phase 3 Readiness

| Item | Status |
|---|---|
| Best model file (`best_efficientnet_b0.pth`) | ✅ Ready |
| Class mapping (`HDPE:0, LDPE:1, PET:2, PP:3, PS:4, PVC:5`) | ✅ Confirmed |
| Input format (224×224 RGB, ImageNet normalized) | ✅ Confirmed |
| Output format (6-class softmax probabilities) | ✅ Confirmed |
| Confidence threshold for "uncertain" prediction | 🔲 Recommend ≥ 0.65 as minimum confidence |

> **Phase 3 (FastAPI Backend) can proceed using EfficientNet-B0. The model is production-ready for all classes except PP, which should be flagged as "lower confidence" to the end user until more PP data is collected.**
