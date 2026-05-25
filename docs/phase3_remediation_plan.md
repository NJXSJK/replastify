# Replastify Remediation & Optimization Plan
**Scope:** Open ML & Security Issues | **Phase:** Phase 3 to Phase 4 Transition

---

## 1. Machine Learning Remediation Plan

### Issue: Poor PP Class Generalisation (F1 = 0.606) & Validation Overfitting
- **Goal:** Elevate PP class F1-score to $>75\%$ and reduce the gap between training and validation accuracy.

#### Action 1.1: Targeted Data Collection & Balance Scaling
- **Execution:** Collect 300 additional images of Polypropylene (PP) representing diverse real-world items (yogurt containers, takeaway boxes, bottle caps, medicine bottles) in various lighting and backgrounds.
- **Why it works:** Augmentation only perturbs existing samples. Genuinely new images provide the model with new visual structures, reducing overfitting.

#### Action 1.2: Model Calibration via Temperature Scaling
- **Execution:** Implement post-hoc temperature scaling on the validation set. Find a single scalar parameter $T > 0$ to scale logits: $\hat{p}_i = \text{softmax}(z_i / T)$.
- **Why it works:** Prevents the model from outputting overconfident predictions (e.g. 99% confident when actually incorrect), aligning prediction confidence with real accuracy.

#### Action 1.3: Label Smoothing Regularization
- **Execution:** Update `03_train_model.py` to use `nn.CrossEntropyLoss(label_smoothing=0.1)`.
- **Why it works:** Softens the target distribution from hard one-hot vectors, preventing the model from becoming excessively confident on training samples.

#### Action 1.4: Test-Time Augmentation (TTA)
- **Execution:** During backend prediction, run inference on the original image, a horizontally flipped version, and a slightly zoomed version, then average their output probabilities.
- **Why it works:** Typically yields a 1–2% gain in test accuracy without retraining.

---

## 2. API & Security Hardening Plan

### Issue: Incomplete File-Type Validation & Abuse Vulnerability
- **Goal:** Harden the upload boundary against spoofed files and prevent brute-force API exploitation.

#### Action 2.1: Double-Gate Input Validation (Extension + MIME-Type)
- **Execution:** Modify `image_utils.py` to inspect the `file.content_type` header (checking for `image/jpeg`, `image/png`, `image/webp`).
- **Logic:**
  ```python
  # Verify either extension or MIME type matches
  mime_ok = file.content_type in {"image/jpeg", "image/png", "image/webp"}
  ext_ok = ext in settings.allowed_extensions
  if not (mime_ok or ext_ok):
      raise HTTPException(status_code=400, detail="Invalid image payload format.")
  ```

#### Action 2.2: Rate Limiting via SlowAPI
- **Execution:** Install `slowapi` and decorate the `/predict` route with a limit of 10 requests per minute per IP address.
- **Why it works:** Prevents malicious automated scraping, denial of service attacks, and API key consumption abuse.

---

## 3. Observability & Monitoring Plan

### Issue: Lack of Metrics and Drift Detection
- **Goal:** Provide administrators with real-time insight into performance.

#### Action 3.1: Prometheus Performance Metrics
- **Execution:** Expose a `/metrics` endpoint to monitor prediction count per class, response latencies, and fallback database usage rate.
- **Action 3.2: Prediction Auditing Log Stream**
  - Stream logs to Elasticsearch/Grafana Loki to monitor confidence distributions over time, highlighting when confidence falls below the 0.70 threshold.
