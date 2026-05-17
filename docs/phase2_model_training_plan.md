# Replastify — Phase 2: Model Training & Evaluation (Comprehensive Technical Plan)

---

## 1. Dataset Analysis & Constraints

### 1.1 Dataset Profile

| Property | Value |
|---|---|
| **Total images** | 2,295 |
| **Active classes** | 6 (PET, HDPE, PVC, LDPE, PP, PS) |
| **Skipped** | "Other" (only 4 images — insufficient for training) |
| **Largest class** | LDPE — 516 images (22.4%) |
| **Smallest class** | PP — 149 images (6.5%) |
| **Imbalance ratio** | 3.5:1 (LDPE:PP) |
| **Image dimensions** | 524–640px, mean ~587×586 |
| **Split** | Train: 1,834 / Val: 226 / Test: 235 |
| **Color space** | RGB (3 channels) |

### 1.2 Class Distribution (Train Set)

```
PET  ████████████████████████████████   324 (17.7%)
HDPE ████████████████████████████████████████   403 (22.0%)
PVC  ██████████████████   180  (9.8%)
LDPE █████████████████████████████████████████   412 (22.5%)
PP   ████████████   119  (6.5%)
PS   ███████████████████████████████████████   396 (21.6%)
```

### 1.3 Technical Challenges

| Challenge | Impact | Severity |
|---|---|---|
| **Small dataset (~2.3K)** | CNNs need thousands per class; high overfitting risk | 🔴 Critical |
| **Class imbalance (3.5:1)** | Model biases toward LDPE/HDPE; ignores PP | 🟡 Moderate |
| **Visual similarity** | PET (transparent bottles) vs PVC (transparent sheets) share visual features | 🔴 Critical |
| **Intra-class variation** | Same plastic type appears as bottles, bags, containers, films | 🟡 Moderate |
| **Variable backgrounds** | Dataset images have mixed lighting, angles, backgrounds | 🟡 Moderate |

### 1.4 Future Data Scaling Considerations

The training pipeline will be designed to handle dataset growth:
- **Modular data loading**: `ImageFolder` + configurable paths, so adding new images = just dropping files into class folders
- **Automatic class weight recalculation**: Weights computed dynamically from actual counts, not hardcoded
- **Re-splitting script**: Can re-run `02_split_dataset.py` anytime after adding new images
- **When data grows beyond ~5K images**: Can increase batch size, reduce augmentation intensity, unfreeze more layers
- **When data grows beyond ~10K images**: Can consider training from scratch (no transfer learning) or try larger models (EfficientNet-B3, ResNet101)

> **Rule of thumb**: With transfer learning, aim for **at least 100 images per class** for reasonable performance, **500+** for good performance, and **1000+** for robust production-quality models.

---

## 2. Model Architecture Analysis

### 2.1 Why Transfer Learning is Mandatory

**Transfer learning** uses a model pre-trained on ImageNet (1.2M images, 1000 classes) and adapts it to our task. This works because:

1. **Early CNN layers** learn universal features (edges, textures, colors) useful for ANY image task
2. **Middle layers** learn compositional features (corners, shapes, patterns)
3. **Deep layers** learn task-specific features (object parts, semantics)

By reusing layers 1-2 and retraining layer 3, we get the benefit of being trained on millions of images while only needing ~2K of our own.

**Mathematical intuition**: A CNN with `N` parameters trained on `M` images has an effective capacity ratio of `M/N`. For ResNet50 (25.6M params) trained on our 1,834 images: `1834/25600000 = 0.00007`. This is catastrophically low — the model has far more capacity than data, guaranteeing overfitting. Transfer learning "pre-fills" most parameters with useful values, so we only train a fraction of them.

---

### 2.2 Candidate Architectures — Technical Deep Dive

#### 🏆 EfficientNet-B0 (PRIMARY — Recommended)

**Core Innovation: Compound Scaling**

Traditional CNNs scale by increasing only depth (more layers), width (more channels), or resolution. EfficientNet scales ALL THREE simultaneously using a compound coefficient `φ`:

```
depth:      d = α^φ
width:      w = β^φ  
resolution: r = γ^φ

Constraint: α · β² · γ² ≈ 2   (keeps FLOPs at 2^φ growth)

For B0: α=1.2, β=1.1, γ=1.15, φ=1
```

**Why this matters for us**: Compound scaling means EfficientNet-B0 gets maximum feature richness from minimum parameters. With only 5.3M params (vs ResNet50's 25.6M), it's far less likely to overfit our small dataset.

**Building Blocks — MBConv (Mobile Inverted Bottleneck)**

Each MBConv block contains:
```
Input (C channels)
  │
  ▼
1×1 Conv (Expand: C → C×expansion_ratio)    ← Increase dimensions
  │
  ▼
Depthwise 3×3/5×5 Conv                      ← Spatial filtering (cheap)
  │
  ▼
Squeeze-and-Excitation Block                 ← Channel attention
  │  ┌──────────────────────┐
  │  │ GlobalAvgPool → C×r  │
  │  │ FC → ReLU → FC → σ  │  ← Learn which channels matter
  │  │ Multiply × features  │
  │  └──────────────────────┘
  ▼
1×1 Conv (Project: C×expansion → C_out)      ← Reduce dimensions
  │
  ▼
Skip Connection (if input/output match)      ← Gradient highway
```

**Squeeze-and-Excitation (SE) explained**: This is a channel attention mechanism. It asks "which feature channels are most important for this image?" by:
1. **Squeeze**: Global average pooling compresses each channel's spatial info into a single number
2. **Excite**: Two FC layers learn inter-channel dependencies and output per-channel weights (0 to 1)
3. **Scale**: Multiply original features by these weights — amplify useful channels, suppress noise

**For plastic classification**: SE blocks help the model learn that texture channels matter more than color channels (since many plastics are transparent/white), improving fine-grained discrimination.

| Spec | Value |
|---|---|
| Parameters | 5.3M |
| ImageNet Top-1 | 77.1% |
| FLOPs | 0.39B |
| Model file size | ~20 MB |
| Inference speed | ~4ms (GPU) / ~50ms (CPU) |
| PyTorch class | `torchvision.models.efficientnet_b0` |
| Feature dim (before classifier) | 1280 |

**Pros for our case**: Best params-to-accuracy; SE attention aids fine-grained plastic texture recognition; small model = fast web deployment.

**Cons**: Swish/SiLU activation + SE adds ~10% training overhead vs ResNet; less debugging community compared to ResNet.

**Relevance: ★★★★★ (5/5)**

---

#### 🥈 ResNet50 (BASELINE — For comparison)

**Core Innovation: Residual (Skip) Connections**

The key insight: in very deep networks, gradients vanish/explode. Residual blocks solve this:

```
Input x
  │
  ├───────────────────┐
  ▼                   │ (skip/shortcut)
Conv → BN → ReLU      │
  ▼                   │
Conv → BN             │
  ▼                   │
  + ←─────────────────┘ (element-wise addition)
  ▼
ReLU
  ▼
Output = F(x) + x     ← Network learns the RESIDUAL F(x), not full mapping
```

**Why residuals work**: Instead of learning `H(x)` directly, the network learns `F(x) = H(x) - x`. If the optimal transformation is close to identity (common in deep networks), learning `F(x) ≈ 0` is much easier than learning `H(x) ≈ x`. This makes training very deep networks stable.

**ResNet50 Architecture**:
```
[Conv 7×7, stride 2] → [BN → ReLU → MaxPool]
     ↓
[Bottleneck Block × 3]  (64→256 channels)    ← Layer 1
     ↓
[Bottleneck Block × 4]  (128→512 channels)   ← Layer 2
     ↓
[Bottleneck Block × 6]  (256→1024 channels)  ← Layer 3
     ↓
[Bottleneck Block × 3]  (512→2048 channels)  ← Layer 4
     ↓
[AdaptiveAvgPool → FC(2048 → 1000)]
```

Each **Bottleneck Block** has: `1×1 conv (reduce) → 3×3 conv (process) → 1×1 conv (expand) + skip`

| Spec | Value |
|---|---|
| Parameters | 25.6M |
| ImageNet Top-1 | 76.1% |
| FLOPs | 4.1B |
| Model file size | ~98 MB |
| Inference speed | ~8ms (GPU) / ~120ms (CPU) |
| PyTorch class | `torchvision.models.resnet50` |
| Feature dim (before classifier) | 2048 |

**Pros**: Rock-solid training stability; most studied CNN ever; easy to explain in reports; massive community.

**Cons**: 5× more parameters than EfficientNet = higher overfitting risk; 10× more FLOPs; larger deployment footprint.

**Relevance: ★★★★☆ (4/5)**

---

#### 🥉 MobileNetV2 (OPTIONAL — Speed priority)

**Core Innovation: Inverted Residuals + Linear Bottlenecks**

Unlike ResNet which narrows→widens, MobileNetV2 widens→narrows:

```
Input (thin)
  ▼
1×1 Conv (Expand)        ← Increase channels (expansion factor 6)
  ▼
Depthwise 3×3 Conv       ← Spatial filtering (1 filter per channel = very cheap)
  ▼
1×1 Conv (Project, LINEAR — no ReLU!)  ← Compress back (ReLU would destroy low-dim info)
  ▼
Skip connection (thin → thin)
```

**Depthwise separable convolution cost**: Standard conv on `H×W×C_in → C_out` costs `H·W·C_in·C_out·K²` operations. Depthwise separable costs `H·W·C_in·K² + H·W·C_in·C_out` — typically **8-9× cheaper**.

| Spec | Value |
|---|---|
| Parameters | 3.4M |
| ImageNet Top-1 | 72.0% |
| FLOPs | 0.3B |
| Model file size | ~14 MB |
| Feature dim | 1280 |

**Pros**: Smallest and fastest; lowest overfitting risk.
**Cons**: Depthwise convolutions process each channel independently — may miss cross-channel texture patterns important for plastic type discrimination; 5% lower ImageNet baseline means less rich pre-trained features.

**Relevance: ★★★☆☆ (3/5)** — Only if deployment speed is critical.

---

### 2.3 Final Selection

| Role | Model | Why |
|---|---|---|
| **Primary** | **EfficientNet-B0** | Best accuracy/params ratio; SE attention aids texture discrimination; proven on plastic datasets |
| **Baseline** | **ResNet50** | Industry standard comparison; strengthens assignment by showing architectural comparison |

We train both and compare. The better performer becomes the deployed model.

---

## 3. Training Strategy — Technical Details

### 3.1 Data Pipeline

#### Training Transforms
```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # Why: Simulates zoom/crop variation; scale=(0.8,1.0) prevents
    # extreme crops that lose the object entirely
    
    transforms.RandomHorizontalFlip(p=0.5),
    # Why: Plastics have no inherent left-right orientation
    # Note: NO vertical flip — plastics rarely appear upside-down
    
    transforms.RandomRotation(15),
    # Why: Simulates slight camera tilt; 15° is conservative to avoid
    # unrealistic orientations
    
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Why: Real-world lighting varies; hue=0.1 is small because
    # color IS somewhat informative for plastic type
    
    transforms.ToTensor(),
    # Converts PIL Image (0-255) → Tensor (0.0-1.0)
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Why: ImageNet pre-trained models EXPECT these exact stats.
    # Using different normalization = feeding garbage to the model.
    # mean/std were computed across 1.2M ImageNet images.
])
```

#### Validation/Test Transforms (NO augmentation)
```python
val_transforms = transforms.Compose([
    transforms.Resize(256),
    # Why: Resize shortest edge to 256, maintaining aspect ratio
    
    transforms.CenterCrop(224),
    # Why: Deterministic crop ensures reproducible evaluation
    # (unlike RandomResizedCrop used in training)
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

> **Critical**: NEVER apply random augmentation to val/test sets. This would make metrics non-reproducible and artificially inflate/deflate results.

#### Advanced Augmentation (Fallback — if standard isn't enough)

| Technique | How It Works | When to Use | PyTorch Implementation |
|---|---|---|---|
| **Random Erasing** | Masks random rectangles with noise/mean | Model overfits to specific regions | `transforms.RandomErasing(p=0.3)` |
| **Mixup** | Blends two images: `x̃ = λ·x_i + (1-λ)·x_j`, labels too | Decision boundaries too sharp | Custom in training loop |
| **CutMix** | Pastes patch from image B onto image A; mixes labels by area ratio | Like Mixup but preserves local structure | Custom in training loop |
| **RandAugment** | Applies N random transforms at magnitude M | Want automated augmentation search | `transforms.RandAugment(num_ops=2, magnitude=9)` |

### 3.2 Handling Class Imbalance

#### Strategy 1: Weighted Random Sampler (Default)

```python
# Each sample gets a weight inversely proportional to its class frequency
# Effect: minority classes (PP=119) get sampled ~3.5× more often than majority (LDPE=412)

class_counts = [324, 403, 180, 412, 119, 396]  # PET, HDPE, PVC, LDPE, PP, PS
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

# Assign weight to each sample based on its class
sample_weights = [class_weights[label] for _, label in dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# replacement=True is important: allows the same PP image to appear multiple times per epoch
# This effectively oversamples minority classes without duplicating files
```

**Mathematically**: After weighting, the effective probability of sampling from each class becomes approximately `1/6 ≈ 16.7%` regardless of actual class size.

#### Strategy 2: Weighted Loss Function (If Strategy 1 isn't enough)

```python
# CrossEntropyLoss: L = -Σ w_c · y_c · log(p_c)
# Without weights: all classes contribute equally to loss per-sample
# With weights: misclassifying PP costs 3.5× more than misclassifying LDPE

weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum() * len(class_counts)  # normalize to sum=6
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# Resulting weights (approximate):
# PET: 1.05, HDPE: 0.85, PVC: 1.90, LDPE: 0.83, PP: 2.87, PS: 0.86
# → PP errors are penalized ~3.4× more than LDPE errors
```

### 3.3 Two-Stage Transfer Learning

#### Stage 1: Feature Extraction (Frozen Backbone)

**What happens**: All pre-trained backbone weights are LOCKED. Only the new classification head trains. The backbone acts as a fixed feature extractor.

```python
# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head (trainable)
model.classifier = nn.Sequential(
    nn.Dropout(0.4),           # 40% neurons randomly zeroed each forward pass
    nn.Linear(1280, 256),      # 1280 = EfficientNet-B0 feature dim
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 6),         # 6 output classes
)

# CRITICAL: Only pass trainable params to optimizer
optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
```

| Setting | Value | Rationale |
|---|---|---|
| Frozen layers | ALL backbone | Prevent destroying pre-trained features |
| Trainable params | ~330K (head only) | Small number = fast training, low overfit risk |
| Optimizer | AdamW | Decoupled weight decay (proper L2 regularization) |
| Learning rate | 1e-3 | Aggressive is fine — only training random-init head |
| Weight decay | 1e-2 | Standard for AdamW; prevents head weights from exploding |
| Epochs | 10 | Usually converges in 5-8 epochs |
| Batch size | 32 | Good balance of speed and gradient noise |
| Scheduler | CosineAnnealingLR(T_max=10) | Smooth decay: starts at 1e-3, ends near 0 |

#### Stage 2: Fine-Tuning (Partial Unfreeze)

**What happens**: We unfreeze the last few backbone blocks so they can adapt to plastic-specific features. Early layers (edges, textures) stay frozen — they're universal. Deep layers adapt to our domain.

```python
# Unfreeze last 2 blocks of EfficientNet-B0
# B0 has blocks named features.0 through features.8
for param in model.features[6:].parameters():
    param.requires_grad = True

# CRITICAL: BatchNorm gotcha (see Section 3.4)
# Keep frozen BN layers in eval() mode to preserve ImageNet statistics
for module in model.features[:6].modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

# Differential learning rate: backbone gets 10× lower LR than head
optimizer = optim.AdamW([
    {'params': model.features[6:].parameters(), 'lr': 1e-4},   # backbone: cautious
    {'params': model.classifier.parameters(),   'lr': 1e-3},   # head: aggressive
], weight_decay=1e-2)
```

| Setting | Value | Rationale |
|---|---|---|
| Unfrozen | Last 2-3 backbone blocks | Deepest layers learn domain-specific features |
| Backbone LR | 1e-4 | 10× lower — don't destroy pre-trained weights |
| Head LR | 1e-3 | Head still needs aggressive updates |
| Epochs | 20 (early stopping patience=7) | More epochs but we stop when val plateaus |
| Scheduler | CosineAnnealingWarmRestarts(T_0=5) | Periodic warm restarts help escape local minima |

**Why differential LR**: The backbone was trained on 1.2M images — its weights are valuable. The head was randomly initialized. If both use the same high LR, the backbone's careful representations get destroyed before the head learns to use them. Solution: backbone gets 10× smaller learning steps.

### 3.4 Critical Technical Precautions

#### ⚠️ BatchNorm Trap (Most Common Fine-Tuning Bug)

BatchNorm layers have TWO separate mechanisms:
1. **Learnable params** (`weight`, `bias`) — controlled by `requires_grad`
2. **Running statistics** (`running_mean`, `running_var`) — controlled by `train()/eval()` mode

**The trap**: Setting `requires_grad=False` does NOT stop running stats from updating. If frozen BN layers stay in `train()` mode, they'll update their running statistics based on our tiny plastic dataset, corrupting the ImageNet statistics.

```python
# WRONG — running stats will silently corrupt
for param in backbone.parameters():
    param.requires_grad = False
# BN layers are still in train() mode → running stats update!

# CORRECT — also freeze BN behavior
for param in backbone.parameters():
    param.requires_grad = False
for module in backbone.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()  # ← THIS is what actually freezes BN statistics
```

#### Other Key Precautions

| # | Precaution | Technical Detail |
|---|---|---|
| 1 | **Set random seeds** | `torch.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)`, `torch.backends.cudnn.deterministic = True` |
| 2 | **Save best model, not last** | `if val_acc > best_val_acc: torch.save(model.state_dict(), 'best.pth')` |
| 3 | **Monitor train-val gap** | If `train_loss << val_loss` → overfitting. If both high → underfitting |
| 4 | **Gradient clipping** | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` — prevents gradient explosions |
| 5 | **Mixed precision** | `torch.cuda.amp.autocast()` — 2× faster training, less GPU memory. Use if GPU available |
| 6 | **No augmentation on val/test** | Augmentation on eval data makes metrics unreproducible |
| 7 | **Verify normalization** | Wrong mean/std = feeding garbage to model. Always use ImageNet stats for pre-trained models |
| 8 | **GPU memory check** | EfficientNet-B0 @ batch=32 needs ~3 GB VRAM. ResNet50 needs ~5 GB. Check before training |
| 9 | **torch.no_grad() for eval** | `with torch.no_grad():` during validation — saves memory, prevents accidental gradient computation |
| 10 | **Data leakage check** | Verify train/val/test have zero file overlap. Our split script handles this, but double-check |

### 3.5 AdamW vs Adam — Why AdamW

Standard Adam applies weight decay incorrectly — it applies L2 penalty INSIDE the adaptive learning rate, weakening the regularization effect for parameters with large gradients.

**AdamW** decouples weight decay from the gradient update:

```
Adam:   θ = θ - lr · (m̂/(√v̂+ε) + λ·θ)     ← λ·θ is coupled with adaptive rate
AdamW:  θ = θ - lr · m̂/(√v̂+ε) - lr · λ·θ  ← λ·θ is independent (proper L2)
```

For transfer learning on small datasets, proper regularization matters significantly. Always use AdamW.

---

## 4. Evaluation Metrics — Complete Technical Reference

### 4.1 Core Metrics

#### Accuracy (Sanity Check Only)

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Limitation for our case**: If we predict LDPE for everything, accuracy = 412/1834 = 22.5%. Not zero, but meaningless. Accuracy hides per-class failures. **Never use accuracy alone with imbalanced data.**

#### Precision, Recall, F1-Score (Per-Class)

For each class `c`:

```
Precision_c = TP_c / (TP_c + FP_c)
  → "When we predict PET, how often is it actually PET?"
  → High precision = few false alarms

Recall_c = TP_c / (TP_c + FN_c)
  → "Of all actual PET items, how many did we catch?"
  → High recall = few missed items

F1_c = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)
  → Harmonic mean — punishes extreme imbalance between P and R
  → F1 = 0.50 if Precision=1.0 but Recall=0.33 (catches few items)
```

**Why harmonic mean, not arithmetic**: Arithmetic mean of P=1.0 and R=0.33 = 0.67 (looks okay). Harmonic mean = 0.50 (correctly penalizes the poor recall). The harmonic mean is always ≤ arithmetic mean, and drops sharply when either component is low.

#### Macro vs Weighted Averaging

| Averaging | Formula | When to Use |
|---|---|---|
| **Macro** | `mean(F1_c for all c)` — each class counts equally | **Our primary metric** — treats PP (149 images) as important as LDPE (516) |
| **Weighted** | `Σ (n_c/N) × F1_c` — weighted by class size | Secondary metric — reflects real-world distribution |
| **Micro** | Compute globally: `TP_total / (TP_total + FP_total)` | Equivalent to accuracy for single-label classification |

> **Our primary overall metric: Macro-Avg F1**. This ensures the model performs well for ALL plastic types, not just the common ones.

#### Cohen's Kappa

```
κ = (p_o - p_e) / (1 - p_e)

p_o = observed agreement (accuracy)
p_e = expected agreement by chance
```

**Why it matters**: Unlike accuracy, Kappa accounts for agreement that would happen by random guessing. A model that always predicts the majority class gets high accuracy but κ ≈ 0.

| κ Value | Interpretation |
|---|---|
| < 0.20 | Poor |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost perfect |

#### Top-K Accuracy

```
Top-K Accuracy = (samples where true label is in top K predictions) / (total samples)
```

**Why Top-3 matters for us**: If the model predicts "PET" but the item is actually "PVC" (a common confusion due to visual similarity), Top-3 accuracy tells us whether PVC was at least in the model's top 3 guesses. This is useful for the app — we can show "most likely X, also could be Y or Z".

### 4.2 Visualizations We'll Generate

| # | Visualization | Technical Details | Purpose |
|---|---|---|---|
| 1 | **Training curves** | Plot `train_loss`, `val_loss`, `train_acc`, `val_acc` per epoch | Detect overfitting (train↓, val↑) or underfitting (both high) |
| 2 | **Confusion matrix** | 6×6 heatmap, `sklearn.metrics.confusion_matrix` normalized by true labels | See which plastics confuse each other |
| 3 | **Classification report** | `sklearn.metrics.classification_report` — P, R, F1 per class | Numerical breakdown |
| 4 | **Per-class F1 bar chart** | Horizontal bar chart with threshold line at 0.50 | Quickly spot failing classes |
| 5 | **Confidence histogram** | Distribution of `max(softmax(logits))` for correct vs incorrect predictions | Model calibration — is it confident when right, uncertain when wrong? |
| 6 | **Misclassified samples** | Grid of worst errors: image + true label + predicted label + confidence | Visual debugging — understand WHY errors happen |
| 7 | **Grad-CAM heatmaps** | Overlay attention heatmap on input images | See WHERE the model looks — is it looking at the plastic or the background? |
| 8 | **Learning rate plot** | LR value over training steps | Verify scheduler behaves correctly |
| 9 | **Model comparison** | Side-by-side table of all metrics: EfficientNet vs ResNet50 | Choose the best model |
| 10 | **ROC curves** | One-vs-Rest ROC curve per class with AUC | Per-class discriminative ability |

### 4.3 Grad-CAM — Model Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes WHICH image regions the model uses for its prediction.

**How it works**:
1. Forward pass: get the feature maps `A` from the last convolutional layer
2. Backward pass: compute gradients of target class score w.r.t. `A`
3. Weight each feature map by the global-average-pooled gradient: `α_k = (1/Z) · Σ ∂y^c/∂A^k`
4. Weighted combination + ReLU: `L = ReLU(Σ α_k · A^k)`
5. Upsample to input size and overlay on original image

**Why it's critical for our project**:
- Validates the model looks at the **plastic object**, not the background
- Explains misclassifications: "It predicted PVC because it focused on the shiny surface, not the object shape"
- Provides visual evidence for assignment/report — demonstrates model understanding

```python
# Implementation using pytorch-grad-cam library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# For EfficientNet-B0, target the last convolutional layer
target_layer = model.features[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

grayscale_cam = cam(input_tensor=img_tensor)
visualization = show_cam_on_image(original_img, grayscale_cam[0])
```

### 4.4 Target Benchmarks

| Metric | Minimum Viable | Good | Excellent |
|---|---|---|---|
| **Overall Accuracy** | >75% | >85% | >92% |
| **Macro-Avg F1** | >0.70 | >0.82 | >0.90 |
| **Worst-class F1** | >0.55 | >0.70 | >0.80 |
| **Cohen's Kappa (κ)** | >0.65 | >0.80 | >0.90 |
| **Top-3 Accuracy** | >90% | >95% | >98% |

> **Red flag**: Any class F1 < 0.50 means the model is essentially failing for that plastic type. Immediate intervention needed (Section 5).

---

## 5. Improvement Strategy — Systematic Escalation

If the model underperforms, follow this escalation ladder (cheapest fixes first):

### Debugging Decision Tree

```
Model underperforms
│
├─ Train acc HIGH (>90%), Val acc LOW (<75%)
│  → OVERFITTING
│  ├─ Level 1: ↑ Dropout (0.4→0.6), ↑ Weight decay (1e-2→5e-2)
│  ├─ Level 1: ↓ Batch size (32→16) for more gradient noise
│  ├─ Level 2: Add Random Erasing, Mixup, or CutMix
│  └─ Level 3: Freeze MORE layers, use smaller model
│
├─ Train acc LOW (<75%), Val acc LOW (<70%)
│  → UNDERFITTING
│  ├─ Level 1: ↑ Epochs, check if LR too low
│  ├─ Level 3: Unfreeze MORE layers
│  ├─ Level 3: Use larger model (EfficientNet-B2 or ResNet101)
│  └─ Level 3: Check data pipeline — wrong normalization? Corrupt images?
│
├─ Overall acc HIGH, specific class F1 < 0.50
│  → CLASS-SPECIFIC FAILURE
│  ├─ Check confusion matrix: which class is it confused with?
│  ├─ Level 2: ↑ Class weight for failing class
│  ├─ Level 4: Add more training images for that class
│  └─ Level 4: Targeted augmentation for that specific class
│
└─ Metrics plateau, no improvement
   → SATURATION
   ├─ Level 2: Add Mixup/CutMix (smooths decision boundaries)
   ├─ Level 3: CosineAnnealingWarmRestarts (escape local minima)
   ├─ Level 3: ↑ Input resolution (224→300 or 380)
   ├─ Level 3: Label smoothing (ε=0.1)
   └─ Level 5: Ensemble two different architectures
```

### Improvement Levels Detailed

#### Level 1 — Hyperparameter Tweaks (Minutes)

| Fix | Change | When |
|---|---|---|
| ↑ Dropout | 0.4 → 0.5 or 0.6 | train_acc >> val_acc |
| ↑ Weight decay | 1e-2 → 5e-2 | Same overfitting signal |
| ↓ Batch size | 32 → 16 | Smaller batches = more gradient noise = implicit regularization |
| ↓ Learning rate | 1e-4 → 5e-5 | Val loss oscillating/unstable |
| ↑ Epochs | 20 → 30 | Val acc still improving at epoch 20 |

#### Level 2 — Augmentation (30 min each)

| Technique | Implementation | Effect |
|---|---|---|
| **Random Erasing** | `transforms.RandomErasing(p=0.3, scale=(0.02,0.2))` | Forces model to not fixate on single region |
| **Mixup** | In training loop: `x = λ·x_i + (1-λ)·x_j; loss = λ·L(x,y_i) + (1-λ)·L(x,y_j)` | Smoother decision boundaries; better generalization |
| **CutMix** | Cut patch from image B, paste onto A; mix labels by area | Like Mixup but preserves local spatial structure |
| **TTA** | At test time: predict on original + flipped + cropped, average results | Free accuracy boost (~1-2%) without retraining |

#### Level 3 — Architecture & Strategy (Hours)

| Fix | Details | When |
|---|---|---|
| Different backbone | Try EfficientNet-B2, DenseNet121, or ConvNeXt-Tiny | Current model plateaus |
| ↑ Input resolution | 224→300 or 380; `transforms.Resize(380), CenterCrop(380)` | Model misses fine texture differences |
| Label smoothing | `nn.CrossEntropyLoss(label_smoothing=0.1)` — soft targets instead of hard 0/1 | Model overconfident on wrong predictions |
| Unfreeze more | Unfreeze all but first 2 blocks | Features too generic for plastic domain |

#### Level 4 — Data Interventions (Days)

| Fix | Details | When |
|---|---|---|
| Collect more images | Web scrape or photograph more PP/PVC specifically | Class F1 consistently < 0.60 |
| Clean dataset | Manual review for mislabeled images | Confusion matrix shows unexpected patterns |
| Merge external dataset | Combine with Kaggle waste datasets | Overall dataset too small |

#### Level 5 — Advanced (Last resort)

| Fix | Details | When |
|---|---|---|
| **Ensemble** | Average predictions from EfficientNet + ResNet50 | Each model has different strengths |
| **K-Fold CV** | Split data into 5 folds, train 5 models, average metrics | Need robust metric estimates |
| **Progressive resizing** | Train on 128px → 224px → 300px gradually | Helps model learn coarse→fine features |

---

## 6. Experiment Tracking Framework

### Per-Experiment Log Template

```
═══════════════════════════════════════════
 Experiment: exp_XXX
 Date: YYYY-MM-DD
═══════════════════════════════════════════
 Model:           EfficientNet-B0
 Stage:           Fine-tuning
 Frozen layers:   features[:6]
 ─────────────────────────────────────────
 Hyperparameters:
   LR (backbone):  1e-4
   LR (head):      1e-3
   Weight decay:   1e-2
   Batch size:     32
   Epochs:         20
   Scheduler:      CosineAnnealingWarmRestarts(T_0=5)
   Augmentation:   Standard + RandomErasing
   Sampler:        WeightedRandomSampler
   Loss:           CrossEntropyLoss (unweighted)
 ─────────────────────────────────────────
 Results:
   Best epoch:     14
   Train acc:      XX.X%
   Val acc:        XX.X%
   Val F1 (macro): X.XXX
   Val Kappa:      X.XXX
   Per-class F1:   PET=X.XX  HDPE=X.XX  PVC=X.XX
                   LDPE=X.XX  PP=X.XX  PS=X.XX
   Worst class:    XX (F1=X.XX)
 ─────────────────────────────────────────
 Observations:
   - [what worked / didn't work]
   - [overfitting noticed at epoch X]
   - [specific class issues]
 ─────────────────────────────────────────
 Next steps:
   - [what to try in next experiment]
═══════════════════════════════════════════
```

### Comparison Table (Filled After All Experiments)

| # | Model | Stage | LR | Aug | Val Acc | Macro F1 | Kappa | Worst Class | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 001 | EffNet-B0 | Feat. Ext. | 1e-3 | Std | — | — | — | — | Baseline |
| 002 | EffNet-B0 | Fine-tune | 1e-4 | Std | — | — | — | — | — |
| 003 | ResNet50 | Feat. Ext. | 1e-3 | Std | — | — | — | — | Baseline |
| 004 | ResNet50 | Fine-tune | 1e-4 | Std | — | — | — | — | — |
| 005 | Best model | Fine-tune | — | +Mixup | — | — | — | — | If needed |

---

## 7. Final Deliverables

| Deliverable | Format | Description |
|---|---|---|
| `03_train_model.py` | `.py` | Full training pipeline (both models, both stages) |
| `04_evaluate_model.py` | `.py` | Generates all metrics, plots, and Grad-CAM |
| `model_best.pth` | `.pth` | Best model state_dict |
| `training_curves.png` | `.png` | Loss/accuracy over epochs |
| `confusion_matrix.png` | `.png` | 6×6 heatmap for both models |
| `classification_report.txt` | `.txt` | Per-class P/R/F1 |
| `gradcam_samples.png` | `.png` | Grad-CAM overlays for each class |
| `misclassified.png` | `.png` | Grid of worst errors |
| `model_comparison.md` | `.md` | EfficientNet vs ResNet50 comparison |
| `experiment_log.md` | `.md` | All experiments with results |

---

## 8. Timeline

| Task | Estimated Time |
|---|---|
| Data loaders + augmentation pipeline | 1–2 hours |
| Training script (Stage 1 + 2, both models) | 2–3 hours |
| Model training (GPU dependent) | 2–4 hours |
| Evaluation script + all visualizations | 2–3 hours |
| Grad-CAM interpretability | 1 hour |
| Iterate/improve if needed | 2–4 hours |
| Documentation + experiment log | 1 hour |
| **Total** | **~1.5–2.5 days** |

> Training times assume a consumer GPU (GTX 1650+) or Google Colab free tier. CPU-only training: multiply by 3-5×.
