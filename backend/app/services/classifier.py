# backend/app/services/classifier.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dataclasses import dataclass, field

from app.config import settings

# ── Class order MUST match training output ──────────────────────────────────
# From 03_train_model.py: {'HDPE': 0, 'LDPE': 1, 'PET': 2, 'PP': 3, 'PS': 4, 'PVC': 5}
CLASS_NAMES: list[str] = ["HDPE", "LDPE", "PET", "PP", "PS", "PVC"]

# ── ImageNet normalisation — MUST match val_transform in 03_train_model.py ──
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Inference transform (deterministic — NOT the augmented train transform) ──
# Sizes driven by config to stay in sync if model is retrained at different input size
# Matches val_tfm: Resize(resize_size) → CenterCrop(input_size)
_inference_transform = transforms.Compose([
    transforms.Resize(settings.resize_size),
    transforms.CenterCrop(settings.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# ── Device selection ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Singleton model reference ────────────────────────────────────────────────
_model: nn.Module | None = None



# ── Return types ─────────────────────────────────────────────────────────────

@dataclass
class Top3Prediction:
    plastic_type: str
    confidence: float


@dataclass
class PredictionResult:
    plastic_type: str
    confidence: float
    is_uncertain: bool
    uncertainty_message: str | None
    top3: list[Top3Prediction] = field(default_factory=list)
    all_probabilities: dict[str, float] = field(default_factory=dict)


# ── Model construction ────────────────────────────────────────────────────────

def _build_model() -> nn.Module:
    """
    Reconstruct EfficientNet-B0 with the custom head used in training.
    Architecture MUST be identical to build_model() in 03_train_model.py.
    """
    model = models.efficientnet_b0(weights=None)  # weights loaded separately

    # Read feature dim from the original head before replacing it
    feature_dim = model.classifier[1].in_features  # 1280 for EfficientNet-B0

    # Custom classification head — identical to training script
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, settings.num_classes),
    )
    return model


# ── Singleton loader ─────────────────────────────────────────────────────────

def load_model() -> nn.Module:
    """
    Load model weights from disk. Called once at application startup via lifespan.

    Design notes:
    - map_location=device: loads to CUDA/CPU dynamically
    - weights_only=True: security hardening — prevents pickle-based code execution
    - model.to(device): transfers model parameters to the selected device
    - model.eval(): disables Dropout randomness and BatchNorm batch-statistics
    """
    global _model
    model = _build_model()

    state_dict = torch.load(
        settings.model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    _model = model
    return _model


def get_model() -> nn.Module:
    """Return the loaded model singleton. Raises if load_model() was not called."""
    if _model is None:
        raise RuntimeError(
            "Model not loaded. Ensure load_model() is called during app startup."
        )
    return _model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(image: Image.Image) -> PredictionResult:
    """
    Run inference on a PIL Image.

    Steps:
      1. Apply val transform (Resize → CenterCrop → ToTensor → Normalize)
      2. Move input tensor to active device (CPU or GPU)
      3. Forward pass under no_grad (no computation graph — prevents memory leak)
      4. Softmax over logits → probabilities
      5. Build top-1, top-3, uncertainty flags
    """
    model = get_model()

    # Preprocess: [H, W, C] PIL → [1, 3, 224, 224] tensor, moved to model's device
    tensor = _inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)                         # [1, 6]
        probs_tensor = torch.softmax(logits, dim=1)    # [1, 6], sums to 1.0
        probs_tensor = probs_tensor.squeeze(0)         # [6]

    probs_list: list[float] = probs_tensor.tolist()

    # Full probability dict
    all_probs = {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, probs_list)}

    # Top-1
    top1_idx = int(probs_tensor.argmax())
    top1_class = CLASS_NAMES[top1_idx]
    top1_conf = round(probs_list[top1_idx], 4)

    # Top-3 (sorted by confidence, highest first)
    top3_indices = probs_tensor.topk(3).indices.tolist()
    top3 = [
        Top3Prediction(CLASS_NAMES[i], round(probs_list[i], 4))
        for i in top3_indices
    ]

    # Uncertainty flags (from Phase 2 analysis — PP F1=0.606)
    is_uncertain = top1_conf < settings.confidence_threshold
    uncertainty_msg: str | None = None

    if is_uncertain:
        uncertainty_msg = (
            f"Model confidence is {top1_conf:.0%}, below the reliable threshold of "
            f"{settings.confidence_threshold:.0%}. "
            "Consider checking the resin code printed on the item."
        )
    elif top1_class == "PP" and settings.pp_extra_warning:
        uncertainty_msg = (
            "PP detection has reduced accuracy in this model. "
            "Verify by checking for resin code #5 on the item."
        )

    return PredictionResult(
        plastic_type=top1_class,
        confidence=top1_conf,
        is_uncertain=is_uncertain,
        uncertainty_message=uncertainty_msg,
        top3=top3,
        all_probabilities=all_probs,
    )
