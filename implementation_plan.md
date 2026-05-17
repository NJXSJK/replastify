# Replastify — 4-Phase Implementation Plan

> A comprehensive, step-by-step plan to build Replastify from data preparation through deployment, organized into 4 clear phases.

---

## Available Asset

| Asset | Location | Notes |
|---|---|---|
| Dataset (7 plastic classes) | `MINOR/Plastic/plastic/` | Genuine labeled images — PET, HDPE, PVC, LDPE, PP, PS, Other |

Everything else (backend, frontend, model) will be **built from scratch**.

---

## Phase 1 — Project Setup & Data Pipeline

**Goal**: Establish a clean project structure, prepare the dataset, and build a robust data pipeline.

### 1.1 Project Structure Setup

Create the following directory structure inside `replastify/`:

```
replastify/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── predict.py       # /predict endpoint
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── classifier.py    # Model loading & inference
│   │   │   ├── gemini.py        # Gemini API integration
│   │   │   └── plastic_info.py  # Static plastic knowledge base
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── image_utils.py   # Image preprocessing helpers
│   │   └── config.py            # Settings / env variables
│   ├── models/                  # Trained model files (.pth)
│   ├── .env                     # API keys (gitignored)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── app.js
│   └── assets/                  # Images, icons
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── data/
│   ├── raw/                     # Original dataset (from MINOR/Plastic/plastic/)
│   │   ├── PET/
│   │   ├── HDPE/
│   │   ├── PVC/
│   │   ├── LDPE/
│   │   ├── PP/
│   │   ├── PS/
│   │   └── Other/
│   ├── processed/               # Augmented & split data
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── README.md                # Dataset source & description
├── docs/
│   ├── project_details.md       # (already exists)
│   └── implementation_plan.md   # (this file)
├── .gitignore
├── README.md
└── info.md                      # (already exists)
```

### 1.2 Environment Setup

- [ ] Create Python virtual environment (`python -m venv .venv`)
- [ ] Create `requirements.txt`:
  ```
  # Backend
  fastapi>=0.115.0
  uvicorn>=0.30.0
  python-multipart>=0.0.12
  Pillow>=10.0.0
  python-dotenv>=1.0.0
  google-genai>=1.0.0

  # ML
  torch>=2.2.0
  torchvision>=0.17.0
  scikit-learn>=1.4.0
  matplotlib>=3.8.0
  seaborn>=0.13.0

  # Notebooks
  jupyter>=1.0.0
  ```
- [ ] Create `.env` file for API keys:
  ```
  GEMINI_API_KEY=your_key_here
  ```
- [ ] Create `.gitignore` (exclude `.env`, `.venv/`, `__pycache__/`, `models/*.pth`, `data/raw/`, temp files)

### 1.3 Data Preparation

- [ ] Copy dataset from `MINOR/Plastic/plastic/` → `replastify/data/raw/` with cleaned folder names (rename to `PET/`, `HDPE/`, etc.)
- [ ] Create `01_data_exploration.ipynb`:
  - Count images per class
  - Visualize sample images from each class
  - Identify class imbalance
  - Check image dimensions and quality
  - Plot class distribution bar chart
- [ ] Split data into train/val/test (70/15/15 or 80/10/10)
- [ ] Implement data augmentation pipeline:
  - Random horizontal flip
  - Random rotation (±15°)
  - Color jitter (brightness, contrast, saturation)
  - Random resized crop
  - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- [ ] Handle class imbalance:
  - Weighted random sampler OR
  - Class-weighted loss function
  - Oversample minority classes through augmentation

### 1.4 Deliverables for Phase 1

| Deliverable | Description |
|---|---|
| Clean project structure | All directories and config files in place |
| Virtual environment | Working Python env with all dependencies |
| Data exploration notebook | Visualizations of dataset, class distributions |
| Processed dataset | train/val/test splits ready for training |
| `.gitignore` and `.env` | Proper secret management |

---

## Phase 2 — Model Training & Evaluation

**Goal**: Train an improved classification model, evaluate thoroughly, and export for deployment.

### 2.1 Model Architecture

- [ ] Use **transfer learning** with a pre-trained backbone (choose one):
  - **ResNet50** — strong baseline, reliable for academic projects
  - **EfficientNet-B0** — best accuracy-to-size ratio
  - **MobileNetV2** — if prioritizing inference speed

- [ ] Custom classification head:
  ```
  Pre-trained backbone (frozen initially)
       ↓
  AdaptiveAvgPool2d(1)
       ↓
  Dropout(0.5)
       ↓
  Linear(backbone_features → 256)
       ↓
  ReLU + Dropout(0.3)
       ↓
  Linear(256 → 7)    # 7 plastic classes
  ```

### 2.2 Training Strategy

- [ ] Create `02_model_training.ipynb`:
  - **Stage 1 — Feature extraction** (5–10 epochs):
    - Freeze all backbone layers
    - Train only the classification head
    - Learning rate: `1e-3` with Adam optimizer
    - Loss: CrossEntropyLoss (with class weights if imbalanced)
  - **Stage 2 — Fine-tuning** (10–20 epochs):
    - Unfreeze last 2–3 blocks of backbone
    - Lower learning rate: `1e-4` or `1e-5`
    - Use ReduceLROnPlateau or CosineAnnealingLR scheduler
    - Early stopping (patience=5) based on val_loss
  - Track metrics: train_loss, val_loss, train_acc, val_acc per epoch
  - Save best model checkpoint based on val_acc

### 2.3 Evaluation

- [ ] Create `03_model_evaluation.ipynb`:
  - Load best model checkpoint
  - Evaluate on test set
  - Generate:
    - **Confusion matrix** (heatmap)
    - **Classification report** (precision, recall, F1 per class)
    - **Per-class accuracy** bar chart
    - **Misclassification examples** — show images the model got wrong
    - **Confidence distribution** — histogram of prediction probabilities
  - Compare against a simple baseline (e.g., untrained ResNet18 or random guessing)
  - Document final accuracy numbers
  - Summarize findings for assignment report

### 2.4 Model Export

- [ ] Save best model as `model_best.pth` (state_dict only, for smaller file)
- [ ] Also save as full model for easier loading: `model_full.pth`
- [ ] Copy to `backend/models/` for deployment
- [ ] Document model metadata:
  ```
  Architecture: ResNet50 (or chosen model)
  Input size: 224×224×3
  Output: 7 classes [PET, HDPE, PVC, LDPE, PP, PS, Other]
  Accuracy: XX.X%
  File size: XX MB
  ```

### 2.5 Deliverables for Phase 2

| Deliverable | Description |
|---|---|
| Training notebook | Full pipeline with training curves, hyperparameters |
| Evaluation notebook | Confusion matrix, classification report, misclassifications |
| Trained model file | `.pth` file in `backend/models/` |
| Model comparison | Baseline ResNet18 vs new model accuracy table |
| Training logs | Loss/accuracy plots over epochs |

---

## Phase 3 — Backend Development

**Goal**: Build a clean, modular FastAPI backend with proper model inference, Gemini integration, and a static plastic knowledge base.

### 3.1 Configuration & Settings

- [ ] `backend/app/config.py`:
  - Load env variables using `python-dotenv`
  - Define settings: `GEMINI_API_KEY`, `MODEL_PATH`, `ALLOWED_EXTENSIONS`, `MAX_FILE_SIZE`
  - No hardcoded secrets

### 3.2 Model Inference Service

- [ ] `backend/app/services/classifier.py`:
  - Load model once at startup (singleton pattern)
  - Preprocessing pipeline (resize, normalize with ImageNet stats)
  - `predict(image_path)` → returns `{ "class": "PET", "confidence": 0.94, "all_probabilities": {...} }`
  - Use `torch.nn.functional.softmax` for confidence scores
  - Add top-3 predictions with probabilities
  - Handle edge cases: non-image files, corrupted images, very small images

### 3.3 Plastic Knowledge Base

- [ ] `backend/app/services/plastic_info.py`:
  - Static dictionary/JSON with info for all 7 types:
    ```python
    PLASTIC_DATABASE = {
        "PET": {
            "full_name": "Polyethylene Terephthalate",
            "resin_code": 1,
            "common_uses": ["Water bottles", "Soda bottles", "Food jars"],
            "recyclability": "Widely recycled",
            "recyclability_score": 5,  # out of 5
            "health_concerns": "Safe for single use; may leach antimony if reused",
            "decomposition_time": "~450 years",
            "recycling_tips": ["Rinse before recycling", "Remove cap", "Flatten to save space"],
            "reuse_ideas": ["Planters", "Bird feeders", "Storage containers"],
            "eco_alternatives": ["Glass bottles", "Stainless steel", "Aluminum cans"],
            "fun_fact": "Recycling one PET bottle saves enough energy to power a laptop for 25 minutes"
        },
        # ... same for HDPE, PVC, LDPE, PP, PS, Other
    }
    ```
  - Function: `get_plastic_info(class_name)` → returns full info dict

### 3.4 Gemini API Service

- [ ] `backend/app/services/gemini.py`:
  - Initialize Gemini client using env variable (NOT hardcoded key)
  - Improved prompt:
    - Request structured JSON-like response
    - Ask for: recycle methods, reuse ideas, environmental impact, condition assessment
    - Include the detected plastic type in the prompt for better context
  - Error handling: timeout, API rate limits, fallback to static knowledge base if API fails
  - Response parsing: clean up Gemini output for frontend consumption

### 3.5 API Routes

- [ ] `backend/app/routes/predict.py`:
  ```
  POST /predict
  - Accepts: multipart/form-data (image file)
  - Validates: file type (jpg/png/webp), file size (<10 MB)
  - Process:
    1. Save temp file
    2. Run model inference → class + confidence
    3. Fetch static plastic info from knowledge base
    4. Call Gemini API for context-specific suggestions
    5. Clean up temp file
  - Returns JSON:
    {
      "prediction": {
        "plastic_type": "PET",
        "full_name": "Polyethylene Terephthalate",
        "resin_code": 1,
        "confidence": 0.94,
        "top_3": [
          {"type": "PET", "confidence": 0.94},
          {"type": "HDPE", "confidence": 0.04},
          {"type": "PVC", "confidence": 0.01}
        ]
      },
      "info": {
        "common_uses": [...],
        "recyclability": "Widely recycled",
        "recyclability_score": 5,
        "health_concerns": "...",
        "decomposition_time": "~450 years",
        "fun_fact": "..."
      },
      "suggestions": {
        "recycle": ["...", "...", "..."],
        "reuse": ["...", "...", "..."],
        "alternatives": ["...", "...", "..."]
      },
      "ai_analysis": "..."  // Gemini's response
    }
  ```

- [ ] `GET /` → Serve frontend (index.html)
- [ ] `GET /health` → Health check endpoint
- [ ] `GET /plastic-types` → Return all 7 plastic types info (for frontend reference pages)

### 3.6 Temp File Cleanup

- [ ] Auto-delete uploaded temp files after processing
- [ ] Use `try/finally` or context manager for cleanup
- [ ] Use a dedicated `temp/` directory (not the root Backend folder)

### 3.7 Deliverables for Phase 3

| Deliverable | Description |
|---|---|
| FastAPI application | Modular, clean code with proper separation |
| Model inference service | Returns class + confidence + top-3 |
| Plastic knowledge base | Complete info for all 7 types |
| Gemini integration | Secure API key, structured prompts, error handling |
| API documentation | Auto-generated via FastAPI `/docs` (Swagger UI) |

---

## Phase 4 — Frontend & Integration

**Goal**: Build a polished, responsive frontend from scratch that beautifully presents all prediction results, and integrate everything end-to-end.

### 4.1 Frontend Design Improvements

Build a clean, modern frontend with:

- [ ] **Structured result display** — Present results in organized cards:
  - Prediction card with type name, resin code badge, confidence meter
  - Info panel with tabbed sections (About / Recycle / Reuse / Alternatives)
  - Environmental impact visual (decomposition timeline bar)
  - Confidence bar (animated, color-coded: green >80%, yellow 50-80%, red <50%)
- [ ] **Loading state** — Skeleton loader while API processes
- [ ] **Error states** — Friendly error messages for failed uploads, API errors, low-confidence predictions
- [ ] **Image preview improvements** — Show uploaded image alongside results
- [ ] **Responsive design** — Ensure mobile layout works well
- [ ] **Accessibility** — Proper alt text, keyboard navigation, screen reader support

### 4.2 Frontend JavaScript Updates

- [ ] `frontend/js/app.js`:
  - Parse the new structured JSON response
  - Render result cards dynamically:
    - Plastic type header with confidence badge
    - Tabbed interface for Recycle / Reuse / Alternatives
    - Animated confidence bar
    - Gemini AI analysis section (formatted, not raw text)
  - Handle edge cases:
    - Low confidence warning (<60%): "The model is not very sure about this result"
    - API error fallback: show static info from plastic knowledge base
  - Add image drag-and-drop (already exists, verify it works)
  - Add copy/share results button

### 4.3 CSS Enhancements

- [ ] `frontend/css/styles.css`:
  - Animated confidence meter (gradient bar)
  - Tab component styles for result sections
  - Skeleton loading animation
  - Toast notification improvements
  - Dark mode toggle (optional stretch)
  - Print-friendly styles for result sharing

### 4.4 End-to-End Integration

- [ ] Connect frontend to backend:
  - Update API URL (use relative paths or configurable base URL)
  - Test full flow: upload → predict → display results
  - Handle CORS properly
- [ ] Serve frontend from FastAPI static files
- [ ] Test with all 7 plastic types (use test images)
- [ ] Test edge cases:
  - Non-plastic images
  - Very large files
  - Blurry/dark images
  - Multiple rapid uploads

### 4.5 Documentation & Finishing

- [ ] `README.md`:
  - Project description and motivation
  - Screenshots / demo GIF
  - Tech stack overview
  - Setup instructions (step-by-step)
  - API documentation summary
  - Team members / credits
  - Future scope
- [ ] Code comments and docstrings
- [ ] Clean up unused files and test artifacts
- [ ] Final testing checklist

### 4.6 Optional Enhancements (if time permits)

- [ ] Docker Compose setup (backend + frontend)
- [ ] Prediction history (localStorage on frontend)
- [ ] Camera capture button (using `getUserMedia` API)
- [ ] PDF export of results
- [ ] Statistics page with environmental impact facts

### 4.7 Deliverables for Phase 4

| Deliverable | Description |
|---|---|
| Polished frontend | Structured result cards, animations, responsive |
| Full integration | Frontend ↔ Backend working end-to-end |
| README.md | Complete project documentation with screenshots |
| Demo-ready app | Can be presented for assignment |

---

## Phase Summary

| Phase | Focus | Key Output | Estimated Time |
|---|---|---|---|
| **Phase 1** | Setup & Data | Project structure, cleaned dataset, exploration notebook | 2–3 days |
| **Phase 2** | Model Training | Trained model, evaluation metrics, comparison | 3–4 days |
| **Phase 3** | Backend | FastAPI app, inference API, Gemini integration, knowledge base | 3–4 days |
| **Phase 4** | Frontend & Polish | UI overhaul, integration, documentation | 3–4 days |

**Total estimated time: ~2 weeks**

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Low model accuracy | Results aren't trustworthy | Try different architectures; augment data more; add more training images |
| Gemini API quota/cost | Suggestions stop working | Static knowledge base as fallback; cache common responses |
| Class imbalance | Model biased toward PET/HDPE | Weighted sampling, class-weighted loss, augment minority classes |
| Low confidence predictions | User gets wrong info | Show confidence score prominently; add "uncertain" warning below 60% |
| API key exposure | Security risk | `.env` file + `.gitignore`; never commit keys |

---

## Quick Start Commands

```bash
# Clone and setup
cd replastify
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# Run backend
cd backend
uvicorn app.main:app --reload --port 8000

# Open in browser
# http://localhost:8000
```
