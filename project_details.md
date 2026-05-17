# Replastify — Project Details

## 1. Project Overview

**Replastify** is an image-based plastic identification system powered by Machine Learning. Users upload or capture an image of a plastic item, and the system:

1. **Identifies the plastic type** (Resin Identification Code 1–7)
2. **Displays detailed information** about that plastic — properties, common uses, and health/safety concerns
3. **Highlights environmental impact** — recyclability, pollution contribution, and decomposition timeline
4. **Suggests recycling/reuse methods** specific to that plastic type
5. **Recommends eco-friendly alternatives** to replace that plastic

---

## 2. Problem Statement & Motivation

### Why This Matters

- **450 million tonnes** of plastic are produced globally each year (2025 estimate)
- **Less than 10%** of all plastic ever produced has been recycled
- **130 million tonnes** of plastic pollute the environment (land, air, water) annually
- **8–23 million tonnes** of plastic leak into oceans every year — equivalent to a garbage truck every minute
- **75–199 million tonnes** of plastic currently circulate in the ocean; 92% of it is microplastics
- Over **1 million seabirds** and **100,000 marine mammals** die annually from plastic pollution
- By **2050**, plastic could outweigh all fish in the ocean if trends continue

### The Core Problem

Most people **cannot identify the type of plastic** they're holding. Without knowing the resin code, they don't know:
- Whether it's actually recyclable in their area
- If it's safe for food contact or reuse
- What the environmental consequences are
- What sustainable alternatives exist

**Replastify bridges this knowledge gap** by making plastic identification instant and actionable.

---

## 3. Plastic Types — Detailed Reference

### 3.1 — PET / PETE (Polyethylene Terephthalate)

| Property | Details |
|---|---|
| **Resin Code** | #1 |
| **Common Uses** | Water/soda bottles, food jars, polyester clothing fibers |
| **Recyclability** | ✅ Widely recycled (curbside accepted almost everywhere) |
| **Health Concerns** | Best for single use only; can harbor bacteria and leach antimony if reused or heated |
| **Decomposition** | ~450 years |
| **Recycling Method** | Rinse, remove cap, flatten, place in recycling bin. Recycled into fiber, carpet, new bottles |
| **Eco Alternatives** | Glass bottles, stainless steel bottles, aluminum cans |

### 3.2 — HDPE (High-Density Polyethylene)

| Property | Details |
|---|---|
| **Resin Code** | #2 |
| **Common Uses** | Milk jugs, detergent bottles, shampoo bottles, plastic lumber |
| **Recyclability** | ✅ Widely recycled |
| **Health Concerns** | Considered one of the safest plastics; non-toxic, durable |
| **Decomposition** | ~500 years |
| **Recycling Method** | Rinse container, recycle curbside. Recycled into pens, plastic lumber, fencing |
| **Eco Alternatives** | Glass containers, stainless steel, refillable systems |

### 3.3 — PVC (Polyvinyl Chloride)

| Property | Details |
|---|---|
| **Resin Code** | #3 |
| **Common Uses** | Pipes, shower curtains, cling wrap, some food packaging, vinyl flooring |
| **Recyclability** | ❌ Difficult to recycle (rarely accepted curbside) |
| **Health Concerns** | ⚠️ Contains harmful phthalates and can release dioxins when burned. Avoid for food/drink contact |
| **Decomposition** | ~1,000 years |
| **Recycling Method** | Specialized facilities only. Often downcycled into decking, flooring, or speed bumps |
| **Eco Alternatives** | Beeswax wraps (for cling wrap), silicone, glass, metal pipes |

### 3.4 — LDPE (Low-Density Polyethylene)

| Property | Details |
|---|---|
| **Resin Code** | #4 |
| **Common Uses** | Grocery bags, bread bags, food wraps, squeezable bottles |
| **Recyclability** | ⚠️ Limited (not typically curbside; drop-off at grocery stores) |
| **Health Concerns** | Generally considered safe; does not leach harmful chemicals under normal use |
| **Decomposition** | ~500 years |
| **Recycling Method** | Return to store drop-off bins. Recycled into trash can liners, shipping envelopes, furniture |
| **Eco Alternatives** | Cloth/canvas tote bags, silicone food bags, beeswax wraps |

### 3.5 — PP (Polypropylene)

| Property | Details |
|---|---|
| **Resin Code** | #5 |
| **Common Uses** | Yogurt tubs, bottle caps, straws, microwave-safe containers, medicine bottles |
| **Recyclability** | ✅ Increasingly accepted in curbside programs |
| **Health Concerns** | Considered safe; heat-resistant, does not leach chemicals at normal temperatures |
| **Decomposition** | ~20–30 years |
| **Recycling Method** | Check local acceptance. Recycled into brooms, brushes, auto parts, trays |
| **Eco Alternatives** | Glass jars, stainless steel containers, ceramic, bamboo straws |

### 3.6 — PS (Polystyrene / Styrofoam)

| Property | Details |
|---|---|
| **Resin Code** | #6 |
| **Common Uses** | Disposable cups, plates, takeout containers, packing peanuts, CD cases |
| **Recyclability** | ❌ Difficult to recycle (rarely accepted; breaks into microplastics easily) |
| **Health Concerns** | ⚠️ Can leach styrene (a possible carcinogen), especially when heated |
| **Decomposition** | ~500+ years |
| **Recycling Method** | Very limited specialized facilities. Most ends up in landfill |
| **Eco Alternatives** | Paper/fiber-based containers, compostable plates, stainless steel, reusable cups |

### 3.7 — Other (Mixed / Specialty Plastics)

| Property | Details |
|---|---|
| **Resin Code** | #7 |
| **Common Uses** | Polycarbonate (water cooler jugs, baby bottles), bioplastics (PLA), multi-layer packaging |
| **Recyclability** | ❌ Rarely recyclable (catch-all category; too varied for standard processing) |
| **Health Concerns** | ⚠️ Some contain BPA (bisphenol-A), an endocrine disruptor. Bio-plastics may be safer but require industrial composting |
| **Decomposition** | Varies widely (100–1,000+ years) |
| **Recycling Method** | Typically not recyclable through conventional means |
| **Eco Alternatives** | Stainless steel, glass, wood, natural fibers (cotton, hemp, jute) |

---

## 4. ML / Deep Learning Approach

### 4.1 Problem Type

**Multi-class image classification** — Given an input image of a plastic item, predict which of the 7 resin codes (or broader waste categories) it belongs to.

### 4.2 Recommended Model Architectures

| Model | Why Consider It | Trade-off |
|---|---|---|
| **MobileNetV2** | Lightweight, fast inference, ideal for web/mobile deployment | Slightly lower accuracy than heavier models |
| **ResNet50** | Strong feature extraction, well-proven for image classification | Larger model size, slower inference |
| **EfficientNet** | Best accuracy-to-compute ratio, state-of-the-art performance | Can be complex to fine-tune |
| **YOLOv8/v9** | Real-time object detection + classification in one pass | More complex setup; overkill if only classifying single images |
| **VGG19** | Simple architecture, easy to implement for beginners | Very large model, slow training |

> [!TIP]
> **Recommended approach**: Use **Transfer Learning with MobileNetV2** or **EfficientNet**. Pre-train on ImageNet, freeze base layers, and fine-tune the top layers on your plastic dataset. This gives strong accuracy with limited training data and fast inference for a web app.

### 4.3 Datasets

| Dataset | Description | Size |
|---|---|---|
| **WaDaBa Dataset** | Benchmark with RGB images of PET, HDPE, PP, PS | Thousands of images |
| **Recyclable & Household Waste (Kaggle)** | 15,000 images across 30 categories including plastic sub-types | 15K images |
| **Plastic Recycling Codes (Kaggle)** | Focused on 7 RIC codes specifically | Varies |
| **Garbage Classification (Kaggle)** | Plastic, glass (green/brown/white) separation | ~2.5K images |
| **Plastic Waste Images (Kaggle)** | 8,000 annotated images (bottles, bags, cups) | 8K images |
| **Roboflow Universe** | Pre-labeled plastic detection datasets, exportable for YOLO | Various |

> [!IMPORTANT]
> You may need to **combine multiple datasets** or **augment existing ones** to get sufficient coverage across all 7 resin codes. Consider also collecting a small custom dataset for underrepresented categories.

### 4.4 Training Pipeline

```
1. Data Collection & Preparation
   ├── Gather images from Kaggle datasets + custom collection
   ├── Organize into folders: /train, /val, /test per class
   ├── Resize to uniform dimensions (e.g., 224×224)
   └── Normalize pixel values (0–1 range)

2. Data Augmentation
   ├── Random rotation (±30°)
   ├── Horizontal/vertical flip
   ├── Zoom, brightness, contrast shifts
   └── Handle class imbalance (oversample minority classes or use weighted loss)

3. Model Setup (Transfer Learning)
   ├── Load pre-trained MobileNetV2 (ImageNet weights)
   ├── Freeze base layers
   ├── Add custom classification head:
   │     GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.5) → Dense(7, Softmax)
   └── Compile with Adam optimizer, categorical cross-entropy loss

4. Training
   ├── Train top layers first (5–10 epochs)
   ├── Unfreeze some base layers, fine-tune (10–20 more epochs)
   ├── Use learning rate scheduler + early stopping
   └── Monitor val_accuracy and val_loss

5. Evaluation
   ├── Test on held-out test set
   ├── Generate confusion matrix
   ├── Report accuracy, precision, recall, F1-score per class
   └── Identify misclassification patterns (e.g., PET vs PVC)

6. Export
   └── Save as .h5 or SavedModel format for Flask integration
```

### 4.5 Key Challenges

- **Class imbalance**: PET/HDPE images are abundant; PVC/PS/Other are scarce
- **Visual similarity**: Some plastics (especially transparent ones) look nearly identical
- **Environmental variability**: Real-world images have varied lighting, backgrounds, angles, deformation
- **Dataset quality**: Public datasets may contain mislabeled or low-quality images

---

## 5. Tech Stack

### Backend
| Component | Technology | Purpose |
|---|---|---|
| Language | **Python 3.10+** | Core programming language |
| ML Framework | **TensorFlow / Keras** | Model training and inference |
| Web Framework | **Flask** | Lightweight API server |
| Image Processing | **OpenCV / Pillow (PIL)** | Resize, normalize, preprocess uploaded images |
| Model Format | **.h5 or SavedModel** | Serialized trained model |

### Frontend
| Component | Technology | Purpose |
|---|---|---|
| Structure | **HTML5** | Page structure |
| Styling | **CSS3** (or Bootstrap) | Responsive, modern UI |
| Interactivity | **JavaScript** | Image upload, API calls, result display |
| UX Enhancements | Drag-and-drop upload, camera capture, loading animations | Polished user experience |

### Optional / Advanced
| Component | Technology | Purpose |
|---|---|---|
| Containerization | **Docker** | Reproducible deployment |
| Production Server | **Gunicorn** | WSGI server for Flask |
| Model Optimization | **TensorFlow Lite** | Faster inference on limited hardware |
| Database | **SQLite / PostgreSQL** | Store prediction history (optional) |

---

## 6. Application Architecture

```
┌─────────────────────────────────────────────────┐
│                   FRONTEND                       │
│  ┌─────────────┐  ┌────────────┐  ┌───────────┐ │
│  │ Image Upload │  │  Camera    │  │  Result   │ │
│  │ (Drag/Drop) │  │  Capture   │  │  Display  │ │
│  └──────┬──────┘  └─────┬──────┘  └─────▲─────┘ │
│         │               │               │       │
│         └───────┬───────┘               │       │
│                 │ POST /predict          │       │
└─────────────────┼───────────────────────┼───────┘
                  │                       │
                  ▼                       │
┌─────────────────────────────────────────────────┐
│                FLASK BACKEND                     │
│                                                  │
│  1. Receive image                                │
│  2. Preprocess (resize, normalize)               │
│  3. Run model inference                          │
│  4. Look up plastic info from knowledge base     │
│  5. Return JSON response:                        │
│     {                                            │
│       "plastic_type": "PET (#1)",                │
│       "confidence": 0.94,                        │
│       "description": "...",                      │
│       "environmental_impact": "...",             │
│       "recycling_methods": ["..."],              │
│       "alternatives": ["..."]                    │
│     }                                            │
└─────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│              ML MODEL (TensorFlow)               │
│                                                  │
│  Pre-trained MobileNetV2 / EfficientNet          │
│  Fine-tuned on plastic classification dataset    │
│  Input: 224×224×3 image                          │
│  Output: 7-class probability vector              │
└─────────────────────────────────────────────────┘
```

---

## 7. Feature List

### Core Features (MVP)
- [ ] Image upload (file picker)
- [ ] Plastic type identification (7 classes)
- [ ] Confidence score display
- [ ] Detailed plastic information panel (properties, uses, health concerns)
- [ ] Recycling/reuse guidance for detected plastic
- [ ] Eco-friendly alternatives display

### Enhanced Features
- [ ] Camera capture (mobile/webcam)
- [ ] Drag-and-drop image upload
- [ ] Prediction history / recent scans
- [ ] Environmental impact statistics dashboard
- [ ] Share results (social media / PDF export)
- [ ] Multi-language support

### Stretch Goals
- [ ] Real-time video detection (YOLO-based)
- [ ] Gamification (track plastics identified, earn eco-badges)
- [ ] Location-based recycling center finder
- [ ] Community contributions (user-submitted images to improve dataset)

---

## 8. Project Structure (Proposed)

```
replastify/
├── app/
│   ├── __init__.py            # Flask app factory
│   ├── routes.py              # API endpoints (/predict, /info, etc.)
│   ├── model.py               # Model loading & inference logic
│   ├── preprocess.py          # Image preprocessing utilities
│   └── plastic_data.py        # Knowledge base (plastic info, recycling, alternatives)
├── models/
│   └── plastic_classifier.h5  # Trained model file
├── static/
│   ├── css/
│   │   └── style.css          # Application styles
│   ├── js/
│   │   └── app.js             # Frontend logic (upload, API calls, rendering)
│   └── images/                # UI assets
├── templates/
│   └── index.html             # Main page template
├── notebooks/
│   ├── data_exploration.ipynb # Dataset analysis
│   └── model_training.ipynb   # Training pipeline
├── data/
│   ├── train/                 # Training images (by class)
│   ├── val/                   # Validation images
│   └── test/                  # Test images
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container setup (optional)
├── README.md                  # Project documentation
└── info.md                    # Original project notes
```

---

## 9. Development Workflow

### Phase 1 — Research & Data (Week 1)
- Finalize dataset selection and download
- Explore and clean data
- Set up project structure and virtual environment

### Phase 2 — Model Development (Week 2)
- Implement data preprocessing and augmentation pipeline
- Train baseline model with transfer learning
- Evaluate and iterate on model performance
- Export final model

### Phase 3 — Backend Development (Week 3)
- Set up Flask application
- Integrate trained model for inference
- Build plastic knowledge base (JSON/dict with all 7 types' info)
- Create API endpoints

### Phase 4 — Frontend Development (Week 3–4)
- Design and build UI (upload, results, info panels)
- Connect frontend to Flask API
- Add responsive design and polish

### Phase 5 — Testing & Deployment (Week 4)
- End-to-end testing with real-world images
- Performance optimization
- Documentation and README
- (Optional) Docker containerization and deployment

---

## 10. Key Dependencies

```
tensorflow>=2.15.0
keras>=3.0.0
flask>=3.0.0
opencv-python>=4.9.0
Pillow>=10.0.0
numpy>=1.26.0
matplotlib>=3.8.0
scikit-learn>=1.4.0
gunicorn>=21.0.0          # Production server
```

---

## 11. References & Resources

- **Datasets**: Kaggle (WaDaBa, Recyclable Waste, Garbage Classification, Plastic Recycling Codes)
- **Models**: TensorFlow Hub (MobileNetV2, EfficientNet, ResNet50)
- **Plastic Info**: ASTM Resin Identification Coding System, EPA Recycling Guidelines
- **Research Papers**: CNN-based plastic waste classification (arXiv), YOLO for real-time waste detection
- **Environmental Data**: UNEP Plastic Pollution Statistics, Breaking the Plastic Wave (2025)
