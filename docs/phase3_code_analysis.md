# Phase 3 — Backend Code Analysis
**Date:** 2026-05-25 | **Status:** Implementation Complete | **Files:** 7 Python modules + Dockerfile

---

## 1. Overview

Phase 3 produced a fully modular FastAPI backend that wires together the Phase 2 EfficientNet-B0 model, a static plastic knowledge base, and an optional Gemini AI enrichment layer. All 11 Python files pass syntax validation. The server is ready to run once `best_efficientnet_b0.pth` is placed in `backend/models/`.

### File Inventory

| File | Lines | Role |
|---|---|---|
| `app/config.py` | 45 | All settings from `.env` via pydantic-settings |
| `app/utils/image_utils.py` | 58 | Upload validation → clean RGB PIL Image |
| `app/services/plastic_info.py` | 110 | Static knowledge base (6 plastic types) |
| `app/services/classifier.py` | 169 | EfficientNet-B0 singleton + inference |
| `app/services/gemini.py` | 151 | Gemini AI with fallback |
| `app/routes/predict.py` | 157 | 3 API endpoints + Pydantic response models |
| `app/main.py` | 114 | App factory, lifespan, CORS, error handler |

---

## 2. Request Lifecycle — `POST /predict`

Every image upload travels through 5 clearly separated stages:

```
Client uploads image
        │
        ▼
[1] image_utils.validate_and_load_image()
    ├─ Extension check (.jpg/.png/.webp only)        → 400 if bad
    ├─ Size check (≤ 10 MB)                          → 413 if large
    ├─ PIL.verify() (corrupt/truncated detection)    → 422 if broken
    └─ .convert("RGB") (normalises RGBA/grayscale)
        │
        ▼
[2] classifier.predict(image)
    ├─ Val transform: Resize(256) → CenterCrop(224) → ToTensor → Normalize
    ├─ Forward pass under torch.no_grad()
    ├─ Softmax over 6 logits → probabilities
    ├─ Top-1 class + confidence
    ├─ Top-3 sorted predictions
    └─ Uncertainty flag (conf < 0.70) + PP disclaimer
        │
        ▼
[3] plastic_info.get_plastic_info(class_name)
    └─ Returns static dict: resin code, uses, recyclability, health, fun_fact
        │
        ▼
[4] gemini.get_ai_suggestions() — async, max 8 seconds
    ├─ If API key missing  → static fallback immediately
    ├─ If Gemini responds  → parse JSON → return suggestions + source="ai"
    ├─ If timeout          → static fallback + log warning
    └─ If any exception    → static fallback + log warning
        │
        ▼
[5] Assemble PredictResponse (Pydantic model)
    └─ Return JSON to client
```

---

## 3. Design Pattern Analysis

### 3.1 Singleton Pattern — `classifier.py`

The model is loaded **once at startup** and reused for all requests via a module-level `_model` variable:

```python
_model: nn.Module | None = None

def load_model() -> nn.Module:   # called once in lifespan
    global _model
    ...
    _model = model
    return _model

def get_model() -> nn.Module:    # called on every request
    if _model is None:
        raise RuntimeError(...)
    return _model
```

**Why correct:** Loading a 20 MB PyTorch model takes ~0.5–1s. Loading it per-request would make the API unusable. The singleton is thread-safe because Python's GIL prevents concurrent writes to the global during loading, and all reads happen after startup completes.

**Risk:** If the server ever spawns multiple worker processes (e.g. `--workers 4`), each process gets its own copy of `_model`. This multiplies memory usage. The current `--workers 1` Dockerfile setting prevents this.

---

### 3.2 Fail-Fast Startup — `main.py`

```python
if not settings.model_path.exists():
    raise FileNotFoundError(...)
```

The server **refuses to start** if the model file is missing, rather than starting and crashing on the first request with an opaque 500 error. This is a critical operational improvement — it gives the developer an immediate, actionable error message.

---

### 3.3 Lifespan Context Manager — `main.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()    # startup
    yield           # server runs
    # shutdown code here if needed
```

This replaces the deprecated `@app.on_event("startup")` pattern (deprecated since FastAPI 0.93). The `yield` is the key — code before `yield` runs on startup, code after runs on shutdown. Currently there is nothing to clean up on shutdown, but the pattern is ready for adding database connections or cache flushes in future.

---

### 3.4 Gemini as Decoupled Optional Layer — `gemini.py`

Three design decisions make Gemini production-safe:

1. **Lazy client initialisation:** `_get_client()` returns `None` if no API key is set, without importing `google.genai` at all. This means the server works even if the package isn't installed.

2. **Thread-pool offload:** `loop.run_in_executor(None, ...)` runs the synchronous Gemini SDK call in a thread, leaving the async event loop free to handle other requests during the API round-trip.

3. **Total exception swallowing:** Every failure path (timeout, quota exceeded, network error, invalid API key, JSON parse failure) logs a warning and returns the static fallback. The `/predict` endpoint **always returns HTTP 200** with a complete response — Gemini is never visible to the client as a failure point.

**Minor concern:** Swallowing all exceptions means a misconfigured API key produces no visible error to the developer during testing. The log warning (`logger.warning(...)`) is the only signal. Consider adding a startup warning if the key looks invalid.

---

### 3.5 Image Processing Pipeline — `image_utils.py`

Two subtle correctness decisions:

**The verify-then-reopen pattern:**
```python
image = Image.open(io.BytesIO(contents))
image.verify()                              # ← reads entire stream to end
image = Image.open(io.BytesIO(contents))   # ← must re-open from same bytes
```
`PIL.verify()` is destructive — it exhausts the stream. Re-opening from the same `contents` bytes is mandatory. Forgetting the second open would pass the corruption check but then crash on `.convert("RGB")`.

**Always converting to RGB:**
The model's first convolutional layer expects exactly 3 channels. Without `convert("RGB")`:
- RGBA PNG → 4 channels → dimension mismatch crash
- Grayscale → 1 channel → dimension mismatch crash
- CMYK → 4 channels → dimension mismatch crash

---

### 3.6 Transform Correctness — `classifier.py`

```python
_inference_transform = transforms.Compose([
    transforms.Resize(256),       # matches: Resize(input_size + 32) where input_size=224
    transforms.CenterCrop(224),   # matches: CenterCrop(input_size)
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])
```

This exactly mirrors `val_tfm` in `03_train_model.py`. Cross-verified against the training script. Using the training transform (which includes `RandomResizedCrop` and `RandomHorizontalFlip`) at inference would make the same image produce different predictions on different calls — a silent but serious correctness bug.

---

### 3.7 Pydantic Response Models — `routes/predict.py`

Six Pydantic models define the full response schema. Benefits:
- FastAPI auto-generates accurate Swagger UI at `/docs` with no extra work
- Response validation catches shape mismatches at development time, not in production
- Type annotations serve as living documentation

The `source: str` field in `SuggestionsResponse` (`"ai"` or `"static"`) lets the frontend optionally show a badge indicating whether suggestions came from Gemini or the static knowledge base.

---

## 4. Potential Issues & Gaps

### 4.1 `map_location="cpu"` Always — Moderate
The current `classifier.py` always loads the model onto CPU regardless of whether a GPU is available on the server. For a local development machine with a GPU, inference would be slower than necessary. A simple fix:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dict = torch.load(settings.model_path, map_location=device, weights_only=True)
model = model.to(device)
# Also move tensor in predict(): tensor = tensor.to(device)
```

For Phase 3 (local dev / demo deployment), CPU is fine. For Phase 4 deployment, this should be addressed.

### 4.2 No Rate Limiting
A user can spam `/predict` with hundreds of requests per second. Each request runs a full EfficientNet forward pass (~50–100ms on CPU). This could saturate the server. For a student project this is acceptable, but production would need `slowapi` or a reverse proxy rate limit.

### 4.3 No Input Logging for Debugging
Currently there is no log of what image was uploaded or what prediction was returned. In production, logging `{plastic_type, confidence, source}` per request would make debugging misclassifications much easier.

### 4.4 `GET /health` Has a Local Import
```python
@router.get("/health")
async def health_check():
    from app.services.classifier import get_model  # ← local import
```
This works but is unusual. The import should be at the top of the file. It was structured this way to avoid a circular import concern that doesn't actually exist here — move it to the top.

### 4.5 `SuggestionsResponse.source` Has No Enum Validation
`source: str` accepts any string. If `get_ai_suggestions` ever returned a typo like `"statiic"`, Pydantic would not catch it. Should be `source: Literal["ai", "static"]` for strict validation.

### 4.6 `all_probabilities` Always Returned
The full 6-probability dict is returned on every `/predict` response. This is useful for debugging but unnecessary data for the frontend (which only needs top-3). Consider making it optional or removing it from the production response to reduce payload size.

---

## 5. Security Assessment

| Concern | Status | Implementation |
|---|---|---|
| API key in code | ✅ Safe | Loaded from `.env`, not hardcoded |
| `.env` in git | ✅ Safe | `.gitignore` excludes it |
| `torch.load` pickle injection | ✅ Mitigated | `weights_only=True` |
| Path traversal via filename | ✅ Safe | Bytes read to memory, filename never used |
| Large file DoS | ✅ Mitigated | 10 MB check before processing |
| Corrupt file crash | ✅ Handled | `PIL.verify()` catches it |
| CORS wildcard `*` | ✅ Not used | Explicit origin allowlist |
| Raw traceback to client | ✅ Prevented | Global exception handler returns generic 500 |

---

## 6. What's Working Well

- **Full separation of concerns** — each file has one job. Adding a new model or a new data source doesn't require touching other files.
- **Gemini never breaks the API** — the fallback chain is watertight.
- **Phase 2 findings are embedded in the code** — the PP disclaimer and 0.70 confidence threshold directly implement the analysis from `phase2_results_analysis.md`.
- **Syntax-verified** — all 11 files passed AST syntax check before commit.
- **4 incremental commits** — no single large dump; history is traceable.

---

## 7. Before Running Locally

```bash
# 1. Copy model file (download from Kaggle)
cp ~/Downloads/best_efficientnet_b0.pth backend/models/

# 2. Set up environment
cd backend
cp .env.example .env
# Optionally add: GEMINI_API_KEY=your_key

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
uvicorn app.main:app --reload --port 8000

# 5. Verify
curl http://localhost:8000/health
# Open: http://localhost:8000/docs
```

---

## 8. Fixes to Apply Before Phase 4

| Priority | Fix |
|---|---|
| High | Add `device` auto-detection (CPU vs GPU) in `classifier.py` |
| Medium | Move `get_model` import to top of `routes/predict.py` |
| Medium | Change `source: str` to `source: Literal["ai", "static"]` |
| Low | Add per-request logging (`plastic_type`, `confidence`, `source`) |
| Low | Consider removing `all_probabilities` from production response |
