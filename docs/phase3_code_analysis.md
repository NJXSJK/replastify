# Phase 3 — Complete Deep Code Analysis & Unified Fix Plan
**Date:** 2026-05-25 | **Scope:** End-to-end Python backend codebase, security, concurrency, and resource constraints.

---

## Part 1 — Expanded Issue Registry & Security Audit

This section details all vulnerabilities, concurrency issues, performance bottlenecks, and code quality issues identified in the backend.

### 1.1 Critical Vulnerabilities & Concurrency Bugs

#### ISSUE-01 · Memory Exhaustion DoS via Unbounded File Reads (`image_utils.py:32`)
- **Severity:** High
- **Vulnerability:** `contents = await file.read()` reads the entire uploaded file into RAM in a single operation.
- **Root Cause:** Size validation occurs *after* the file is fully read into memory.
- **Impact:** A malicious client uploading a 1 GB file will cause the server to allocate 1 GB of memory. Under high concurrency or limited RAM, this immediately triggers a kernel Out-Of-Memory (OOM) event and crashes the entire Docker container (Denial of Service).
- **Fix:** Read the stream incrementally in 1 MB chunks. Raise an HTTP 413 error immediately if the accumulated bytes exceed the configured limit.

#### ISSUE-02 · Blocking CPU-bound Inference on Async Event Loop (`routes/predict.py:83`)
- **Severity:** High
- **Vulnerability:** `result = predict(image)` runs PyTorch model inference directly inside an `async def` handler.
- **Root Cause:** FastAPI uses a single-threaded event loop for async routes. Regular sync calls block the loop.
- **Impact:** The event loop is blocked for 80-200ms per prediction. While one prediction is running, no other requests, background tasks, or health checks can run.
- **Fix:** Wrap the synchronous `predict` call in `asyncio.to_thread` to run it in FastAPI's default worker thread pool.

#### ISSUE-03 · Global Exception Handler Swallows Custom API Errors (`main.py:89`)
- **Severity:** Medium
- **Vulnerability:** The general `@app.exception_handler(Exception)` catches and masks all exceptions, including `HTTPException`.
- **Root Cause:** FastAPI/Starlette `HTTPException` is a subclass of `Exception`. Registering a catch-all handler without explicitly returning or re-raising HTTP exceptions overrides their custom status codes and details.
- **Impact:** Validation errors (400, 413, 422) raised within utility functions are returned to the user as generic "500 Internal Server Error" responses with masked detail messages.
- **Fix:** Explicitly inspect the exception class and return the appropriate HTTP response structure if it is a Starlette or FastAPI `HTTPException`.

---

### 1.2 Performance & System Gaps

#### ISSUE-04 · Hardcoded CPU Inference (`classifier.py:88`)
- **Severity:** Medium
- **Vulnerability:** `map_location="cpu"` is hardcoded for model weight loading, and input tensors are not transferred to the active device.
- **Root Cause:** Assumption that deployment is CPU-only, ignoring GPU availability in local or staging environments.
- **Impact:** The backend fails to utilize CUDA acceleration even if GPUs are present.
- **Fix:** Add dynamic device auto-detection (`cuda` vs `cpu`) and explicitly transfer both the model and the preprocessed image tensor to the detected device.

#### ISSUE-05 · Lack of Input/Output Logging (`routes/predict.py`)
- **Severity:** Medium
- **Vulnerability:** API routes process uploads and return predictions silently without logging.
- **Root Cause:** Omission of operational logging.
- **Impact:** Operational blindness. Administrators cannot monitor model performance, detect drift, or debug user-reported misclassifications in production.
- **Fix:** Add structured `logger.info` stating the identified class, model confidence, uncertainty status, and the suggestion provider source.

#### ISSUE-06 · Missing Class Consistency Verification on Startup (`main.py:40`)
- **Severity:** Medium
- **Vulnerability:** No automated check to ensure classifier classes map to knowledge base entries.
- **Root Cause:** Lack of dependency validation during startup.
- **Impact:** If `CLASS_NAMES` in `classifier.py` and `PLASTIC_DATABASE` in `plastic_info.py` drift, the application starts successfully but throws an uncaught KeyError at runtime when a mismatched class is predicted.
- **Fix:** Add a validation loop in the lifespan startup handler to assert that every class in `CLASS_NAMES` exists as a key in `PLASTIC_DATABASE`.

---

### 1.3 Code Quality & Dead Code

#### ISSUE-07 · Dead Code: Unused Temp Directory (`main.py:41`)
- **Severity:** Low
- **Vulnerability:** Startup lifespan creates a `temp/` folder on disk that is never used.
- **Root Cause:** Legacy design where uploaded files were saved to disk before classification.
- **Impact:** Wasted storage/IO setup, creates confusion for maintainers.
- **Fix:** Remove `temp_dir` from `config.py` and cleanup directories creation in `main.py`.

#### ISSUE-08 · Type Constraint Gaps in API Schemas (`routes/predict.py:60`)
- **Severity:** Low
- **Vulnerability:** Fields like `HealthResponse.status` use unconstrained strings.
- **Root Cause:** Omission of strict schema typing.
- **Impact:** API documentation shows generic string types, and invalid status values could pass validation.
- **Fix:** Apply `Literal` constraints (e.g. `status: Literal["ok", "degraded"]`).

---

## Part 2 — Action Plan & Rethinking

We will implement all fixes sequentially in a single pass to ensure consistency.

### 2.1 Implementation Checklist

- [ ] **Step 1: Configuration (`config.py`)**
  - Verify `temp_dir` is removed.
  - Verify model dimensions are present.

- [ ] **Step 2: Security & Helper Optimizations (`utils/image_utils.py`)**
  - Replace `file.read()` with chunk-by-chunk stream loading (1 MB buffer).
  - Enforce size bounds during the streaming process to mitigate DoS.
  - Clean up redundant exception handlers.

- [ ] **Step 3: Device-Aware Model Inference (`services/classifier.py`)**
  - Auto-detect the target device (`cuda` or `cpu`).
  - Move model weights and input tensors to the target device.

- [ ] **Step 4: API Routes Verification & Blocking Call Offloading (`routes/predict.py`)**
  - Use `asyncio.to_thread` for the synchronous `predict` call.
  - Apply `Literal` constraint to `HealthResponse.status`.
  - Add structured request logging with fields: `filename`, `prediction`, `confidence`, `is_uncertain`, `suggestions_source`.

- [ ] **Step 5: App Factory Lifecycle & Lifespan Context Manager (`main.py`)**
  - Add `CLASS_NAMES` to `PLASTIC_DATABASE` check at startup.
  - Remove folder creation logic for `temp_dir`.
  - Update `global_exception_handler` to properly pass `HTTPException` detail and status codes.
