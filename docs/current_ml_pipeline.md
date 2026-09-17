# SVACS Current ML Pipeline — Audit Report

**Audit Date:** 2026-09-17

---

## End-to-End Data Flow

```
                         ┌─────────────────┐
                         │  Image Upload    │
                         │  (bytes/base64)  │
                         └────────┬────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Preprocessing             │
                    │  decode_image_bytes()      │
                    │  preprocess_for_inference()│
                    │  (currently pass-through)  │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
   ┌──────────▼──────────┐  ┌────▼──────────┐        │
   │  Stage 2: OCR       │  │  Stage 3:     │        │
   │  EasyOCR            │  │  Detection    │        │
   │  extract_text()     │  │  YOLO predict │        │
   │  (non-fatal)        │  │  + classifier │        │
   └──────────┬──────────┘  └────┬──────────┘        │
              │                  │                    │
              │    ┌─────────────▼─────────────┐      │
              │    │  For each YOLO detection:  │      │
              │    │                            │      │
              │    │  1. Crop vessel region     │      │
              │    │  2. EfficientNet classify  │      │
              │    │     (crop + whole-image)   │      │
              │    │  3. Refinement heuristics  │      │
              │    │  4. Label assignment       │      │
              │    └─────────────┬─────────────┘      │
              │                  │                    │
              └───────┬──────────┘                    │
                      │                               │
           ┌──────────▼──────────┐                    │
           │  Stage 4:           │                    │
           │  Explainability     │                    │
           │  draw_evidence()    │                    │
           │  (red/green boxes)  │                    │
           └──────────┬──────────┘                    │
                      │                               │
           ┌──────────▼──────────┐                    │
           │  Stage 5:           │                    │
           │  Build Response     │                    │
           │  VisionAnalysis     │                    │
           │  Response           │                    │
           └──────────┬──────────┘                    │
                      │                               │
     ┌────────────────┼────────────────┐              │
     │                │                │              │
┌────▼────┐    ┌──────▼──────┐   ┌─────▼─────┐       │
│ Stage 6 │    │  Stage 7    │   │  main.py  │       │
│ Replay  │    │  Bucket     │   │  POST     │       │
│ Save    │    │  Write      │   │  handler  │       │
│ (disk)  │    │  (HTTP)     │   │           │       │
└─────────┘    └─────────────┘   └─────┬─────┘       │
                                       │              │
                              ┌────────▼────────┐     │
                              │  Assemble final  │     │
                              │  JSON response   │     │
                              │  + vessel_store  │     │
                              └────────┬────────┘     │
                                       │              │
                              ┌────────▼────────┐     │
                              │  HTTP Response   │     │
                              │  to Frontend     │     │
                              └─────────────────┘     │
```

---

## Component Details

### 1. Input Handling

**Entry Points:**
- `POST /intelligence/image` — UploadFile (primary, used by frontend)
- `POST /api/v1/analyze` — UploadFile (structured response)
- `POST /api/v1/batch-analyze` — Base64 batch

**Processing:**
```python
# vision_orchestrator.process_bytes()
image_bytes → decode_image_bytes() → np.ndarray (BGR)
            → preprocess_for_inference() → np.ndarray (currently identity)
```

### 2. YOLO Detector

**Architecture:**
- YOLOv8-nano (Ultralytics 8.2.70)
- Custom-trained on single class: `front_vessel`
- Falls back to COCO class 8 (`boat`) if COCO model loaded

**Inference Flow:**
```python
results = model.predict(image, conf=0.35, iou=0.45, imgsz=640)
# If no detections → retry at conf=0.25
# Filter: only class 0 (custom) or class 8 (COCO)
```

**Quick Mode:**
- Skips YOLO entirely
- Runs EfficientNet on whole image
- Returns single detection with full-image bounding box

### 3. EfficientNet Classifier

**Architecture:**
- `torchvision.models.efficientnet_v2_s(weights=None)`
- Final linear layer: 1280 → 8 classes
- Loaded from checkpoint with `model_state_dict` + `classes`

**Inference Flow:**
```python
# Whole-image classification:
Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet)
→ model(input) → softmax → top-3

# Crop classification:
Expand crop by 15% → Pad to square → Same transforms → model(input)

# Refinement heuristics:
Container Ship → Cruise Ship (if cruise conf >= 12% and AR >= 0.35)
Container Ship → Passenger Ferry (if ferry conf >= 12% and AR >= 0.30)
Fishing Vessel → Oil Tanker/Container/Ferry (if conf >= 20% and AR >= 2.0)
Oil Tanker/Container → Cruise Ship (if cruise conf >= 15% and AR >= 0.35)
```

### 4. OCR Service

**Engine:** EasyOCR v1.7.1
- CRAFT text detection (83 MB)
- English recognition (15 MB)
- Runs on CPU
- Images downscaled to max 1000px

**Output:**
```python
List[OCRResult]:
  text: str
  confidence: float
  bounding_box: BoundingBox
```

### 5. Explainability

**Simple overlay:**
- Red bounding boxes for detections with label + confidence
- Green bounding boxes for OCR results with text
- Output: Base64-encoded JPEG

### 6. Replay Service

**Saves per request:**
- `input.jpg` — original image
- `contract.json` — response (without base64 image)
- `metadata.json` — timestamp, model version, model path
- Location: `/tmp/svacs_replays/{replay_id}/`

### 7. Bucket Client

**Real HTTP client:**
- POST to `https://bhiv-bucket-i1l6.onrender.com/bucket/artifact`
- SHA-256 hash verification
- Chain-state linking
- Non-fatal (warnings only)

---

## Model Loading Strategy

```
Application startup → NO model loading (lifespan is empty)
    │
First POST /intelligence/image request
    │
    ├── inference_service.initialize()
    │   ├── Threading lock acquired
    │   ├── YOLO model loaded from vessel_front_model.pt
    │   ├── EfficientNet loaded from efficientnet_vessel_best.pth
    │   └── Lock released
    │
    └── ocr_service.initialize() (if OCR enabled)
        ├── Threading lock acquired
        ├── EasyOCR Reader constructed
        └── Lock released
```

---

## What SVACS Does NOT Currently Have

1. **No Indian Navy vessel recognition** — No ship-level identity
2. **No embedding/retrieval system** — No FAISS, no SigLIP2
3. **No identity database** — No vessel metadata
4. **No open-set recognition** — No UNKNOWN handling
5. **No military vessel classification** — Only commercial vessel types
6. **No vessel taxonomy** — Hardcoded 8 classes
7. **No multi-vessel pipeline** — Processes largest detection only (in main.py response)
8. **No decision engine** — Direct label → response mapping
9. **No GPU support** — CPU-only in requirements.txt
10. **No structured tests** — Only `test_api.py` and `test_bug.py`
11. **No evaluation pipeline** — No metrics reporting
12. **No dataset validation** — No quality checks
13. **No model versioning** — No manifest
14. **No configurable taxonomy** — Hardcoded class lists

---

## Conflicts with Specification

| Specification Requirement | Current State | Resolution |
|---------------------------|---------------|------------|
| GPU/CUDA support | CPU-only PyTorch | Need separate local requirements with CUDA |
| Multiple vessel processing | Only best detection in response | Refactor main.py POST handler |
| Identity recognition | None | New services needed |
| SigLIP2 + FAISS | None | New dependencies + services |
| Maritime ViT classifier | EfficientNetV2-S only | Add as additional classifier |
| Decision engine | None | New service needed |
| Structured tests | Minimal | New test suite needed |
| Dataset structure | Flat raw images | Restructure needed |
| Configurable taxonomy | Hardcoded | New config file needed |

---

## Key Preservation Points

These must remain functional during upgrade:
1. ✅ `POST /intelligence/image` endpoint
2. ✅ `POST /api/v1/analyze` endpoint
3. ✅ YOLO detection pipeline
4. ✅ EfficientNet classification (as baseline)
5. ✅ EasyOCR integration
6. ✅ Explainability overlay
7. ✅ Replay service
8. ✅ Bucket client
9. ✅ Frontend Signals page upload flow
10. ✅ Lazy model loading pattern
11. ✅ Thread-safe initialization
12. ✅ `render.yaml` deployment config
13. ✅ Health endpoint
14. ✅ All GET endpoints
