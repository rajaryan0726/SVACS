# SVACS Current Architecture — Audit Report

**Audit Date:** 2026-09-17  
**Repository:** BHIV-Engineering-Exchange/bhiv-SVACS  
**Auditor:** Phase 1 — Full Repository Inspection

---

## Repository Structure

```
bhiv-SVACS/
├── backend/                          # FastAPI vision runtime (Python 3.11)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI entry point (539 lines)
│   │   ├── core/
│   │   │   └── config.py             # Settings via pydantic-settings
│   │   ├── models/
│   │   │   └── schemas.py            # Pydantic response/request models
│   │   └── services/
│   │       ├── inference_service.py   # YOLO + EfficientNet (629 lines)
│   │       ├── vision_orchestrator.py # Pipeline orchestrator (200 lines)
│   │       ├── ocr_service.py         # EasyOCR wrapper (172 lines)
│   │       ├── preprocessing.py       # Image decode/encode utilities
│   │       ├── explainability.py      # Bounding-box overlay drawing
│   │       ├── replay_service.py      # Save input+output for replay
│   │       └── bucket_client.py       # Bucket artifact writer
│   ├── scripts/
│   │   ├── scrape_wikimedia.py        # Wikimedia Commons image scraper
│   │   ├── train_classifier.py        # EfficientNet training (advanced)
│   │   ├── add_image_to_classifier.py # Add images to classifier dataset
│   │   ├── download_ocr_models.py     # Pre-bundle EasyOCR models
│   │   └── import_drive_images.py     # Import images from Google Drive
│   ├── train_model.py                 # YOLO training script
│   ├── train_classifier.py            # Simpler EfficientNet training
│   ├── auto_label.py                  # Auto-label using COCO YOLO
│   ├── evaluate_model.py              # Model evaluation
│   ├── predict.py                     # Standalone prediction
│   ├── debug_yolo.py                  # YOLO debugging
│   ├── test_api.py                    # API integration test
│   ├── test_bug.py                    # Bug reproduction test
│   ├── requirements.txt               # Python deps (CPU-only PyTorch)
│   ├── runtime.txt                    # python-3.11.9
│   ├── vessel_front_model.pt          # Custom YOLO model (6.5 MB)
│   ├── yolov8n.pt                     # Pretrained YOLOv8-nano (6.5 MB)
│   ├── efficientnet_vessel_best.pth   # EfficientNetV2-S weights (81.6 MB)
│   ├── ocr_models/                    # Pre-bundled EasyOCR weights
│   │   ├── craft_mlt_25k.pth         # 83.1 MB — text detection
│   │   └── english_g2.pth            # 15.1 MB — text recognition
│   ├── replays/                       # 186 saved replay directories
│   ├── runs/                          # YOLO training outputs (gitignored)
│   └── operational_integration_sprint/# Sprint documentation
│
├── dataset/                           # 37 raw ship images (JPG, ~2-5 MB each)
│
├── frontend/                          # Vite + React + TypeScript dashboard
│   ├── src/
│   │   ├── App.tsx                    # React Router (12 pages)
│   │   ├── main.tsx                   # Entry point
│   │   ├── env.ts                     # Environment configuration
│   │   ├── api/
│   │   │   ├── adapter.ts             # Mock/Real API adapter pattern
│   │   │   ├── client.ts              # Axios client
│   │   │   └── queryClient.ts         # React Query setup
│   │   ├── domain/types.ts            # Zod schemas for domain types
│   │   ├── components/                # UI components
│   │   ├── pages/                     # 13 page components
│   │   ├── lib/                       # Utility functions
│   │   ├── mock/                      # Mock data generators
│   │   └── store/                     # Zustand state management
│   ├── package.json                   # React 18, TailwindCSS 3, Recharts
│   ├── tailwind.config.js
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── render.yaml                        # Render deployment config
├── README.md
├── run-backend.cmd                    # Windows helper scripts
└── run-frontend.cmd
```

---

## Current ML Pipeline

### 1. YOLO Detector
- **Model:** Custom-trained YOLOv8-nano (`vessel_front_model.pt`, 6.5 MB)
- **Base:** `yolov8n.pt` (COCO pretrained)
- **Training:** `train_model.py` — 10 epochs, batch=4, imgsz=640, CPU
- **Classes:** Single class: `0 = front_vessel`
- **Fallback:** COCO class 8 (boat) when using generic `yolov8n.pt`
- **Dual-threshold:** Primary 0.35, fallback 0.25

### 2. EfficientNet Classifier
- **Model:** EfficientNetV2-S (`efficientnet_vessel_best.pth`, 81.6 MB)
- **Training:** Two scripts exist:
  - `train_classifier.py` (root) — simpler, 10 epochs
  - `scripts/train_classifier.py` — advanced, 50 epochs, augmentation
- **Classes (8):**
  1. Chemical Tanker
  2. Container Ship
  3. Cruise Ship
  4. Fishing Trawler
  5. Fishing Vessel
  6. LPG Carrier
  7. Oil Tanker
  8. Passenger Ferry
- **Inference:** Whole-image + crop classification with refinement heuristics
- **Minimum confidence:** 0.60 (configurable)

### 3. OCR Service
- **Engine:** EasyOCR (v1.7.1)
- **Languages:** English (`en`)
- **Models:** Pre-bundled in `ocr_models/` (CRAFT + English recognition)
- **Feature:** Lazy-loaded, thread-safe, configurable on/off
- **Optimization:** Images resized to max 1000px before OCR

### 4. Processing Pipeline
```
Image bytes → decode → preprocess → OCR extraction → YOLO detection
    → EfficientNet classification (per crop + whole image)
    → Label refinement heuristics → Explainability overlay
    → Replay save → Bucket write → API response
```

---

## Current API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Root — service status |
| GET | `/health` | Health check — models loaded status |
| POST | `/intelligence/image` | **Primary** — Image upload + analysis |
| POST | `/api/v1/analyze` | File upload analysis (structured response) |
| POST | `/api/v1/batch-analyze` | Batch base64 image analysis |
| GET | `/signals` | Signal chunks (empty) |
| GET | `/perception` | Perception events (empty) |
| GET | `/intelligence` | Intelligence events (empty) |
| GET | `/state-events` | State events (empty) |
| GET | `/vessels` | Vessel store (populated by uploads) |
| GET | `/alerts` | Alerts (empty) |
| GET | `/bucket/status` | Bucket sync status |
| GET | `/stage-metrics` | Pipeline stage metrics (hardcoded) |
| GET | `/events-over-time` | Events timeline (hardcoded) |
| GET | `/validation-breakdown` | Validation stats (hardcoded) |
| GET | `/trace/{trace_id}` | Trace lifecycle lookup |

---

## Current Pydantic Schemas

### Response Models
- `BoundingBox`: x_min, y_min, x_max, y_max
- `TopPrediction`: class_name, confidence
- `DetectionResult`: label, confidence, bounding_box, top_predictions
- `OCRResult`: text, confidence, bounding_box
- `VisionAnalysisResponse`: replay_id, detections[], ocr_results[], explainable_image_base64
- `UploadImageResponse`: trace_id, vessel_class, confidence_score, ocr_text, risk_level, etc.

### Request Models
- `VisionAnalysisRequest`: image_base64, return_explainable_image

---

## Current Configuration (config.py)

| Variable | Default | Purpose |
|----------|---------|---------|
| `YOLO_MODEL_PATH` | `vessel_front_model.pt` | YOLO weights path |
| `YOLO_IMAGE_SIZE` | 640 | Inference image size |
| `YOLO_CONFIDENCE_THRESHOLD` | 0.35 | Primary detection threshold |
| `YOLO_FALLBACK_CONFIDENCE_THRESHOLD` | 0.25 | Fallback threshold |
| `YOLO_USE_FALLBACK` | true | Enable fallback detection |
| `YOLO_IOU_THRESHOLD` | 0.45 | NMS IoU threshold |
| `YOLO_MAX_DETECTIONS` | 100 | Max detections per image |
| `YOLO_MIN_ACCEPTED_CONFIDENCE` | 0.25 | Minimum accepted confidence |
| `CLASSIFIER_MODEL_PATH` | `efficientnet_vessel_best.pth` | Classifier weights |
| `CLASSIFIER_MIN_CONFIDENCE` | 0.60 | Minimum classifier confidence |
| `OCR_ENABLED` | true | Enable/disable OCR |
| `OCR_LANGUAGES` | `["en"]` | OCR languages |
| `REPLAY_STORAGE_DIR` | `/tmp/svacs_replays` | Replay storage location |

---

## Current Frontend Architecture

- **Framework:** Vite + React 18 + TypeScript
- **Styling:** TailwindCSS 3
- **State:** Zustand
- **Data fetching:** TanStack React Query
- **Routing:** React Router v6
- **Charts:** Recharts
- **Adapter pattern:** Mock/Real adapter switching via `VITE_USE_MOCK`

### Frontend Pages (13)
1. Overview — Dashboard with KPIs
2. LivePipeline — Pipeline visualization
3. Signals — **Image upload + analysis** (primary interaction)
4. Perception — Perception events
5. Intelligence — Intelligence events with validation
6. StateEngine — State transitions
7. Vessels — Vessel tracking table
8. Alerts — Alert management
9. TraceExplorer — Trace lifecycle viewer
10. BucketStatus — Storage sync status
11. SystemHealth — Health monitoring
12. Settings — Configuration
13. MaritimeCommandCenter — Command overview

### Image Upload Flow (Signals.tsx)
1. User uploads image (file, camera, photo library)
2. POST to `/intelligence/image?quick=true`
3. Displays: vessel_class, confidence, top_predictions, risk_level, OCR text, explanation, explainable image

---

## Current Dataset

### `dataset/` (root)
- 37 raw JPG images from camera (July 2026)
- File sizes: 1.1 MB — 4.8 MB each
- No labels, no metadata, no structure
- Appears to be photos taken at a port/harbor

### Training Dataset (gitignored)
- `backend/dataset/` is gitignored
- Expected structure: `dataset/{train,val,test}/{images,labels}/`
- `data.yaml` expected for YOLO training
- `dataset/classifier/` expected for EfficientNet (ImageFolder format)

### Scraper
- `scripts/scrape_wikimedia.py` downloads images from 35 Wikimedia categories
- Target: 2000 images per class from Wikimedia Commons

---

## Current Model Weights

| File | Size | Purpose |
|------|------|---------|
| `vessel_front_model.pt` | 6.5 MB | Custom YOLO (vessel detection) |
| `yolov8n.pt` | 6.5 MB | Pretrained YOLOv8-nano (COCO) |
| `efficientnet_vessel_best.pth` | 81.6 MB | EfficientNetV2-S (8-class vessel type) |
| `ocr_models/craft_mlt_25k.pth` | 83.1 MB | CRAFT text detection |
| `ocr_models/english_g2.pth` | 15.1 MB | English OCR recognition |

---

## Deployment

- **Platform:** Render (Free tier)
- **Backend:** Python web service, CPU-only PyTorch
- **Frontend:** Static site (Vite build)
- **Constraints:** 512 MB RAM limit, no GPU, lazy model loading required
- **CORS:** Configured for localhost and Render domains

---

## Key Design Decisions Already Made
1. **Lazy model loading** — Models load on first request, not at startup (memory constraint)
2. **CPU-only** — PyTorch CPU wheels to fit Render Free tier
3. **Thread-safe singletons** — Lock-based initialization for models
4. **Non-fatal OCR** — OCR failure doesn't crash the pipeline
5. **Replay system** — Full input/output saved for reproducibility
6. **Bucket integration** — Real artifact writing to external Bucket service
7. **Quick mode** — Skip YOLO detection, use classifier only
