import os
from pydantic_settings import BaseSettings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_YOLO_MODEL_PATH = os.path.join(BASE_DIR, "vessel_front_model.pt")
DEFAULT_CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "efficientnet_vessel_best.pth")

# Load .env file if present (for local development)
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

class Settings(BaseSettings):
    PROJECT_NAME: str = "BHIV Vision Intelligence Runtime"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    # ── YOLO Detector ─────────────────────────────────────────
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", DEFAULT_YOLO_MODEL_PATH)
    YOLO_IMAGE_SIZE: int = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.35"))
    YOLO_FALLBACK_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_FALLBACK_CONFIDENCE_THRESHOLD", "0.25"))
    YOLO_USE_FALLBACK: bool = str(os.getenv("YOLO_USE_FALLBACK", "true")).lower() in ("1", "true", "yes")
    YOLO_IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
    YOLO_MAX_DETECTIONS: int = int(os.getenv("YOLO_MAX_DETECTIONS", "100"))
    YOLO_MIN_ACCEPTED_CONFIDENCE: float = float(os.getenv("YOLO_MIN_ACCEPTED_CONFIDENCE", "0.25"))

    # ── EfficientNet Classifier (baseline) ────────────────────
    CLASSIFIER_MODEL_PATH: str = os.getenv(
        "CLASSIFIER_MODEL_PATH", DEFAULT_CLASSIFIER_MODEL_PATH
    )
    CLASSIFIER_MIN_CONFIDENCE: float = float(
        os.getenv("CLASSIFIER_MIN_CONFIDENCE", "0.60")
    )

    # ── Gemini Vision API ─────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    GEMINI_ENABLED: bool = str(os.getenv("GEMINI_ENABLED", "true")).lower() in ("1", "true", "yes")
    GEMINI_TIMEOUT: int = int(os.getenv("GEMINI_TIMEOUT", "45"))

    # ── OpenRouter API (Fallback) ─────────────────────────────
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash:free")

    # ── Local Gemini Proxy API ────────────────────────────────
    LOCAL_GEMINI_API_URL: str = os.getenv("LOCAL_GEMINI_API_URL", "http://127.0.0.1:8081/v1/chat/completions")
    LOCAL_GEMINI_API_KEY: str = os.getenv("LOCAL_GEMINI_API_KEY", "sk-my-custom-key")

    # ── Device Configuration ──────────────────────────────────
    # auto | cuda | cpu
    VISION_DEVICE: str = os.getenv("VISION_DEVICE", "auto")

    # ── OCR config ────────────────────────────────────────────
    OCR_LANGUAGES: list[str] = ["en"]
    OCR_ENABLED: bool = str(os.getenv("OCR_ENABLED", "true")).lower() in (
        "1",
        "true",
        "yes",
    )

    # ── Replay storage ────────────────────────────────────────
    REPLAY_STORAGE_DIR: str = os.getenv("REPLAY_STORAGE_DIR", "/tmp/svacs_replays")

    class Config:
        case_sensitive = True

settings = Settings()

# Ensure replay directory exists — wrapped so a failure here never prevents startup
try:
    os.makedirs(settings.REPLAY_STORAGE_DIR, exist_ok=True)
except Exception:
    pass  # Non-fatal: replays will be skipped gracefully if the directory is not writable

