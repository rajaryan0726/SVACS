from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

# ── Existing schemas (preserved for backward compatibility) ────────

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class TopPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    class_name: str = Field(..., alias="class")
    confidence: float

class DetectionResult(BaseModel):
    label: str
    confidence: float
    bounding_box: BoundingBox
    top_predictions: List[TopPrediction] = []

class OCRResult(BaseModel):
    text: str
    confidence: float
    bounding_box: BoundingBox

class VisionAnalysisResponse(BaseModel):
    replay_id: str
    detections: List[DetectionResult]
    ocr_results: List[OCRResult]
    explainable_image_base64: Optional[str] = Field(None, description="Base64 encoded image with bounding boxes drawn")

class VisionAnalysisRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image to analyze")
    return_explainable_image: bool = Field(True, description="Whether to return the base64 encoded image with visual evidence")

class DetectionDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    class_name: str = Field(..., alias="class")
    confidence: float
    bbox: BoundingBox

class UploadImageResponse(BaseModel):
    trace_id: str
    validation_status: str
    vessel_detected: bool
    vessel_class: str
    confidence_score: float
    ocr_text: Optional[str] = None
    operator: Optional[str] = None
    risk_level: str
    classification_source: str
    detections: List[DetectionDetail]
    top_predictions: List[TopPrediction] = []
    explanation: List[str]
    explainable_image_base64: Optional[str] = None


# ── NEW: Indian Navy Vessel Identification schemas ─────────────────

class VesselTypeResult(BaseModel):
    """Broad vessel classification result."""
    category: Optional[str] = None  # "Military Vessel", "Commercial Vessel", etc.
    subtype: Optional[str] = None   # "Destroyer", "Aircraft Carrier", etc.
    confidence: float = 0.0

class VesselIdentityResult(BaseModel):
    """Indian Navy ship-level identity result."""
    known: bool = False
    ship_id: Optional[str] = None
    name: Optional[str] = None          # "INS Kolkata"
    ship_class: Optional[str] = None    # "Kolkata-class"
    pennant_number: Optional[str] = None  # "D63"
    similarity: Optional[float] = None

class VesselDetectionItem(BaseModel):
    """A single vessel detection with full identification pipeline results."""
    bounding_box: BoundingBox
    detection_confidence: float = 0.0

    # Classification
    vessel_type: Optional[VesselTypeResult] = None

    # Identity
    identity: Optional[VesselIdentityResult] = None

    # OCR
    ocr_text: List[str] = []

    # Decision
    organization: Optional[str] = None  # INDIAN_NAVY, FOREIGN_MILITARY, CIVILIAN, UNKNOWN
    status: str = "UNKNOWN"             # IDENTIFIED, UNKNOWN, VESSEL_DETECTED, NO_VESSEL, LOW_CONFIDENCE
    nation: Optional[str] = None

    # Evidence
    visual_evidence: List[str] = []
    description: Optional[str] = None
    confidence_level: Optional[str] = None  # high, medium, low

    # Backward compat
    label: str = "Unknown"              # EfficientNet label
    confidence: float = 0.0             # EfficientNet confidence
    top_predictions: List[TopPrediction] = []

class EnhancedAnalysisResponse(BaseModel):
    """Enhanced response with Indian Navy identification."""
    success: bool = True
    trace_id: str
    vessel_detected: bool = False
    total_detections: int = 0
    detections: List[VesselDetectionItem] = []
    ocr_results: List[OCRResult] = []
    explainable_image_base64: Optional[str] = None
    classification_source: str = "Gemini Vision API + YOLO + EfficientNet"
    model_version: str = "2.0.0"

    # Backward compatibility fields
    validation_status: str = "OK"
    vessel_class: Optional[str] = None
    confidence_score: float = 0.0
    risk_level: str = "LOW"
    explanation: List[str] = []

