"""decision_engine.py — Combines detection, Gemini analysis, and OCR into final vessel decision.

Takes outputs from:
  1. YOLO detector (bounding boxes)
  2. Gemini Vision API (vessel identification)
  3. EfficientNet classifier (baseline type classification)
  4. EasyOCR (text extraction)

Produces a final decision with:
  - vessel_type + subtype
  - organization (INDIAN_NAVY / FOREIGN_MILITARY / CIVILIAN / UNKNOWN)
  - identity (known ship or unknown)
  - status (IDENTIFIED / UNKNOWN / VESSEL_DETECTED / NO_VESSEL)
  - combined confidence
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Status values ─────────────────────────────────────────────
STATUS_NO_VESSEL = "NO_VESSEL"
STATUS_IDENTIFIED = "IDENTIFIED"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_VESSEL_DETECTED = "VESSEL_DETECTED"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"

# ── Organization values ───────────────────────────────────────
ORG_INDIAN_NAVY = "INDIAN_NAVY"
ORG_FOREIGN_MILITARY = "FOREIGN_MILITARY"
ORG_CIVILIAN = "CIVILIAN"
ORG_UNKNOWN = "UNKNOWN"


class DecisionEngine:
    """Fuses evidence from multiple sources into a final vessel identification decision."""

    def decide(
        self,
        gemini_result: Optional[dict],
        classifier_label: str = "Unknown",
        classifier_confidence: float = 0.0,
        ocr_texts: Optional[list[str]] = None,
        detection_confidence: float = 0.0,
    ) -> dict:
        """Produce the final decision for a single detected vessel.

        Args:
            gemini_result: Parsed JSON from Gemini Vision API (or None).
            classifier_label: EfficientNet label (baseline fallback).
            classifier_confidence: EfficientNet confidence.
            ocr_texts: List of OCR-detected text strings.
            detection_confidence: YOLO detection confidence.

        Returns:
            Decision dict with vessel_type, identity, organization, status, etc.
        """
        ocr_texts = ocr_texts or []

        # ── Case 1: Gemini provided a result ──────────────────
        if gemini_result and gemini_result.get("vessel_detected"):
            return self._decide_with_gemini(
                gemini_result, classifier_label, classifier_confidence,
                ocr_texts, detection_confidence
            )

        # ── Case 2: Gemini unavailable but YOLO detected something ──
        if detection_confidence > 0:
            return self._decide_without_gemini(
                classifier_label, classifier_confidence,
                ocr_texts, detection_confidence
            )

        # ── Case 3: Nothing detected ──────────────────────────
        return {
            "vessel_type": {"category": None, "subtype": None, "confidence": 0.0},
            "identity": {
                "known": False,
                "ship_id": None,
                "name": None,
                "ship_class": None,
                "pennant_number": None,
                "similarity": None,
            },
            "organization": None,
            "status": STATUS_NO_VESSEL,
            "ocr_text": ocr_texts,
            "visual_evidence": [],
            "description": "No vessel detected in image.",
            "nation": None,
            "gemini_used": False,
        }

    def _decide_with_gemini(
        self,
        gemini: dict,
        classifier_label: str,
        classifier_confidence: float,
        ocr_texts: list[str],
        detection_confidence: float,
    ) -> dict:
        """Build decision when Gemini analysis is available."""
        identity_info = gemini.get("identity", {})
        is_known = identity_info.get("known", False)
        ship_name = identity_info.get("name")
        ship_class = identity_info.get("ship_class")
        pennant_number = identity_info.get("pennant_number")
        organization = gemini.get("organization", ORG_UNKNOWN)
        confidence_level = gemini.get("confidence_level", "low")
        visual_evidence = gemini.get("visual_evidence", [])
        description = gemini.get("description", "")
        nation = gemini.get("nation")

        # Map confidence level to numeric
        confidence_map = {"high": 0.95, "medium": 0.75, "low": 0.50}
        confidence_numeric = confidence_map.get(confidence_level, 0.50)

        # ── OCR corroboration ─────────────────────────────────
        # If OCR found text matching the pennant number, boost confidence
        ocr_corroborates = False
        if pennant_number and ocr_texts:
            for text in ocr_texts:
                if pennant_number.lower() in text.lower().replace(" ", ""):
                    ocr_corroborates = True
                    if confidence_level != "high":
                        confidence_numeric = min(confidence_numeric + 0.10, 0.99)
                    logger.info(
                        "OCR corroborates pennant number: %s found in OCR text '%s'",
                        pennant_number, text,
                    )
                    break

        # ── Determine status ──────────────────────────────────
        if is_known and ship_name and confidence_level in ("high", "medium"):
            status = STATUS_IDENTIFIED
        elif organization in (ORG_INDIAN_NAVY, ORG_FOREIGN_MILITARY):
            status = STATUS_UNKNOWN if not is_known else STATUS_IDENTIFIED
        elif confidence_level == "low":
            status = STATUS_LOW_CONFIDENCE
        else:
            status = STATUS_VESSEL_DETECTED

        return {
            "vessel_type": {
                "category": gemini.get("vessel_type", "Unknown Vessel"),
                "subtype": gemini.get("subtype"),
                "confidence": confidence_numeric,
            },
            "identity": {
                "known": is_known,
                "ship_id": None,  # Could be mapped from metadata DB
                "name": ship_name,
                "ship_class": ship_class,
                "pennant_number": pennant_number,
                "similarity": confidence_numeric,
            },
            "organization": organization,
            "status": status,
            "ocr_text": ocr_texts,
            "ocr_corroborates": ocr_corroborates,
            "visual_evidence": visual_evidence,
            "description": description,
            "nation": nation,
            "confidence_level": confidence_level,
            "gemini_used": True,
            "classifier_label": classifier_label,
            "classifier_confidence": classifier_confidence,
            "detection_confidence": detection_confidence,
        }

    def _decide_without_gemini(
        self,
        classifier_label: str,
        classifier_confidence: float,
        ocr_texts: list[str],
        detection_confidence: float,
    ) -> dict:
        """Fallback decision when Gemini is unavailable — uses EfficientNet only."""
        # Map EfficientNet classes to vessel categories
        military_keywords = ["naval", "patrol", "military", "warship", "destroyer", "frigate"]
        is_military = any(kw in classifier_label.lower() for kw in military_keywords)

        if is_military:
            organization = ORG_UNKNOWN  # Can't determine nationality without Gemini
            category = "Military Vessel"
        elif classifier_label in ("Unknown", "Unknown Vessel Type"):
            organization = ORG_UNKNOWN
            category = "Unknown Vessel"
        else:
            organization = ORG_CIVILIAN
            category = "Commercial Vessel"

        return {
            "vessel_type": {
                "category": category,
                "subtype": classifier_label if classifier_label != "Unknown" else None,
                "confidence": classifier_confidence,
            },
            "identity": {
                "known": False,
                "ship_id": None,
                "name": None,
                "ship_class": None,
                "pennant_number": None,
                "similarity": None,
            },
            "organization": organization,
            "status": STATUS_VESSEL_DETECTED if classifier_label != "Unknown" else STATUS_UNKNOWN,
            "ocr_text": ocr_texts,
            "visual_evidence": [],
            "description": f"Classified as {classifier_label} by EfficientNet baseline (Gemini unavailable).",
            "nation": None,
            "gemini_used": False,
            "classifier_label": classifier_label,
            "classifier_confidence": classifier_confidence,
            "detection_confidence": detection_confidence,
        }


# Module-level singleton
decision_engine = DecisionEngine()
