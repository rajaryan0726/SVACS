"""explainability.py — Draws bounding boxes and identity labels on images.

Two modes:
  1. draw_evidence() — Original overlay (red detection boxes, green OCR boxes)
  2. draw_enhanced_evidence() — Color-coded identity overlay with ship names
"""
import cv2
import numpy as np
from typing import List

from app.models.schemas import DetectionResult, OCRResult, VesselDetectionItem


def draw_evidence(image: np.ndarray, detections: List[DetectionResult], ocr_results: List[OCRResult]) -> np.ndarray:
    """Draws bounding boxes and labels on the image for explainability (legacy)."""
    output_image = image.copy()

    # Draw detections (red boxes)
    for det in detections:
        bbox = det.bounding_box
        cv2.rectangle(
            output_image,
            (int(bbox.x_min), int(bbox.y_min)),
            (int(bbox.x_max), int(bbox.y_max)),
            (0, 0, 255),
            2
        )
        label_text = f"{det.label} {det.confidence:.2f}"
        cv2.putText(
            output_image,
            label_text,
            (int(bbox.x_min), int(bbox.y_min) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    # Draw OCR results (green boxes)
    for ocr in ocr_results:
        bbox = ocr.bounding_box
        cv2.rectangle(
            output_image,
            (int(bbox.x_min), int(bbox.y_min)),
            (int(bbox.x_max), int(bbox.y_max)),
            (0, 255, 0),
            2
        )
        label_text = f"OCR: {ocr.text}"
        cv2.putText(
            output_image,
            label_text,
            (int(bbox.x_min), int(bbox.y_max) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return output_image


# ── Status → Color mapping ─────────────────────────────────────────
STATUS_COLORS = {
    "IDENTIFIED":      (0, 220, 0),     # Green — known Indian Navy ship
    "UNKNOWN":         (0, 180, 255),    # Orange — military but unknown
    "VESSEL_DETECTED": (255, 200, 0),    # Cyan — civilian vessel
    "LOW_CONFIDENCE":  (100, 100, 255),  # Light red — low confidence
    "NO_VESSEL":       (128, 128, 128),  # Gray
}

ORG_COLORS = {
    "INDIAN_NAVY":     (0, 220, 0),     # Green
    "FOREIGN_MILITARY": (0, 140, 255),   # Orange
    "CIVILIAN":        (255, 200, 0),    # Cyan/yellow
    "UNKNOWN":         (180, 180, 180),  # Gray
}


def draw_enhanced_evidence(
    image: np.ndarray,
    detections: List[VesselDetectionItem],
    ocr_results: List[OCRResult],
) -> np.ndarray:
    """Draw color-coded bounding boxes with ship identity labels.

    Colors:
      - Green: IDENTIFIED Indian Navy vessel
      - Orange: UNKNOWN military vessel
      - Cyan: CIVILIAN vessel
      - Gray: Insufficient information
    """
    output_image = image.copy()
    img_h, img_w = output_image.shape[:2]

    # Scale font/line thickness based on image size
    scale = max(img_w, img_h) / 1000.0
    thickness = max(2, int(scale * 2))
    font_scale = max(0.4, scale * 0.5)
    label_pad = max(5, int(scale * 5))

    for i, det in enumerate(detections):
        bbox = det.bounding_box
        x1, y1 = int(bbox.x_min), int(bbox.y_min)
        x2, y2 = int(bbox.x_max), int(bbox.y_max)

        # Color based on organization/status
        org = det.organization or "UNKNOWN"
        color = ORG_COLORS.get(org, (180, 180, 180))

        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        lines = []

        # Line 1: Identity or vessel type
        if det.identity and det.identity.known and det.identity.name:
            lines.append(det.identity.name)
        elif det.vessel_type and det.vessel_type.subtype:
            lines.append(det.vessel_type.subtype)
        elif det.label and det.label != "Unknown":
            lines.append(det.label)

        # Line 2: Organization + Status
        org_display = {
            "INDIAN_NAVY": "Indian Navy",
            "FOREIGN_MILITARY": "Foreign Military",
            "CIVILIAN": "Civilian",
            "UNKNOWN": "Unknown",
        }.get(org, org)

        status_display = det.status or "UNKNOWN"
        lines.append(f"{org_display} | {status_display}")

        # Line 3: Ship class or pennant
        if det.identity and det.identity.ship_class:
            class_info = det.identity.ship_class
            if det.identity.pennant_number:
                class_info += f" ({det.identity.pennant_number})"
            lines.append(class_info)

        # Draw label background + text above the box
        y_text = y1
        for line_idx, line in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_y = y_text - label_pad - (len(lines) - line_idx - 1) * (th + label_pad)

            if text_y - th - label_pad < 0:
                # Put text inside the box if no room above
                text_y = y1 + (line_idx + 1) * (th + label_pad) + label_pad

            # Background rectangle
            bg_y1 = text_y - th - 4
            bg_y2 = text_y + 4
            bg_x1 = x1
            bg_x2 = x1 + tw + 8

            # Semi-transparent background
            overlay = output_image.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.addWeighted(overlay, 0.6, output_image, 0.4, 0, output_image)

            # Text
            cv2.putText(
                output_image, line,
                (x1 + 4, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                max(1, thickness - 1),
                cv2.LINE_AA,
            )

    # Draw OCR results (green boxes, thinner)
    for ocr in ocr_results:
        bbox = ocr.bounding_box
        cv2.rectangle(
            output_image,
            (int(bbox.x_min), int(bbox.y_min)),
            (int(bbox.x_max), int(bbox.y_max)),
            (0, 255, 0),
            max(1, thickness - 1),
        )
        cv2.putText(
            output_image,
            f"OCR: {ocr.text}",
            (int(bbox.x_min), int(bbox.y_max) + int(15 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.8,
            (0, 255, 0),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    return output_image
