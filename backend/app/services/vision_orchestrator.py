"""vision_orchestrator.py — Orchestrates the full SVACS vision pipeline.

Pipeline (v2.0):
  1. Preprocessing — decode and validate image
  2. OCR extraction — EasyOCR text detection (non-fatal)
  3. YOLO detection — vessel bounding boxes
  4. For each detection:
     a. EfficientNet classification (baseline)
     b. Gemini Vision API analysis (ship identification)
     c. Decision engine fusion
  5. Explainability overlay
  6. Replay save
  7. Bucket write

Every stage is wrapped in its own try/except so a single step failure
produces a clear log entry and gracefully degrades.
"""
import logging
import traceback
import uuid
import time
import numpy as np
import cv2

from app.models.schemas import (
    VisionAnalysisRequest,
    VisionAnalysisResponse,
    VesselDetectionItem,
    VesselTypeResult,
    VesselIdentityResult,
    BoundingBox,
    TopPrediction,
)
from app.services.preprocessing import (
    decode_base64_image,
    preprocess_for_inference,
    encode_image_base64,
    decode_image_bytes,
)
from app.services.ocr_service import ocr_service
from app.services.inference_service import inference_service
from app.services.gemini_vision_service import gemini_vision_service
from app.services.decision_engine import decision_engine
from app.services.explainability import draw_evidence, draw_enhanced_evidence
from app.services.replay_service import replay_service
from app.services.bucket_client import write_vision_artifact

logger = logging.getLogger(__name__)


class VisionOrchestrator:
    # ------------------------------------------------------------------
    # Enhanced pipeline with Gemini + Identity
    # ------------------------------------------------------------------

    def process_image_enhanced(
        self, raw_image: np.ndarray, return_explainable_image: bool, quick: bool = False
    ) -> dict:
        """Run the full enhanced pipeline: YOLO → Gemini → OCR → Decision Engine.

        Returns a dict matching the EnhancedAnalysisResponse schema.
        """
        replay_id = str(uuid.uuid4())
        timings = {}

        if raw_image is None or raw_image.size == 0:
            logger.error("process_image_enhanced() received an empty/None image")
            raise ValueError("Image could not be decoded — the uploaded file may be empty or corrupt.")

        logger.info(
            "process_image_enhanced() start — shape=%s dtype=%s replay_id=%s",
            raw_image.shape, raw_image.dtype, replay_id,
        )

        # ── Stage 1: Preprocessing ────────────────────────────
        t0 = time.time()
        try:
            processed_image = preprocess_for_inference(raw_image)
            logger.info("Stage 1 (preprocessing) OK — output shape: %s", processed_image.shape)
        except Exception as exc:
            logger.exception("Stage 1 (preprocessing) FAILED: %s", exc)
            raise RuntimeError(f"Preprocessing failed: {exc}") from exc
        timings["preprocessing_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Stage 2: OCR extraction (non-fatal) ───────────────
        t0 = time.time()
        ocr_results = []
        all_ocr_texts = []
        try:
            ocr_results = [] if quick else ocr_service.extract_text(processed_image)
            all_ocr_texts = [r.text for r in ocr_results]
            logger.info("Stage 2 (OCR) OK — %d results: %s", len(ocr_results), all_ocr_texts)
        except Exception as exc:
            logger.exception("Stage 2 (OCR) FAILED (non-fatal): %s", exc)
        timings["ocr_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Stage 3: YOLO detection ───────────────────────────
        t0 = time.time()
        detections = []
        try:
            detections = inference_service.detect(processed_image, quick=quick)
            logger.info("Stage 3 (YOLO detection) OK — %d detection(s)", len(detections))
        except Exception as exc:
            logger.exception("Stage 3 (YOLO detection) FAILED: %s", exc)
            # Don't raise — try Gemini on the full image instead
        timings["detection_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Stage 4: For each detection → Gemini + Decision ───
        t0 = time.time()
        enhanced_detections: list[VesselDetectionItem] = []

        if detections:
            for det in detections:
                try:
                    enhanced = self._process_single_detection(
                        processed_image, det, all_ocr_texts
                    )
                    enhanced_detections.append(enhanced)
                except Exception as exc:
                    logger.exception("Failed to process detection: %s", exc)
                    # Fallback: keep the basic detection
                    enhanced_detections.append(
                        VesselDetectionItem(
                            bounding_box=det.bounding_box,
                            detection_confidence=det.confidence,
                            label=det.label,
                            confidence=det.confidence,
                            top_predictions=det.top_predictions,
                            status="VESSEL_DETECTED",
                        )
                    )
        else:
            # No YOLO detections — try Gemini on the full image
            try:
                logger.info("No YOLO detections — running Gemini on full image...")
                gemini_result = gemini_vision_service.analyze_full_image(processed_image)
                if gemini_result and gemini_result.get("vessel_detected"):
                    h, w = processed_image.shape[:2]
                    decision = decision_engine.decide(
                        gemini_result=gemini_result,
                        ocr_texts=all_ocr_texts,
                    )
                    enhanced_detections.append(
                        self._build_detection_item(
                            bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=float(w), y_max=float(h)),
                            detection_confidence=0.0,
                            decision=decision,
                            classifier_label="Unknown",
                            classifier_confidence=0.0,
                            top_predictions=[],
                        )
                    )
                    logger.info("Gemini found vessel in full image: %s", decision.get("status"))
            except Exception as exc:
                logger.exception("Full-image Gemini analysis failed: %s", exc)

        timings["identification_ms"] = round((time.time() - t0) * 1000, 1)

        # ── Stage 5: Explainability overlay ────────────────────
        explainable_base64 = None
        if return_explainable_image:
            try:
                explained_image = draw_enhanced_evidence(
                    raw_image, enhanced_detections, ocr_results
                )
                explainable_base64 = encode_image_base64(explained_image)
                logger.info("Stage 5 (explainability) OK")
            except Exception as exc:
                logger.exception("Stage 5 (explainability) FAILED (non-fatal): %s", exc)
                # Fallback to basic overlay
                try:
                    explained_image = draw_evidence(raw_image, detections, ocr_results)
                    explainable_base64 = encode_image_base64(explained_image)
                except Exception:
                    pass

        # ── Stage 6: Build response ────────────────────────────
        vessel_detected = len(enhanced_detections) > 0
        best_detection = None
        if enhanced_detections:
            # Pick the most important detection (identified > unknown > detected)
            status_priority = {"IDENTIFIED": 3, "UNKNOWN": 2, "VESSEL_DETECTED": 1, "LOW_CONFIDENCE": 0}
            best_detection = max(
                enhanced_detections,
                key=lambda d: status_priority.get(d.status, 0),
            )

        # Backward-compat fields
        vessel_class = best_detection.label if best_detection else "Unknown"
        confidence_score = best_detection.confidence if best_detection else 0.0
        if best_detection and best_detection.identity and best_detection.identity.known:
            vessel_class = best_detection.identity.name or vessel_class

        # Explanation text
        explanation_list = []
        if best_detection:
            if best_detection.status == "IDENTIFIED":
                explanation_list.append(
                    f"Identified as {best_detection.identity.name} "
                    f"({best_detection.identity.ship_class}) — {best_detection.organization}"
                )
            elif best_detection.status == "UNKNOWN":
                explanation_list.append(
                    f"Vessel detected but identity unknown — {best_detection.vessel_type.category or 'Unknown type'}"
                )
            else:
                explanation_list.append(
                    f"Detected: {best_detection.vessel_type.subtype or best_detection.label}"
                )
            if best_detection.description:
                explanation_list.append(best_detection.description)
            for ev in (best_detection.visual_evidence or [])[:3]:
                explanation_list.append(f"Visual evidence: {ev}")
        else:
            explanation_list.append("No vessel detected in the image.")

        validation_status = "OK"
        if not vessel_detected:
            validation_status = "FLAG"
        elif best_detection and best_detection.status == "UNKNOWN":
            validation_status = "FLAG"

        dyn_classification_source = "YOLO + EfficientNet (AI unavailable)"
        if best_detection and best_detection.llm_provider:
             dyn_classification_source = f"{best_detection.llm_provider} + YOLO + EfficientNet"
        elif gemini_vision_service.is_available:
             dyn_classification_source = "Gemini Vision API + YOLO + EfficientNet"

        result = {
            "success": True,
            "trace_id": replay_id,
            "vessel_detected": vessel_detected,
            "total_detections": len(enhanced_detections),
            "detections": [d.model_dump() for d in enhanced_detections],
            "ocr_results": [o.model_dump() for o in ocr_results],
            "explainable_image_base64": explainable_base64,
            "classification_source": dyn_classification_source,
            "model_version": "2.0.0",
            # Backward compatibility
            "validation_status": validation_status,
            "vessel_class": vessel_class,
            "confidence_score": confidence_score,
            "risk_level": "LOW",
            "explanation": explanation_list,
            "timings": timings,
        }

        # ── Stage 7: Replay save (non-fatal) ──────────────────
        try:
            legacy_response = VisionAnalysisResponse(
                replay_id=replay_id,
                detections=detections,
                ocr_results=ocr_results,
                explainable_image_base64=explainable_base64,
            )
            replay_service.save_replay(raw_image, legacy_response)
        except Exception as exc:
            logger.warning("Replay save FAILED (non-fatal): %s", exc)

        # ── Stage 8: Bucket write (non-fatal) ─────────────────
        try:
            bucket_payload = {
                "detections": result["detections"],
                "ocr_results": result["ocr_results"],
            }
            write_vision_artifact(
                trace_id=replay_id,
                artifact_type="vision_detection",
                payload=bucket_payload,
            )
        except Exception as exc:
            logger.warning("Bucket write FAILED (non-fatal): %s", exc)

        logger.info(
            "process_image_enhanced() completed — trace_id=%s vessel_detected=%s total=%d",
            replay_id, vessel_detected, len(enhanced_detections),
        )
        return result

    def _process_single_detection(
        self,
        image: np.ndarray,
        detection,
        ocr_texts: list[str],
    ) -> VesselDetectionItem:
        """Process a single YOLO detection through Gemini + Decision Engine."""
        bbox = detection.bounding_box

        # Crop the vessel region (with padding)
        h, w = image.shape[:2]
        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(w, int(bbox.x_max))
        y2 = min(h, int(bbox.y_max))

        # Add 10% padding
        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = image[y1:y2, x1:x2]

        # ── Gemini analysis ────────────────────────────────────
        gemini_result = None
        if crop.size > 0:
            ocr_context = ", ".join(ocr_texts) if ocr_texts else ""
            gemini_result = gemini_vision_service.analyze_vessel(crop, context=ocr_context)

        # ── Decision engine ────────────────────────────────────
        decision = decision_engine.decide(
            gemini_result=gemini_result,
            classifier_label=detection.label,
            classifier_confidence=detection.confidence,
            ocr_texts=ocr_texts,
            detection_confidence=detection.confidence,
        )

        return self._build_detection_item(
            bbox=detection.bounding_box,
            detection_confidence=detection.confidence,
            decision=decision,
            classifier_label=detection.label,
            classifier_confidence=detection.confidence,
            top_predictions=detection.top_predictions,
        )

    def _build_detection_item(
        self,
        bbox: BoundingBox,
        detection_confidence: float,
        decision: dict,
        classifier_label: str,
        classifier_confidence: float,
        top_predictions: list,
    ) -> VesselDetectionItem:
        """Build a VesselDetectionItem from a decision dict."""
        vt = decision.get("vessel_type", {})
        ident = decision.get("identity", {})

        return VesselDetectionItem(
            bounding_box=bbox,
            detection_confidence=detection_confidence,
            vessel_type=VesselTypeResult(
                category=vt.get("category"),
                subtype=vt.get("subtype"),
                confidence=vt.get("confidence", 0.0),
            ),
            identity=VesselIdentityResult(
                known=ident.get("known", False),
                ship_id=ident.get("ship_id"),
                name=ident.get("name"),
                ship_class=ident.get("ship_class"),
                pennant_number=ident.get("pennant_number"),
                similarity=ident.get("similarity"),
            ),
            ocr_text=decision.get("ocr_text", []),
            organization=decision.get("organization"),
            status=decision.get("status", "UNKNOWN"),
            nation=decision.get("nation"),
            visual_evidence=decision.get("visual_evidence", []),
            description=decision.get("description"),
            confidence_level=decision.get("confidence_level"),
            label=classifier_label,
            confidence=classifier_confidence,
            top_predictions=top_predictions,
            llm_provider=decision.get("llm_provider"),
        )

    # ------------------------------------------------------------------
    # Legacy pipeline (preserved for /api/v1/analyze compatibility)
    # ------------------------------------------------------------------

    def process_image(
        self, raw_image: np.ndarray, return_explainable_image: bool, quick: bool = False
    ) -> VisionAnalysisResponse:
        """Run the legacy pipeline on a decoded BGR numpy image."""
        replay_id = str(uuid.uuid4())

        if raw_image is None or raw_image.size == 0:
            logger.error("process_image() received an empty/None image (replay_id=%s)", replay_id)
            raise ValueError("Image could not be decoded — the uploaded file may be empty or corrupt.")

        logger.info("process_image() start — shape=%s replay_id=%s", raw_image.shape, replay_id)

        try:
            processed_image = preprocess_for_inference(raw_image)
        except Exception as exc:
            raise RuntimeError(f"Preprocessing failed: {exc}") from exc

        ocr_results = []
        try:
            ocr_results = [] if quick else ocr_service.extract_text(processed_image)
        except Exception as exc:
            logger.exception("OCR FAILED (non-fatal): %s", exc)

        try:
            detections = inference_service.detect(processed_image, quick=quick)
        except Exception as exc:
            raise RuntimeError(f"Vessel detection failed: {exc}") from exc

        explainable_base64 = None
        if return_explainable_image:
            try:
                explained_image = draw_evidence(raw_image, detections, ocr_results)
                explainable_base64 = encode_image_base64(explained_image)
            except Exception as exc:
                logger.exception("Explainability FAILED (non-fatal): %s", exc)

        response = VisionAnalysisResponse(
            replay_id=replay_id,
            detections=detections,
            ocr_results=ocr_results,
            explainable_image_base64=explainable_base64,
        )

        try:
            replay_service.save_replay(raw_image, response)
        except Exception as exc:
            logger.warning("Replay save FAILED (non-fatal): %s", exc)

        try:
            bucket_payload = {
                "detections": [d.model_dump() for d in detections],
                "ocr_results": [o.model_dump() for o in ocr_results],
            }
            write_vision_artifact(trace_id=replay_id, artifact_type="vision_detection", payload=bucket_payload)
        except Exception as exc:
            logger.warning("Bucket write FAILED (non-fatal): %s", exc)

        return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, request: VisionAnalysisRequest) -> VisionAnalysisResponse:
        """Accepts a base64-encoded image from the VisionAnalysisRequest schema."""
        logger.info("process() called via base64 path")
        try:
            raw_image = decode_base64_image(request.image_base64)
        except Exception as exc:
            raise ValueError(f"Invalid base64 image: {exc}") from exc
        return self.process_image(raw_image, request.return_explainable_image)

    def process_bytes(
        self, image_bytes: bytes, return_explainable_image: bool, quick: bool = False
    ) -> VisionAnalysisResponse:
        """Accepts raw image bytes — LEGACY endpoint compatibility."""
        logger.info("process_bytes() called — %d bytes", len(image_bytes))
        if not image_bytes:
            raise ValueError("Uploaded file is empty — no image data received.")
        try:
            raw_image = decode_image_bytes(image_bytes)
        except Exception as exc:
            raise ValueError(f"OpenCV could not decode the uploaded image: {exc}") from exc
        if raw_image is None or raw_image.size == 0:
            raise ValueError("OpenCV decoded an empty image.")
        return self.process_image(raw_image, return_explainable_image, quick=quick)

    def process_bytes_enhanced(
        self, image_bytes: bytes, return_explainable_image: bool, quick: bool = False
    ) -> dict:
        """Accepts raw image bytes — ENHANCED pipeline with Gemini identification."""
        logger.info("process_bytes_enhanced() called — %d bytes", len(image_bytes))
        if not image_bytes:
            raise ValueError("Uploaded file is empty — no image data received.")
        try:
            raw_image = decode_image_bytes(image_bytes)
        except Exception as exc:
            raise ValueError(f"OpenCV could not decode the uploaded image: {exc}") from exc
        if raw_image is None or raw_image.size == 0:
            raise ValueError("OpenCV decoded an empty image.")
        return self.process_image_enhanced(raw_image, return_explainable_image, quick=quick)


vision_orchestrator = VisionOrchestrator()
