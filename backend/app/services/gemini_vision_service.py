"""gemini_vision_service.py — Gemini API-based vessel identification service.

Uses the google-genai SDK (v2.x) which properly supports the new AQ. API
key format. Falls back gracefully if the API is unavailable.
"""

import json
import logging
import base64
import threading
import concurrent.futures
import requests
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()

VESSEL_ANALYSIS_PROMPT = """You are an expert maritime vessel identification analyst specializing in Indian Navy and international naval vessels.

Analyze this vessel image carefully and return a JSON response with the following structure:

{
  "vessel_detected": true,
  "vessel_type": "<category>",
  "subtype": "<specific type>",
  "organization": "<org>",
  "identity": {
    "known": true/false,
    "name": "<ship name or null>",
    "ship_class": "<class name or null>",
    "pennant_number": "<hull/pennant number or null>"
  },
  "nation": "<country or null>",
  "confidence_level": "<high|medium|low>",
  "visual_evidence": ["<list of visual cues used>"],
  "description": "<brief description of what you observe>"
}

Field definitions:
- vessel_type: One of "Military Vessel", "Commercial Vessel", "Fishing Vessel", "Passenger Vessel", "Support Vessel", "Unknown Vessel"
- subtype: Specific type like "Aircraft Carrier", "Destroyer", "Frigate", "Corvette", "Submarine", "Patrol Vessel", "Container Ship", "Tanker", "Bulk Carrier", "Cruise Ship", "Fishing Trawler", "Tugboat", "Yacht", etc.
- organization: One of "INDIAN_NAVY", "FOREIGN_MILITARY", "CIVILIAN", "UNKNOWN"
- identity.known: true ONLY if you can confidently identify the SPECIFIC ship
- identity.name: Ship name (e.g., "INS Vikrant", "INS Kolkata") or null if unknown
- identity.ship_class: Class name (e.g., "Vikrant-class", "Kolkata-class") or null
- identity.pennant_number: Hull/pennant number visible or known (e.g., "D63", "R11") or null
- nation: Country of origin if identifiable, null otherwise
- confidence_level: "high" if very confident, "medium" if reasonably sure, "low" if uncertain
- visual_evidence: List of specific visual features you used

CRITICAL RULES:
1. Only identify a vessel as Indian Navy if you have strong visual evidence
2. Do NOT force every military vessel into an Indian Navy identity
3. If you cannot identify the specific ship, set known=false and name=null
4. Return "UNKNOWN" organization when evidence is insufficient
5. If NO vessel is visible, return: {"vessel_detected": false, "description": "No vessel detected in image"}
6. Be honest about confidence — use "low" when uncertain
7. Return ONLY valid JSON — no markdown, no code blocks

Known Indian Navy vessels (only confirm if visual evidence matches):
- INS Vikrant (R11), INS Vikramaditya (R33) — Aircraft Carriers
- INS Kolkata (D63), INS Kochi (D64), INS Chennai (D65) — Kolkata-class destroyers
- INS Visakhapatnam (D66), INS Mormugao (D67) — Visakhapatnam-class destroyers
- INS Delhi (D61), INS Mysore (D60), INS Mumbai (D62) — Delhi-class destroyers
- INS Shivalik (F47), INS Satpura (F48), INS Sahyadri (F49) — Shivalik-class frigates
- INS Talwar (F40), INS Trishul (F43), INS Tabar (F44) — Talwar-class frigates
- INS Nilgiri (F41) — Nilgiri-class (Project 17A) frigate
- INS Kamorta (P28), INS Kadmatt (P29), INS Kiltan (P30), INS Kavaratti (P31) — Kamorta-class corvettes
- INS Arihant (S2), INS Arighat (S3) — Arihant-class SSBNs
- INS Kalvari (S21) through INS Vagsheer (S26) — Kalvari-class (Scorpene) submarines

Return ONLY the JSON object."""


class GeminiVisionService:
    """Analyzes vessel images using google-genai SDK v2.x."""

    def __init__(self):
        self._client = None
        self._initialized: bool = False
        self._available: bool = False

    def initialize(self):
        """Initialize. Thread-safe and idempotent."""
        if self._initialized:
            return
        with _init_lock:
            if self._initialized:
                return
            if not settings.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not set — Gemini DISABLED.")
                self._initialized = True
                self._available = False
                return
            if not settings.GEMINI_ENABLED:
                logger.info("Gemini disabled via GEMINI_ENABLED=false.")
                self._initialized = True
                self._available = False
                return
            try:
                from google import genai
                self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
                self._available = True
                self._initialized = True
                logger.info(
                    "Gemini Vision Service ready — model=%s (google-genai SDK v2)",
                    settings.GEMINI_MODEL,
                )
            except Exception as exc:
                logger.exception("Failed to initialize Gemini client: %s", exc)
                self._initialized = True
                self._available = False

    @property
    def is_available(self) -> bool:
        self.initialize()
        return self._available

    def analyze_vessel(self, image: np.ndarray, context: str = "") -> Optional[dict]:
        """Analyze a vessel image using Gemini. Returns parsed JSON dict or None."""
        self.initialize()
        if not self._available:
            return None

        try:
            from google.genai import types

            # Encode image to JPEG bytes
            success, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("Failed to encode image to JPEG")
                return None

            image_bytes = buffer.tobytes()

            # Build prompt
            prompt = VESSEL_ANALYSIS_PROMPT
            if context:
                prompt += f"\n\nAdditional context from OCR: {context}"

            logger.info("Sending image to Gemini (%d KB)...", len(image_bytes) // 1024)

            # Enforce strict timeout and fallback using ThreadPoolExecutor
            response = None
            gemini_failed = False
            
            def _call_gemini():
                # Internal retry logic just for transient Google errors
                max_retries = 3
                retry_delay = 2
                for attempt in range(max_retries):
                    try:
                        return self._client.models.generate_content(
                            model=settings.GEMINI_MODEL,
                            contents=[
                                types.Content(
                                    parts=[
                                        types.Part.from_text(text=prompt),
                                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                                    ]
                                )
                            ],
                            config=types.GenerateContentConfig(
                                temperature=0.1,
                                max_output_tokens=1024,
                            ),
                        )
                    except Exception as api_exc:
                        err_msg = str(api_exc)
                        if "503" in err_msg or "429" in err_msg:
                            if attempt < max_retries - 1:
                                logger.warning(f"Gemini API returned {err_msg[:50]} (Attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds...")
                                import time
                                time.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                        raise
                return None

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_call_gemini)
                    response = future.result(timeout=settings.GEMINI_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logger.error(f"Gemini API timed out after {settings.GEMINI_TIMEOUT} seconds.")
                gemini_failed = True
            except Exception as e:
                logger.error("Gemini API call failed completely: %s", e)
                gemini_failed = True
                
            if gemini_failed or not response or not response.text:
                logger.warning("Gemini failed or returned empty. Executing OpenRouter Fallback...")
                return self._analyze_with_openrouter(image_bytes, prompt)

            raw_text = response.text.strip()

            # Strip markdown code blocks if present
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw_text = "\n".join(lines)

            result = json.loads(raw_text)
            logger.info(
                "Gemini OK — detected=%s type=%s org=%s name=%s",
                result.get("vessel_detected"), result.get("vessel_type"),
                result.get("organization"), result.get("identity", {}).get("name"),
            )
            return result

        except json.JSONDecodeError as exc:
            logger.error("Gemini returned invalid JSON: %s", exc)
            return None
        except Exception as exc:
            logger.exception("Gemini vessel analysis failed: %s", exc)
            return None

    def analyze_full_image(self, image: np.ndarray) -> Optional[dict]:
        """Analyze the full image for all vessels."""
        return self.analyze_vessel(image)

    def _analyze_with_openrouter(self, image_bytes: bytes, prompt: str) -> Optional[dict]:
        """Fallback to OpenRouter API (Free Model) if Gemini fails."""
        if not settings.OPENROUTER_API_KEY:
            logger.error("OpenRouter API key not configured. Fallback failed.")
            return None
        
        fallback_models = [
            "google/gemma-4-31b-it:free",
            "google/gemma-4-26b-a4b-it:free",
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "openrouter/free"
        ]
        
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{base64_image}"
        
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "SVACS",
            "Content-Type": "application/json"
        }
        
        for model_name in fallback_models:
            logger.info("Initiating OpenRouter fallback (Model: %s)...", model_name)
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1
            }
            
            try:
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
                if resp.status_code != 200:
                    logger.warning("OpenRouter model %s failed: %s %s", model_name, resp.status_code, resp.text[:100])
                    continue
                    
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"].strip()
                
                # Clean markdown
                if raw_text.startswith("```"):
                    lines = raw_text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw_text = "\n".join(lines)
                
                result = json.loads(raw_text)
                logger.info("OpenRouter Fallback OK (Model: %s) — detected=%s type=%s", model_name, result.get("vessel_detected"), result.get("vessel_type"))
                return result
            except json.JSONDecodeError:
                logger.warning("OpenRouter model %s returned invalid JSON.", model_name)
                continue
            except Exception as e:
                logger.warning("OpenRouter model %s request failed: %s", model_name, e)
                continue
                
        logger.error("All OpenRouter fallback models failed.")
        return None


# Module-level singleton
gemini_vision_service = GeminiVisionService()
