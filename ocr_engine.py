from __future__ import annotations
import os
import math
import numpy as np
import cv2
from PIL import Image
import io

_vision_client = None
MAX_OCR_DIM = 1600


def _get_vision_client():
    global _vision_client
    if _vision_client is None:
        from google.cloud import vision
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            from google.api_core.client_options import ClientOptions
            _vision_client = vision.ImageAnnotatorClient(
                client_options=ClientOptions(api_key=api_key)
            )
        else:
            _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


# ── Image helpers ─────────────────────────────────────────────────────────────

def _array_to_bytes(image_array: np.ndarray) -> bytes:
    pil = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _rotate_image(image_array: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image_array
    if degrees == 90:
        return cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image_array, cv2.ROTATE_180)
    return cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _resize(image_array: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image_array.shape[:2]
    if max(h, w) <= max_dim:
        return image_array
    scale = max_dim / max(h, w)
    return cv2.resize(image_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _infer_rotation_from_response(response) -> int:
    """
    Google Vision reads text at any orientation. We can derive the card's
    rotation for display purposes from the bounding-box vertex order of the
    detected text (cheap — uses data we already have, no extra API call).
    """
    if not response.full_text_annotation or not response.full_text_annotation.pages:
        return 0
    angles = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            verts = block.bounding_box.vertices
            if len(verts) < 2:
                continue
            dx = verts[1].x - verts[0].x
            dy = verts[1].y - verts[0].y
            angle = math.degrees(math.atan2(dy, dx))
            angles.append(angle)
    if not angles:
        return 0
    avg = sum(angles) / len(angles)
    # Snap to nearest 90°
    if -45 <= avg < 45:
        return 0
    if 45 <= avg < 135:
        return 270  # text leans down-right → card rotated 270° clockwise
    if avg >= 135 or avg < -135:
        return 180
    return 90


# ── Public API ────────────────────────────────────────────────────────────────

def detect_best_rotation(image_array: np.ndarray) -> int:
    """Backwards-compat shim — actual rotation now derived from Vision response."""
    return 0


def extract_text(image_array: np.ndarray, auto_rotate: bool = True):
    """
    Returns (extracted_text, rotation_for_display).

    Single Google Vision API call — Google reads text at any orientation, and
    we derive rotation from the response's bounding boxes for display only.
    """
    from google.cloud import vision

    image_array = _resize(image_array, MAX_OCR_DIM)
    image_bytes = _array_to_bytes(image_array)

    client = _get_vision_client()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Google Vision API error: {response.error.message}")

    full_text = response.full_text_annotation.text if response.full_text_annotation else ""
    rotation = _infer_rotation_from_response(response) if auto_rotate else 0
    return full_text.strip(), rotation
