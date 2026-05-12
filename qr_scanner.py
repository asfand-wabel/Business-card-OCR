from __future__ import annotations
import numpy as np
import cv2

try:
    from pyzbar import pyzbar as _pyzbar
    _PYZBAR_OK = True
except Exception:
    _PYZBAR_OK = False


def scan_qr_codes(image_array: np.ndarray, rotation: int = 0) -> list:
    """
    Fast QR scan: pyzbar (which is fast and reliable) on a small set of
    preprocessing variants. Bails out the moment one decode succeeds.
    """
    for processed, scale in _generate_attempts(image_array):
        results = _decode(processed, image_array, scale)
        if results:
            return results
    return []


# ── Attempt generator (≤4 variants for speed) ────────────────────────────────

def _generate_attempts(image_array: np.ndarray):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    max_dim = max(h, w)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    if max_dim > 2000:
        # Huge phone photos: try a couple of downscale targets — zbar prefers
        # ~1000-2000px but the optimum varies with QR code size.
        for target in (1500, 2000, 1000):
            scale = target / max_dim
            small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            yield small, scale
            yield clahe.apply(small), scale
            _, otsu = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            yield otsu, scale
    else:
        yield gray, 1.0
        yield clahe.apply(gray), 1.0
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield otsu, 1.0
        if max_dim < 1200:
            big = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            yield big, 2.0


# ── Decoder (pyzbar primary, OpenCV fallback) ────────────────────────────────

def _decode(processed, original_rgb, scale):
    if _PYZBAR_OK:
        results = _decode_pyzbar(processed, original_rgb, scale)
        if results:
            return results
    return _decode_opencv(processed, original_rgb, scale)


def _decode_pyzbar(processed, original_rgb, scale):
    img8 = processed if len(processed.shape) == 2 else cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    decoded = _pyzbar.decode(img8)
    out = []
    for obj in decoded:
        if obj.type != "QRCODE":
            continue
        data = obj.data.decode("utf-8", errors="replace")
        if not data:
            continue
        pts = np.array([(p.x, p.y) for p in obj.polygon], dtype=float)
        crop = _crop(original_rgb, pts, scale, processed.shape, original_rgb.shape)
        out.append({"data": data, "type": obj.type, "crop": crop})
    return out


def _decode_opencv(processed, original_rgb, scale):
    detector = cv2.QRCodeDetector()
    try:
        data, points, _ = detector.detectAndDecode(processed)
        if data and points is not None:
            crop = _crop(original_rgb, points[0], scale, processed.shape, original_rgb.shape)
            return [{"data": data, "type": "QRCODE", "crop": crop}]
    except cv2.error:
        pass
    return []


# ── Coordinate mapping ────────────────────────────────────────────────────────

def _crop(original_rgb, pts, scale, processed_shape, original_shape):
    try:
        pts = np.array(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        ph, pw = processed_shape[:2]
        oh, ow = original_shape[:2]

        pts /= scale
        sx = ow / (pw / scale)
        sy = oh / (ph / scale)
        if scale != 1:
            pts[:, 0] *= sx / scale
            pts[:, 1] *= sy / scale

        pts = pts.astype(int)
        PAD = 16
        x1 = max(0, pts[:, 0].min() - PAD)
        x2 = min(ow, pts[:, 0].max() + PAD)
        y1 = max(0, pts[:, 1].min() - PAD)
        y2 = min(oh, pts[:, 1].max() + PAD)
        crop = original_rgb[y1:y2, x1:x2]
        return crop if crop.size > 0 else None
    except Exception:
        return None
