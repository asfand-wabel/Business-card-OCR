import numpy as np
import cv2


def segment_cards(image_array: np.ndarray) -> list:
    """
    Detects individual business cards in a photo that may contain one or more.
    Returns a list of (crop_array, label) tuples.
    Falls back to [(image_array, "Card")] when only one card is found.
    """
    cards = _find_card_regions(image_array)
    if len(cards) >= 2:
        return [(crop, f"Card {i + 1}") for i, (_, _, crop) in enumerate(cards)]
    return [(image_array, "Card")]


def _find_card_regions(image_array: np.ndarray) -> list:
    h, w = image_array.shape[:2]
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Estimate background colour from image corners
    margin = max(25, min(h, w) // 30)
    corner_patches = [
        gray[:margin, :margin],
        gray[:margin, w - margin:],
        gray[h - margin:, :margin],
        gray[h - margin:, w - margin:],
    ]
    bg_val = int(np.median([np.median(p) for p in corner_patches]))

    # Mask pixels that differ significantly from background
    diff = cv2.absdiff(gray, np.full_like(gray, bg_val))
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Close gaps inside cards, then open to remove small noise
    close_k = max(15, min(h, w) // 35)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8), iterations=3
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((max(5, close_k // 3), max(5, close_k // 3)), np.uint8)
    )

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = h * w
    candidates = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 0.04 * img_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        long_s = max(cw, ch)
        short_s = min(cw, ch)
        if short_s == 0:
            continue
        aspect = long_s / short_s
        # Business cards are roughly 1.4–3.5 in aspect ratio (landscape or portrait)
        if not (1.2 <= aspect <= 4.0):
            continue
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        candidates.append((y, x, image_array[y1:y2, x1:x2]))

    # Sort top-to-bottom then left-to-right
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates
