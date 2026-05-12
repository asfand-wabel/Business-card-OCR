import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

from PIL import Image
import numpy as np
from card_segmenter import segment_cards
from ocr_engine import extract_text
from qr_scanner import scan_qr_codes
from parser import parse_business_cards

images = sorted([
    f"/Users/apple/Downloads/{f}"
    for f in os.listdir("/Users/apple/Downloads")
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

for path in images:
    print(f"\n{'='*60}")
    print(f"IMAGE: {os.path.basename(path)}")
    print('='*60)
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    cards = segment_cards(arr)
    print(f"CARDS DETECTED: {len(cards)}")

    for card_arr, label in cards:
        print(f"\n  --- {label} ---")
        raw, rotation = extract_text(card_arr, auto_rotate=True)
        if rotation:
            print(f"  Auto-rotated: {rotation}°")
        print("  RAW OCR:\n" + raw)

        contacts = parse_business_cards(raw)
        for ci, parsed in enumerate(contacts, 1):
            tag = f" (contact {ci})" if len(contacts) > 1 else ""
            print(f"\n  PARSED{tag}:")
            for k, v in parsed.items():
                if v:
                    print(f"    {k:12s}: {v}")

        qrs = scan_qr_codes(card_arr, rotation=rotation)
        if qrs:
            print(f"\n  QR CODES ({len(qrs)} found):")
            for q in qrs:
                print(f"    [{q['type']}] {q['data'][:120]}")
        else:
            print("\n  QR: none detected")
