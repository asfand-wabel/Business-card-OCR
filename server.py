"""Minimal web server for the Business Card Scanner — Python standard library only.

Serves a single static page (index.html) and one JSON endpoint (/scan) that runs
the OCR + parsing + QR pipeline. No web framework.

Run:  python server.py        →  http://127.0.0.1:8000
"""
from __future__ import annotations
import io
import re
import json
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

from card_segmenter import segment_cards
from ocr_engine import extract_text
from qr_scanner import scan_qr_codes
from parser import parse_business_cards

HOST, PORT = "127.0.0.1", 8000
MAX_UPLOAD = 25 * 1024 * 1024  # 25 MB


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_data_url(arr, max_dim: int = 900):
    """Encode a numpy RGB image as a (down-scaled) JPEG data: URL."""
    if arr is None or getattr(arr, "size", 0) == 0:
        return None
    pil = Image.fromarray(arr).convert("RGB")
    if max(pil.size) > max_dim:
        s = max_dim / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_multipart(body: bytes, boundary: bytes) -> dict:
    """Tiny multipart/form-data parser. File fields → bytes, text fields → str."""
    out: dict = {}
    for chunk in body.split(b"--" + boundary):
        if not chunk or chunk in (b"--", b"--\r\n", b"\r\n"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        if b"\r\n\r\n" not in chunk:
            continue
        raw_headers, content = chunk.split(b"\r\n\r\n", 1)
        cd = ""
        for line in raw_headers.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition:"):
                cd = line.decode("latin-1", "replace")
                break
        m = re.search(r'name="([^"]*)"', cd)
        if not m:
            continue
        name = m.group(1)
        if 'filename="' in cd:
            out[name] = content                                   # raw bytes
        else:
            out[name] = content.decode("utf-8", "replace").strip()
    return out


def run_pipeline(pil_image: Image.Image, default_region):
    """Full OCR → parse → QR pipeline for one uploaded image."""
    arr = np.array(pil_image.convert("RGB"))
    cards = segment_cards(arr)
    out_cards = []
    for card_arr, label in cards:
        raw_text, rotation = extract_text(card_arr, auto_rotate=True)
        qr_results = scan_qr_codes(card_arr, rotation=rotation)
        contacts = parse_business_cards(raw_text, default_region=default_region)
        out_cards.append({
            "label": label,
            "preview": _to_data_url(card_arr),
            "rotation": rotation,
            "raw_text": raw_text,
            "contacts": contacts,
            "qr_codes": [
                {
                    "data": q.get("data", ""),
                    "type": q.get("type", "QRCODE"),
                    "crop": _to_data_url(q.get("crop"), max_dim=340),
                }
                for q in qr_results
            ],
        })
    return {"cards": out_cards}


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "BusinessCardScanner/1.0"

    # -- responders -----------------------------------------------------------
    def _json(self, obj, status: int = 200):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _static(self, path: str, content_type: str):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes ---------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._static("index.html", "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json({"ok": True})
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        if self.path != "/scan":
            self.send_error(404, "Not found")
            return

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._json({"error": "Expected a multipart/form-data upload."}, 400)
            return
        m = re.search(r"boundary=([^;]+)", ctype)
        if not m:
            self._json({"error": "Missing multipart boundary."}, 400)
            return
        boundary = m.group(1).strip().strip('"').encode("latin-1")

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._json({"error": "Empty request body."}, 400)
            return
        if length > MAX_UPLOAD:
            self._json({"error": "Image too large (max 25 MB)."}, 413)
            return

        body = self.rfile.read(length)
        fields = _parse_multipart(body, boundary)

        image_bytes = fields.get("image")
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            self._json({"error": "No image uploaded."}, 400)
            return
        default_region = (fields.get("region") or "").strip() or None

        try:
            pil = Image.open(io.BytesIO(bytes(image_bytes)))
            pil.load()
        except Exception:
            self._json({"error": "Could not read that image file."}, 400)
            return

        try:
            self._json(run_pipeline(pil, default_region))
        except Exception as e:  # noqa: BLE001 — surface pipeline errors to the UI
            self._json({"error": f"Processing failed: {e}"}, 500)

    def log_message(self, fmt, *args):  # slightly quieter logging
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Business Card Scanner — open  http://{HOST}:{PORT}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
