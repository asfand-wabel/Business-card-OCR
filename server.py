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
import parser_ai
import zoho_crm
import queue_store

HOST, PORT = "127.0.0.1", 8000
MAX_UPLOAD = 100 * 1024 * 1024  # 100 MB total across all images in one upload


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
    """Tiny multipart/form-data parser.
    File fields (filename present) → list of (filename, bytes) tuples.
    Text fields → str (last wins).
    """
    files: dict[str, list] = {}
    text: dict[str, str] = {}
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
        fn_match = re.search(r'filename="([^"]*)"', cd)
        if fn_match is not None:
            files.setdefault(name, []).append((fn_match.group(1), content))
        else:
            text[name] = content.decode("utf-8", "replace").strip()
    return {"files": files, "text": text}


def process_one_image(pil_image: Image.Image, source: str,
                      default_region=None) -> list[dict]:
    """OCR → parse → QR → CRM-preview, returning a list of queue items
    (one item per detected contact)."""
    arr = np.array(pil_image.convert("RGB"))
    cards = segment_cards(arr)
    crm_on = zoho_crm.is_configured()
    items: list[dict] = []
    for card_arr, label in cards:
        raw_text, rotation = extract_text(card_arr, auto_rotate=True)
        qr_results = scan_qr_codes(card_arr, rotation=rotation)
        contacts = parse_business_cards(raw_text, default_region=default_region)
        preview = _to_data_url(card_arr)
        qrs = [
            {"data": q.get("data", ""),
             "type": q.get("type", "QRCODE"),
             "crop": _to_data_url(q.get("crop"), max_dim=340)}
            for q in qr_results
        ]
        for c in contacts:
            crm = None
            if crm_on:
                try:
                    crm = zoho_crm.crm_preview_for_contact(c)
                except zoho_crm.ZohoError as e:
                    crm = {"error": str(e)}
                except Exception as e:  # noqa: BLE001
                    crm = {"error": f"CRM preview failed: {e}"}
            items.append({
                "source": source,
                "card_label": label,
                "preview": preview,
                "rotation": rotation,
                "raw_text": raw_text,
                "qr_codes": qrs,
                "fields": c,
                "crm": crm,
            })
    return items


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
            self._json({
                "ok": True,
                "crm_enabled": zoho_crm.is_configured(),
                "crm_environment": zoho_crm.environment_label(),
                "crm_api_base": zoho_crm.API_BASE,
                "crm_ui_base": zoho_crm.UI_BASE,
            })
        elif self.path == "/crm/options":
            if not zoho_crm.is_configured():
                self._json({"error": "Zoho not configured."}, 400)
                return
            try:
                self._json(zoho_crm.get_form_options())
            except zoho_crm.ZohoError as e:
                self._json({"error": str(e)}, 502)
        elif self.path.startswith("/queue"):
            self._json({
                "items": queue_store.list_items(),
                "counts": queue_store.counts(),
                "crm_enabled": zoho_crm.is_configured(),
            })
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        if self.path == "/crm/push":
            self._handle_push()
            return
        if self.path == "/queue/delete":
            self._handle_queue_delete()
            return
        if self.path == "/queue/clear":
            queue_store.purge_all()
            self._json({"ok": True})
            return
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
            self._json({"error": "Total upload too large (max 25 MB)."}, 413)
            return

        body = self.rfile.read(length)
        form = _parse_multipart(body, boundary)
        files = form["files"].get("image") or form["files"].get("images") or []
        if not files:
            self._json({"error": "No image uploaded."}, 400)
            return
        default_region = (form["text"].get("region") or "").strip() or None

        new_items: list[dict] = []
        errors: list[dict] = []
        for filename, image_bytes in files:
            try:
                pil = Image.open(io.BytesIO(bytes(image_bytes)))
                pil.load()
            except Exception:
                errors.append({"source": filename or "?", "error": "Could not read image file."})
                continue
            try:
                items = process_one_image(pil, filename or "upload", default_region)
                new_items.extend(items)
            except parser_ai.RateLimitError as e:
                self._json({
                    "error": "AI parser rate-limited.",
                    "retry_after_seconds": e.retry_after,
                    "limit_kind": e.kind,
                    "processed_before_limit": len(new_items),
                }, 429)
                if new_items:
                    queue_store.add_items(new_items)
                return
            except Exception as e:  # noqa: BLE001
                errors.append({"source": filename or "?", "error": f"Processing failed: {e}"})
        if new_items:
            new_items = queue_store.add_items(new_items)
        self._json({"ok": True, "added": len(new_items), "errors": errors,
                    "items": new_items})

    def _handle_queue_delete(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
        except Exception:
            self._json({"error": "Invalid JSON."}, 400); return
        item_id = (body.get("id") or "").strip()
        if not item_id:
            self._json({"error": "Missing id."}, 400); return
        ok = queue_store.delete_item(item_id)
        self._json({"ok": ok})

    # -- /crm/push --------------------------------------------------------
    def _handle_push(self):
        if not zoho_crm.is_configured():
            self._json({"error": "Zoho not configured."}, 400)
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._json({"error": "Expected application/json."}, 400)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._json({"error": "Empty request."}, 400)
            return
        try:
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._json({"error": "Invalid JSON."}, 400)
            return

        item_id = body.get("id")
        choices = body.get("choices") or {}
        # If an `id` is given, push the queued item; otherwise expect inline `fields`.
        if item_id:
            item = queue_store.get_item(item_id)
            if not item:
                self._json({"error": "Queue item not found."}, 404); return
            parsed = item.get("fields") or {}
        else:
            parsed = body.get("fields") or {}

        for k in ("stage", "pipeline"):
            if not choices.get(k):
                self._json({"error": f"Missing required choice: {k}"}, 400); return
        try:
            result = zoho_crm.crm_push(parsed, choices)
        except zoho_crm.ZohoError as e:
            self._json({"error": str(e), "zoho_body": e.body}, 502); return
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"Push failed: {e}"}, 500); return

        if item_id:
            queue_store.update_item(item_id, {
                "status": "approved",
                "choices": choices,
                "push_result": result,
            })
        self._json({"ok": True, "result": result, "id": item_id})

    def log_message(self, fmt, *args):  # slightly quieter logging
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main():
    # Refuse to start if Zoho is configured but the URL isn't a sandbox
    # and the user hasn't explicitly opted into production writes.
    if zoho_crm.is_configured():
        try:
            zoho_crm.assert_safe_environment()
        except zoho_crm.ProductionWriteBlocked as e:
            print(f"\n❌  REFUSING TO START: {e}\n", flush=True)
            raise SystemExit(2)
        env = zoho_crm.environment_label()
        marker = "🟢" if env == "SANDBOX" else "🔴"
        print(f"{marker}  Zoho environment: {env}  ({zoho_crm.API_BASE})", flush=True)

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
