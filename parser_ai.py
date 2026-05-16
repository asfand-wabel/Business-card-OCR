"""LLM-based field extractor — calls Google Gemini Flash to turn a card's OCR
text into a structured list of contact dicts. Multilingual, layout-agnostic.

Used by parser.parse_business_cards() as the primary path; the heuristic
fallback in parser.py kicks in if this is unavailable.
"""
from __future__ import annotations
import os
import sys
import json
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
_TIMEOUT = 90

# Free-tier per-minute limit for gemini-flash-latest is ~10 RPM. Keep a sliding
# window of recent call timestamps and sleep just long enough to stay under it
# when uploading a batch. Override with GEMINI_RPM env var if you upgrade.
_RPM = int(os.environ.get("GEMINI_RPM", "10"))
_call_times: list[float] = []
_throttle_lock = __import__("threading").Lock()


def _throttle():
    """Block (briefly) when the per-minute call budget would be exceeded."""
    while True:
        now = time.time()
        with _throttle_lock:
            cutoff = now - 60.0
            # Drop timestamps older than 60s
            while _call_times and _call_times[0] < cutoff:
                _call_times.pop(0)
            if len(_call_times) < _RPM:
                _call_times.append(now)
                return
            wait = max(0.5, _call_times[0] + 60.5 - now)
        print(f"[parser_ai] throttling: sleeping {wait:.1f}s to stay under {_RPM} RPM", file=sys.stderr)
        time.sleep(wait)

# Keys the rest of the app expects on every contact
_FIELDS = ("name", "company", "job_title", "email", "phone", "mobile",
           "fax", "website", "linkedin", "twitter", "address")

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "contacts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {k: {"type": "string", "nullable": True} for k in _FIELDS},
            },
        }
    },
    "required": ["contacts"],
}

class RateLimitError(Exception):
    """Surface this to the UI so the user is told what kind of limit was hit
    and roughly when it'll be available again."""
    def __init__(self, retry_after: int = 60, kind: str = "minute", message: str = ""):
        # `kind` is "minute" or "day" — the UI uses it to pick the right copy.
        self.retry_after = max(1, int(retry_after))
        self.kind = kind
        super().__init__(message or f"Gemini rate-limited ({kind}), retry in {retry_after}s")


def _parse_quota_failure(body_text: str) -> tuple[int, str]:
    """Returns (retry_after_seconds, kind) where kind is 'minute' or 'day'."""
    retry_after = 30
    kind = "minute"
    try:
        obj = json.loads(body_text)
    except Exception:
        return retry_after, kind
    for d in (obj.get("error", {}) or {}).get("details", []) or []:
        t = (d.get("@type") or "").split("/")[-1]
        if t == "QuotaFailure":
            for v in d.get("violations", []) or []:
                qid = (v.get("quotaId") or v.get("quotaMetric") or "")
                if "PerDay" in qid:
                    kind = "day"
                elif "PerMinute" in qid:
                    kind = "minute"
        elif t == "RetryInfo":
            rd = d.get("retryDelay") or d.get("retry_delay")
            if isinstance(rd, str) and rd.endswith("s"):
                try:
                    retry_after = max(1, int(float(rd[:-1])))
                except Exception:
                    pass
    # Daily reset can be hours away — clamp the displayed wait to something
    # reasonable; the UI will tell them it's the daily quota anyway.
    if kind == "day":
        retry_after = max(retry_after, 60)
    return retry_after, kind


_PROMPT = """You are extracting structured contact data from the OCR text of a business card.
The card may be in any language (English, Italian, French, German, Spanish, Arabic, etc.).
Sometimes the OCR text covers TWO business cards photographed together — split them into two contacts.

For each distinct person on the card(s), return one object with these fields
(use null when a field is genuinely absent):
  - name        : the person's full name. NEVER the company.
  - company     : the legal entity / organisation, e.g. "Acme Ltd", "TUMMINELLO srl",
                  "Bonizzi s.r.l.", "F.lli Veroni fu Angelo S.p.A.", "Müller GmbH",
                  "Société Foo SARL". Strip trailing punctuation.
  - job_title   : the role (Director, Export Manager, etc.). NEVER a person's name.
  - email       : prefer a personal email (first.last@) over a generic (info@/sales@) when
                  both are present for the same person. Comma-join only if multiple
                  personal emails for the SAME person.
  - phone       : office / landline number. Comma-join if there are several.
                  Format in international "+<country code> ..." form when you can
                  confidently determine the country (from the address, an explicit
                  +cc on another number, the TLD, the language). Otherwise leave as
                  printed — do NOT guess a wrong country code.
  - mobile      : cell / mobile number. Same formatting rules.
  - fax         : fax number. Same formatting rules.
  - website     : the company website (keep the host as printed).
  - linkedin    : LinkedIn URL or handle if present.
  - twitter     : Twitter / X URL or handle if present.
  - address     : the postal address, joined with ", ".

Important rules:
- Italian P.IVA / C.F. / Codice Univoco, German USt-IdNr, French SIRET / TVA,
  Spanish CIF / NIF, UK VAT — these are TAX IDENTIFIERS, never phones.
  Do not put them anywhere in phone/mobile/fax.
- A card with ONE person but TWO emails (info@ + first.last@) is ONE contact.
- A photo with TWO cards (two distinct persons, each with their own contact info)
  is TWO contacts.
- If the photo also has other cards visible at the edges (bleed-through), do NOT
  pull websites/emails from those neighbouring cards into the main contact. Only
  use text that clearly belongs to the card(s) in focus.
- If the card is company-only (no person), return one contact with name=null
  and the company / address / phone / etc. populated.

OCR text:
\"\"\"%s\"\"\"

Return ONLY the JSON object."""


def is_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def extract(ocr_text: str) -> list[dict] | None:
    """Return a list of contact dicts (one per person) or None on failure."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key or not ocr_text.strip():
        return None

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}"
           f":generateContent?key={quote(key)}")
    body = json.dumps({
        "contents": [{"parts": [{"text": _PROMPT % ocr_text}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": _RESPONSE_SCHEMA,
            "temperature": 0,
        },
    }).encode()

    # Self-throttle before each call so a batch upload doesn't trip the limit.
    _throttle()

    # 429 → raise RateLimitError immediately (the UI tells the user to wait).
    # Transient 5xx / network error → one retry with a short backoff.
    data = None
    for attempt in (1, 2):
        try:
            req = Request(url, data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=_TIMEOUT) as r:
                data = json.loads(r.read())
            break
        except HTTPError as e:
            body_txt = ""
            try:
                body_txt = e.read().decode("utf-8", "replace")
            except Exception:
                pass
            if e.code == 429:
                ra, kind = _parse_quota_failure(body_txt)
                print(f"[parser_ai] 429 ({kind}-quota), retry in {ra}s", file=sys.stderr)
                raise RateLimitError(retry_after=ra, kind=kind)
            if e.code in (500, 502, 503, 504) and attempt == 1:
                print(f"[parser_ai] HTTP {e.code}, retrying once", file=sys.stderr)
                time.sleep(1.5)
                continue
            print(f"[parser_ai] HTTP {e.code}: {body_txt[:200]!r}", file=sys.stderr)
            return None
        except (URLError, TimeoutError) as e:
            if attempt == 1:
                print(f"[parser_ai] network error, retrying: {e}", file=sys.stderr)
                time.sleep(1.0)
                continue
            print(f"[parser_ai] network error (giving up): {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"[parser_ai] unexpected error: {e}", file=sys.stderr)
            return None
    if data is None:
        return None

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(text)
        contacts = parsed.get("contacts") or []
    except Exception:
        return None

    # Normalise: ensure every dict has every expected key, null → None
    out: list[dict] = []
    for c in contacts:
        if not isinstance(c, dict):
            continue
        clean = {}
        for k in _FIELDS:
            v = c.get(k)
            if isinstance(v, str):
                v = v.strip() or None
            clean[k] = v
        # Skip entirely empty rows
        if any(clean.get(k) for k in _FIELDS):
            out.append(clean)

    return out or None
