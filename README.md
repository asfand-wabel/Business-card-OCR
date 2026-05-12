# Business Card OCR

Scan one or more business cards from a photo, extract structured contact fields, and read QR codes — built with Streamlit and Google Cloud Vision.

## Features

- **OCR via Google Cloud Vision** (`document_text_detection`) — handles any font, colour, and orientation; rotation is inferred from the response, no separate pass needed (~1.5 s/card).
- **Multi-card / multi-contact** — a photo with two cards (or one card listing two people) is split into separate contacts, using distinct email addresses or repeated-email + multiple-name signals.
- **Field extraction** — name, company, job title, email(s), phone, mobile, fax, website, LinkedIn, Twitter/X, postal address. Multiple phone/mobile numbers per card are all captured.
- **Phone normalisation** — numbers that carry their own country code (`+44…`, `0044…`, bare `44…`) are rewritten to international `+<code>` form via Google's `phonenumbers` library and validated. Purely-local numbers are left as printed unless a default country is selected in the UI — it never guesses the country wrong.
- **QR codes** — decoded with pyzbar (OpenCV fallback), shown with the cropped QR image and its data.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Google Cloud API key (the Vision API must be enabled and billing active on the project — the first 1,000 scans/month are free):

```
GOOGLE_API_KEY=your_api_key_here
```

A service-account JSON works too — set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` instead.

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501, upload a business-card photo, and the extracted fields + QR data appear.

## Project layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI |
| `card_segmenter.py` | Splits a photo into individual card crops |
| `ocr_engine.py` | Google Vision OCR + rotation inference |
| `parser.py` | Field extraction, multi-contact splitting, phone normalisation |
| `qr_scanner.py` | QR-code detection / decoding |
| `test_pipeline.py` | Batch test over a folder of images |
