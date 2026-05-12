from __future__ import annotations
import streamlit as st
import numpy as np
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

from card_segmenter import segment_cards
from ocr_engine import extract_text
from qr_scanner import scan_qr_codes
from parser import parse_business_cards

st.set_page_config(
    page_title="Business Card Scanner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .field-card {
        background: #f8f9fa;
        border-left: 4px solid #4a90e2;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
    }
    .field-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .field-value {
        font-size: 1rem;
        color: #1a1a2e;
        font-weight: 500;
    }
    .name-value {
        font-size: 1.4rem;
        color: #1a1a2e;
        font-weight: 700;
    }
    .qr-section {
        background: #fff8e7;
        border: 1px solid #f0c040;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .section-divider {
        margin: 1.5rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
    .tag-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .rotation-badge {
        display: inline-block;
        background: #e6f4ea;
        color: #1e7e34;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


def render_field(label: str, value: str, icon: str = "", name_style: bool = False):
    val_class = "name-value" if name_style else "field-value"
    st.markdown(f"""
    <div class="field-card">
        <div class="field-label">{icon} {label}</div>
        <div class="{val_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def _qr_data_label(data: str) -> str:
    if data.startswith(("http://", "https://")):
        return "URL"
    if data.startswith("BEGIN:VCARD"):
        return "vCard"
    if data.startswith("WIFI:"):
        return "Wi-Fi"
    if data.startswith("mailto:"):
        return "Email"
    if data.startswith("tel:"):
        return "Phone"
    return "Text"


def render_qr_section(qr_results: list):
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### QR Code Detected")

    for idx, qr in enumerate(qr_results, 1):
        col_img, col_data = st.columns([1, 2], gap="large")
        data = qr["data"]
        label = _qr_data_label(data)

        with col_img:
            if qr.get("crop") is not None and qr["crop"].size > 0:
                st.image(qr["crop"], caption=f"QR Code {idx}", use_container_width=True)
            else:
                st.info("QR image not available")

        with col_data:
            st.markdown(f'<div class="tag-badge">{label}</div>', unsafe_allow_html=True)

            if label == "URL":
                st.markdown(f"""
                <div class="field-card">
                    <div class="field-label">🔗 QR CODE DATA</div>
                    <div class="field-value"><a href="{data}" target="_blank">{data}</a></div>
                </div>
                """, unsafe_allow_html=True)
            elif label == "vCard":
                st.markdown(f"""
                <div class="field-card">
                    <div class="field-label">👤 QR CODE DATA</div>
                    <div class="field-value"><pre style="margin:0;font-size:0.82rem;white-space:pre-wrap">{data}</pre></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                display = data.replace("mailto:", "").replace("tel:", "").replace("WIFI:", "")
                st.markdown(f"""
                <div class="field-card">
                    <div class="field-label">📋 QR CODE DATA</div>
                    <div class="field-value">{display}</div>
                </div>
                """, unsafe_allow_html=True)


_FIELD_MAP = [
    ("name",      "Name",        "👤", True),
    ("company",   "Company",     "🏢", False),
    ("job_title", "Job Title",   "💼", False),
    ("email",     "Email",       "📧", False),
    ("phone",     "Phone",       "📞", False),
    ("mobile",    "Mobile",      "📱", False),
    ("fax",       "Fax",         "🖨️", False),
    ("website",   "Website",     "🌐", False),
    ("linkedin",  "LinkedIn",    "🔗", False),
    ("twitter",   "Twitter / X", "🐦", False),
    ("address",   "Address",     "📍", False),
]


def _render_contact(parsed: dict):
    any_field = False
    for key, label, icon, name_style in _FIELD_MAP:
        val = parsed.get(key)
        if val:
            render_field(label, val, icon, name_style)
            any_field = True
    if not any_field:
        st.warning("No structured fields could be extracted. Check raw text below.")


def process_and_render_card(card_array: np.ndarray, default_region: str | None = None):
    """Run full pipeline on one card crop and render the results."""
    with st.spinner("Running OCR and QR scan…"):
        raw_text, rotation = extract_text(card_array, auto_rotate=True)
        qr_results = scan_qr_codes(card_array, rotation=rotation)
        contacts = parse_business_cards(raw_text, default_region=default_region)

    col_card, col_results = st.columns([1, 1], gap="large")

    with col_card:
        st.markdown("#### Card Preview")
        st.image(card_array, use_container_width=True)
        if rotation:
            st.markdown(
                f'<div class="rotation-badge">↻ Auto-rotated {rotation}°</div>',
                unsafe_allow_html=True,
            )

    with col_results:
        if len(contacts) == 1:
            st.markdown("#### Extracted Information")
            _render_contact(contacts[0])
        else:
            st.markdown(f"#### Extracted Information — {len(contacts)} contacts found")
            for n, parsed in enumerate(contacts, 1):
                name = parsed.get("name") or f"Contact {n}"
                st.markdown(f'<div class="tag-badge">Contact {n}: {name}</div>',
                            unsafe_allow_html=True)
                _render_contact(parsed)
                if n < len(contacts):
                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        with st.expander("Raw OCR Text"):
            st.code(raw_text if raw_text else "(no text detected)", language=None)

    # QR section below the two-column layout
    if qr_results:
        render_qr_section(qr_results)
    else:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.caption("No QR code detected on this card.")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">Business Card Scanner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a photo containing one or more business cards. '
    'Fields, QR codes, and orientation are detected automatically.</div>',
    unsafe_allow_html=True,
)

# Fallback country: used only for phone normalisation when the card itself
# gives no hint (no +<code>, no country in the address, no recognisable TLD).
_REGION_CHOICES = {
    "Auto-detect (recommended)": None,
    "United Kingdom (+44)": "GB",
    "Ireland (+353)": "IE",
    "United States / Canada (+1)": "US",
    "United Arab Emirates (+971)": "AE",
    "Saudi Arabia (+966)": "SA",
    "Qatar (+974)": "QA",
    "India (+91)": "IN",
    "Germany (+49)": "DE",
    "France (+33)": "FR",
    "Netherlands (+31)": "NL",
    "Belgium (+32)": "BE",
    "Australia (+61)": "AU",
    "Singapore (+65)": "SG",
}

col_up, col_region = st.columns([2, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Upload Business Card Image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    )
with col_region:
    region_label = st.selectbox(
        "Default country for phone numbers",
        list(_REGION_CHOICES.keys()),
        help="Only used when a card has no country code, no address country, "
             "and no recognisable website/email TLD. Otherwise the country is "
             "detected automatically from the card.",
    )
default_region = _REGION_CHOICES[region_label]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    with st.spinner("Detecting cards in image…"):
        cards = segment_cards(image_array)

    if len(cards) == 1:
        # Single card — render directly
        process_and_render_card(cards[0][0], default_region=default_region)
    else:
        # Multiple cards — use tabs
        st.info(f"{len(cards)} business cards detected in this image.")
        tab_labels = [label for _, label in cards]
        tabs = st.tabs(tab_labels)
        for tab, (card_array, _) in zip(tabs, cards):
            with tab:
                process_and_render_card(card_array, default_region=default_region)
