from __future__ import annotations
import re
from typing import Optional

try:
    import phonenumbers
    from phonenumbers import PhoneNumberFormat as _PNF
    _PHONENUMBERS_OK = True
except Exception:
    _PHONENUMBERS_OK = False

# ── Label prefix patterns ─────────────────────────────────────────────────────

_PHONE_LABEL_RE = re.compile(
    r'^(?:M|Mob|Mobile|Cell|T|Tel|Tel\.|Phone|Ph|HQ|Office|Dir|Direct|Fax|F)\s*[.:]\s*',
    re.IGNORECASE,
)
_MOBILE_LABEL_RE = re.compile(r'^(?:M|Mob|Mobile|Cell)\s*[.:]\s*', re.IGNORECASE)
_FAX_LABEL_RE = re.compile(r'^(?:Fax|F)\s*[.:]\s*', re.IGNORECASE)
_EMAIL_LABEL_RE = re.compile(r'^e\s*[.:]\s*', re.IGNORECASE)
_WEB_LABEL_RE = re.compile(r'^w\s*[.:]\s*', re.IGNORECASE)


def _strip_label(line: str, pattern: re.Pattern) -> Optional[str]:
    m = pattern.match(line)
    return line[m.end():].strip() if m else None


def _fix_phone(phone: str) -> str:
    phone = re.sub(r'\((\d)\b(?!\))', r'(\1)', phone)
    return re.sub(r'\s+', ' ', phone).strip()


# ── Phone normalisation (international '+<country code> …' form) ───────────────
#
# Conservative by design: a number is rewritten with a +<code> ONLY when the
# country is unambiguous —
#   • the number already carries its own country code: "+44…", "0044…", or a
#     bare leading code like "44 7352…"   →  identified from the number itself
#     by the `phonenumbers` library (and validated);
#   • OR the caller supplied an explicit `region` (the "default country"
#     selector in the UI), used only for purely-local numbers.
# Anything else — a bare local number on a card with no explicit country, e.g.
# "07970 176996" or "020 7946 0958" — is left exactly as printed (just cleaned:
# the "(0)" trunk marker and stray whitespace removed). We never guess the
# country from the website TLD, the address text, or other numbers on the card.

def _normalize_one_number(raw: str, region: Optional[str]) -> str:
    cleaned = re.sub(r"\s+", " ", raw.replace("(0)", " ").replace("( 0 )", " ")).strip()
    if not _PHONENUMBERS_OK:
        return cleaned  # at least the (0) trunk marker is gone

    digits = re.sub(r"\D", "", cleaned)

    # Candidates, in priority order. Each must validate before it's accepted.
    candidates = []
    if cleaned.startswith("+"):
        candidates.append((cleaned, None))                  # +44 7352…
    else:
        if digits.startswith("00") and len(digits) > 5:
            candidates.append(("+" + digits[2:], None))     # 0044 7352… → +44…
        candidates.append(("+" + digits, None))             # 44 7352… (bare code)
        if region:
            candidates.append((cleaned, region))            # local + explicit region

    for cand, reg in candidates:
        try:
            p = phonenumbers.parse(cand, reg)
        except phonenumbers.NumberParseException:
            continue
        if phonenumbers.is_valid_number(p):
            return phonenumbers.format_number(p, _PNF.INTERNATIONAL)

    # Country not certain → leave it exactly as printed (cleaned).
    return cleaned


def _normalize_number_csv(value: Optional[str], region: Optional[str]) -> Optional[str]:
    if not value:
        return value
    out, seen = [], set()
    for part in value.split(", "):
        norm = _normalize_one_number(part, region)
        key = re.sub(r"\D", "", norm)
        if key and key not in seen:
            seen.add(key)
            out.append(norm)
    return ", ".join(out) if out else value


# ── Shared regex objects ──────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r'\+?[\d][\d\s\(\)\-\.]{6,20}[\d]')
_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9\-]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?",
    re.IGNORECASE,
)
_STREET_RE = re.compile(
    r"\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Park|Way)\b"
    r"|P\.?O\.?\s*Box\s+\d+",
    re.IGNORECASE,
)
_POSTCODE_RE = re.compile(
    r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b"   # UK: BD20 0EE
    r"|\b\d{5}(?:-\d{4})?\b"                         # US ZIP
    r"|LV-\d{4}"                                     # Latvia
    r"|\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b"                # Canada
    r"|\bD\d{2}\b",                                  # Ireland Eircode prefix
    re.IGNORECASE,
)
_NOT_ADDRESS_RE = re.compile(r"[@]|(?:\+\d)|(?:https?://)")
_ALLCAPS_HEADING_RE = re.compile(r'^[A-Z][A-Z\s]{3,}$')   # short ALL-CAPS heading

# Person-name shape (Unicode-aware): 1-4 capitalized word tokens
_PERSON_NAME_RE = re.compile(
    r"^[^\W\d_][^\W\d_'\-\.]*(?:\s+[^\W\d_][^\W\d_'\-\.]*){0,3}$",
    re.UNICODE,
)


def _looks_like_name(line: str) -> bool:
    s = line.strip()
    return bool(
        s and s[0].isupper()
        and 1 <= len(s.split()) <= 4
        and _PERSON_NAME_RE.match(s)
    )

_TITLE_KEYWORDS = [
    "CEO", "CTO", "CFO", "COO", "CMO", "CIO", "CISO", "CPO",
    "President", "Vice President", "VP", "EVP", "SVP",
    "Director", "Managing Director", "Executive Director",
    "Manager", "General Manager", "Project Manager", "Product Manager",
    "Brand Manager", "Export Development",
    "Engineer", "Software Engineer", "Lead Engineer",
    "Developer", "Software Developer", "Web Developer",
    "Designer", "UX Designer", "UI Designer", "Graphic Designer",
    "Architect", "Solution Architect",
    "Consultant", "Senior Consultant",
    "Analyst", "Business Analyst", "Data Analyst",
    "Founder", "Co-Founder", "Partner", "Principal",
    "Officer", "Executive", "Coordinator", "Specialist",
    "Head of", "Lead", "Supervisor", "Administrator",
    "Representative", "Advisor", "Strategist", "Associate",
    "Export", "Import", "Sales", "Marketing", "Finance",
    "Operations", "Procurement", "Logistics", "Account",
]

_COMPANY_SUFFIXES = [
    "Inc", "LLC", "Ltd", "Corp", "Co", "GmbH", "AG", "PLC",
    "Pvt", "Pty", "Group", "Solutions", "Technologies", "Services",
    "Consulting", "Systems", "Digital", "Labs", "Studio", "Studios",
    "Ventures", "Holdings", "Enterprises", "International", "Limited",
]


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_business_card(text: str, default_region: Optional[str] = None) -> dict:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    result = {
        "name": None, "company": None, "job_title": None,
        "email": None, "phone": None, "mobile": None,
        "fax": None, "website": None, "address": None,
        "linkedin": None, "twitter": None,
    }
    used: set = set()

    # 1. Email (detect before website so email domains aren't grabbed as URLs).
    # Collect all distinct emails; mark EVERY email-bearing line as used so a
    # duplicated email (two-sided cards) doesn't leak into job_title etc.
    _emails_found: list = []
    for i, line in enumerate(lines):
        bare = _strip_label(line, _EMAIL_LABEL_RE) or line
        m = _EMAIL_RE.search(bare)
        if m:
            addr = m.group()
            if addr.lower() not in {e.lower() for e in _emails_found}:
                _emails_found.append(addr)
            used.add(i)
    if _emails_found:
        result["email"] = ", ".join(_emails_found)

    # 2. LinkedIn (detect before website)
    for i, line in enumerate(lines):
        lower = line.lower()
        if ("linkedin" in lower or "linkedin.com" in lower) and not result["linkedin"]:
            result["linkedin"] = line.strip()
            used.add(i)

    # 3. Twitter / X
    _twitter_handle_re = re.compile(r'^@\w{3,}$')
    for i, line in enumerate(lines):
        lower = line.lower()
        stripped = line.strip()
        if (
            "twitter" in lower or "twitter.com" in lower
            or "x.com" in lower
            or _twitter_handle_re.match(stripped)
        ) and not result["twitter"]:
            result["twitter"] = stripped
            used.add(i)

    # 4. Phone / Mobile / Fax — collect ALL numbers (a card can list several)
    phones: list = []
    mobiles: list = []
    faxes: list = []

    def _digits(s: str) -> str:
        return re.sub(r'\D', '', s)

    for i, line in enumerate(lines):
        if i in used:          # skip email / linkedin / twitter lines
            continue
        lower = line.lower()
        matched = False

        fax_bare = _strip_label(line, _FAX_LABEL_RE)
        if fax_bare:
            m = _PHONE_RE.search(fax_bare)
            if m and len(_digits(m.group())) >= 7:
                faxes.append(_fix_phone(m.group()))
                used.add(i)
                matched = True

        if not matched:
            mob_bare = _strip_label(line, _MOBILE_LABEL_RE)
            if mob_bare:
                m = _PHONE_RE.search(mob_bare)
                if m and len(_digits(m.group())) >= 7:
                    mobiles.append(_fix_phone(m.group()))
                    used.add(i)
                    matched = True

        if not matched:
            ph_bare = _strip_label(line, _PHONE_LABEL_RE)
            if ph_bare:
                m = _PHONE_RE.search(ph_bare)
                if m and len(_digits(m.group())) >= 7:
                    phones.append(_fix_phone(m.group()))
                    used.add(i)
                    matched = True

        if not matched:
            for m in _PHONE_RE.finditer(line):
                raw = m.group()
                if len(_digits(raw)) < 7:
                    continue
                cleaned = _fix_phone(raw)
                if re.search(r'\bfax\b|f:', lower):
                    faxes.append(cleaned)
                    used.add(i)
                elif re.search(r'\b(mobile|cell|mob)\b|(?:^|\s)m:', lower):
                    mobiles.append(cleaned)
                    used.add(i)
                else:
                    phones.append(cleaned)
                    used.add(i)

    def _dedup(lst: list) -> list:
        seen, out = set(), []
        for x in lst:
            d = _digits(x)
            if d and d not in seen:
                seen.add(d)
                out.append(x)
        return out

    phones, mobiles, faxes = _dedup(phones), _dedup(mobiles), _dedup(faxes)
    result["phone"] = ", ".join(phones) if phones else None
    result["mobile"] = ", ".join(mobiles) if mobiles else None
    result["fax"] = ", ".join(faxes) if faxes else None

    # 5. Website (after email + linkedin so their lines are already in `used`)
    for i, line in enumerate(lines):
        if i in used:
            continue
        bare = _strip_label(line, _WEB_LABEL_RE) or line
        m = _URL_RE.search(bare)
        if m and "@" not in m.group() and not result["website"]:
            result["website"] = m.group().strip()
            used.add(i)

    # 6. Job title
    for i, line in enumerate(lines):
        if i in used:
            continue
        for kw in _TITLE_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", line, re.IGNORECASE):
                result["job_title"] = line.strip()
                used.add(i)
                break
        if result["job_title"]:
            break

    # 7. Address
    for i, line in enumerate(lines):
        if _STREET_RE.search(line):
            addr_parts = [line.strip()]
            used.add(i)
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j].strip()
                if not nxt or _NOT_ADDRESS_RE.search(nxt):
                    break
                if any(kw in nxt.lower() for kw in ("scan", "browse", "visit", "follow")):
                    break
                if len(re.sub(r'\D', '', nxt)) >= 8:
                    break
                # Stop if we hit an ALL-CAPS heading (e.g. "EU DISTRIBUTION")
                if _ALLCAPS_HEADING_RE.match(nxt) and not _POSTCODE_RE.search(nxt):
                    break
                addr_parts.append(nxt)
                used.add(j)
            result["address"] = ", ".join(addr_parts)
            break

    if not result["address"]:
        for i, line in enumerate(lines):
            if i in used:
                continue
            if _POSTCODE_RE.search(line) and not _NOT_ADDRESS_RE.search(line):
                result["address"] = line.strip()
                used.add(i)
                break

    # 8a. If the address line starts with "<Company Suffix>," extract company
    if result["address"] and not result["company"]:
        first_part = result["address"].split(",")[0].strip()
        for sfx in _COMPANY_SUFFIXES:
            if re.search(r"\b" + re.escape(sfx) + r"\.?\b", first_part, re.IGNORECASE):
                result["company"] = first_part.rstrip(".")
                break

    # 8. Company — explicit suffix first
    _only_suffix_re = re.compile(
        r"^(?:" + "|".join(re.escape(s) for s in _COMPANY_SUFFIXES) + r")\.?$",
        re.IGNORECASE,
    )
    # Logo abbreviations: 1-2 char words at line start (any case)
    # e.g. "hj"/"HJ" read as "hi"/"HI" by OCR, "bm", "lb", "HJ", "BM" etc.
    _logo_prefix_re = re.compile(r'^[A-Za-z]{1,2}\s+')

    for i, line in enumerate(lines):
        if i in used:
            continue
        if _STREET_RE.search(line):
            continue
        for sfx in _COMPANY_SUFFIXES:
            if re.search(r"\b" + re.escape(sfx) + r"\.?\b", line, re.IGNORECASE):
                candidate = line.strip()
                # If the line is ONLY a suffix word (e.g. "International" alone),
                # try to merge with the preceding non-used line for the full name.
                if _only_suffix_re.match(candidate) and i > 0:
                    prev_idx = i - 1
                    while prev_idx >= 0 and prev_idx in used:
                        prev_idx -= 1
                    if prev_idx >= 0 and prev_idx not in used:
                        prev_line = _logo_prefix_re.sub('', lines[prev_idx].strip())
                        if prev_line and not _STREET_RE.search(prev_line):
                            candidate = prev_line + " " + candidate
                            used.add(prev_idx)
                # Strip any leading logo abbreviation from the final candidate
                candidate = _logo_prefix_re.sub('', candidate)
                # If line has commas and suffix is in the first part,
                # trim to just the company name (drop address fragments).
                if ',' in candidate:
                    first = candidate.split(',')[0].strip()
                    for s2 in _COMPANY_SUFFIXES:
                        if re.search(r"\b" + re.escape(s2) + r"\.?\b", first, re.IGNORECASE):
                            candidate = first
                            break
                result["company"] = candidate.rstrip(".")
                used.add(i)
                break
        if result["company"]:
            break

    # 9. Name — first unused line that looks like a person's name (Unicode-aware)
    _name_re = re.compile(
        r"^[^\W\d_][^\W\d_'\-\.]*(?:\s+[^\W\d_][^\W\d_'\-\.]*){0,3}$",
        re.UNICODE,
    )
    name_idx = None
    for i, line in enumerate(lines):
        if i in used:
            continue
        stripped = line.strip()
        if (
            _name_re.match(stripped)
            and 1 <= len(stripped.split()) <= 4
            and stripped[0].isupper()
        ):
            result["name"] = stripped
            used.add(i)
            name_idx = i
            break

    # 9b. If only the first part of a 2-line name was grabbed (e.g. "JOHN" on
    # one line, "RYAN" on the next), merge the next single-word capitalized line.
    if name_idx is not None and len(result["name"].split()) == 1:
        for j in range(name_idx + 1, min(name_idx + 3, len(lines))):
            if j in used:
                continue
            nxt = lines[j].strip()
            if (
                _name_re.match(nxt)
                and len(nxt.split()) == 1
                and nxt[0].isupper()
                and len(nxt) >= 3
            ):
                result["name"] = result["name"] + " " + nxt
                used.add(j)
                break

    # 10. Company fallback — ALL-CAPS brand (only if name not already grabbed it)
    if not result["company"]:
        _noise_words = {"SCAN", "BUY", "ORDER", "INFO", "TEL", "FAX", "EMAIL",
                        "NEW", "OLD", "VIP", "CALL", "VISIT", "FOLLOW"}
        for i, line in enumerate(lines):
            if i in used:
                continue
            stripped = line.strip()
            words = stripped.split()
            if (
                1 <= len(words) <= 4
                and all(w.isupper() and w.isalpha() and len(w) > 2 for w in words)
                and stripped.upper() not in _noise_words
                # Single-word brands must be ≥ 6 chars to avoid noise like "SCAN"
                and (len(words) >= 2 or len(stripped) >= 6)
            ):
                result["company"] = stripped
                used.add(i)
                break

    # 11. Normalise phone numbers to international '+<country code> …' form —
    # but only when the country is certain (the number carries its own code, or
    # the caller supplied an explicit default_region). Otherwise leave as-is.
    for key in ("phone", "mobile", "fax"):
        result[key] = _normalize_number_csv(result[key], default_region)

    return result


# ── Multi-contact splitting ───────────────────────────────────────────────────

def _is_company_ish(line: str) -> bool:
    for sfx in _COMPANY_SUFFIXES:
        if re.search(r"\b" + re.escape(sfx) + r"\.?\b", line, re.IGNORECASE):
            return True
    return False


def _has_contact_info_after(lines: list, idx: int, window: int = 6) -> bool:
    """True if an email or a phone-like number appears within `window` lines."""
    for j in range(idx + 1, min(idx + 1 + window, len(lines))):
        if _EMAIL_RE.search(lines[j]):
            return True
        m = _PHONE_RE.search(lines[j])
        if m and len(re.sub(r'\D', '', m.group())) >= 7:
            return True
    return False


def _blocks_to_contacts(lines: list, anchor_indices: list) -> list:
    """Split `lines` at each anchor index and parse each block."""
    split_points = [0] + list(anchor_indices[1:]) + [len(lines)]
    blocks = []
    for k in range(len(split_points) - 1):
        chunk = lines[split_points[k]:split_points[k + 1]]
        if chunk:
            blocks.append("\n".join(chunk))
    return blocks


def parse_business_cards(text: str, default_region: Optional[str] = None) -> list:
    """
    Parse OCR text that may contain MORE THAN ONE contact — e.g. a photo with
    two business cards, or a card listing two people.

    Two splitting strategies, tried in order:
      1. Two or more distinct email addresses → split before the name line that
         introduces each email.
      2. Two or more well-separated person-name lines (each followed by contact
         info) → split at each name line. Handles cards that share one email.

    Always returns a list with at least one dict.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return [parse_business_card(text, default_region)]

    # ── Strategy 1: distinct email addresses ──────────────────────────────────
    email_hits = [(i, m.group().lower())
                  for i, line in enumerate(lines)
                  if (m := _EMAIL_RE.search(line))]
    distinct_emails = list(dict.fromkeys(e for _, e in email_hits))
    if len(distinct_emails) >= 2:
        first_idx_of = {}
        for idx, email in email_hits:
            first_idx_of.setdefault(email, idx)
        email_line_indices = sorted(first_idx_of[e] for e in distinct_emails)

        anchors = [0]
        for ei in email_line_indices[1:]:
            boundary = ei
            for j in range(ei - 1, anchors[-1] - 1, -1):
                if _looks_like_name(lines[j]):
                    boundary = j
                    break
            anchors.append(boundary)
        blocks = _blocks_to_contacts(lines, anchors)
        if len(blocks) >= 2:
            return [parse_business_card(b, default_region) for b in blocks]

    # ── Strategy 2: one email repeated → N contacts sharing it ────────────────
    # (e.g. two cards from the same company, each printed with the company's
    #  generic sales@ address but a different person's name and phone.)
    if len(distinct_emails) == 1 and len(email_hits) >= 2:
        name_anchors = []
        for i, line in enumerate(lines):
            if not _looks_like_name(line):
                continue
            if _is_company_ish(line):
                continue
            if not _has_contact_info_after(lines, i):
                continue
            # Skip lines too close to the previous anchor (first/last name split
            # across two lines, or a name immediately followed by a title).
            if name_anchors and i - name_anchors[-1] < 5:
                continue
            name_anchors.append(i)

        # Only split if the number of name anchors matches the number of email
        # occurrences — a strong signal that each is a distinct contact.
        if len(name_anchors) == len(email_hits) >= 2:
            blocks = _blocks_to_contacts(lines, name_anchors)
            if all(_EMAIL_RE.search(b) for b in blocks) and len(blocks) >= 2:
                return [parse_business_card(b, default_region) for b in blocks]

    return [parse_business_card(text, default_region)]
