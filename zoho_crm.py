"""Zoho CRM client — token refresh, search, create.

Used by server.py for the business-card → Potential workflow:
  1. search_account()  / create_account()
  2. search_contact()  / create_contact()
  3. create_deal()     (Deal = "Potential" in this org's renamed schema)

All Zoho HTTP details live here. The token is cached in-process and
refreshed automatically. Picklist / user / layout lookups feed the UI
dropdowns via the /crm/options endpoint in server.py.
"""
from __future__ import annotations
import os
import json
import time
import threading
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

_CID = os.environ.get("ZOHO_CLIENT_ID", "")
_CS  = os.environ.get("ZOHO_CLIENT_SECRET", "")
_RT  = os.environ.get("ZOHO_REFRESH_TOKEN", "")
_DC  = os.environ.get("ZOHO_DC", "com")

# Critical: this is what actually decides sandbox vs production. The OAuth
# credentials are workspace-scoped (same for both); ZOHO_API_BASE is the only
# thing routing the request to one or the other.
_DEFAULT_API_BASE = "https://crmsandbox.zoho.com/crm/v8"
API_BASE = (os.environ.get("ZOHO_API_BASE") or _DEFAULT_API_BASE).rstrip("/")
UI_BASE  = (os.environ.get("ZOHO_UI_BASE") or "https://crmsandbox.zoho.com/crm/datateamhp").rstrip("/")
ALLOW_PRODUCTION = (os.environ.get("ZOHO_ALLOW_PRODUCTION", "").lower()
                    in ("1", "true", "yes", "y"))

_TOKEN_URL = f"https://accounts.zoho.{_DC}/oauth/v2/token"

LAYOUT_NAME_PREF = "Layout Clean"


def is_sandbox() -> bool:
    return "sandbox" in API_BASE.lower()


def environment_label() -> str:
    return "SANDBOX" if is_sandbox() else "PRODUCTION"


class ProductionWriteBlocked(RuntimeError):
    pass


def assert_safe_environment() -> None:
    """Refuse to operate if pointing at production without an explicit opt-in.
    Called at server startup and again on every push, belt-and-suspenders."""
    if not is_sandbox() and not ALLOW_PRODUCTION:
        raise ProductionWriteBlocked(
            f"ZOHO_API_BASE='{API_BASE}' is not a sandbox URL. "
            f"Set ZOHO_ALLOW_PRODUCTION=true if you really mean to write to live data."
        )


def is_configured() -> bool:
    return bool(_CID and _CS and _RT)


# ── Token cache ──────────────────────────────────────────────────────────────

class _TokenCache:
    def __init__(self) -> None:
        self._token: str | None = None
        self._api_domain: str | None = None
        self._exp: float = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        """Returns just the access token. API base URL is fixed by ZOHO_API_BASE."""
        with self._lock:
            now = time.time()
            if self._token and now < self._exp:
                return self._token  # type: ignore[return-value]
            body = urlencode({
                "refresh_token": _RT,
                "client_id": _CID,
                "client_secret": _CS,
                "grant_type": "refresh_token",
            }).encode()
            with urlopen(Request(_TOKEN_URL, data=body, method="POST"), timeout=20) as r:
                data = json.loads(r.read())
            if "access_token" not in data:
                raise ZohoError(0, data)
            self._token = data["access_token"]
            self._exp = now + int(data.get("expires_in", 3600)) - 60
            return self._token  # type: ignore[return-value]


_cache = _TokenCache()


class ZohoError(Exception):
    def __init__(self, code: int, body) -> None:
        super().__init__(f"Zoho HTTP {code}: {body}")
        self.code = code
        self.body = body


# ── Generic HTTP ─────────────────────────────────────────────────────────────

def _request(method: str, path: str, *, params=None, body=None):
    token = _cache.get()
    # `path` is now relative to ZOHO_API_BASE (e.g. "/Accounts/search").
    if not path.startswith("/"):
        path = "/" + path
    url = f"{API_BASE}{path}" + (("?" + urlencode(params)) if params else "")
    data = json.dumps(body).encode() if body is not None else None
    headers = {
        "Authorization": f"Zoho-oauthtoken {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as r:
            if r.status == 204:
                return None
            raw = r.read()
            return json.loads(raw) if raw else None
    except HTTPError as e:
        if e.code == 204:
            return None
        try:
            err_body = json.loads(e.read().decode("utf-8", "replace") or "{}")
        except Exception:
            err_body = "<unparseable response>"
        raise ZohoError(e.code, err_body) from None
    except URLError as e:
        raise ZohoError(0, f"Network error: {e.reason}") from None


# ── Metadata for dropdowns ───────────────────────────────────────────────────

def get_active_users() -> list[dict]:
    data = _request("GET", "/users", params={"type": "ActiveConfirmedUsers"})
    out = []
    for u in (data or {}).get("users", []):
        out.append({
            "id": u["id"],
            "name": u.get("full_name", ""),
            "email": u.get("email", ""),
            "role": (u.get("role") or {}).get("name", ""),
        })
    out.sort(key=lambda x: x["name"].lower())
    return out


def get_picklist(module: str, api_name: str) -> list[str]:
    data = _request("GET", "/settings/fields", params={"module": module})
    for f in (data or {}).get("fields", []):
        if f.get("api_name") == api_name:
            return [pv["display_value"] for pv in f.get("pick_list_values", []) if pv.get("display_value")]
    return []


def get_layout_id(module: str, layout_name: str) -> str | None:
    data = _request("GET", "/settings/layouts", params={"module": module})
    for L in (data or {}).get("layouts", []):
        if L.get("name") == layout_name and L.get("status") == "active":
            return L["id"]
    return None


# Convenience: bundle of everything the UI needs
def get_form_options() -> dict:
    return {
        "users": get_active_users(),
        "pipeline":            get_picklist("Deals", "Pipeline"),
        "stage":               get_picklist("Deals", "Stage"),
        "potential_type":      get_picklist("Deals", "Potential_Type"),
        "met_by_consultant_at":get_picklist("Deals", "Met_By_Consultant_At"),
        "potential_target":    get_picklist("Deals", "Potential_Target"),
        "lead_source":         get_picklist("Deals", "Lead_Source"),
        "zoho_company_type":   get_picklist("Accounts", "Zoho_Company_Type"),
        "deal_layout_id":      get_layout_id("Deals", LAYOUT_NAME_PREF),
    }


# ── Search ───────────────────────────────────────────────────────────────────

def _escape_criteria(v: str) -> str:
    return (v.replace("\\", "\\\\")
             .replace("(", "\\(")
             .replace(")", "\\)")
             .replace(":", "\\:")
             .replace(",", "\\,"))


def search_account(name: str | None, website: str | None = None) -> dict | None:
    """Search by Account_Name (equals). Returns the first hit's record dict or None."""
    name = (name or "").strip()
    if not name:
        return None
    crit = f"(Account_Name:equals:{_escape_criteria(name)})"
    try:
        data = _request("GET", "/Accounts/search", params={"criteria": crit})
    except ZohoError as e:
        if e.code in (204,):
            return None
        # equals didn't find anything? Try a starts_with as a fallback
        try:
            crit2 = f"(Account_Name:starts_with:{_escape_criteria(name)})"
            data = _request("GET", "/Accounts/search", params={"criteria": crit2})
        except ZohoError:
            return None
    if not data or not data.get("data"):
        return None
    return data["data"][0]


def search_contact(email: str | None) -> dict | None:
    email = (email or "").strip()
    if not email:
        return None
    try:
        data = _request("GET", "/Contacts/search", params={"email": email})
    except ZohoError as e:
        if e.code in (204,):
            return None
        raise
    if not data or not data.get("data"):
        return None
    return data["data"][0]


# ── Create ───────────────────────────────────────────────────────────────────

def _first_result(data) -> dict:
    if not data or not data.get("data"):
        raise ZohoError(0, data)
    r = data["data"][0]
    if r.get("code") == "SUCCESS":
        return r.get("details", {})
    raise ZohoError(0, r)


def split_full_name(full_name: str | None) -> tuple[str | None, str]:
    name = (full_name or "").strip()
    if not name:
        return (None, "Unknown")
    parts = name.split()
    if len(parts) == 1:
        return (None, parts[0])
    return (parts[0], " ".join(parts[1:]))


def create_account(
    *,
    name: str,
    website: str | None = None,
    phone: str | None = None,
    billing_country: str | None = None,
    billing_street: str | None = None,
    tradeshows: list[str] | None = None,
    owner_id: str | None = None,
) -> dict:
    rec: dict = {"Account_Name": name}
    if website: rec["Website"] = website
    if phone: rec["Phone"] = phone
    if billing_country: rec["Billing_Country"] = billing_country
    if billing_street: rec["Billing_Street"] = billing_street
    if tradeshows: rec["Tradeshows"] = tradeshows
    if owner_id: rec["Owner"] = {"id": owner_id}
    return _first_result(_request("POST", "/Accounts", body={"data": [rec]}))


def create_contact(
    *,
    last_name: str,
    first_name: str | None = None,
    account_id: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    mobile: str | None = None,
    contact_position: str | None = None,
    linkedin: str | None = None,
    owner_id: str | None = None,
) -> dict:
    rec: dict = {"Last_Name": last_name or "Unknown"}
    if first_name: rec["First_Name"] = first_name
    if account_id: rec["Account_Name"] = {"id": account_id}
    if email: rec["Email"] = email
    if phone: rec["Phone"] = phone
    if mobile: rec["Mobile"] = mobile
    if contact_position: rec["Contact_Position"] = contact_position
    if linkedin: rec["URL_Linkedin_Profile"] = linkedin
    if owner_id: rec["Owner"] = {"id": owner_id}
    return _first_result(_request("POST", "/Contacts", body={"data": [rec]}))


def create_deal(
    *,
    deal_name: str,
    stage: str,
    pipeline: str,
    account_id: str | None = None,
    contact_id: str | None = None,
    potential_type: str | None = None,
    met_by_consultant_at: list[str] | None = None,
    lead_source: str | None = None,
    potential_target: str | None = None,
    owner_id: str | None = None,
    layout_id: str | None = None,
) -> dict:
    rec: dict = {
        "Deal_Name": deal_name,
        "Stage": stage,
        "Pipeline": pipeline,
    }
    if account_id: rec["Account_Name"] = {"id": account_id}
    if contact_id: rec["Contact_Name"] = {"id": contact_id}
    if potential_type: rec["Potential_Type"] = potential_type
    if met_by_consultant_at: rec["Met_By_Consultant_At"] = met_by_consultant_at
    if lead_source: rec["Lead_Source"] = lead_source
    if potential_target: rec["Potential_Target"] = potential_target
    if owner_id: rec["Owner"] = {"id": owner_id}
    if layout_id: rec["Layout"] = {"id": layout_id}
    return _first_result(_request("POST", "/Deals", body={"data": [rec]}))


# ── High-level helpers used by server.py ─────────────────────────────────────

def crm_preview_for_contact(parsed: dict) -> dict:
    """Look up Account (by company name) and Contact (by email).
    Returns a dict the UI can render directly."""
    company = parsed.get("company")
    email = parsed.get("email")
    # Multi-email comma-joined: just use the first
    if email and "," in email:
        email = email.split(",")[0].strip()

    account = search_account(company) if company else None
    contact = search_contact(email) if email else None

    return {
        "company_searched": company,
        "company_status": "exists" if account else "new",
        "account_id": account.get("id") if account else None,
        "account_name": account.get("Account_Name") if account else None,
        "email_searched": email,
        "contact_status": "exists" if contact else "new",
        "contact_id": contact.get("id") if contact else None,
        "contact_existing_account_id": (contact.get("Account_Name") or {}).get("id") if contact else None,
        "contact_existing_account_name": (contact.get("Account_Name") or {}).get("name") if contact else None,
    }


def crm_push(parsed: dict, choices: dict) -> dict:
    """Create Account (if missing), Contact (if missing), then Deal.
    `choices` carries the user-selected Stage/Pipeline/Summit/Owner/etc.
    """
    # Belt-and-suspenders: re-check the environment on every write.
    assert_safe_environment()

    company = (parsed.get("company") or "").strip() or "Unknown Company"
    email = parsed.get("email") or None
    if email and "," in email:
        email = email.split(",")[0].strip()
    phone  = parsed.get("phone")
    mobile = parsed.get("mobile")
    if phone and "," in phone:  phone  = phone.split(",")[0].strip()
    if mobile and "," in mobile: mobile = mobile.split(",")[0].strip()
    address = parsed.get("address") or ""
    country = None
    if address and "," in address:
        # crude country pick: last comma chunk that isn't a postcode
        parts = [p.strip() for p in address.split(",") if p.strip()]
        if parts:
            country = parts[-1]

    owner_id = choices.get("owner_id") or None
    summit_type = choices.get("potential_type") or None  # string
    met_at = choices.get("met_by_consultant_at") or None  # may be string OR list
    if isinstance(met_at, str) and met_at:
        met_at = [met_at]
    layout_id = choices.get("layout_id") or get_layout_id("Deals", LAYOUT_NAME_PREF)

    # 1) Account
    account = search_account(company)
    created_account = False
    if account:
        account_id = account["id"]
        account_name = account.get("Account_Name", company)
    else:
        details = create_account(
            name=company,
            website=parsed.get("website"),
            phone=phone or mobile,
            billing_country=country,
            billing_street=address,
            tradeshows=[summit_type] if summit_type else None,
            owner_id=owner_id,
        )
        account_id = details["id"]
        account_name = company
        created_account = True

    # 2) Contact
    contact = search_contact(email)
    created_contact = False
    if contact:
        contact_id = contact["id"]
        contact_name = (contact.get("Full_Name")
                        or f"{contact.get('First_Name','')} {contact.get('Last_Name','')}".strip())
    else:
        first, last = split_full_name(parsed.get("name"))
        details = create_contact(
            last_name=last, first_name=first,
            account_id=account_id,
            email=email, phone=phone, mobile=mobile,
            contact_position=parsed.get("job_title"),
            linkedin=parsed.get("linkedin"),
            owner_id=owner_id,
        )
        contact_id = details["id"]
        contact_name = f"{first or ''} {last}".strip()
        created_contact = True

    # 3) Deal (Potential)
    deal_name_pattern = choices.get("deal_name") or f"{account_name} - {summit_type or 'Card'}"
    details = create_deal(
        deal_name=deal_name_pattern,
        stage=choices["stage"],
        pipeline=choices["pipeline"],
        account_id=account_id,
        contact_id=contact_id,
        potential_type=summit_type,
        met_by_consultant_at=met_at,
        lead_source=choices.get("lead_source") or "Trade Show",
        potential_target=choices.get("potential_target"),
        owner_id=owner_id,
        layout_id=layout_id,
    )
    deal_id = details["id"]

    deal_url = f"{UI_BASE}/tab/Potentials/{deal_id}"
    return {
        "account_id": account_id,
        "account_created": created_account,
        "account_name": account_name,
        "account_url": f"{UI_BASE}/tab/Accounts/{account_id}",
        "contact_id": contact_id,
        "contact_created": created_contact,
        "contact_name": contact_name,
        "contact_url": f"{UI_BASE}/tab/Contacts/{contact_id}",
        "deal_id": deal_id,
        "deal_name": deal_name_pattern,
        "deal_url": deal_url,
        "environment": environment_label(),
    }
