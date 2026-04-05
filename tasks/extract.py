"""
tasks/extract.py
----------------
Extract task for the LGBTIQ+ compliance pipeline.

Reads legal_inventory.csv, resolves the actual PDF/text URL from each
fichaOrdenamiento.php viewer page, extracts text via pdfplumber,
and saves per-state .txt files to data/raw/{state_name}/.

Morelos is handled separately using local PDFs in data/morelos/.
"""

from __future__ import annotations

import logging
import pathlib
import tempfile
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
import pdfplumber
from dotenv import load_dotenv
from urllib.parse import urljoin

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE         = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT  = _HERE.parent
INVENTORY_CSV = PROJECT_ROOT / "data" / "output" / "legal_inventory.csv"
MORELOS_DIR   = PROJECT_ROOT / "data" / "morelos"
RAW_DIR       = PROJECT_ROOT / "data" / "raw"

# ── State list ────────────────────────────────────────────────────────────────
# Must match the values in the `state` column of legal_inventory.csv exactly.
STATES: List[str] = [
    "Aguascalientes",
    "Baja California",
    "Baja California Sur",
    "Campeche",
    "Chiapas",
    "Chihuahua",
    "Ciudad de Mexico",
    "Coahuila",
    "Colima",
    "Durango",
    "Guanajuato",
    "Guerrero",
    "Hidalgo",
    "Jalisco",
    "Estado de Mexico",
    "Michoacan",
    "Morelos",
    "Nayarit",
    "Nuevo Leon",
    "Oaxaca",
    "Puebla",
    "Queretaro",
    "Quintana Roo",
    "San Luis Potosi",
    "Sinaloa",
    "Sonora",
    "Tabasco",
    "Tamaulipas",
    "Tlaxcala",
    "Veracruz",
    "Yucatan",
    "Zacatecas",
]

# ── Pilot mode ────────────────────────────────────────────────────────────────
# Set to True to process only Morelos (local PDFs) for pipeline testing.
# Set to False for full 32-state production run.
PILOT_MODE = False

# ── HTTP session ──────────────────────────────────────────────────────────────
BASE_URL    = "http://www.ordenjuridico.gob.mx"
CRAWL_DELAY = 0.5   # seconds between requests
TIMEOUT     = 30    # seconds per request

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; LGBTIQPipeline/1.0; "
        "legal-compliance-research)"
    )
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_pdf_url(ficha_url: str) -> str | None:
    """
    Fetch a fichaOrdenamiento.php viewer page and return the direct PDF URL.

    Page structure (confirmed via debug_ficha.py):
    - No <iframe>; the PDF link is a plain <a href> ending in .pdf
    - The href may use the pattern /./Estatal/... which must be normalised
      to /Estatal/... to produce a valid URL
    """
    try:
        resp = _SESSION.get(ficha_url, timeout=TIMEOUT)
        resp.encoding = "iso-8859-1"
        soup = BeautifulSoup(resp.text, "html.parser")

        # Priority 1 — iframe src (kept as fallback for other page variants)
        iframe = soup.find("iframe", src=True)
        if iframe:
            return urljoin(BASE_URL, iframe["src"])

        # Priority 2 — <a> link whose href ends in .pdf or .htm
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith((".pdf", ".htm", ".html")):
                return urljoin(BASE_URL, href)

    except Exception as exc:
        log.warning("Could not resolve %s: %s", ficha_url, exc)

    return None


def _extract_with_pdfplumber(url_or_path: str) -> str:
    """
    Extract text from a digital PDF using pdfplumber.

    If url_or_path is an HTTP(S) URL, the file is downloaded to a temp file
    first, then opened with pdfplumber.  Local paths are opened directly.
    Returns full text as a single string. Returns empty string on failure.
    """
    try:
        if url_or_path.startswith("http"):
            response = _SESSION.get(url_or_path, timeout=TIMEOUT)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            with pdfplumber.open(tmp_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            pathlib.Path(tmp_path).unlink(missing_ok=True)
        else:
            with pdfplumber.open(url_or_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages)
    except Exception as exc:
        log.warning("pdfplumber extraction failed for %s: %s", url_or_path, exc)
        return ""


def _save_text(state_name: str, file_stem: str, text: str) -> pathlib.Path:
    """
    Save extracted text to data/raw/{state_name}/{file_stem}.txt.
    Returns the path written.
    """
    out_dir = RAW_DIR / _sanitise(state_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_sanitise(file_stem)}.txt"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _sanitise(name: str) -> str:
    """Replace characters unsafe for directory/file names."""
    return (
        name.replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "")
    )


# ── Main extraction logic ─────────────────────────────────────────────────────

def _extract_state_from_inventory(
    state: str,
    rows: pd.DataFrame,
) -> List[Tuple[str, str]]:
    """
    Extract all documents for a single state from the inventory CSV.

    For each row the function:
    1. Resolves the direct document URL from the fichaOrdenamiento.php page.
    2. Skips the document if a .txt file already exists for it.
    3. Passes the resolved URL to Docling.
    4. Saves the result and appends it to the return list.

    Returns a list of (file_name, extracted_text) tuples.
    """
    results: List[Tuple[str, str]] = []
    extracted = 0
    skipped   = 0

    for _, row in rows.iterrows():
        file_stem = _sanitise(str(row["law_title"])[:80])
        out_path  = RAW_DIR / _sanitise(state) / f"{file_stem}.txt"

        if out_path.exists():
            text = out_path.read_text(encoding="utf-8")
            results.append((str(out_path), text))
            skipped += 1
            continue

        ficha_url = str(row["url"])

        # Fast path: direct PDF URL — skip the viewer page resolution
        if ficha_url.lower().endswith(".pdf"):
            direct_url = ficha_url
        else:
            direct_url = _resolve_pdf_url(ficha_url)
            time.sleep(CRAWL_DELAY)

        if not direct_url:
            log.warning("[%s] Could not resolve direct URL for: %s", state, row["law_title"])
            continue

        text = _extract_with_pdfplumber(direct_url)
        time.sleep(CRAWL_DELAY)

        if not text.strip():
            log.warning("[%s] Empty extraction for: %s", state, row["law_title"])
            continue

        _save_text(state, file_stem, text)
        results.append((str(out_path), text))
        extracted += 1

    log.info("[%s] extracted=%d  skipped=%d  total=%d", state, extracted, skipped, len(results))
    return results


def _extract_morelos_local() -> List[Tuple[str, str]]:
    """
    Extract text from the Morelos fallback PDFs stored in data/morelos/.
    Files already extracted (data/raw/Morelos/*.txt) are skipped.

    Returns a list of (file_name, extracted_text) tuples.
    """
    results: List[Tuple[str, str]] = []
    extracted = 0
    skipped   = 0

    for pdf_path in sorted(MORELOS_DIR.glob("*.pdf")):
        file_stem = pdf_path.stem
        out_path  = RAW_DIR / "Morelos" / f"{file_stem}.txt"

        if out_path.exists():
            text = out_path.read_text(encoding="utf-8")
            results.append((str(out_path), text))
            skipped += 1
            continue

        text = _extract_with_pdfplumber(str(pdf_path))

        if not text.strip():
            log.warning("[Morelos] Empty extraction for: %s", pdf_path.name)
            continue

        _save_text("Morelos", file_stem, text)
        results.append((str(out_path), text))
        extracted += 1

    log.info("[Morelos] extracted=%d  skipped=%d  total=%d", extracted, skipped, len(results))
    return results


# ── Airflow callables ─────────────────────────────────────────────────────────

def extract_state(state_name: str) -> Dict[str, List[str]]:
    """
    Airflow task callable — extract documents for a single state.

    Called once per state by the parallel extract tasks in the DAG.
    Morelos uses local PDFs from data/morelos/; all other states read from
    legal_inventory.csv and resolve URLs via ordenjuridico.gob.mx.

    In PILOT_MODE, only Morelos produces real output; all other states
    return an empty list immediately without hitting the network.

    Returns
    -------
    dict
        {state_name: [absolute_file_path_strings]}
        Passed to the transform task via XCom (file paths only, no text).
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    # ── Pilot mode: skip non-Morelos states ───────────────────────────────────
    if PILOT_MODE and state_name.strip().lower() != "morelos":
        log.info("PILOT_MODE=True — skipping %s", state_name)
        return {state_name: []}

    # ── Morelos: read local PDFs ───────────────────────────────────────────────
    if state_name.strip().lower() == "morelos":
        tuples = _extract_morelos_local()
        return {"Morelos": [fp for fp, _ in tuples]}

    # ── All other states: read from inventory CSV ─────────────────────────────
    if not INVENTORY_CSV.exists():
        raise FileNotFoundError(
            f"Inventory CSV not found: {INVENTORY_CSV}\n"
            "Run notebook 01 and export the CSV first."
        )

    inventory = pd.read_csv(INVENTORY_CSV, encoding="utf-8-sig")
    group = inventory[
        inventory["state"].str.strip().str.lower() == state_name.strip().lower()
    ]

    if group.empty:
        log.warning("[%s] No rows found in inventory — check that the state name matches the CSV exactly.", state_name)
        return {state_name: []}

    tuples = _extract_state_from_inventory(state_name, group)
    return {state_name: [fp for fp, _ in tuples]}


def extract_all_states() -> Dict[str, List[str]]:
    """
    Airflow task callable — extract documents for all 32 states sequentially.

    Iterates over STATES, calling extract_state() for each one, and merges
    the results into a single dict.  Pool-based concurrency is enforced at
    the Airflow task level (pool="extract_pool"), so this function itself
    runs single-threaded.

    Returns
    -------
    dict
        {state_name: [absolute_file_path_strings]} for all states.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    all_results: Dict[str, List[str]] = {}
    for state in STATES:
        result = extract_state(state)
        all_results.update(result)

    total = sum(len(v) for v in all_results.values())
    log.info("extract_all_states complete: %d states, %d documents", len(all_results), total)
    return all_results


def extract_pdfs() -> Dict[str, List[str]]:
    """
    Morelos-only fallback callable (kept for ad-hoc use / backward compat).
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    tuples = _extract_morelos_local()
    return {"Morelos": [fp for fp, _ in tuples]}
