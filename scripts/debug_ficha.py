"""
scripts/debug_ficha.py
----------------------
Debug helper: fetches one fichaOrdenamiento.php viewer page and prints
its structure so _resolve_pdf_url() can be tuned to the real HTML layout.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
from bs4 import BeautifulSoup

URL = (
    "http://www.ordenjuridico.gob.mx/"
    "fichaOrdenamiento.php?idArchivo=25642&ambito=estatal"
)

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; LGBTIQPipeline/1.0; "
        "legal-compliance-research)"
    )
})

resp = session.get(URL, timeout=30)
resp.encoding = "iso-8859-1"
html = resp.text

print("=" * 60)
print("STATUS CODE:", resp.status_code)
print("ENCODING:   ", resp.encoding)
print("=" * 60)

print("\n-- RAW HTML (first 3000 chars) " + "-" * 30)
print(html[:3000])

soup = BeautifulSoup(html, "html.parser")

print("\n── <a> tags with href " + "─" * 38)
for tag in soup.find_all("a", href=True):
    print(" ", tag["href"][:200])

print("\n── <iframe> tags with src " + "─" * 34)
for tag in soup.find_all("iframe", src=True):
    print(" ", tag["src"][:200])

print("\n── <script> tags (first 200 chars each) " + "─" * 20)
for tag in soup.find_all("script"):
    content = (tag.string or "").strip()
    if content:
        print(" ", content[:200])
        print()
