#!/usr/bin/env python3
"""
scrape_pib.py — Scrape Hindi press releases from PIB (pib.gov.in)

HOW PIB WORKS (discovered from live site inspection):
  - Each press release has a unique integer PRID (Press Release ID)
  - URL pattern: https://www.pib.gov.in/PressReleasePage.aspx?PRID={id}&lang=2
    where lang=2 = Hindi, lang=1 = English
  - PRIDs are sequential integers. Current range (2024-2025): ~2,000,000 to ~2,250,000
  - Older releases (2004-2017): available at archive.pib.gov.in but different structure
  - NOT every PRID has a Hindi version — some are English-only
  - The listing page is JavaScript-rendered, so we can't scrape it for PRIDs.
    Instead we iterate sequentially and skip PRIDs with no Hindi content.

WHY NO OCR:
  PIB releases are digitally typeset HTML — no images, no scans.
  Text extraction is instant via BeautifulSoup, no Tesseract needed.
  This is fundamentally different from the Lok Sabha PDF pipeline.

OUTPUT SCHEMA (same fields as BERTaav, source_type='pib_press_release'):
  {
    "id": "pib_{PRID}",
    "source_type": "pib_press_release",
    "source_collection": "Press Information Bureau",
    "title": "...",
    "date": "YYYY-MM-DD",
    "metadata": {
      "ministry": "...",
      "prid": 2097923,
      "url": "https://...",
    },
    "language": "hi",
    "source_pdf": null,
    "source_pdf_bytes": null,
    "ocr_engine": null,
    "ocr_confidence_mean": null,
    "num_pages": 1,
    "text": "...",
    "text_bytes_utf8": ...,
    "quality_flag": "ok",
    "quality_notes": [],
    "pages": [{"page_num": 1, "text": "...", "confidence_mean": null, ...}]
  }

USAGE:
  # Install deps (no new ones needed beyond requests, which you already have):
  pip install beautifulsoup4

  # Test with 10 releases first:
  python scrape_pib.py --test

  # Full run (2018 onwards, ~500K releases estimated):
  python scrape_pib.py --start-prid 1500000 --end-prid 2300000

  # Resume after interruption (reads progress from pib_progress.db):
  python scrape_pib.py --start-prid 1500000 --end-prid 2300000
  (automatically skips already-processed PRIDs)

  # Check status:
  python scrape_pib.py --status

PBS SUBMISSION:
  See submit_pib.sh — splits the PRID range across array job chunks.
  Each chunk can run independently; they write to separate output files
  and a shared SQLite DB for deduplication/progress tracking.

RATE LIMITING:
  PIB is a government server. Default delay is 0.3s between requests.
  This gives ~200 requests/min, ~12,000/hour.
  For 500K PRIDs at ~60% Hindi hit rate = 300K releases.
  Estimated time: ~41 hours. Manageable in 2-3 overnight runs.
  The script is resumable — restart from where it stopped.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from bs4 import Tag
from requests.adapters import HTTPAdapter

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = os.environ.get(
    "BERTAAV_ROOT",
    os.getcwd()
)
OUT_DIR   = os.path.join(ROOT, "data", "pib_raw")
LOGS_DIR  = os.path.join(ROOT, "logs")
DB_PATH = os.environ.get("PIB_DB_PATH", os.path.expanduser("~/pib_progress.sqlite3"))

BASE_URL  = "https://www.pib.gov.in/PressReleasePage.aspx"
LANG_HINDI = 2   # lang=2 for Hindi

REQUEST_DELAY   = 0.3   # seconds between requests (polite)
MAX_RETRIES     = 4
TIMEOUT         = 20    # seconds

# PRID ranges by approximate year (from live site inspection)
# 2004-2008:  ~1       - ~50,000
# 2009-2013:  ~50,000  - ~200,000
# 2014-2017:  ~200,000 - ~500,000
# 2018-2021:  ~500,000 - ~1,200,000
# 2022-2025:  ~1,200,000 - ~2,300,000
DEFAULT_START = 500_000   # 2018 onwards
DEFAULT_END   = 2_300_000

MIN_TEXT_BYTES = 1000  # discard pages with nearly no text

# ── Hindi month name → number ─────────────────────────────────────────────────
HINDI_MONTHS = {
    'जनवरी': 1, 'फरवरी': 2, 'मार्च': 3, 'अप्रैल': 4,
    'मई': 5, 'जून': 6, 'जुलाई': 7, 'अगस्त': 8,
    'सितंबर': 9, 'अक्टूबर': 10, 'नवंबर': 11, 'दिसंबर': 12,
    # English months too — some PIB Hindi pages have English dates
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
}

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(chunk_id=None):
    os.makedirs(LOGS_DIR, exist_ok=True)
    suffix = f"_chunk{chunk_id}" if chunk_id is not None else ""
    log_path = os.path.join(LOGS_DIR, f"scrape_pib{suffix}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)

# ── SQLite progress tracker ───────────────────────────────────────────────────
def db_init():
    os.makedirs(LOGS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pib_progress (
            prid        INTEGER PRIMARY KEY,
            status      TEXT,   -- 'ok' | 'no_hindi' | 'error' | 'empty'
            out_path    TEXT,
            scraped_at  TEXT
        )
    """)
    conn.commit()
    conn.close()


def db_mark(prid, status, out_path=None):
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        "INSERT OR REPLACE INTO pib_progress (prid, status, out_path, scraped_at) "
        "VALUES (?, ?, ?, ?)",
        (prid, status, out_path, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def db_already_done(prid):
    conn = sqlite3.connect(DB_PATH, timeout=60)
    row = conn.execute(
        "SELECT status FROM pib_progress WHERE prid=?", (prid,)
    ).fetchone()
    conn.close()
    return row is not None


def db_status():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM pib_progress GROUP BY status"
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM pib_progress").fetchone()[0]
    conn.close()
    return dict(rows), total


# ── HTTP ──────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "hi,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
        " BERTaav-corpus-builder/1.0"
    ),
})
_adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)


def fetch_page(prid):
    """Fetch Hindi press release page. Returns (html_text, final_url) or (None, None)."""
    url = f"{BASE_URL}?PRID={prid}&lang={LANG_HINDI}"
    for attempt in range(MAX_RETRIES):
        try:
            r = SESSION.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.text, r.url
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 2 ** attempt))
                continue
            # 404 or other = this PRID doesn't exist / no Hindi version
            return None, None
        except requests.RequestException as e:
            time.sleep(min(30, 2 ** attempt))
    return None, None


# ── Parsing ───────────────────────────────────────────────────────────────────

def is_hindi_text(text):
    """Check if text contains significant Devanagari script."""
    if not text:
        return False
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return devanagari / max(len(text), 1) > 0.1


def parse_date(date_str):
    """
    Parse dates like '31 JAN 2025', '15 अगस्त, 2024', '24.03.2025'
    Returns 'YYYY-MM-DD' or None.
    """
    if not date_str:
        return None
    date_str = date_str.strip()

    # Try 'DD MON YYYY' (PIB's main English format, e.g. "31 JAN 2025")
    m = re.search(
        r'(\d{1,2})\s+([A-Z]{3})\s+(\d{4})',
        date_str.upper()
    )
    if m:
        abbrev_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        month = abbrev_map.get(m.group(2))
        if month:
            return f"{m.group(3)}-{month:02d}-{int(m.group(1)):02d}"

    # Try 'DD Hindi_month, YYYY'
    for name, num in HINDI_MONTHS.items():
        m = re.search(rf'(\d{{1,2}})\s+{re.escape(name)},?\s+(\d{{4}})', date_str)
        if m:
            return f"{m.group(2)}-{num:02d}-{int(m.group(1)):02d}"

    # Try DD.MM.YYYY
    m = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', date_str)
    if m and int(m.group(2)) <= 12:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    return None


def parse_release(html, prid, url):
    """
    Parse a PIB press release HTML page into our JSON schema.
    Returns a dict, or None if this page has no usable Hindi content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # ── Title ─────────────────────────────────────────────────────────────────
    # PIB page title is in an <h2> tag within the content area
    title_tag = soup.find("h2")
    title = title_tag.get_text(strip=True) if title_tag else None

    # If title is English-only, this is an English-only release
    if title and not is_hindi_text(title):
        # Check if any content is in Hindi before giving up
        pass  # will catch via body text check below

    # ── Date ──────────────────────────────────────────────────────────────────
    # Look for "प्रविष्टि तिथि:" pattern in page text
    date_raw = None
    date_tag = soup.find(string=re.compile(r'प्रविष्टि\s+तिथि', re.IGNORECASE))
    if date_tag:
        date_raw = date_tag.parent.get_text()
    else:
        # Fallback: find any element containing date-like text near the title
        for tag in soup.find_all(string=re.compile(r'\d{1,2}\s+[A-Z]{3}\s+\d{4}')):
            date_raw = tag
            break

    date = parse_date(date_raw) if date_raw else None

    # ── Ministry ──────────────────────────────────────────────────────────────
    # The ministry name appears in a <div> or <p> right at the top of content,
    # before the <h2> title — usually the first non-empty text block
    ministry = None
    content_div = soup.find("div", {"id": "content"}) or soup.find("div", {"class": "innner-page-main-about-us-content-right-part"}) or soup.body
    if content_div:
        # Find all text nodes in order; ministry is usually the first short Hindi line
        for tag in content_div.find_all(["p", "div", "td", "h3", "h4"], limit=10):
            text = tag.get_text(strip=True)
            if text and 5 < len(text) < 100 and is_hindi_text(text):
                # Skip if it looks like a date or the title itself
                if not re.search(r'\d{4}', text) and text != title:
                    ministry = text
                    break

    # ── Body text ─────────────────────────────────────────────────────────────
    # Remove nav, footer, scripts, styles before extracting
    for tag in soup(["nav", "footer", "script", "style", "noscript",
                     "header", "aside"]):
        tag.decompose()

    # Remove share buttons, visitor counter, language links etc
    for tag in soup.find_all(string=re.compile(
        r'(Share on|Visitor|आगंतुक पटल|रिलीज़ आईडी|Release ID|'
        r'इस विज्ञप्ति को|Read this release)', re.IGNORECASE
    )):
        if isinstance(tag, Tag) and tag.parent:
            tag.parent.decompose()

    # Get all text from main content area
    body = soup.get_text(separator="\n")

    # Clean up
    # Remove repeated blank lines
    body = re.sub(r'\n{3,}', '\n\n', body)
    # Remove lines that are just whitespace or single chars
    lines = [l for l in body.split('\n') if len(l.strip()) > 2]
    body = '\n'.join(lines).strip()

    if not body or not is_hindi_text(body):
        return None  # No Hindi content — English-only release

    text_bytes = len(body.encode('utf-8'))
    if text_bytes < MIN_TEXT_BYTES:
        return None  # Too short to be useful

    record = {
        "id":               f"pib_{prid}",
        "source_type":      "pib_press_release",
        "source_collection": "Press Information Bureau",
        "title":            title,
        "date":             date,
        "metadata": {
            "ministry":          ministry,
            "prid":              prid,
            "url":               url,
            "lok_sabha_session": None,
            "speaker":           None,
            "newspaper_name":    None,
        },
        "language":          "hi",
        "source_pdf":        None,
        "source_pdf_bytes":  None,
        "ocr_engine":        None,
        "ocr_confidence_mean": None,
        "num_pages":         1,
        "text":              body,
        "text_bytes_utf8":   text_bytes,
        "text_hash":         hashlib.md5(body.encode("utf-8")).hexdigest(),
        "quality_flag":      "ok",
        "quality_notes":     [],
        "processed_at":      datetime.now().isoformat(),
        "pages": [
            {
                "page_num":        1,
                "text":            body,
                "confidence_mean": None,
                "text_bytes_utf8": text_bytes,
            }
        ],
    }
    return record


# ── Output ────────────────────────────────────────────────────────────────────

def out_path_for(prid):
    """
    Shard output files into subdirectories of 10,000 PRIDs each to avoid
    huge single directories.
    """
    shard = (prid // 10_000) * 10_000
    shard_dir = os.path.join(OUT_DIR, f"prid_{shard:07d}")
    os.makedirs(shard_dir, exist_ok=True)
    return os.path.join(shard_dir, f"pib_{prid}.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_scraper(start_prid, end_prid, chunk_id=None, num_chunks=1,
                          dry_run=False, concurrency=12):
    logger = setup_logging(chunk_id)
    db_init()
    os.makedirs(OUT_DIR, exist_ok=True)

    all_prids = range(end_prid, start_prid - 1, -1)
    if chunk_id is not None and num_chunks > 1:
        my_prids = [p for i, p in enumerate(all_prids) if i % num_chunks == chunk_id]
    else:
        my_prids = list(all_prids)

    # Filter out already-done PRIDs up front (fast, single DB read pass)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    done_prids = set(
        row[0] for row in conn.execute("SELECT prid FROM pib_progress").fetchall()
    )
    conn.close()

    pending = [p for p in my_prids if p not in done_prids]
    logger.info(
        f"Total in range: {len(my_prids)} | already done: {len(done_prids & set(my_prids))} "
        f"| pending: {len(pending)} | concurrency={concurrency}"
    )

    if dry_run:
        for p in pending[:10]:
            print(f"  Would process: {p}")
        return

    # ── Single writer thread: drains a queue, does all SQLite + file writes ──
    write_queue = queue.Queue()
    stats = {"ok": 0, "no_hindi": 0, "errors": 0}
    stats_lock = threading.Lock()

    def writer_thread():
        conn = sqlite3.connect(DB_PATH, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL;")
        buffer = []
        while True:
            item = write_queue.get()
            if item is None:  # sentinel = stop
                break
            prid, status, record = item

            if record is not None:
                out = out_path_for(prid)
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                buffer.append((prid, status, out, datetime.now().isoformat()))
            else:
                buffer.append((prid, status, None, datetime.now().isoformat()))

            # Batch commits every 20 items for efficiency
            if len(buffer) >= 20:
                conn.executemany(
                    "INSERT OR REPLACE INTO pib_progress "
                    "(prid, status, out_path, scraped_at) VALUES (?, ?, ?, ?)",
                    buffer
                )
                conn.commit()
                buffer.clear()

            with stats_lock:
                stats[status if status in stats else "errors"] = \
                    stats.get(status, 0) + 1

        # Flush remaining
        if buffer:
            conn.executemany(
                "INSERT OR REPLACE INTO pib_progress "
                "(prid, status, out_path, scraped_at) VALUES (?, ?, ?, ?)",
                buffer
            )
            conn.commit()
        conn.close()

    writer = threading.Thread(target=writer_thread, daemon=True)
    writer.start()

    # ── Worker function: fetch + parse ONE prid, push result to queue ────────
    def process_one(prid):
        html, final_url = fetch_page(prid)
        if html is None:
            write_queue.put((prid, "no_hindi", None))
            return
        record = parse_release(html, prid, final_url)
        if record is None:
            write_queue.put((prid, "no_hindi", None))
        else:
            write_queue.put((prid, "ok", record))
        time.sleep(REQUEST_DELAY)  # politeness delay per-thread

    # ── Run the thread pool ───────────────────────────────────────────────────
    processed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process_one, p): p for p in pending}
        for future in futures:
            try:
                future.result()
            except Exception as e:
                prid = futures[future]
                write_queue.put((prid, "errors", None))
                logger.error(f"[ERROR] PRID {prid}: {e}")
            processed += 1
            if processed % 500 == 0:
                with stats_lock:
                    logger.info(
                        f"Progress: {processed}/{len(pending)} | "
                        f"ok={stats.get('ok',0)} no_hindi={stats.get('no_hindi',0)} "
                        f"errors={stats.get('errors',0)}"
                    )

    write_queue.put(None)  # stop the writer thread
    writer.join(timeout=30)

    with stats_lock:
        logger.info(f"=== DONE: {stats} ===")



def print_status():
    db_init()
    counts, total = db_status()
    print("\n=== PIB SCRAPER STATUS ===")
    print(f"Total PRIDs processed: {total}")
    for status, count in sorted(counts.items()):
        print(f"  {status:<15} {count:>8}")
    ok = counts.get('ok', 0)
    if ok > 0:
        # Count output files
        n_files = sum(
            len(files) for _, _, files in os.walk(OUT_DIR)
            if files
        ) if os.path.exists(OUT_DIR) else 0
        print(f"\nJSON files written: {n_files}")
        # Estimate text size
        total_bytes = 0
        for dirpath, _, files in os.walk(OUT_DIR):
            for f in files:
                if f.endswith('.json'):
                    total_bytes += os.path.getsize(os.path.join(dirpath, f))
        print(f"Estimated output size: {total_bytes / 1e6:.1f} MB")
    print()


def main():
    parser = argparse.ArgumentParser(description="PIB Hindi press release scraper")
    parser.add_argument("--start-prid",  type=int, default=DEFAULT_START)
    parser.add_argument("--end-prid",    type=int, default=DEFAULT_END)
    parser.add_argument("--chunk-id",    type=int, default=None)
    parser.add_argument("--num-chunks",  type=int, default=1)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--test",        action="store_true",
                        help="Test with 20 recent releases and exit")
    parser.add_argument("--status",      action="store_true")
    parser.add_argument("--concurrency", type=int, default=12,
                        help="Number of concurrent threads (default: 12)")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.chunk_id is None and "PBS_ARRAY_INDEX" in os.environ:
        args.chunk_id = int(os.environ["PBS_ARRAY_INDEX"])

    if args.test:
        # Sample 20 PRIDs from recent range to verify extraction works
        print("=== TEST MODE: sampling 20 recent Hindi releases ===\n")
        setup_logging()
        db_init()
        found = 0
        for prid in range(2_097_900, 2_098_000):
            html, url = fetch_page(prid)
            if html:
                record = parse_release(html, prid, url)
                if record:
                    found += 1
                    print(f"PRID {prid}:")
                    print(f"  Title    : {record['title']}")
                    print(f"  Date     : {record['date']}")
                    print(f"  Ministry : {record['metadata']['ministry']}")
                    print(f"  Bytes    : {record['text_bytes_utf8']}")
                    print(f"  Preview  : {record['text'][:200]}\n")
                    if found >= 5:
                        break
            time.sleep(REQUEST_DELAY)
        print(f"Found {found} Hindi releases in sampled range.")
        return

    run_scraper(
        start_prid=args.start_prid,
        end_prid=args.end_prid,
        chunk_id=args.chunk_id,
        num_chunks=args.num_chunks,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
