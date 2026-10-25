#!/usr/bin/env python3
"""
pipeline_db.py — Central state tracker for the PDF→Image→OCR pipeline.

This module is imported by all other pipeline scripts. It owns the SQLite DB
that tracks every PDF through its processing stages. No script touches the
filesystem state of another script's output directly — they only read/write
this DB, which is the single source of truth for "what has been done."

DB lives at: $BERTAAV_ROOT/logs/pipeline.db
(on the HPC native filesystem — NOT under /mnt/... which has I/O issues)
"""

import os
import sqlite3
import threading
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = os.environ.get(
    "BERTAAV_ROOT",
    "/home/USER/projects/BERTaav"
)

RAW_PDFS_DIR   = os.path.join(ROOT, "data", "raw_pdfs")
IMAGES_DIR     = os.path.join(ROOT, "data", "images")
OCR_OUTPUT_DIR = os.path.join(ROOT, "data", "ocr_output")
LOGS_DIR       = os.path.join(ROOT, "logs")
DB_PATH        = os.path.join(LOGS_DIR, "pipeline.db")

# Pipeline stages (column names in the DB = status values)
STATUS_PENDING       = "pending"          # discovered, not yet processed
STATUS_IMAGES_DONE   = "images_done"      # PDF→images complete
STATUS_OCR_DONE      = "ocr_done"         # OCR complete, JSON written
STATUS_FAILED_IMAGES = "failed_images"    # PDF→images failed
STATUS_FAILED_OCR    = "failed_ocr"       # OCR failed

# ── Thread-safe DB access ─────────────────────────────────────────────────────
_DB_LOCK = threading.Lock()


def get_conn():
    os.makedirs(LOGS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call multiple times."""
    with _DB_LOCK:
        conn = get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                -- Identity
                item_uuid        TEXT NOT NULL,
                filename         TEXT NOT NULL,
                source_pdf_path  TEXT NOT NULL PRIMARY KEY,

                -- Source metadata (populated at manifest-build time)
                source_type      TEXT DEFAULT 'lok_sabha_qa',
                source_collection TEXT DEFAULT 'Part 1 (Questions And Answers)',
                title            TEXT,
                source_pdf_bytes INTEGER,

                -- Pipeline state
                status           TEXT DEFAULT 'pending',
                num_pages        INTEGER,

                -- Image conversion
                images_dir       TEXT,
                images_done_at   TEXT,
                images_error     TEXT,

                -- OCR
                ocr_json_path    TEXT,
                ocr_confidence   REAL,
                ocr_text_bytes   INTEGER,
                ocr_done_at      TEXT,
                ocr_error        TEXT,

                -- Housekeeping
                discovered_at    TEXT DEFAULT (datetime('now')),
                updated_at       TEXT
            )
        """)
        conn.commit()
        conn.close()


# ── Manifest builder ──────────────────────────────────────────────────────────

def sync_manifest(source_type="lok_sabha_qa",
                  source_collection="Part 1 (Questions And Answers)"):
    """
    Scan RAW_PDFS_DIR for PDFs not yet in the DB and INSERT them as 'pending'.
    Safe to call repeatedly as new files arrive — only inserts truly new files
    (INSERT OR IGNORE), never touches existing rows.

    Returns: (new_files_added, total_in_db)
    """
    init_db()
    pdfs = [
        f for f in os.listdir(RAW_PDFS_DIR)
        if f.endswith(".pdf") and not f.endswith(".part")
    ]

    new_count = 0
    with _DB_LOCK:
        conn = get_conn()
        for fname in pdfs:
            fpath = os.path.join(RAW_PDFS_DIR, fname)
            fsize = os.path.getsize(fpath)

            # Filename convention: {item_uuid}__{original_name}.pdf
            # Fall back gracefully if the naming convention differs.
            if "__" in fname:
                item_uuid, original = fname.split("__", 1)
                # Use the question title from the filename if available
                # (the downloader uses the original filename as-is from DSpace,
                # so title metadata must come from the OCR manifest or the
                # downloader's SQLite DB; we use None here and enrich later)
                title = None
            else:
                item_uuid = fname.replace(".pdf", "")
                original  = fname
                title     = None

            cur = conn.execute(
                """INSERT OR IGNORE INTO files
                   (item_uuid, filename, source_pdf_path,
                    source_type, source_collection, title, source_pdf_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (item_uuid, original, fpath,
                 source_type, source_collection, title, fsize)
            )
            new_count += cur.rowcount

        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        conn.close()

    return new_count, total


# ── Status helpers ────────────────────────────────────────────────────────────

def get_pending_images(limit=None):
    """Return files where PDF→image conversion hasn't been done yet."""
    conn = get_conn()
    q = "SELECT * FROM files WHERE status = 'pending' ORDER BY source_pdf_path"
    if limit:
        q += f" LIMIT {limit}"
    rows = [dict(r) for r in conn.execute(q).fetchall()]
    conn.close()
    return rows


def get_pending_ocr(limit=None):
    """Return files where images exist but OCR hasn't been done yet."""
    conn = get_conn()
    q = "SELECT * FROM files WHERE status = 'images_done' ORDER BY source_pdf_path"
    if limit:
        q += f" LIMIT {limit}"
    rows = [dict(r) for r in conn.execute(q).fetchall()]
    conn.close()
    return rows


def mark_images_done(source_pdf_path, images_dir, num_pages):
    with _DB_LOCK:
        conn = get_conn()
        conn.execute(
            """UPDATE files SET status=?, images_dir=?, num_pages=?,
               images_done_at=?, updated_at=datetime('now')
               WHERE source_pdf_path=?""",
            (STATUS_IMAGES_DONE, images_dir, num_pages,
             datetime.now().isoformat(), source_pdf_path)
        )
        conn.commit()
        conn.close()


def mark_images_failed(source_pdf_path, error):
    with _DB_LOCK:
        conn = get_conn()
        conn.execute(
            """UPDATE files SET status=?, images_error=?, updated_at=datetime('now')
               WHERE source_pdf_path=?""",
            (STATUS_FAILED_IMAGES, str(error)[:2000], source_pdf_path)
        )
        conn.commit()
        conn.close()


def mark_ocr_done(source_pdf_path, ocr_json_path, ocr_confidence, ocr_text_bytes):
    with _DB_LOCK:
        conn = get_conn()
        conn.execute(
            """UPDATE files SET status=?, ocr_json_path=?, ocr_confidence=?,
               ocr_text_bytes=?, ocr_done_at=?, updated_at=datetime('now')
               WHERE source_pdf_path=?""",
            (STATUS_OCR_DONE, ocr_json_path, ocr_confidence, ocr_text_bytes,
             datetime.now().isoformat(), source_pdf_path)
        )
        conn.commit()
        conn.close()


def mark_ocr_failed(source_pdf_path, error):
    with _DB_LOCK:
        conn = get_conn()
        conn.execute(
            """UPDATE files SET status=?, ocr_error=?, updated_at=datetime('now')
               WHERE source_pdf_path=?""",
            (STATUS_FAILED_OCR, str(error)[:2000], source_pdf_path)
        )
        conn.commit()
        conn.close()


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_status():
    init_db()
    conn = get_conn()

    counts = dict(conn.execute(
        "SELECT status, COUNT(*) FROM files GROUP BY status"
    ).fetchall())

    size_stats = conn.execute("""
        SELECT
            source_type,
            COUNT(*) as n_files,
            SUM(source_pdf_bytes) as total_pdf_bytes,
            SUM(ocr_text_bytes) as total_text_bytes
        FROM files
        GROUP BY source_type
    """).fetchall()

    failed_images = conn.execute(
        "SELECT filename, images_error FROM files WHERE status='failed_images' LIMIT 5"
    ).fetchall()

    failed_ocr = conn.execute(
        "SELECT filename, ocr_error FROM files WHERE status='failed_ocr' LIMIT 5"
    ).fetchall()

    conn.close()

    print("\n=== PIPELINE STATUS ===")
    print(f"  DB: {DB_PATH}")
    total = sum(counts.values())
    print(f"  Total files tracked: {total}")
    print()

    stage_order = [
        STATUS_PENDING, STATUS_IMAGES_DONE, STATUS_OCR_DONE,
        STATUS_FAILED_IMAGES, STATUS_FAILED_OCR
    ]
    labels = {
        STATUS_PENDING:       "Pending (not started)",
        STATUS_IMAGES_DONE:   "Images done, OCR pending",
        STATUS_OCR_DONE:      "OCR done ✓",
        STATUS_FAILED_IMAGES: "FAILED at PDF→image",
        STATUS_FAILED_OCR:    "FAILED at OCR",
    }
    for s in stage_order:
        n = counts.get(s, 0)
        bar = "█" * min(40, int(40 * n / max(total, 1)))
        print(f"  {labels[s]:<35} {n:>6}  {bar}")

    print("\n--- Per-source breakdown ---")
    print(f"  {'Source':<30} {'Files':>8} {'PDF size':>12} {'Text size':>12}")
    print("  " + "-" * 64)
    for row in size_stats:
        pdf_mb  = (row["total_pdf_bytes"]  or 0) / 1e6
        text_mb = (row["total_text_bytes"] or 0) / 1e6
        print(f"  {row['source_type']:<30} {row['n_files']:>8} "
              f"{pdf_mb:>10.1f}MB {text_mb:>10.1f}MB")

    if failed_images:
        print("\n--- Recent PDF→image failures (first 5) ---")
        for f in failed_images:
            print(f"  {f['filename']}: {f['images_error'][:120]}")

    if failed_ocr:
        print("\n--- Recent OCR failures (first 5) ---")
        for f in failed_ocr:
            print(f"  {f['filename']}: {f['ocr_error'][:120]}")

    print()


if __name__ == "__main__":
    print_status()
