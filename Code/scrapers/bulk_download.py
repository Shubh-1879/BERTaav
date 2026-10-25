#!/usr/bin/env python3
"""
bulk_download.py — Step 2: Download ALL Hindi PDFs from "Part 1 (Questions
And Answers)" on eLibrary Sansad (~1.15M items).

ONLY RUN THIS AFTER test_run.py HAS WORKED CORRECTLY AND YOU HAVE EYEBALLED
A FEW DOWNLOADED PDFs TO CONFIRM looks_hindi() IS DETECTING CORRECTLY.

DESIGN GOALS (why it's built this way):

1. RESUMABLE: progress is tracked in a local SQLite DB (progress.sqlite3).
   If your HPC job dies, gets killed by the scheduler, or you Ctrl-C it,
   re-running this script picks up exactly where it left off instead of
   re-downloading or re-scanning ~1.15M items from page 0.

2. POLITE CONCURRENCY: uses a bounded thread pool (default 8 workers) with
   a small per-request delay and exponential backoff on errors/429s. This
   is a government server, not a CDN -- being a good citizen here also
   protects you from getting IP-banned partway through a multi-day job.
   Tune CONCURRENCY up cautiously; watch error rates before increasing.

3. TWO-PHASE (crawl, then fetch) using a SQLite work-queue:
   Phase A "discover": paginate the Discover API, and for every item found,
       fetch its bitstream list, decide which are Hindi, and INSERT a row
       per Hindi bitstream into the `files` table (status='pending').
       This phase is itself resumable via a `last_page_done` checkpoint.
   Phase B "fetch": pull all status='pending' rows and download them,
       marking each 'done' or 'failed' as it completes.
   Running with --mode both (the default) interleaves: it discovers a
   batch of items, immediately queues their downloads to the thread pool,
   and keeps going. This avoids needing to hold 1.15M URLs in memory at
   once and lets downloading start immediately rather than after a full
   day of crawling metadata first.

4. SAFE FILENAMES + MANIFEST: every saved file is named
   "<item_uuid>__<original_filename>" inside OUT_DIR, and every row in the
   SQLite DB records item uuid, item title, bitstream uuid, filename,
   bytes downloaded, and timestamp -- so you end up with a clean manifest
   of exactly what you have, suitable for later cross-referencing against
   the parliamentary question numbers.

5. CHECKSUMS (optional but recommended for ~1M files): if VERIFY_SIZE is
   True, after download we compare bytes-on-disk to the Content-Length
   header DSpace returned, and re-queue the file as 'failed' (for retry)
   on mismatch, rather than silently keeping a truncated PDF.

6. HPC SYNC (added after discovering HPC compute/login nodes have no
   internet access): since this script now runs on your LAPTOP, not the
   HPC, files are downloaded to a local STAGING_DIR first. A background
   thread periodically rsyncs completed files from STAGING_DIR to the HPC
   over SSH, then deletes the local copies that were successfully synced
   -- so your laptop's disk usage stays bounded near zero regardless of
   how many files have been downloaded in total. This requires passwordless
   SSH key auth to the HPC (see SSH KEY SETUP below) since it must run
   unattended for potentially days; a script blocking on a password prompt
   every sync interval will hang forever with no one watching.

USAGE ON YOUR LAPTOP (this script no longer runs on the HPC):

    # 1. Install deps:
    pip install requests

    # 2. One-time: set up passwordless SSH key auth to the HPC, so rsync
    #    doesn't prompt for a password every sync cycle (it runs unattended
    #    for hours/days). From your laptop:
    #
    #      ssh-keygen -t ed25519          # if you don't already have a key
    #                                      # (press Enter through the prompts)
    #      ssh-copy-id shubham.agarwal_phd24@10.1.4.95
    #
    #    Then test that this works with NO password prompt:
    #      ssh shubham.agarwal_phd24@10.1.4.95 "echo connected ok"
    #
    #    If ssh-copy-id isn't available (e.g. on Windows/PowerShell without
    #    it installed), manually append the contents of your local
    #    ~/.ssh/id_ed25519.pub to ~/.ssh/authorized_keys on the HPC.

    # 3. Edit the HPC_* settings below (host, user, remote path) to match
    #    your setup -- they're already filled in based on what you've told
    #    me, but double check the remote path exists:
    #      ssh shubham.agarwal_phd24@10.1.4.95 "mkdir -p ~/projects/BERTaav/data/raw_pdfs"

    # 4. Run it. Just let it run in a terminal -- on Windows this means
    #    keeping the terminal window open (or use Windows Terminal / WSL +
    #    `nohup`/`screen`/`tmux` if you want it survivable across a laptop
    #    sleep/reboot, which matters a lot for a job this long):
    python bulk_download.py

    # 5. Monitor progress at any time, from another terminal, while it runs:
    python bulk_download.py --status

    # 4. If it stops for any reason, just re-run the same command --
    #    it resumes automatically from the SQLite checkpoint.

TUNING:
    - CONCURRENCY: number of parallel download threads. Start at 8.
    - REQUEST_DELAY: seconds to sleep between *metadata* API calls per
      worker thread (politeness throttle independent of download speed).
    - PAGE_SIZE: items per Discover API page. 100 is a reasonable default;
      DSpace typically caps this around 100-200 server-side regardless of
      what you ask for.
"""

import argparse
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ----------------------------------------------------------------------------
# CONFIG -- tune these
# ----------------------------------------------------------------------------
BASE = "https://elibrary.sansad.in"
API = f"{BASE}/server/api"
COLLECTION_UUID = "75228d43-3a98-4b7d-b90f-ffec6aa9fa11"

OUT_DIR = os.environ.get("SANSAD_OUT_DIR", "./staging_pdfs")  # LOCAL staging only
DB_PATH = os.environ.get("SANSAD_DB_PATH", "./progress.sqlite3")

PAGE_SIZE = 100
CONCURRENCY = 8            # parallel download workers
REQUEST_DELAY = 0.05       # polite delay between metadata calls (seconds)
MAX_RETRIES = 6
VERIFY_SIZE = True

# --- HPC sync settings ---------------------------------------------------
# Files land in OUT_DIR locally, then get rsync'd to the HPC over SSH and
# deleted locally once confirmed transferred. This keeps laptop disk usage
# bounded regardless of total corpus size.
HPC_USER = os.environ.get("SANSAD_HPC_USER", "shubham.agarwal_phd24")
HPC_HOST = os.environ.get("SANSAD_HPC_HOST", "10.1.4.95")
HPC_REMOTE_DIR = os.environ.get("SANSAD_HPC_DIR", "projects/BERTaav/data/raw_pdfs")
# ^ relative to the HPC user's home directory, since that's how you described it.
#   Use an absolute path here instead if you'd rather be explicit, e.g.
#   "/home/shubham.agarwal_phd24/projects/BERTaav/data/raw_pdfs"

SYNC_INTERVAL_SECONDS = int(os.environ.get("SANSAD_SYNC_INTERVAL", "300"))  # every 5 min
SYNC_ENABLED = os.environ.get("SANSAD_SYNC_ENABLED", "1") != "0"  # set to 0 to disable, e.g. for local-only testing

# ----------------------------------------------------------------------------

SESSION_LOCAL = threading.local()


def get_session():
    """One requests.Session per thread (connection pooling, thread-safety)."""
    if not hasattr(SESSION_LOCAL, "s"):
        s = requests.Session()
        s.headers.update({
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) sansad-bulk-download-script/1.0",
        })
        SESSION_LOCAL.s = s
    return SESSION_LOCAL.s


def get_json(url, params=None, retries=MAX_RETRIES):
    s = get_session()
    for attempt in range(retries):
        try:
            r = s.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(60, 2 ** attempt)
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"Failed GET {url} after {retries} attempts")


# ----------------------------------------------------------------------------
# SQLite work-queue
# ----------------------------------------------------------------------------

DB_LOCK = threading.Lock()


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")  # better concurrent access
    return conn


def db_init():
    conn = db_connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            bitstream_uuid TEXT PRIMARY KEY,
            item_uuid TEXT,
            item_title TEXT,
            filename TEXT,
            dest_path TEXT,
            status TEXT DEFAULT 'pending',   -- pending | done | failed
            bytes INTEGER,
            error TEXT,
            updated_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crawl_progress (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_page_done INTEGER DEFAULT -1,
            total_pages INTEGER
        )
    """)
    conn.execute("INSERT OR IGNORE INTO crawl_progress (id, last_page_done) VALUES (1, -1)")
    conn.commit()
    conn.close()


def db_get_crawl_progress():
    conn = db_connect()
    row = conn.execute("SELECT last_page_done, total_pages FROM crawl_progress WHERE id=1").fetchone()
    conn.close()
    return row  # (last_page_done, total_pages)


def db_set_crawl_progress(last_page_done, total_pages=None):
    with DB_LOCK:
        conn = db_connect()
        if total_pages is not None:
            conn.execute(
                "UPDATE crawl_progress SET last_page_done=?, total_pages=? WHERE id=1",
                (last_page_done, total_pages),
            )
        else:
            conn.execute(
                "UPDATE crawl_progress SET last_page_done=? WHERE id=1",
                (last_page_done,),
            )
        conn.commit()
        conn.close()


def db_enqueue_file(bitstream_uuid, item_uuid, item_title, filename, dest_path):
    with DB_LOCK:
        conn = db_connect()
        conn.execute(
            """INSERT OR IGNORE INTO files
               (bitstream_uuid, item_uuid, item_title, filename, dest_path, status)
               VALUES (?, ?, ?, ?, ?, 'pending')""",
            (bitstream_uuid, item_uuid, item_title, filename, dest_path),
        )
        conn.commit()
        conn.close()


def db_mark(bitstream_uuid, status, bytes_=None, error=None):
    with DB_LOCK:
        conn = db_connect()
        conn.execute(
            "UPDATE files SET status=?, bytes=?, error=?, updated_at=? WHERE bitstream_uuid=?",
            (status, bytes_, error, time.time(), bitstream_uuid),
        )
        conn.commit()
        conn.close()


def db_pending_files(limit=None):
    conn = db_connect()
    q = "SELECT bitstream_uuid, item_uuid, filename, dest_path FROM files WHERE status='pending'"
    if limit:
        q += f" LIMIT {limit}"
    rows = conn.execute(q).fetchall()
    conn.close()
    return rows


def db_status_counts():
    conn = db_connect()
    rows = conn.execute("SELECT status, COUNT(*) FROM files GROUP BY status").fetchall()
    conn.close()
    return dict(rows)


# ----------------------------------------------------------------------------
# DSpace API helpers (same logic as test_run.py, kept in sync)
# ----------------------------------------------------------------------------

HINDI_FILENAME_HINTS = re.compile(r"(hin|hindi|_hi\b|-hi\b)", re.IGNORECASE)


def looks_hindi(bitstream):
    name = bitstream.get("name", "") or ""
    metadata = bitstream.get("metadata", {}) or {}
    lang_vals = metadata.get("dc.language.iso", [])
    for v in lang_vals:
        val = (v.get("value") or "").lower()
        if val in ("hi", "hin", "hindi"):
            return True
    return bool(HINDI_FILENAME_HINTS.search(name))


def extract_items_from_search(search_json):
    objs = search_json["_embedded"]["searchResult"]["_embedded"]["objects"]
    items = []
    for obj in objs:
        item = obj.get("_embedded", {}).get("indexableObject", obj)
        items.append(item)
    return items


def fetch_items_page(page, size=PAGE_SIZE):
    url = f"{API}/discover/search/objects"
    params = {"scope": COLLECTION_UUID, "dsoType": "item", "page": page, "size": size}
    return get_json(url, params=params)


def fetch_item_bitstreams(item):
    bitstreams = []
    bundles_href = item.get("_links", {}).get("bundles", {}).get("href")
    if not bundles_href:
        return bitstreams
    bundles_json = get_json(bundles_href)
    for bundle in bundles_json.get("_embedded", {}).get("bundles", []):
        if bundle.get("name") not in ("ORIGINAL", None):
            continue
        bs_href = bundle.get("_links", {}).get("bitstreams", {}).get("href")
        if not bs_href:
            continue
        bjson = get_json(bs_href)
        bitstreams.extend(bjson.get("_embedded", {}).get("bitstreams", []))
    return bitstreams


def safe_filename(name, fallback):
    name = (name or fallback).strip()
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name or fallback


def item_title_of(item):
    md = item.get("metadata", {})
    titles = md.get("dc.title", [])
    if titles:
        return titles[0].get("value", "")
    return item.get("name", "")


# ----------------------------------------------------------------------------
# Phase A: crawl + enqueue
# ----------------------------------------------------------------------------

def crawl_and_enqueue(start_page, end_page=None, stop_event=None):
    """
    Walk Discover API pages starting at start_page, fetch each item's
    bitstreams, enqueue Hindi ones into the SQLite `files` table, and
    checkpoint progress after every page so a restart skips completed pages.
    """
    page = start_page
    while True:
        if stop_event and stop_event.is_set():
            return
        try:
            search_json = fetch_items_page(page)
        except Exception as e:
            print(f"[crawl] FATAL error fetching page {page}: {e}", flush=True)
            time.sleep(5)
            continue

        page_info = search_json["_embedded"]["searchResult"]["page"]
        total_pages = page_info["totalPages"]
        items = extract_items_from_search(search_json)

        if not items:
            print(f"[crawl] page {page} empty -- assuming end of collection.")
            db_set_crawl_progress(page, total_pages)
            break

        for item in items:
            item_uuid = item.get("uuid")
            title = item_title_of(item)
            try:
                bitstreams = fetch_item_bitstreams(item)
            except Exception as e:
                print(f"[crawl] error fetching bitstreams for item {item_uuid}: {e}", flush=True)
                continue

            for b in bitstreams:
                if not looks_hindi(b):
                    continue
                bs_uuid = b["uuid"]
                fname = safe_filename(b.get("name"), f"{bs_uuid}.pdf")
                dest = os.path.join(OUT_DIR, f"{item_uuid}__{fname}")
                db_enqueue_file(bs_uuid, item_uuid, title, fname, dest)

            time.sleep(REQUEST_DELAY)

        db_set_crawl_progress(page, total_pages)
        if page % 50 == 0:
            print(f"[crawl] checkpoint: finished page {page}/{total_pages}", flush=True)

        page += 1
        if end_page is not None and page > end_page:
            break
        if page >= total_pages:
            print(f"[crawl] reached final page {page} of {total_pages}. Crawl complete.")
            break


# ----------------------------------------------------------------------------
# Phase B: download workers
# ----------------------------------------------------------------------------

def download_one(bitstream_uuid, dest_path):
    s = get_session()
    url = f"{API}/core/bitstreams/{bitstream_uuid}/content"
    for attempt in range(MAX_RETRIES):
        try:
            with s.get(url, stream=True, timeout=120) as r:
                if r.status_code != 200:
                    if r.status_code in (429, 500, 502, 503, 504):
                        time.sleep(min(60, 2 ** attempt))
                        continue
                    raise RuntimeError(f"HTTP {r.status_code}")

                expected_len = r.headers.get("Content-Length")
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                tmp_path = dest_path + ".part"
                size = 0
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        size += len(chunk)

                if VERIFY_SIZE and expected_len and int(expected_len) != size:
                    os.remove(tmp_path)
                    raise RuntimeError(f"size mismatch: expected {expected_len}, got {size}")

                os.replace(tmp_path, dest_path)
                return size
        except Exception as e:
            last_err = str(e)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"download failed after {MAX_RETRIES} attempts: {last_err}")


def run_download_workers(concurrency=CONCURRENCY, batch_limit=5000):
    """
    Repeatedly pulls batches of pending files from SQLite and downloads
    them with a thread pool, until no pending files remain (the crawler
    may still be adding more in parallel, in which case run this in a loop
    from main()).
    """
    pending = db_pending_files(limit=batch_limit)
    if not pending:
        return 0

    done_count = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {}
        for bs_uuid, item_uuid, fname, dest in pending:
            if os.path.exists(dest):  # already on disk from a prior run
                db_mark(bs_uuid, "done", os.path.getsize(dest))
                done_count += 1
                continue
            fut = pool.submit(download_one, bs_uuid, dest)
            futures[fut] = bs_uuid

        for fut in as_completed(futures):
            bs_uuid = futures[fut]
            try:
                size = fut.result()
                db_mark(bs_uuid, "done", size)
                done_count += 1
            except Exception as e:
                db_mark(bs_uuid, "failed", error=str(e))
                print(f"[download] FAILED {bs_uuid}: {e}", flush=True)

    return done_count


# ----------------------------------------------------------------------------
# HPC sync (rsync over SSH)
# ----------------------------------------------------------------------------

def check_rsync_available():
    """
    rsync ships by default on macOS/Linux. On native Windows it usually does
    NOT (unless installed via WSL, Git Bash with extra tools, cwRsync, or
    similar) -- so we check up front and fail with a clear message rather
    than a cryptic 'rsync not found' deep inside a background thread.
    """
    return shutil.which("rsync") is not None


def db_mark_synced(bitstream_uuid):
    with DB_LOCK:
        conn = db_connect()
        conn.execute("UPDATE files SET status='synced' WHERE bitstream_uuid=?", (bitstream_uuid,))
        conn.commit()
        conn.close()


def db_done_unsynced_paths():
    """Files fully downloaded ('done') but not yet pushed to the HPC."""
    conn = db_connect()
    rows = conn.execute(
        "SELECT bitstream_uuid, dest_path FROM files WHERE status='done'"
    ).fetchall()
    conn.close()
    return rows


def sync_to_hpc_once():
    """
    One rsync pass: push every locally-'done' file to the HPC, then -- only
    for files rsync confirms it actually transferred -- delete the local
    copy and mark it 'synced' in SQLite. Uses `rsync --remove-source-files`
    so deletion only happens on confirmed successful transfer (rsync itself
    guards against deleting on a failed/interrupted transfer).

    We rsync the whole OUT_DIR in one shot rather than file-by-file, which
    is dramatically faster -- one SSH handshake instead of thousands.
    """
    pending = db_done_unsynced_paths()
    if not pending:
        return 0

    existing_paths = [p for _, p in pending if os.path.exists(p)]
    if not existing_paths:
        # Files vanished locally for some reason (already manually moved?) --
        # just mark them synced so we don't loop on them forever.
        for uuid, _ in pending:
            db_mark_synced(uuid)
        return 0

    remote = f"{HPC_USER}@{HPC_HOST}:{HPC_REMOTE_DIR}/"
    cmd = [
        "rsync",
        "-av",                     # archive mode, verbose
        "--remove-source-files",   # delete local file ONLY after confirmed transfer
        "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",  # never prompt for a password; fail fast if key auth isn't set up
        OUT_DIR + "/",
        remote,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        print("[sync] rsync timed out after 30 min -- will retry next cycle.", flush=True)
        return 0

    if result.returncode != 0:
        print(f"[sync] rsync FAILED (exit {result.returncode}): {result.stderr.strip()[:500]}", flush=True)
        print("[sync] Check: is the HPC reachable? Is SSH key auth set up (BatchMode=yes "
              "means it will fail immediately rather than hang on a password prompt)?", flush=True)
        return 0

    # rsync succeeded; figure out which of our 'done' rows are now gone
    # locally (i.e. successfully transferred + removed by --remove-source-files)
    # and mark exactly those as synced.
    synced = 0
    for uuid, path in pending:
        if not os.path.exists(path):
            db_mark_synced(uuid)
            synced += 1

    # rsync leaves now-empty subdirectories behind sometimes; OUT_DIR itself
    # we keep (next batch of downloads needs it), but it's harmless either way.
    print(f"[sync] pushed {synced} file(s) to {HPC_HOST}:{HPC_REMOTE_DIR}", flush=True)
    return synced


def sync_loop(stop_event):
    """Runs in a background thread for the lifetime of the program."""
    if not SYNC_ENABLED:
        print("[sync] SANSAD_SYNC_ENABLED=0 -- sync to HPC disabled, files will stay local in OUT_DIR.")
        return
    if not check_rsync_available():
        print("[sync] WARNING: 'rsync' not found on PATH. Files will accumulate locally in "
              f"{OUT_DIR} and NOT be pushed to the HPC. On Windows, install rsync via WSL, "
              "Git for Windows (with rsync), or cwRsync, then re-run. Falling back to "
              "local-only mode for this run.", flush=True)
        return

    while not stop_event.is_set():
        try:
            sync_to_hpc_once()
        except Exception as e:
            print(f"[sync] unexpected error: {e}", flush=True)
        stop_event.wait(SYNC_INTERVAL_SECONDS)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def print_status():
    counts = db_status_counts()
    last_page, total_pages = db_get_crawl_progress()
    print("=== STATUS ===")
    print(f"Crawl: last completed page = {last_page} / {total_pages}")
    print(f"Files: {counts}")
    print(f"  (status meanings: pending=queued, done=downloaded locally but not yet synced, "
          f"synced=confirmed on HPC, failed=needs retry)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="print progress and exit")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--end-page", type=int, default=None, help="stop crawl after this page (debug)")
    parser.add_argument("--no-sync", action="store_true", help="disable rsync-to-HPC, keep files local only")
    args = parser.parse_args()

    db_init()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.status:
        print_status()
        return

    global SYNC_ENABLED
    if args.no_sync:
        SYNC_ENABLED = False

    last_page, _ = db_get_crawl_progress()
    start_page = last_page + 1
    print(f"Resuming crawl from page {start_page} (0-indexed). Local staging dir: {OUT_DIR}")
    if SYNC_ENABLED:
        print(f"Sync target: {HPC_USER}@{HPC_HOST}:{HPC_REMOTE_DIR}  (every {SYNC_INTERVAL_SECONDS}s)")
        if platform.system() == "Windows" and not check_rsync_available():
            print("WARNING: rsync not found. On Windows, the simplest fix is usually to run "
                  "this script from inside WSL (Windows Subsystem for Linux), which has rsync "
                  "preinstalled and can still reach the HPC over your normal network. "
                  "Continuing in local-only mode for now.", flush=True)
    else:
        print("Sync to HPC is DISABLED for this run -- files will only accumulate in OUT_DIR locally.")

    stop_event = threading.Event()

    crawl_thread = threading.Thread(
        target=crawl_and_enqueue,
        kwargs={"start_page": start_page, "end_page": args.end_page, "stop_event": stop_event},
        daemon=True,
    )
    crawl_thread.start()

    sync_thread = threading.Thread(target=sync_loop, args=(stop_event,), daemon=True)
    sync_thread.start()

    try:
        # Interleave: keep draining the download queue while the crawler fills it.
        while crawl_thread.is_alive():
            n = run_download_workers(concurrency=args.concurrency)
            if n == 0:
                time.sleep(5)  # nothing pending yet, wait for crawler to add more
        # Crawl finished -- drain any remaining queued downloads.
        while True:
            n = run_download_workers(concurrency=args.concurrency)
            if n == 0:
                break
    except KeyboardInterrupt:
        print("\nInterrupted -- progress is saved in SQLite. Re-run the same command to resume "
              "(it will pick up the crawl and downloads exactly where it left off).")
        stop_event.set()
        # Give the sync thread a moment to notice stop_event rather than killing it mid-rsync.
        sync_thread.join(timeout=5)
        sys.exit(1)

    print("Downloads complete. Running final sync pass to push any remaining files to the HPC...")
    stop_event.set()
    if SYNC_ENABLED and check_rsync_available():
        while True:
            n = sync_to_hpc_once()
            if n == 0:
                break

    print_status()
    print("All done.")


if __name__ == "__main__":
    main()