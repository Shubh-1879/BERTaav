#!/usr/bin/env python3
"""
postprocess.py — Clean and enrich OCR JSON files before HuggingFace push.

For each JSON in ocr_output/, this script:

  1. PARSES HEADER METADATA from the OCR text itself (since DSpace metadata
     was sparse — title/date/ministry were all null). The Lok Sabha Q&A
     documents have a completely consistent header structure that makes this
     reliable even with minor OCR errors.

  2. CLEANS TEXT:
     - Strips trailing noise (the page-border decoration Tesseract reads as
       random short syllables at the end of every document)
     - Normalises whitespace (multiple blank lines → single blank line)
     - Strips the "मूल/मूत्र हिंदी में" OCR artifact header line
     - Strips URLs that OCR garbled into mixed-script garbage

  3. QUALITY FILTERING:
     - Marks documents below confidence threshold as low_quality (kept but
       flagged — never silently dropped, so you can audit them)
     - Marks documents with almost no text as empty (also kept, flagged)
     - Records all decisions in a quality_flag field + quality_notes list

  4. DEDUPLICATION:
     - Detects exact-duplicate texts (same hash) and flags them
     - Does NOT silently delete — flags with quality_flag='duplicate' so
       you can inspect before deciding

  5. PRODUCES A FINAL SCHEMA matching what was agreed, with all fields
     populated as fully as possible, plus new fields:
       - quality_flag: 'ok' | 'low_confidence' | 'low_text' | 'duplicate' | 'empty'
       - quality_notes: list of specific issues found
       - question_type: 'unstarred' | 'starred' | 'unknown'
       - question_number: integer (parsed from header)
       - questioner: name of the MP who asked (parsed from text)
       - ministry_parsed: ministry name extracted from header

  6. WRITES output to processed/ directory as one JSON per document,
     plus a manifest CSV and a quality report.

USAGE:
  python scripts/postprocess.py                    # process all
  python scripts/postprocess.py --dry-run          # show stats without writing
  python scripts/postprocess.py --min-confidence 70  # set quality threshold
  python scripts/postprocess.py --report-only      # just print quality report

RESUMABLE: already-processed files (present in processed/) are skipped.
Re-run safely at any time as new OCR files arrive.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = os.environ.get(
    "BERTAAV_ROOT",
    "/home/shubham.agarwal_phd24/projects/BERTaav"
)
OCR_DIR       = os.path.join(ROOT, "data", "ocr_output")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
LOGS_DIR      = os.path.join(ROOT, "logs")
MANIFEST_PATH = os.path.join(PROCESSED_DIR, "manifest.csv")
REPORT_PATH   = os.path.join(PROCESSED_DIR, "quality_report.json")

MIN_CONFIDENCE_DEFAULT = 70.0   # below this → low_confidence flag
MIN_TEXT_BYTES         = 100    # below this → low_text / empty flag

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(LOGS_DIR, "postprocess.log"), mode="a"
            ),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────

# The page-border decoration Tesseract reads as random syllables.
# Matches 2+ tokens at end of text that are each 1-4 chars long and look like
# Hindi syllables (short words like जे, ने, के, ते, नै, में, का, की, को).
# We require at least 3 such tokens to avoid accidentally stripping real text.
_TRAILING_NOISE_RE = re.compile(
    r'[\n\s]*(?:[\u0900-\u097F]{1,4}\s+){2,}[\u0900-\u097F]{1,4}\s*$'
)

# "मूल हिंदी में" header line — also catches OCR variants like "मूत्र हिंदी में"
_MOOL_HINDI_RE = re.compile(
    r'^[मम][ूु][लत्र]{1,3}\s+हिंदी\s+में\s*\n?',
    re.MULTILINE
)

# Garbled URLs — mixed Devanagari/Latin/special chars in a URL-like pattern
_GARBLED_URL_RE = re.compile(
    r'(?:https?://|www\.|(?:[\[\|]){1,2}[\d०-९]{2,}[:/])'  # URL start signals
    r'[^\n]{5,}',                                             # followed by junk
    re.IGNORECASE
)

# Multiple consecutive blank lines → single blank line
_MULTI_BLANK_RE = re.compile(r'\n{3,}')

# Lone punctuation lines (page numbers, stray chars)
_LONE_PUNCT_RE = re.compile(r'^\s*[।॥\-_=*#|]{1,5}\s*$', re.MULTILINE)


def clean_text(text: str) -> tuple[str, list[str]]:
    """
    Clean a single text string. Returns (cleaned_text, list_of_changes_made).
    """
    notes = []
    original_len = len(text)

    # Strip मूल हिंदी में header artifact
    cleaned = _MOOL_HINDI_RE.sub('', text)
    if len(cleaned) != len(text):
        notes.append("stripped_mool_hindi_header")
    text = cleaned

    # Strip garbled URLs
    cleaned = _GARBLED_URL_RE.sub('[URL]', text)
    if cleaned != text:
        notes.append("replaced_garbled_urls")
    text = cleaned

    # Strip trailing noise
    cleaned = _TRAILING_NOISE_RE.sub('', text).rstrip()
    if cleaned != text.rstrip():
        notes.append("stripped_trailing_noise")
    text = cleaned

    # Strip lone punctuation lines
    cleaned = _LONE_PUNCT_RE.sub('', text)
    if cleaned != text:
        notes.append("stripped_lone_punctuation")
    text = cleaned

    # Normalise multiple blank lines
    cleaned = _MULTI_BLANK_RE.sub('\n\n', text)
    if cleaned != text:
        notes.append("normalised_whitespace")
    text = cleaned.strip()

    return text, notes


# ── Metadata extraction ───────────────────────────────────────────────────────

# Question type detection
_UNSTARRED_RE = re.compile(r'अतारांकित\s+प्रश्न', re.IGNORECASE)
_STARRED_RE   = re.compile(r'तारांकित\s+प्रश्न',  re.IGNORECASE)

# Question number — e.g. "प्रश्न सं. 3339" or "प्रश्न संख्या 4043"
_QNUM_RE = re.compile(
    r'प्रश्न\s+(?:सं\.?|संख्या)\s*[।.]?\s*(\d+)',
    re.IGNORECASE
)

# Date — handles formats like "2 मार्च, 2023" / "24.03.2025" / "03.42.2024"
_HINDI_MONTHS = {
    'जनवरी': 1, 'फरवरी': 2, 'मार्च': 3, 'अप्रैल': 4,
    'मई': 5, 'जून': 6, 'जुलाई': 7, 'अगस्त': 8,
    'सितंबर': 9, 'अक्टूबर': 10, 'नवंबर': 11, 'दिसंबर': 12,
}
_DATE_HINDI_RE = re.compile(
    r'(\d{1,2})\s+(' + '|'.join(_HINDI_MONTHS.keys()) + r'),?\s+(\d{4})'
)
_DATE_NUMERIC_RE = re.compile(r'(\d{2})\.(\d{2})\.(\d{4})')

# Ministry — first non-"भारत सरकार" line after "भारत सरकार"
# We capture 1-3 lines after "भारत सरकार" as ministry candidates
_BHARAT_SARKAR_RE = re.compile(r'भारत\s+सरकार')

# Question title — the line between the date line and the question number line
# Pattern: line ending with "के लिए" or "के लिए नियत", followed by title line
_TITLE_AFTER_DATE_RE = re.compile(
    r'(?:के\s+लिए(?:\s+नियत)?)\s*\n\s*\n?(.+?)(?:\n|$)',
    re.DOTALL
)

# Questioner — "NNNN. श्री/श्रीमती/डॉ. <Name>:" pattern
_QUESTIONER_RE = re.compile(
    r'\d+\.\s+((?:श्री|श्रीमती|डॉ\.?|कुमारी|प्रो\.?)\s+[^\n:]{3,60})(?:\s*:|\s*\n)',
    re.UNICODE
)


def extract_metadata(text: str) -> dict:
    """
    Parse structured metadata from the OCR text of a Lok Sabha Q&A document.
    Returns a dict with keys: title, date, question_number, question_type,
    questioner, ministry_parsed.
    All values are None if not found — never raises.
    """
    result = {
        "title":           None,
        "date":            None,
        "question_number": None,
        "question_type":   None,
        "questioner":      None,
        "ministry_parsed": None,
    }

    if not text:
        return result

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # ── Question type ─────────────────────────────────────────────────────────
    if _UNSTARRED_RE.search(text):
        result["question_type"] = "unstarred"
    elif _STARRED_RE.search(text):
        result["question_type"] = "starred"
    else:
        result["question_type"] = "unknown"

    # ── Question number ───────────────────────────────────────────────────────
    m = _QNUM_RE.search(text)
    if m:
        try:
            result["question_number"] = int(m.group(1))
        except ValueError:
            pass

    # ── Date ─────────────────────────────────────────────────────────────────
    # Try Hindi month name format first (more reliable)
    m = _DATE_HINDI_RE.search(text)
    if m:
        day, month_name, year = m.group(1), m.group(2), m.group(3)
        month = _HINDI_MONTHS.get(month_name)
        if month and year.isdigit():
            try:
                result["date"] = f"{year}-{month:02d}-{int(day):02d}"
            except Exception:
                pass

    # Fall back to numeric date (DD.MM.YYYY)
    if result["date"] is None:
        m = _DATE_NUMERIC_RE.search(text)
        if m:
            day, month, year = m.group(1), m.group(2), m.group(3)
            # Validate: months > 12 are OCR errors (e.g. "03.42.2024")
            if int(month) <= 12 and int(day) <= 31:
                result["date"] = f"{year}-{month}-{day}"

    # ── Ministry ──────────────────────────────────────────────────────────────
    # Find "भारत सरकार" and take the next 1-2 non-empty lines as ministry
    bharat_match = _BHARAT_SARKAR_RE.search(text)
    if bharat_match:
        after = text[bharat_match.end():]
        ministry_lines = []
        for line in after.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Stop at structural keywords
            if any(kw in line for kw in ['लोक सभा', 'राज्य सभा', 'अतारांकित',
                                          'तारांकित', 'प्रश्न']):
                break
            ministry_lines.append(line)
            if len(ministry_lines) == 2:
                break
        if ministry_lines:
            result["ministry_parsed"] = " | ".join(ministry_lines)

    # ── Title ─────────────────────────────────────────────────────────────────
    # Strategy: find the line after the date line and before the questioner line
    # The title is between "के लिए" and the question number line
    title_match = _TITLE_AFTER_DATE_RE.search(text)
    if title_match:
        candidate = title_match.group(1).strip()
        # Must be a real title: reasonable length, mostly Hindi chars
        hindi_ratio = sum(
            1 for c in candidate if '\u0900' <= c <= '\u097F'
        ) / max(len(candidate), 1)
        if 5 < len(candidate) < 200 and hindi_ratio > 0.3:
            result["title"] = candidate

    # Fallback: if question_number found, look for title just before questioner
    if result["title"] is None and result["question_number"] is not None:
        qnum_str = str(result["question_number"])
        for i, line in enumerate(lines):
            if qnum_str in line and any(
                kw in line for kw in ['श्री', 'श्रीमती', 'डॉ']
            ):
                # Title is likely the line just before this
                if i > 0:
                    candidate = lines[i - 1]
                    if 5 < len(candidate) < 200:
                        result["title"] = candidate
                break

    # ── Questioner ───────────────────────────────────────────────────────────
    m = _QUESTIONER_RE.search(text)
    if m:
        result["questioner"] = m.group(1).strip()

    return result


# ── Quality assessment ────────────────────────────────────────────────────────

def assess_quality(record: dict, min_confidence: float) -> tuple[str, list[str]]:
    """
    Assess quality of a record. Returns (flag, notes).
    flag is one of: 'ok', 'low_confidence', 'low_text', 'empty'
    Notes explain what's wrong. Multiple issues are all recorded.
    """
    flag  = "ok"
    notes = []

    text_bytes = record.get("text_bytes_utf8", 0) or 0
    confidence = record.get("ocr_confidence_mean", 0) or 0

    if text_bytes < MIN_TEXT_BYTES:
        flag = "empty"
        notes.append(f"text_bytes={text_bytes} < {MIN_TEXT_BYTES}")

    elif text_bytes < 500:
        if flag == "ok":
            flag = "low_text"
        notes.append(f"text_bytes={text_bytes} very short")

    if confidence < min_confidence:
        if flag == "ok":
            flag = "low_confidence"
        notes.append(f"confidence={confidence:.1f} < {min_confidence}")

    # Check for pages with very low individual confidence
    pages = record.get("pages", [])
    bad_pages = [
        p["page_num"] for p in pages
        if (p.get("confidence_mean") or 0) < min_confidence - 10
    ]
    if bad_pages:
        notes.append(f"low_confidence_pages={bad_pages}")

    return flag, notes


# ── Main processing ───────────────────────────────────────────────────────────

def process_one(
    json_path: str,
    min_confidence: float,
    seen_hashes: dict,
) -> dict | None:
    """
    Load, clean, enrich, and quality-assess one OCR JSON file.
    Returns the processed record dict, or None if file is unreadable.
    seen_hashes: {text_hash: source_pdf} — updated in place for dedup.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            record = json.load(f)
    except Exception as e:
        return None

    quality_notes = []

    # ── Clean document-level text ─────────────────────────────────────────────
    raw_text = record.get("text", "") or ""
    clean, notes = clean_text(raw_text)
    quality_notes.extend(notes)
    record["text"] = clean
    record["text_bytes_utf8"] = len(clean.encode("utf-8"))

    # ── Clean per-page text ───────────────────────────────────────────────────
    for page in record.get("pages", []):
        raw_page = page.get("text", "") or ""
        clean_page, page_notes = clean_text(raw_page)
        page["text"] = clean_page
        page["text_bytes_utf8"] = len(clean_page.encode("utf-8"))

    # ── Extract metadata from text ────────────────────────────────────────────
    meta = extract_metadata(record["text"])

    # Only overwrite null fields — never overwrite what DSpace already gave us
    if record.get("title") is None:
        record["title"] = meta["title"]
    if record.get("date") is None:
        record["date"] = meta["date"]

    # New enrichment fields
    record["question_type"]   = meta["question_type"]
    record["question_number"] = meta["question_number"]
    record["questioner"]      = meta["questioner"]
    record["ministry_parsed"] = meta["ministry_parsed"]

    # ── Deduplication ─────────────────────────────────────────────────────────
    text_hash = hashlib.md5(record["text"].encode("utf-8")).hexdigest()
    record["text_hash"] = text_hash

    if text_hash in seen_hashes and seen_hashes[text_hash] != record.get("source_pdf"):
        record["quality_flag"]  = "duplicate"
        record["quality_notes"] = quality_notes + [
            f"duplicate_of={seen_hashes[text_hash]}"
        ]
        return record
    seen_hashes[text_hash] = record.get("source_pdf", "")

    # ── Quality assessment ────────────────────────────────────────────────────
    flag, q_notes = assess_quality(record, min_confidence)
    quality_notes.extend(q_notes)

    record["quality_flag"]  = flag
    record["quality_notes"] = quality_notes

    # ── Processing timestamp ──────────────────────────────────────────────────
    record["processed_at"] = datetime.now().isoformat()

    return record


def write_manifest(records: list[dict]):
    """Write a flat CSV manifest for easy inspection without loading JSONs."""
    fieldnames = [
        "id", "source_pdf", "title", "date", "ministry_parsed",
        "question_type", "question_number", "questioner",
        "num_pages", "ocr_confidence_mean", "text_bytes_utf8",
        "source_pdf_bytes", "quality_flag", "quality_notes",
    ]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["quality_notes"] = "; ".join(row.get("quality_notes", []))
            writer.writerow(row)


def write_quality_report(records: list[dict]):
    """Write a JSON quality report with aggregated stats."""
    flag_counts   = defaultdict(int)
    note_counts   = defaultdict(int)
    conf_buckets  = defaultdict(int)
    total_text_mb = 0.0
    ok_text_mb    = 0.0

    for r in records:
        flag = r.get("quality_flag", "unknown")
        flag_counts[flag] += 1
        for note in r.get("quality_notes", []):
            # Generalise note to avoid too many unique keys
            key = note.split("=")[0]
            note_counts[key] += 1
        conf = r.get("ocr_confidence_mean") or 0
        bucket = f"{int(conf // 10) * 10}-{int(conf // 10) * 10 + 9}"
        conf_buckets[bucket] += 1
        tb = (r.get("text_bytes_utf8") or 0) / 1e6
        total_text_mb += tb
        if flag == "ok":
            ok_text_mb += tb

    report = {
        "generated_at":    datetime.now().isoformat(),
        "total_documents": len(records),
        "by_quality_flag": dict(flag_counts),
        "by_issue":        dict(note_counts),
        "confidence_distribution": dict(conf_buckets),
        "total_text_mb":   round(total_text_mb, 2),
        "ok_text_mb":      round(ok_text_mb, 2),
        "pct_ok":          round(
            100 * flag_counts["ok"] / max(len(records), 1), 1
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Post-process OCR JSONs before HuggingFace push"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=MIN_CONFIDENCE_DEFAULT,
        help=f"OCR confidence threshold for quality flag (default {MIN_CONFIDENCE_DEFAULT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process everything but don't write output files"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Read existing processed/ files and regenerate quality report only"
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Reprocess all files even if already in processed/"
    )
    args = parser.parse_args()

    logger = setup_logging()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Report-only mode ──────────────────────────────────────────────────────
    if args.report_only:
        logger.info("Report-only mode: reading existing processed/ files...")
        records = []
        for fname in os.listdir(PROCESSED_DIR):
            if not fname.endswith(".json") or fname == "quality_report.json":
                continue
            with open(os.path.join(PROCESSED_DIR, fname), encoding="utf-8") as f:
                records.append(json.load(f))
        report = write_quality_report(records)
        write_manifest(records)
        _print_report(report)
        return

    # ── Main processing ───────────────────────────────────────────────────────
    input_files = [
        f for f in os.listdir(OCR_DIR)
        if f.endswith(".json")
    ]
    logger.info(f"Found {len(input_files)} OCR JSON files to process")

    # Track already-processed files for resumability
    already_done = set()
    if not args.reprocess:
        for fname in os.listdir(PROCESSED_DIR):
            if fname.endswith(".json") and fname != "quality_report.json":
                already_done.add(fname)
        if already_done:
            logger.info(
                f"Skipping {len(already_done)} already-processed files "
                f"(use --reprocess to redo)"
            )

    seen_hashes  = {}
    all_records  = []
    ok = low_conf = low_text = empty = dupes = errors = 0

    for i, fname in enumerate(input_files, start=1):
        if fname in already_done:
            # Load for manifest/report even if skipping
            try:
                with open(
                    os.path.join(PROCESSED_DIR, fname), encoding="utf-8"
                ) as f:
                    all_records.append(json.load(f))
            except Exception:
                pass
            continue

        in_path  = os.path.join(OCR_DIR, fname)
        out_path = os.path.join(PROCESSED_DIR, fname)

        record = process_one(in_path, args.min_confidence, seen_hashes)

        if record is None:
            logger.error(f"[ERROR] Could not process {fname}")
            errors += 1
            continue

        flag = record.get("quality_flag", "unknown")
        if flag == "ok":           ok        += 1
        elif flag == "low_confidence": low_conf += 1
        elif flag == "low_text":   low_text  += 1
        elif flag == "empty":      empty     += 1
        elif flag == "duplicate":  dupes     += 1

        all_records.append(record)

        if not args.dry_run:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

        if i % 500 == 0:
            logger.info(
                f"[{i}/{len(input_files)}] "
                f"ok={ok} low_conf={low_conf} low_text={low_text} "
                f"empty={empty} dupes={dupes} errors={errors}"
            )

    logger.info(
        f"Done. ok={ok} low_conf={low_conf} low_text={low_text} "
        f"empty={empty} dupes={dupes} errors={errors}"
    )

    if not args.dry_run and all_records:
        write_manifest(all_records)
        report = write_quality_report(all_records)
        _print_report(report)
        logger.info(f"Manifest → {MANIFEST_PATH}")
        logger.info(f"Quality report → {REPORT_PATH}")
    elif args.dry_run:
        report = write_quality_report(all_records)
        _print_report(report)
        logger.info("Dry run — no files written.")


def _print_report(report: dict):
    print("\n=== QUALITY REPORT ===")
    print(f"  Total documents : {report['total_documents']}")
    print(f"  By quality flag : {report['by_quality_flag']}")
    print(f"  % OK for HF push: {report['pct_ok']}%")
    print(f"  Total text      : {report['total_text_mb']} MB")
    print(f"  OK text         : {report['ok_text_mb']} MB")
    print(f"  Confidence dist : {report['confidence_distribution']}")
    if report.get("by_issue"):
        print(f"  Issues seen     : {report['by_issue']}")
    print()


if __name__ == "__main__":
    main()
