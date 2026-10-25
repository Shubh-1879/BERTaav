#!/usr/bin/env python3
"""
postprocess_pib.py — Clean and validate PIB press release JSONs before
merging into the BERTaav corpus.

PROBLEMS FIXED:
  1. DUPLICATE TEXT: PIB website renders content in two divs (mobile +
     desktop). Every file has the full text repeated twice. Fixed by
     detecting and removing the second half when it's near-identical.

  2. WRONG LANGUAGE: PIB Mumbai/Chennai/Bengaluru offices release in
     regional languages (Marathi, Tamil, Kannada etc.) but the URL
     lang=2 means "non-English" — not specifically Hindi. The scraper's
     is_hindi_text() check passes Marathi/Nepali/other Devanagari scripts.
     Fixed by using a Hindi-specific bigram/vocabulary check.

  3. BOILERPLATE: Multiple recurring boilerplate patterns need stripping:
     - "Press Release Page | Press Information Bureau" header
     - "(रिलीज़ आईडी: XXXXXXX)"
     - "आगंतुक पटल : N"
     - "इस विज्ञप्ति को इन भाषाओं में पढ़ें:"
     - Language list (English, Odia, Urdu, हिन्दी, ...)
     - PIB office bylines (N.Sapre/N.Chitale etc.)
     - Social media handles (@PIBMumbai, pibmumbai[at]gmail...)
     - Blank lines from removed content

  4. ENCODING ARTIFACTS: \r\n carriage returns, excessive whitespace,
     leading/trailing spaces on lines.

USAGE:
  # On your laptop (Windows path via WSL):
  cd /mnt/c/Users/shubh/Downloads/MSc\ 4th\ Sem/Erasmus\ Mobility/Work/BERTaav

  # Dry run — see stats without writing anything:
  python Code/scrapers/postprocess_pib.py --dry-run

  # Full run:
  python Code/scrapers/postprocess_pib.py

  # Then rsync cleaned files to HPC:
  rsync -av --remove-source-files Data/pib_clean/ \\
        USER@HPC_HOST:projects/BERTaav/data/pib_clean/

OUTPUT:
  Data/pib_clean/  — same shard structure as pib_raw, cleaned JSONs
  Data/pib_clean/postprocess_report.json — quality stats
"""

import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIB_RAW    = os.path.join(BASE_DIR, "Data", "pib_raw")
PIB_CLEAN  = os.path.join(BASE_DIR, "Data", "pib_clean")
REPORT_OUT = os.path.join(PIB_CLEAN, "postprocess_report.json")

MIN_TEXT_BYTES   = 300   # discard after cleaning if too short
MIN_HINDI_RATIO  = 0.25  # minimum fraction of chars that must be Hindi-specific

# ── Language detection ────────────────────────────────────────────────────────
# Marathi-specific common words that appear very frequently in Marathi but
# rarely in Hindi. If these dominate, the document is Marathi, not Hindi.
MARATHI_MARKERS = re.compile(
    r'\b(आहे|आहेत|होते|होती|होता|केले|केली|केला|असून|म्हणाले|म्हणून|'
    r'यांनी|यांचे|यांच्या|त्यांनी|त्यांचे|येथे|नाही|आणि|करण्यात|'
    r'सांगितले|झाले|झाली|झाला|असल्याचे|देतांना|करतांना|घेतले)\b',
    re.UNICODE
)

# Hindi-specific common function words / particles that appear frequently
# in Hindi but rarely in Marathi
HINDI_MARKERS = re.compile(
    r'\b(है|हैं|था|थी|थे|किया|की|के|का|में|से|को|पर|और|यह|वह|'
    r'इस|उस|जो|कि|भी|तो|ने|लिए|हुए|गया|गई|गए|होगा|होगी|'
    r'सरकार|मंत्री|मंत्रालय|प्रधानमंत्री|राज्य|केंद्र)\b',
    re.UNICODE
)

def detect_language(text):
    """
    Returns 'hi' (Hindi), 'mr' (Marathi), or 'other' (Devanagari but
    neither clearly Hindi nor Marathi, e.g. Nepali, Maithili).

    Method: count marker hits for Marathi vs Hindi. A document is
    considered Marathi if Marathi markers substantially outnumber Hindi
    markers. This is intentionally conservative — when in doubt, keep
    the document (flag as 'hi' with a quality note).
    """
    if not text:
        return 'other'

    marathi_hits = len(MARATHI_MARKERS.findall(text))
    hindi_hits   = len(HINDI_MARKERS.findall(text))

    total = marathi_hits + hindi_hits
    if total == 0:
        return 'other'

    marathi_ratio = marathi_hits / total
    if marathi_ratio > 0.60:
        return 'mr'
    elif marathi_ratio > 0.40 and hindi_hits < 5:
        return 'mr'
    else:
        return 'hi'


# ── Boilerplate patterns ──────────────────────────────────────────────────────

# Header line that appears at top of every raw page
HEADER_LINE = re.compile(
    r'^Press Release Page \| Press Information Bureau\s*$',
    re.MULTILINE
)

# Release ID line
RELEASE_ID = re.compile(
    r'\(रिलीज़\s+आईडी\s*[:：]?\s*\d+\)',
    re.UNICODE
)

# Visitor counter
VISITOR_CTR = re.compile(
    r'आगंतुक\s+पटल\s*[:：]?\s*\d*',
    re.UNICODE
)

# "Read this in other languages" line + the language list that follows
LANG_LIST = re.compile(
    r'इस\s+विज्ञप्ति\s+को\s+इन\s+भाषाओं\s+में\s+पढ़ें\s*:.*?(?=\n[^\n]{3,}|\Z)',
    re.DOTALL | re.UNICODE
)

# Individual language names on their own lines (after the "Read in" line)
LANG_NAMES = re.compile(
    r'^\s*(English|Odia|Urdu|हिन्दी|Manipuri|Bengali|Assamese|Punjabi|'
    r'Gujarati|Tamil|Telugu|Kannada|Malayalam|Marathi|Sanskrit|Nepali|'
    r'Bodo|Dogri|Konkani|Maithili|Santali|Sindhi|Kashmiri)\s*$',
    re.MULTILINE | re.UNICODE
)

# PIB office bylines like "N.Sapre/N.Chitale/D.Rane" or "S.Kane/V.Sahajrao"
BYLINES = re.compile(
    r'^[A-Z]\.[A-Za-z]+(?:/[A-Z]\.[A-Za-z]+)*\s*$',
    re.MULTILINE
)

# Social media handles and contact info
SOCIAL_MEDIA = re.compile(
    r'(@PIB\w+|PIB\w+\s*\n|/pib\w+|pib\w+\[at\]\w+|'
    r'सोशल\s+मिडियावर\s+आम्हाला\s+फॉलो\s+करा|'
    r'Follow\s+us\s+on\s+social\s+media)',
    re.IGNORECASE | re.UNICODE
)

# Twitter/X embed artifacts like "— Narendra Modi (@narendramodi) \nNovember 1, 2024"
TWEET_ATTR = re.compile(
    r'—\s+\w[\w\s]+\(@\w+\)\s*\n\w+\s+\d+,\s+\d{4}',
    re.UNICODE
)

# *** separator lines
STAR_SEP = re.compile(r'^\*{2,}\s*$', re.MULTILINE)

# Lines of pure whitespace or single chars after cleaning
SHORT_LINES = re.compile(r'^\s{0,5}$', re.MULTILINE)

# Multiple blank lines → single
MULTI_BLANK = re.compile(r'\n{3,}')

# Leading whitespace on each line (PIB HTML uses lots of indentation)
LINE_INDENT = re.compile(r'^[ \t]+', re.MULTILINE)

# Carriage returns
CRLF = re.compile(r'\r\n?')


def remove_boilerplate(text):
    """Apply all boilerplate removal patterns in order."""
    notes = []

    original = text

    text = CRLF.sub('\n', text)
    text = HEADER_LINE.sub('', text)

    before = len(text)
    text = RELEASE_ID.sub('', text)
    text = VISITOR_CTR.sub('', text)
    text = LANG_LIST.sub('', text)
    text = LANG_NAMES.sub('', text)
    text = BYLINES.sub('', text)
    text = SOCIAL_MEDIA.sub('', text)
    text = TWEET_ATTR.sub('', text)
    text = STAR_SEP.sub('', text)
    if len(text) < before:
        notes.append('removed_boilerplate')

    text = LINE_INDENT.sub('', text)
    text = SHORT_LINES.sub('', text)
    text = MULTI_BLANK.sub('\n\n', text)
    text = text.strip()

    return text, notes


def deduplicate(text):
    """
    Remove duplicate second half. PIB pages render content twice.
    Strategy: split at midpoint, check if the two halves start the
    same way, keep only the first.
    """
    if len(text) < 400:
        return text, False

    half = len(text) // 2
    first_half  = text[:half]
    second_half = text[half:]

    # Compare first 150 chars of each half (after stripping leading whitespace)
    first_start  = first_half.lstrip()[:150]
    second_start = second_half.lstrip()[:150]

    if len(first_start) > 50 and first_start[:80] == second_start[:80]:
        return first_half.rstrip(), True

    # Fallback: look for exact repetition of the first paragraph
    first_para = first_half.strip().split('\n\n')[0]
    if len(first_para) > 100 and second_half.count(first_para[:100]) > 0:
        # Find where the repetition starts in second half and truncate there
        idx = second_half.find(first_para[:80])
        if idx >= 0:
            return (first_half + second_half[:idx]).rstrip(), True

    return text, False


# ── Main processing ───────────────────────────────────────────────────────────

def process_one(in_path, out_path):
    """
    Load, clean, validate one PIB JSON.
    Returns (status, quality_notes) where status is one of:
    'ok', 'marathi', 'too_short', 'not_devanagari', 'error'
    """
    try:
        with open(in_path, encoding='utf-8') as f:
            record = json.load(f)
    except Exception as e:
        return 'error', [str(e)]

    quality_notes = list(record.get('quality_notes', []))
    raw_text = record.get('text', '') or ''

    # Step 1: Deduplicate
    text, was_duped = deduplicate(raw_text)
    if was_duped:
        quality_notes.append('removed_duplicate_half')

    # Step 2: Remove boilerplate
    text, bp_notes = remove_boilerplate(text)
    quality_notes.extend(bp_notes)

    # Step 3: Language detection
    lang = detect_language(text)
    if lang == 'mr':
        return 'marathi', ['detected_marathi_not_hindi']
    if lang == 'other':
        # Check if there's any Devanagari at all
        deva = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        if deva / max(len(text), 1) < 0.1:
            return 'not_devanagari', ['insufficient_devanagari']
        # Ambiguous — keep but flag
        quality_notes.append('language_ambiguous_devanagari')

    # Step 4: Length check
    text_bytes = len(text.encode('utf-8'))
    if text_bytes < MIN_TEXT_BYTES:
        return 'too_short', [f'text_bytes={text_bytes}']

    # Step 5: Update record
    record['text'] = text
    record['text_bytes_utf8'] = text_bytes
    record['text_hash'] = hashlib.md5(text.encode('utf-8')).hexdigest()
    record['quality_notes'] = quality_notes
    record['quality_flag'] = 'ok'
    record['processed_at'] = datetime.now().isoformat()

    # Update pages too
    if record.get('pages'):
        record['pages'][0]['text'] = text
        record['pages'][0]['text_bytes_utf8'] = text_bytes

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return 'ok', quality_notes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--raw-dir',   default=PIB_RAW)
    parser.add_argument('--clean-dir', default=PIB_CLEAN)
    args = parser.parse_args()

    raw_dir   = args.raw_dir
    clean_dir = args.clean_dir

    if not os.path.exists(raw_dir):
        print(f"ERROR: pib_raw directory not found at {raw_dir}")
        print("Set --raw-dir to the correct path.")
        sys.exit(1)

    os.makedirs(clean_dir, exist_ok=True)

    # Collect all input files
    all_files = []
    for shard_dir in sorted(os.listdir(raw_dir)):
        shard_path = os.path.join(raw_dir, shard_dir)
        if not os.path.isdir(shard_path):
            continue
        for fname in os.listdir(shard_path):
            if fname.endswith('.json'):
                all_files.append((shard_dir, fname))

    total = len(all_files)
    print(f"Found {total} PIB JSON files in {raw_dir}")

    if args.dry_run:
        # Process a sample of 100 files spread across shards
        import random
        sample = random.sample(all_files, min(100, total))
        print(f"DRY RUN: sampling {len(sample)} files\n")
    else:
        sample = None

    counts = defaultdict(int)
    note_counts = defaultdict(int)
    processed = 0
    total_bytes_in = 0
    total_bytes_out = 0

    files_to_process = sample if args.dry_run else all_files

    for i, (shard_dir, fname) in enumerate(files_to_process):
        in_path  = os.path.join(raw_dir, shard_dir, fname)
        out_path = os.path.join(clean_dir, shard_dir, fname)

        # Skip already processed
        if not args.dry_run and os.path.exists(out_path):
            counts['skipped'] += 1
            continue

        in_size = os.path.getsize(in_path)
        total_bytes_in += in_size

        status, notes = process_one(in_path, out_path if not args.dry_run else '/dev/null')
        counts[status] += 1
        for note in notes:
            note_counts[note] += 1

        if status == 'ok' and not args.dry_run:
            total_bytes_out += os.path.getsize(out_path)

        processed += 1
        if processed % 5000 == 0 or args.dry_run:
            print(f"  [{processed}/{len(files_to_process)}] "
                  f"ok={counts['ok']} marathi={counts['marathi']} "
                  f"too_short={counts['too_short']} "
                  f"not_devanagari={counts['not_devanagari']} "
                  f"error={counts['error']}")

    # Report
    print(f"\n{'='*50}")
    print(f"{'DRY RUN ' if args.dry_run else ''}POSTPROCESS COMPLETE")
    print(f"{'='*50}")
    print(f"Total files:         {len(files_to_process)}")
    print(f"  OK (kept):         {counts['ok']}")
    print(f"  Marathi (dropped): {counts['marathi']}")
    print(f"  Too short:         {counts['too_short']}")
    print(f"  Not Devanagari:    {counts['not_devanagari']}")
    print(f"  Skipped (done):    {counts['skipped']}")
    print(f"  Errors:            {counts['error']}")
    if not args.dry_run:
        print(f"\nInput size:    {total_bytes_in / 1e6:.1f} MB")
        print(f"Output size:   {total_bytes_out / 1e6:.1f} MB")
    print(f"\nTop cleaning notes:")
    for note, count in sorted(note_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {note:<45} {count:>8}")

    # Write report
    if not args.dry_run:
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_input': len(files_to_process),
            'counts': dict(counts),
            'note_counts': dict(note_counts),
            'input_mb': round(total_bytes_in / 1e6, 1),
            'output_mb': round(total_bytes_out / 1e6, 1),
        }
        with open(REPORT_OUT, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport → {REPORT_OUT}")
        print(f"Clean files → {clean_dir}")
        print(f"\nNext step — rsync to HPC:")
        print(f"  rsync -av Data/pib_clean/ \\")
        print(f"    USER@HPC_HOST:"
              f"projects/BERTaav/data/pib_clean/")


if __name__ == '__main__':
    main()