#!/usr/bin/env python3
"""
run_ocr.py — Stage 2: Run Tesseract OCR on images, write JSON output.

For each file in status='images_done', reads its page PNGs from images/,
runs pytesseract with Hindi language pack, and writes one JSON file to
ocr_output/ matching the agreed schema:

  {
    "id": ...,
    "source_type": ...,
    "source_collection": ...,
    "title": ...,
    "date": null,
    "metadata": {...},
    "language": "hi",
    "source_pdf": ...,
    "source_pdf_bytes": ...,
    "ocr_engine": "tesseract-5.2.0",
    "ocr_confidence_mean": ...,
    "num_pages": ...,
    "text": "...",
    "text_bytes_utf8": ...,
    "pages": [
      {"page_num": 1, "text": "...", "confidence_mean": ..., "text_bytes_utf8": ...},
      ...
    ]
  }

USAGE:
  python3 run_ocr.py                        # process all images_done files
  python3 run_ocr.py --chunk-id 3 --num-chunks 50   # PBS array chunk
  python3 run_ocr.py --retry-failed         # retry failed_ocr files
"""

import argparse
import json
import logging
import os
import sys
import traceback

import pytesseract
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import pipeline_db as db

# ── OCR config ────────────────────────────────────────────────────────────────
TESSERACT_LANG   = "hin"
TESSERACT_CONFIG = "--oem 3 --psm 3"
# oem 3 = default LSTM engine
# psm 3 = fully automatic page segmentation (best for scanned documents)


def setup_logging(chunk_id=None):
    os.makedirs(db.LOGS_DIR, exist_ok=True)
    suffix   = f"_chunk{chunk_id}" if chunk_id is not None else ""
    log_path = os.path.join(db.LOGS_DIR, f"run_ocr{suffix}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def get_tesseract_version():
    try:
        return pytesseract.get_tesseract_version().public
    except Exception:
        return "unknown"


def ocr_one_page(image_path):
    """
    Run Tesseract on a single page image.
    Returns (text: str, confidence_mean: float)
    """
    img = Image.open(image_path)

    # Get both text and confidence data in one call
    data = pytesseract.image_to_data(
        img,
        lang=TESSERACT_LANG,
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    # Filter to words with valid confidence (Tesseract uses -1 for non-word rows)
    confidences = [
        int(c) for c in data["conf"]
        if str(c).lstrip("-").isdigit() and int(c) >= 0
    ]
    text = pytesseract.image_to_string(
        img,
        lang=TESSERACT_LANG,
        config=TESSERACT_CONFIG,
    ).strip()

    conf_mean = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    return text, conf_mean


def ocr_one_file(row, tess_version, logger, chunk_id=0):
    """
    OCR all pages of one document, assemble the JSON schema, write to disk.
    Returns (success: bool, error: str|None)
    """
    source_pdf_path = row["source_pdf_path"]
    item_uuid       = row["item_uuid"]
    images_dir      = row["images_dir"]

    if not images_dir or not os.path.isdir(images_dir):
        err = f"images_dir missing or not found: {images_dir}"
        logger.error(f"[FAIL] {row['filename']}: {err}")
        db.mark_ocr_failed_safe(source_pdf_path, err, chunk_id)
        return False, err

    # Get page images in order
    page_images = sorted([
        f for f in os.listdir(images_dir) if f.endswith(".png")
    ])

    if not page_images:
        err = "No PNG images found in images_dir"
        logger.error(f"[FAIL] {row['filename']}: {err}")
        db.mark_ocr_failed_safe(source_pdf_path, err, chunk_id)
        return False, err

    # OCR each page
    pages_data = []
    all_texts  = []
    all_confs  = []

    for page_file in page_images:
        page_num = int(page_file.replace("page_", "").replace(".png", ""))
        img_path = os.path.join(images_dir, page_file)
        try:
            text, conf = ocr_one_page(img_path)
        except Exception:
            err = traceback.format_exc()
            logger.error(f"[FAIL] {row['filename']} page {page_num}:\n{err[:400]}")
            db.mark_ocr_failed_safe(source_pdf_path, err, chunk_id)
            return False, err

        text_bytes = len(text.encode("utf-8"))
        pages_data.append({
            "page_num":       page_num,
            "text":           text,
            "confidence_mean": conf,
            "text_bytes_utf8": text_bytes,
        })
        all_texts.append(text)
        if conf > 0:
            all_confs.append(conf)

    full_text        = "\n\n".join(all_texts)
    full_text_bytes  = len(full_text.encode("utf-8"))
    overall_conf     = round(sum(all_confs) / len(all_confs), 2) if all_confs else 0.0

    # Assemble the schema
    record = {
        "id":               item_uuid,
        "source_type":      row.get("source_type", "lok_sabha_qa"),
        "source_collection": row.get("source_collection", ""),
        "title":            row.get("title"),
        "date":             None,
        "metadata": {
            "ministry":        None,
            "lok_sabha_session": None,
            "speaker":         None,
            "newspaper_name":  None,
        },
        "language":          "hi",
        "source_pdf":        row["filename"],
        "source_pdf_bytes":  row.get("source_pdf_bytes"),
        "ocr_engine":        f"tesseract-{tess_version}",
        "ocr_confidence_mean": overall_conf,
        "num_pages":         len(pages_data),
        "text":              full_text,
        "text_bytes_utf8":   full_text_bytes,
        "pages":             pages_data,
    }

    # Write JSON output
    os.makedirs(db.OCR_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(db.OCR_OUTPUT_DIR, f"{item_uuid}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception as e:
        err = f"Failed writing JSON: {e}"
        logger.error(f"[FAIL] {row['filename']}: {err}")
        db.mark_ocr_failed_safe(source_pdf_path, err, chunk_id)
        return False, err

    db.mark_ocr_done_safe(source_pdf_path, out_path, overall_conf, full_text_bytes, chunk_id)
    logger.info(
        f"[OK] {row['filename']} | {len(pages_data)}pp | "
        f"conf={overall_conf:.1f} | {full_text_bytes/1024:.1f}KB text"
    )
    return True, None


def get_my_chunk(rows, chunk_id, num_chunks):
    return [r for i, r in enumerate(rows) if i % num_chunks == chunk_id]


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Images → OCR JSON")
    parser.add_argument("--chunk-id",     type=int, default=None)
    parser.add_argument("--num-chunks",   type=int, default=1)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if args.chunk_id is None and "PBS_ARRAY_INDEX" in os.environ:
        args.chunk_id = int(os.environ["PBS_ARRAY_INDEX"])

    logger = setup_logging(args.chunk_id)
    tess_version = get_tesseract_version()
    logger.info(f"=== run_ocr starting | tesseract={tess_version} | "
                f"lang={TESSERACT_LANG} | chunk={args.chunk_id}/{args.num_chunks} ===")

    db.init_db()

    if args.retry_failed:
        conn = db.get_conn()
        conn.execute("UPDATE files SET status='images_done', ocr_error=NULL "
                     "WHERE status='failed_ocr'")
        conn.commit()
        conn.close()
        logger.info("Reset failed_ocr → images_done for retry")

    pending = db.get_pending_ocr()

    if args.chunk_id is not None and args.num_chunks > 1:
        my_work = get_my_chunk(pending, args.chunk_id, args.num_chunks)
        logger.info(f"Chunk {args.chunk_id}/{args.num_chunks}: "
                    f"{len(my_work)}/{len(pending)} files")
    else:
        my_work = pending
        logger.info(f"{len(my_work)} files to OCR")

    if args.dry_run:
        for row in my_work[:20]:
            print(f"  {row['filename']}  ({row['num_pages']} pages)")
        if len(my_work) > 20:
            print(f"  ... and {len(my_work)-20} more")
        return

    if not my_work:
        logger.info("Nothing to do.")
        return

    ok = fail = 0
    for i, row in enumerate(my_work, start=1):
        logger.info(f"[{i}/{len(my_work)}] {row['filename']}")
        success, _ = ocr_one_file(row, tess_version, logger, chunk_id=args.chunk_id or 0)
        if success:
            ok += 1
        else:
            fail += 1

    logger.info(f"=== Done: {ok} ok, {fail} failed ===")


if __name__ == "__main__":
    main()
