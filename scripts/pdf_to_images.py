#!/usr/bin/env python3
"""
pdf_to_images.py — Stage 1: Convert PDFs to per-page PNG images.

Replaces your existing pdf_2_img.py with:
  - Manifest-driven processing (reads from pipeline.db, not os.listdir)
  - Resumability (skips files already in images_done / ocr_done states)
  - Per-file error isolation (one bad PDF doesn't stop the job)
  - PBS array job support (--chunk-id / --num-chunks for parallel execution)
  - Structured logging to pipeline.db + errors written to logs/

USAGE:
  # First run: sync manifest then process
  python3 pdf_to_images.py --sync-first

  # PBS array job (50 parallel chunks):
  python3 pdf_to_images.py --chunk-id $PBS_ARRAY_INDEX --num-chunks 50

  # Dry run to check what would be processed:
  python3 pdf_to_images.py --dry-run
"""

import argparse
import logging
import os
import sys
import traceback

from pdf2image import convert_from_path

sys.path.insert(0, os.path.dirname(__file__))
import pipeline_db as db

DPI          = 300
FMT          = "png"
SAVE_WORKERS = 2


def setup_logging(chunk_id=None):
    os.makedirs(db.LOGS_DIR, exist_ok=True)
    suffix   = f"_chunk{chunk_id}" if chunk_id is not None else ""
    log_path = os.path.join(db.LOGS_DIR, f"pdf_to_images{suffix}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def convert_one_pdf(row, logger, chunk_id=0):
    source_pdf_path = row["source_pdf_path"]
    item_uuid       = row["item_uuid"]
    images_dir      = os.path.join(db.IMAGES_DIR, item_uuid)
    os.makedirs(images_dir, exist_ok=True)

    # Safety net: if images already on disk, just mark done and skip
    existing = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    if existing:
        logger.info(f"[SKIP] {item_uuid} already has {len(existing)} images on disk")
        db.mark_images_done_safe(source_pdf_path, images_dir, len(existing), chunk_id)
        return True, len(existing), None

    try:
        pages = convert_from_path(
            source_pdf_path,
            dpi=DPI,
            fmt=FMT,
            thread_count=SAVE_WORKERS,
        )
    except Exception:
        err = traceback.format_exc()
        logger.error(f"[FAIL] {row['filename']}: pdf2image error:\n{err[:500]}")
        db.mark_images_failed_safe(source_pdf_path, err, chunk_id)
        return False, 0, err

    for i, page in enumerate(pages, start=1):
        save_path = os.path.join(images_dir, f"page_{i:04d}.png")
        try:
            page.save(save_path, "PNG")
        except Exception as e:
            err = f"Failed saving page {i}: {e}"
            logger.error(f"[FAIL] {row['filename']} page {i}: {e}")
            db.mark_images_failed_safe(source_pdf_path, err, chunk_id)
            return False, 0, err

    db.mark_images_done_safe(source_pdf_path, images_dir, len(pages), chunk_id)
    logger.info(f"[OK] {row['filename']} → {len(pages)} pages → {images_dir}")
    return True, len(pages), None


def get_my_chunk(rows, chunk_id, num_chunks):
    return [r for i, r in enumerate(rows) if i % num_chunks == chunk_id]


def main():
    parser = argparse.ArgumentParser(description="Stage 1: PDF → Images")
    parser.add_argument("--sync-first", action="store_true",
                        help="Sync manifest from raw_pdfs/ before processing")
    parser.add_argument("--source-type", default="lok_sabha_qa")
    parser.add_argument("--chunk-id",    type=int, default=None)
    parser.add_argument("--num-chunks",  type=int, default=1)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if args.chunk_id is None and "PBS_ARRAY_INDEX" in os.environ:
        args.chunk_id = int(os.environ["PBS_ARRAY_INDEX"])

    logger = setup_logging(args.chunk_id)
    logger.info(f"=== pdf_to_images starting | chunk={args.chunk_id}/{args.num_chunks} ===")

    if args.sync_first:
        new, total = db.sync_manifest(source_type=args.source_type)
        logger.info(f"Manifest sync: +{new} new, {total} total")

    db.init_db()

    if args.retry_failed:
        conn = db.get_conn()
        conn.execute("UPDATE files SET status='pending', images_error=NULL "
                     "WHERE status='failed_images'")
        conn.commit()
        conn.close()
        logger.info("Reset failed_images → pending for retry")

    pending = db.get_pending_images()

    if args.chunk_id is not None and args.num_chunks > 1:
        my_work = get_my_chunk(pending, args.chunk_id, args.num_chunks)
        logger.info(f"Chunk {args.chunk_id}/{args.num_chunks}: "
                    f"{len(my_work)}/{len(pending)} files")
    else:
        my_work = pending
        logger.info(f"{len(my_work)} files to process")

    if args.dry_run:
        for row in my_work[:20]:
            print(f"  {row['filename']}")
        if len(my_work) > 20:
            print(f"  ... and {len(my_work)-20} more")
        return

    if not my_work:
        logger.info("Nothing to do.")
        return

    ok = fail = 0
    for i, row in enumerate(my_work, start=1):
        logger.info(f"[{i}/{len(my_work)}] {row['filename']}")
        success, _, _ = convert_one_pdf(row, logger, chunk_id=args.chunk_id or 0)
        if success:
            ok += 1
        else:
            fail += 1

    logger.info(f"=== Done: {ok} ok, {fail} failed ===")


if __name__ == "__main__":
    main()
