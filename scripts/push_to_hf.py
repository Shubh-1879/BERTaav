#!/usr/bin/env python3
"""
push_to_hf.py — Push processed JSON files to HuggingFace as a dataset.

PREREQUISITES:
  pip install datasets huggingface_hub
  huggingface-cli login   (or set HF_TOKEN env var)

USAGE:
  # Dry run — shows what would be pushed without actually pushing
  python scripts/push_to_hf.py --dry-run

  # Push all 'ok' quality documents only (recommended for first push)
  python scripts/push_to_hf.py --repo-id YOUR_HF_USERNAME/BERTaav-Hindi-Parliament

  # Push everything including flagged documents (with quality_flag column)
  python scripts/push_to_hf.py --repo-id YOUR_HF_USERNAME/BERTaav-Hindi-Parliament --include-flagged

  # Push a specific quality tier only
  python scripts/push_to_hf.py --repo-id ... --quality ok
  python scripts/push_to_hf.py --repo-id ... --quality low_confidence

WHAT GETS PUSHED:
  The dataset will have these splits (matching RooseBERT's 90/10 approach):
    - train (90%)
    - validation (10%)
  Split is deterministic (seed=42) and stratified by source_type so each
  split has proportional representation from each source.

  Each row in the dataset is one document with all schema fields populated.
  Page-level data is included as a JSON string column (HF doesn't support
  nested lists natively in all viewers, but the data is there).
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.environ.get(
    "BERTAAV_ROOT",
    "/home/shubham.agarwal_phd24/projects/BERTaav"
)
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
REPORT_PATH   = os.path.join(PROCESSED_DIR, "quality_report.json")

# Columns to include in the HuggingFace dataset (in order)
# Pages are serialised to JSON string since HF datasets don't render
# nested lists cleanly in the dataset viewer
HF_COLUMNS = [
    "id",
    "source_type",
    "source_collection",
    "title",
    "date",
    "question_type",
    "question_number",
    "questioner",
    "ministry_parsed",
    "language",
    "source_pdf",
    "source_pdf_bytes",
    "ocr_engine",
    "ocr_confidence_mean",
    "num_pages",
    "text",
    "text_bytes_utf8",
    "text_hash",
    "quality_flag",
    "quality_notes",
    "pages_json",          # serialised pages array
    "processed_at",
]


def load_processed(quality_filter: list[str]) -> list[dict]:
    """Load all processed JSONs matching the quality filter."""
    records = []
    for fname in os.listdir(PROCESSED_DIR):
        if not fname.endswith(".json") or fname == "quality_report.json":
            continue
        fpath = os.path.join(PROCESSED_DIR, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                r = json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not load {fname}: {e}")
            continue
        if r.get("quality_flag") in quality_filter:
            records.append(r)
    return records


def to_hf_row(record: dict) -> dict:
    """Convert a processed record to a flat HF dataset row."""
    row = {}
    for col in HF_COLUMNS:
        if col == "pages_json":
            row[col] = json.dumps(
                record.get("pages", []), ensure_ascii=False
            )
        elif col == "quality_notes":
            # Store as semicolon-separated string for readability in HF viewer
            notes = record.get("quality_notes", [])
            row[col] = "; ".join(notes) if notes else ""
        else:
            row[col] = record.get(col)
    return row


def stratified_split(records: list[dict], test_frac: float = 0.1, seed: int = 42):
    """
    Stratified 90/10 split by source_type.
    Returns (train_records, val_records).
    """
    by_source = defaultdict(list)
    for r in records:
        by_source[r.get("source_type", "unknown")].append(r)

    rng = random.Random(seed)
    train, val = [], []

    for source, recs in by_source.items():
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * test_frac))
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])
        print(f"  {source}: {len(recs)} total → "
              f"{len(recs)-n_val} train / {n_val} val")

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main():
    parser = argparse.ArgumentParser(
        description="Push processed Hindi parliament corpus to HuggingFace"
    )
    parser.add_argument(
        "--repo-id", default=None,
        help="HuggingFace repo ID, e.g. username/BERTaav-Hindi-Parliament"
    )
    parser.add_argument(
        "--quality", nargs="+",
        default=["ok", "low_confidence", "low_text", "duplicate"],
        choices=["ok", "low_confidence", "low_text", "duplicate", "empty", "all"],
        help="Which quality tiers to include (default: ok only)"
    )
    parser.add_argument(
        "--include-flagged", action="store_true",
        help="Include low_confidence documents alongside ok (shorthand)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show stats without pushing"
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Push as a private dataset"
    )
    args = parser.parse_args()

    quality_filter = args.quality
    if "all" in quality_filter:
        quality_filter = ["ok", "low_confidence", "low_text", "duplicate", "empty"]
    if args.include_flagged and "low_confidence" not in quality_filter:
        quality_filter.append("low_confidence")

    print(f"Loading processed documents (quality: {quality_filter})...")
    records = load_processed(quality_filter)
    print(f"Loaded {len(records)} documents")

    if not records:
        print("No documents found. Run postprocess.py first.")
        sys.exit(1)

    # Stats
    total_mb = sum(
        (r.get("text_bytes_utf8") or 0) for r in records
    ) / 1e6
    print(f"Total text: {total_mb:.1f} MB")

    # Split
    print("\nCreating train/validation split (90/10, stratified by source_type):")
    train_records, val_records = stratified_split(records)
    print(f"\nFinal split: {len(train_records)} train / {len(val_records)} validation")

    if args.dry_run:
        print("\nDry run — not pushing. Add --repo-id to push.")
        # Show a sample row
        if train_records:
            sample = to_hf_row(train_records[0])
            print("\nSample row (first train document):")
            for k, v in sample.items():
                val_preview = str(v)[:100] + "..." if str(v) and len(str(v)) > 100 else v
                print(f"  {k}: {val_preview}")
        return

    if not args.repo_id:
        print("Error: --repo-id is required for pushing. "
              "Use --dry-run to preview without pushing.")
        sys.exit(1)

    # Import here so dry-run works without these installed
    try:
        from datasets import Dataset, DatasetDict, Features, Value, Sequence
        from huggingface_hub import HfApi
    except ImportError:
        print("Install required packages: pip install datasets huggingface_hub")
        sys.exit(1)

    print(f"\nPreparing HuggingFace dataset...")
    train_rows = [to_hf_row(r) for r in train_records]
    val_rows   = [to_hf_row(r) for r in val_records]

    train_ds = Dataset.from_list(train_rows)
    val_ds   = Dataset.from_list(val_rows)

    dataset_dict = DatasetDict({
        "train":      train_ds,
        "validation": val_ds,
    })

    print(f"Pushing to {args.repo_id}...")
    dataset_dict.push_to_hub(
        args.repo_id,
        private=args.private,
        commit_message=(
            f"Add Hindi Parliament Q&A corpus: "
            f"{len(train_records)} train + {len(val_records)} val documents, "
            f"{total_mb:.0f}MB text"
        ),
    )

    print(f"\n✓ Dataset pushed to https://huggingface.co/datasets/{args.repo_id}")
    print(f"  Train: {len(train_records)} documents")
    print(f"  Val:   {len(val_records)} documents")
    print(f"  Text:  {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
