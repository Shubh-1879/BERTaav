#!/usr/bin/env python3
"""
Push the filtered + deduplicated Sangraha political subset to the BERTaav HF dataset
repo as its OWN config (default name: 'sangraha_political'). The existing Lok Sabha
config is not touched.

What it does
  * drops pib_like docs (PIB is covered by our own PIB scrape)
  * adds a `tier` column: 'core' (distinct_strong >= 3 and political_score >= 8)
    or 'extended'. Full set = all rows; core set = rows with tier == 'core'.
  * deterministic ~1% validation split chosen from text_md5 (no shuffling, reproducible)
  * streams from disk via a generator (never holds 15 GB in RAM)
  * keeps the Arrow build cache on /storage, NOT /home

Usage
  python scripts/push_sangraha_to_hf.py --dry-run --limit-shards 3       # build only, no upload
  python scripts/push_sangraha_to_hf.py --limit-shards 3 \
        --repo-id Shubh1198/BERTaav-test --private                       # scratch-repo test
  nohup python scripts/push_sangraha_to_hf.py > logs/push_sangraha.log 2>&1 &   # the real push
"""
import os, sys, re, json, glob, argparse

USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE_BASE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")

# Must be set BEFORE `datasets` is imported. We deliberately do NOT override HF_HOME,
# because that is where your login token lives.
HF_CACHE_ROOT = os.path.join(STORAGE_BASE, "hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_CACHE_ROOT, "datasets"))
os.environ.setdefault("HF_XET_CACHE", os.path.join(HF_CACHE_ROOT, "xet"))

# Parquet staging files during upload can be several GB; keep them off a small /tmp.
try:
    _tmp = os.path.join(STORAGE_BASE, "tmp")
    os.makedirs(_tmp, exist_ok=True)
    os.environ.setdefault("TMPDIR", _tmp)
except OSError:
    pass

DEFAULT_IN = os.path.join(STORAGE_BASE, "sangraha_hindi_political_dedup")
VAL_PERCENT = 1                      # ~1% of docs go to validation
CORE_MIN_DISTINCT, CORE_MIN_SCORE = 3, 8.0
MD5_TAIL = re.compile(r'"text_md5": "([0-9a-f]{32})"\}\s*$')


def bucket(md5_hex):
    return int(md5_hex[:8], 16) % 100


def gen(shards, want_val):
    """Yield records for one split. The md5 sits at the end of each line, so we can
    decide the split WITHOUT parsing the JSON (cheap skip for the other split)."""
    for sp in shards:
        with open(sp, encoding="utf-8") as f:
            for line in f:
                m = MD5_TAIL.search(line[-80:])
                if m:
                    md5 = m.group(1)
                else:                                   # fallback if key order ever changes
                    try:
                        md5 = json.loads(line)["text_md5"]
                    except (ValueError, KeyError):
                        continue
                if (bucket(md5) < VAL_PERCENT) != want_val:
                    continue
                r = json.loads(line)
                if r.get("pib_like"):
                    continue
                core = (r["distinct_strong"] >= CORE_MIN_DISTINCT
                        and r["political_score"] >= CORE_MIN_SCORE)
                yield {
                    "text": r["text"],
                    "source": r.get("source", "sangraha_hindi"),
                    "n_tokens": int(r["n_tokens"]),
                    "distinct_strong": int(r["distinct_strong"]),
                    "political_score": float(r["political_score"]),
                    "tier": "core" if core else "extended",
                    "text_md5": md5,
                }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="Shubh1198/BERTaav")
    ap.add_argument("--config-name", default="sangraha_political")
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--limit-shards", type=int, default=None, help="use only the first N shards (testing)")
    ap.add_argument("--dry-run", action="store_true", help="build + report, do not upload")
    ap.add_argument("--private", action="store_true", help="only matters when creating a NEW repo")
    ap.add_argument("--num-proc", type=int, default=None, help="parallel build workers (be gentle on the login node)")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.in_dir, "shard_*.jsonl")))
    if not shards:
        sys.exit(f"No shards in {a.in_dir}")
    if a.limit_shards:
        shards = shards[: a.limit_shards]
    print(f"{len(shards)} shards from {a.in_dir}", flush=True)

    # Fail fast on auth BEFORE a long build.
    if not a.dry_run:
        from huggingface_hub import HfApi
        try:
            who = HfApi().whoami()["name"]
        except Exception as e:
            sys.exit(f"Not logged in to Hugging Face ({e}). Run: huggingface-cli login")
        print(f"Logged in as {who}; pushing config '{a.config_name}' to {a.repo_id}", flush=True)

    from datasets import Dataset, DatasetDict, Features, Value
    features = Features({
        "text": Value("string"), "source": Value("string"),
        "n_tokens": Value("int32"), "distinct_strong": Value("int32"),
        "political_score": Value("float32"), "tier": Value("string"),
        "text_md5": Value("string"),
    })
    cache = os.path.join(HF_CACHE_ROOT, "sangraha_build")
    os.makedirs(cache, exist_ok=True)

    splits = {}
    for name, want_val in (("train", False), ("validation", True)):
        print(f"building {name} ...", flush=True)
        splits[name] = Dataset.from_generator(
            gen, gen_kwargs={"shards": shards, "want_val": want_val},
            features=features, cache_dir=cache, num_proc=a.num_proc)
    dd = DatasetDict(splits)

    for name, ds in dd.items():
        tiers = ds["tier"]
        n_core = sum(1 for t in tiers if t == "core")
        print(f"{name:11s} docs={len(ds):,}  core={n_core:,}  extended={len(ds)-n_core:,}  "
              f"tokens={sum(ds['n_tokens']):,}", flush=True)

    if a.dry_run:
        print("DRY RUN: nothing uploaded.")
        return

    dd.push_to_hub(a.repo_id, config_name=a.config_name, private=a.private)
    print(f"\nPushed. Check: https://huggingface.co/datasets/{a.repo_id}")
    print(f"Verify with: load_dataset('{a.repo_id}', '{a.config_name}')")


if __name__ == "__main__":
    main()
