#!/usr/bin/env python3
"""
Exact-dedup the filtered Sangraha shards (first occurrence wins, shards in sorted
order) and print how many docs / GB would survive at tighter thresholds.

Reads  : <storage>/sangraha_hindi_political/shard_*.jsonl
Writes : <storage>/sangraha_hindi_political_dedup/shard_*.jsonl
Memory : one md5 string per unique doc (~3M docs -> roughly 0.5 GB of RAM)
"""
import os, sys, json, glob, argparse, hashlib

USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE_BASE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")
DEFAULT_IN = os.path.join(STORAGE_BASE, "sangraha_hindi_political")
DEFAULT_OUT = os.path.join(STORAGE_BASE, "sangraha_hindi_political_dedup")

DENS = (4, 5, 6, 8, 10, 12)
DISTINCT = (2, 3, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="write into a non-empty out-dir")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.in_dir, "shard_*.jsonl")))
    if not shards:
        sys.exit(f"No shards in {a.in_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    if glob.glob(os.path.join(a.out_dir, "shard_*.jsonl")) and not a.force:
        sys.exit(f"{a.out_dir} already has shards; remove them or pass --force")

    seen = set()
    total = dups = pib = 0
    tab = {(m, d): [0, 0] for m in DISTINCT for d in DENS}   # [docs, bytes]

    for i, sp in enumerate(shards):
        final = os.path.join(a.out_dir, os.path.basename(sp))
        tmp = final + ".tmp"
        with open(sp, encoding="utf-8") as fi, open(tmp, "w", encoding="utf-8") as fo:
            for line in fi:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                total += 1
                h = r.get("text_md5") or hashlib.md5(" ".join(r["text"].split()).encode("utf-8")).hexdigest()
                if h in seen:
                    dups += 1
                    continue
                seen.add(h)
                fo.write(line)
                if r.get("pib_like"):
                    pib += 1
                nbytes = len(r["text"].encode("utf-8"))
                ds, sc = r.get("distinct_strong", 0), r.get("political_score", 0.0)
                for m in DISTINCT:
                    if ds >= m:
                        for d in DENS:
                            if sc >= d:
                                tab[(m, d)][0] += 1
                                tab[(m, d)][1] += nbytes
        os.replace(tmp, final)
        if (i + 1) % 50 == 0 or i + 1 == len(shards):
            print(f"{i+1}/{len(shards)} shards  unique={len(seen):,}  dups={dups:,}", flush=True)

    uniq = len(seen)
    print(f"\ninput docs:   {total:,}")
    print(f"exact dups:   {dups:,} ({100*dups/max(total,1):.2f}%)")
    print(f"unique docs:  {uniq:,}")
    print(f"pib_like (among unique): {pib:,}")
    print("\nSurvivors if you tighten thresholds (docs in millions / text in GB):")
    print("min_distinct\\min_density " + "".join(f"{d:>13}" for d in DENS))
    for m in DISTINCT:
        row = "".join(f"{tab[(m,d)][0]/1e6:6.2f}M/{tab[(m,d)][1]/1e9:4.1f}G" for d in DENS)
        print(f"{m:>25} {row}")


if __name__ == "__main__":
    main()
