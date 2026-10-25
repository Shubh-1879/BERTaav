#!/usr/bin/env python3
"""
Pre-tokenise GENERIC Hindi text (no strong political terms) for the DAPT control run.

Reads the raw Sangraha Hindi shards (data/sangraha_hindi_raw/shard_*.jsonl), keeps documents with
zero strong political terms (same test as the generic validation set: sangraha_filter.score(t)[2]==0),
and writes uint32 token streams gen_<shard>.npy into /storage/<user>/dapt_full/tokens/.

The shard order is the same shuffle (Random(0)) the pilot used; the first 8 shards of that order built
the generic VALIDATION set, so they are skipped here (no train/val overlap).

Usage: python scripts/pretokenize_generic.py --task 0 --ntasks 20     # one array task
       python scripts/pretokenize_generic.py --limit-shards 1         # quick test
"""
import os, sys, json, glob, random, argparse, time
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SCRIPT_DIR)
from sangraha_filter import score
from pretokenize_dapt import encode, OUT, MODEL, MAX_CHARS
ROOT = os.path.dirname(SCRIPT_DIR)

ap = argparse.ArgumentParser()
ap.add_argument("--task", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", 0)))
ap.add_argument("--ntasks", type=int, default=1)
ap.add_argument("--limit-shards", type=int, default=None)
a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
os.makedirs(OUT, exist_ok=True)
shards = sorted(glob.glob(os.path.join(ROOT, "data", "sangraha_hindi_raw", "shard_*.jsonl")))
random.Random(0).shuffle(shards)
shards = shards[8:]                       # first 8 = generic validation shards
if a.limit_shards: shards = shards[:a.limit_shards]
mine = shards[a.task::a.ntasks]
print(f"task {a.task}/{a.ntasks}: {len(mine)} of {len(shards)} usable shards", flush=True)
t0 = time.time(); total = 0
for sp in mine:
    dst = os.path.join(OUT, "gen_" + os.path.basename(sp).replace(".jsonl", "") + ".npy")
    if os.path.exists(dst): continue
    texts = []
    with open(sp, encoding="utf-8") as f:
        for line in f:
            try: t = json.loads(line)["text"]
            except Exception: continue
            if len(t) >= 300 and score(t)[2] == 0:
                texts.append(t[:MAX_CHARS])
    arr = encode(tok, texts)
    np.save(dst + ".tmp.npy", arr); os.replace(dst + ".tmp.npy", dst)
    total += len(arr)
    print(f"  {os.path.basename(dst)}: {len(texts):,} docs, {len(arr):,} tokens ({time.time()-t0:.0f}s)", flush=True)
print(f"done. this run wrote {total:,} tokens", flush=True)
