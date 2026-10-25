#!/usr/bin/env python3
"""
Pre-tokenise the DAPT training corpus once, on CPU, so GPU time is spent only on training.

Output: uint32 .npy files in /storage/<user>/dapt_full/tokens/  (one file = a long stream of
token ids, documents separated by [SEP]; no padding, no [CLS]).

  sang : Sangraha CORE tier (distinct_strong>=3 and political_score>=8), pib_like dropped,
         validation bucket (md5 % 100 < 1) dropped, read from the local dedup shards.
         Parallel: each array task handles shards[task::ntasks]. Resumable (skips finished files).
  lok  : Lok Sabha Q&A TRAIN split, read from the HF dataset (needs internet + HF cache ->
         run this one on the LOGIN node; ~16M tokens, a few minutes).

Usage
  python scripts/pretokenize_dapt.py --source lok                       # login node
  python scripts/pretokenize_dapt.py --source sang --task 0 --ntasks 20 # one array task
  python scripts/pretokenize_dapt.py --source sang --limit-shards 2     # quick test
"""
import os, sys, re, json, glob, argparse, time
import numpy as np

USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(STORAGE, "hf_cache", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(STORAGE, "hf_cache", "models"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(STORAGE, "hf_cache", "datasets"))
OUT = os.path.join(STORAGE, "dapt_full", "tokens")
DEDUP = os.path.join(STORAGE, "sangraha_hindi_political_dedup")
MODEL = "ai4bharat/IndicBERTv2-MLM-only"
REPO = "Shubh1198/BERTaav"
MD5_TAIL = re.compile(r'"text_md5": "([0-9a-f]{32})"\}\s*$')
CORE_DISTINCT, CORE_SCORE = 3, 8.0
MAX_CHARS = 20000


def encode(tok, texts):
    sep = tok.sep_token_id
    parts = []
    for i in range(0, len(texts), 512):
        for ids in tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]:
            parts.append(np.asarray(ids + [sep], dtype=np.uint32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint32)


def sang_texts(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = MD5_TAIL.search(line[-80:])
            try:
                r = json.loads(line)
            except ValueError:
                continue
            md5 = m.group(1) if m else r.get("text_md5", "")
            if not md5 or int(md5[:8], 16) % 100 < 1:      # validation bucket
                continue
            if r.get("pib_like"):
                continue
            if r["distinct_strong"] >= CORE_DISTINCT and r["political_score"] >= CORE_SCORE:
                yield r["text"][:MAX_CHARS]


def pick_text(row):
    if isinstance(row.get("text"), str):
        return row["text"]
    best = ""
    for v in row.values():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["sang", "lok"], required=True)
    ap.add_argument("--task", type=int, default=int(os.environ.get("PBS_ARRAY_INDEX", 0)))
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    total = 0
    t0 = time.time()

    if a.source == "sang":
        shards = sorted(glob.glob(os.path.join(DEDUP, "shard_*.jsonl")))
        if a.limit_shards:
            shards = shards[:a.limit_shards]
        mine = shards[a.task::a.ntasks]
        print(f"task {a.task}/{a.ntasks}: {len(mine)} of {len(shards)} shards", flush=True)
        for sp in mine:
            name = "sang_" + os.path.basename(sp).replace(".jsonl", "") + ".npy"
            dst = os.path.join(OUT, name)
            if os.path.exists(dst):
                continue
            arr = encode(tok, list(sang_texts(sp)))
            np.save(dst + ".tmp.npy", arr)
            os.replace(dst + ".tmp.npy", dst)
            total += len(arr)
            print(f"  {name}: {len(arr):,} tokens  ({time.time()-t0:.0f}s)", flush=True)
    else:
        from datasets import load_dataset
        ds = load_dataset(REPO, "default", split="train", streaming=True)
        buf, k = [], 0
        def flush():
            nonlocal buf, k, total
            if not buf:
                return
            arr = encode(tok, buf)
            dst = os.path.join(OUT, f"lok_{k:04d}.npy")
            np.save(dst + ".tmp.npy", arr); os.replace(dst + ".tmp.npy", dst)
            total += len(arr); k += 1; buf = []
            print(f"  {dst}: {len(arr):,} tokens  ({time.time()-t0:.0f}s)", flush=True)
        for row in ds:
            t = pick_text(row)
            if len(t) >= 200:
                buf.append(t[:MAX_CHARS])
            if len(buf) >= 5000:
                flush()
        flush()
    print(f"done. this run wrote {total:,} tokens", flush=True)


if __name__ == "__main__":
    main()
