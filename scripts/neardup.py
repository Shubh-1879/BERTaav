#!/usr/bin/env python3
"""
Near-duplicate tooling (numpy MinHash, no extra dependencies).

  pib   : clean + exact-dedup + near-dedup the PIB press-release JSONs
            python neardup.py pib --in-dir <dir with prid_*/pib_*.json> --out-dir <dir>
  dump  : (login node) dump a HF split to jsonl, for the leakage check
            python neardup.py dump --config default --split validation --out lok_val.jsonl
  leak  : how many VALIDATION docs have a near-duplicate in the TRAINING text?
            python neardup.py leak --val lok_val.jsonl --train-jsonl lok_train.jsonl --out leak_lok.json
            python neardup.py leak --val sang_val.jsonl --train-sang          --out leak_sang.json
  Run with PYTHONHASHSEED=0 (token hashing uses Python's hash()).
"""
import os, sys, re, json, glob, argparse, hashlib, time, random
import numpy as np

NP, SH, BANDS = 64, 5, 16            # permutations, words per shingle, LSH bands (16 x 4 rows)
ROWS = NP // BANDS
P = (1 << 61) - 1
_rng = np.random.default_rng(12345)
A = _rng.integers(1, P, NP, dtype=np.uint64); B = _rng.integers(0, P, NP, dtype=np.uint64)
TOK = re.compile(r"[ऀ-ॿ]+|[A-Za-z0-9]+")
DEV = re.compile(r"[ऀ-ॿ]")

def signature(text):
    w = TOK.findall(text)
    if len(w) < SH + 3: return None
    h = np.fromiter((hash(x) & 0xFFFFFFFF for x in w), dtype=np.uint64, count=len(w))
    sh = h[:-SH + 1].copy() if SH > 1 else h.copy()
    for k in range(1, SH):
        sh = sh * np.uint64(1000003) + h[k:len(h) - SH + 1 + k]     # uint64 wraparound = fine as a hash mixer
    sh = sh[(sh % np.uint64(2)) == 0] if len(sh) > 4000 else sh      # sample very long docs (same rule everywhere)
    if len(sh) == 0: return None
    x = (sh[:, None] * A[None, :] + B[None, :]) >> np.uint64(3)       # cheap universal-ish mixing
    return x.min(axis=0)

def band_keys(sig):
    return [(b, hash(sig[b * ROWS:(b + 1) * ROWS].tobytes())) for b in range(BANDS)]

def jaccard(s1, s2): return float((s1 == s2).mean())

# ---------------------------------------------------------------- PIB
BOILER = [re.compile(p) for p in (
    r"^Press Release Page", r"^प्रविष्टि तिथि", r"^Posted On", r"^\d{1,2} [A-Za-z]{3} \d{4}.*by PIB",
    r"^\(?\s*(रिलीज़|रिलीज) आईडी", r"^\(?\s*Release ID", r"^आगंतुक पटल", r"^Visitor Counter")]

def clean_pib(text, title, ministry):
    seen, out = set(), []
    for l in text.replace("\r", "\n").split("\n"):
        l = " ".join(l.split())
        if not l: continue
        if any(p.search(l) for p in BOILER): continue
        if l == (title or "").strip() or l == (ministry or "").strip(): continue
        if len(l) >= 40:
            if l in seen: continue          # the page text is repeated verbatim inside each record
            seen.add(l)
        out.append(l)
    return "\n".join(out)

def dev_ratio(t):
    letters = sum(c.isalpha() for c in t[:5000])
    return len(DEV.findall(t[:5000])) / max(letters, 1)

def pib(a):
    files = sorted(glob.glob(os.path.join(a.in_dir, "**", "pib_*.json"), recursive=True))
    os.makedirs(a.out_dir, exist_ok=True)
    st = dict(files=len(files), unreadable=0, empty_or_short=0, non_hindi=0, exact_dup=0, near_dup=0, kept=0,
              raw_chars=0, clean_chars=0, kept_chars=0)
    seen_md5, buckets, sigs, kept_ids = set(), {}, {}, {}
    dups = open(os.path.join(a.out_dir, "pib_dups_sample.tsv"), "w", encoding="utf-8")
    dups.write("dropped_id\tkept_id\tkind\test_jaccard\n")
    fo = open(os.path.join(a.out_dir, "pib_clean.jsonl"), "w", encoding="utf-8")
    t0 = time.time()
    for i, f in enumerate(files):
        try: r = json.load(open(f, encoding="utf-8"))
        except Exception: st["unreadable"] += 1; continue
        md = r.get("metadata") or {}
        raw = r.get("text") or ""; st["raw_chars"] += len(raw)
        t = clean_pib(raw, r.get("title"), md.get("ministry")); st["clean_chars"] += len(t)
        if len(TOK.findall(t)) < a.min_words: st["empty_or_short"] += 1; continue
        if dev_ratio(t) < 0.5: st["non_hindi"] += 1; continue
        did = r.get("id") or os.path.basename(f)
        h = hashlib.md5(" ".join(t.split()).encode("utf-8")).hexdigest()
        if h in seen_md5:
            st["exact_dup"] += 1; dups.write(f"{did}\t-\texact\t1.0\n"); continue
        seen_md5.add(h)
        sig = signature(t)
        if sig is not None:
            cand = {}
            for k in band_keys(sig):
                for c in buckets.get(k, ()): cand[c] = 1
            hit = next(((c, j) for c in cand if (j := jaccard(sig, sigs[c])) >= a.thr), None)
            if hit:
                st["near_dup"] += 1; dups.write(f"{did}\t{hit[0]}\tnear\t{hit[1]:.2f}\n"); continue
            sigs[did] = sig
            for k in band_keys(sig): buckets.setdefault(k, []).append(did)
        st["kept"] += 1; st["kept_chars"] += len(t)
        fo.write(json.dumps({"id": did, "title": r.get("title"), "date": r.get("date"), "ministry": md.get("ministry"),
                             "prid": md.get("prid"), "url": md.get("url"), "text": t}, ensure_ascii=False) + "\n")
        if (i + 1) % 5000 == 0:
            print(f"{i+1}/{len(files)}  kept {st['kept']}  exact {st['exact_dup']}  near {st['near_dup']}  ({time.time()-t0:.0f}s)", flush=True)
    fo.close(); dups.close()
    json.dump(st, open(os.path.join(a.out_dir, "pib_dedup_stats.json"), "w"), indent=2)
    print(json.dumps(st, indent=2))

# ---------------------------------------------------------------- dump / leak
def dump(a):
    from datasets import load_dataset
    ds = load_dataset("Shubh1198/BERTaav", a.config, split=a.split, streaming=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in ds:
            t = r.get("text") or ""
            if len(t) >= 200: f.write(json.dumps({"text": t[:20000]}, ensure_ascii=False) + "\n"); n += 1
    print("wrote", n, "docs to", a.out)

def train_texts(a):
    if a.train_sang:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pretokenize_dapt import sang_texts, DEDUP
        for sp in sorted(glob.glob(os.path.join(DEDUP, "shard_*.jsonl"))): yield from sang_texts(sp)
    else:
        for line in open(a.train_jsonl, encoding="utf-8"): yield json.loads(line)["text"]

def leak(a):
    vals = [json.loads(l)["text"] for l in open(a.val, encoding="utf-8")]
    vsig, buckets = {}, {}
    for i, t in enumerate(vals):
        s = signature(t)
        if s is None: continue
        vsig[i] = s
        for k in band_keys(s): buckets.setdefault(k, []).append(i)
    print(len(vals), "validation docs,", len(vsig), "with signatures", flush=True)
    best = {}                                          # val id -> best estimated Jaccard seen in train
    t0 = time.time(); n = 0
    for t in train_texts(a):
        n += 1
        if a.max_train and n > a.max_train: break
        s = signature(t)
        if s is None: continue
        c = {}
        for k in band_keys(s):
            for v in buckets.get(k, ()): c[v] = 1
        for v in c:
            j = jaccard(s, vsig[v])
            if j > best.get(v, 0): best[v] = j
        if n % 20000 == 0: print(f"{n} train docs  ({time.time()-t0:.0f}s)  val docs with a candidate: {len(best)}", flush=True)
    res = {"val_docs": len(vals), "val_with_signature": len(vsig), "train_docs_scanned": n}
    for thr in (0.5, 0.7, 0.8, 0.9):
        res[f"val_docs_with_train_near_dup_ge_{thr}"] = sum(1 for j in best.values() if j >= thr)
        res[f"fraction_ge_{thr}"] = round(res[f"val_docs_with_train_near_dup_ge_{thr}"] / max(len(vsig), 1), 4)
    json.dump({"result": res, "best_jaccard_per_val": {str(k): round(v, 3) for k, v in best.items()}}, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------- language split (Devanagari != Hindi)
LW = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")
HI_M = set("है हैं किया गया गयी गई था थे थी नहीं होगा रहा रही रहे कि लिए द्वारा करें".split())
MR_M = set("आहे आहेत आणि मध्ये यांनी केले केली केल्या झाले झाली झाला नाही साठी करण्यात असून तसेच त्यांनी होते होती असे".split())

def lang_guess(t):
    w = LW.findall(t)[:4000]
    h = sum(x in HI_M for x in w); m = sum(x in MR_M for x in w)
    if m > h: return "mr", h, m
    if h == 0: return "unk", h, m
    return "hi", h, m

def split_lang(a):
    cnt = {"hi": 0, "mr": 0, "unk": 0}; ex = {"mr": [], "unk": []}
    os.makedirs(a.out_dir, exist_ok=True)
    fo = {k: open(os.path.join(a.out_dir, f"pib_{k}.jsonl"), "w", encoding="utf-8") for k in cnt}
    for line in open(a.inp, encoding="utf-8"):
        r = json.loads(line); k, h, m = lang_guess(r["text"]); cnt[k] += 1
        fo[k].write(line)
        if k in ex and len(ex[k]) < 8: ex[k].append((r["id"], h, m, (r.get("title") or "")[:60]))
    for f in fo.values(): f.close()
    print(json.dumps(cnt, indent=1)); print(json.dumps(ex, ensure_ascii=False, indent=1))
    json.dump(cnt, open(os.path.join(a.out_dir, "pib_lang_split.json"), "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pib"); p.add_argument("--in-dir", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--thr", type=float, default=0.8); p.add_argument("--min-words", type=int, default=30)
    d = sp.add_parser("dump"); d.add_argument("--config", default="default"); d.add_argument("--split", default="validation"); d.add_argument("--out", required=True)
    l = sp.add_parser("leak"); l.add_argument("--val", required=True); l.add_argument("--train-jsonl"); l.add_argument("--train-sang", action="store_true")
    l.add_argument("--max-train", type=int, default=0); l.add_argument("--out", required=True)
    g = sp.add_parser("split-lang"); g.add_argument("--in", dest="inp", required=True); g.add_argument("--out-dir", required=True)
    a = ap.parse_args(); {"pib": pib, "dump": dump, "leak": leak, "split-lang": split_lang}[a.cmd](a)
