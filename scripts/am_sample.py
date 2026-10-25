#!/usr/bin/env python3
"""
Draw the argument-mining gold-standard excerpts from HELD-OUT text only.

Pool (never used for DAPT training):
  sang : Sangraha core tier, validation bucket (md5[:8] % 100 < 1), pib_like dropped
  lok  : Lok Sabha Q&A, HF config `default`, split `validation`

An "excerpt" = WINDOW consecutive valid sentences (default 10). Segments = sentences.
Strata (target n): sang_random 60, sang_cue 60, lok_random 40, lok_cue 40.
  random : random document, random window
  cue    : window with the most argument-cue words, drawn at random from the top 40%
           of documents by best-window cue count (needs >= 2 cues)
Sets (stratified, per stratum): CAL ~8%  (you + mother, independent, for guideline calibration)
                                A   ~20% (you + mother, independent, blind -> headline kappa)
                                B   rest (you review LLM pre-labels)

Steps
  python scripts/am_sample.py pool-sang            # CPU job (scans the dedup shards)
  python scripts/am_sample.py pool-lok             # login node (needs internet)
  python scripts/am_sample.py select               # quick; writes the sheets
Outputs in /storage/<user>/am_gold/
"""
import os, re, sys, json, glob, argparse, random, hashlib, csv, math, collections

USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(STORAGE, "hf_cache", "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(STORAGE, "hf_cache", "datasets"))
OUT = os.path.join(STORAGE, "am_gold")
DEDUP = os.path.join(STORAGE, "sangraha_hindi_political_dedup")
REPO = "Shubh1198/BERTaav"
MD5_TAIL = re.compile(r'"text_md5": "([0-9a-f]{32})"\}\s*$')
CORE_DISTINCT, CORE_SCORE = 3, 8.0
MAX_CHARS = 20000
MAX_SENTS = 150


def norm(s):
    """Fold chandrabindu->anusvara and drop nukta so spelling variants match."""
    return s.replace("ँ", "ं").replace("़", "")


CUES = ["क्योंकि", "चूंकि", "इसलिए", "इसीलिए", "अतः", "लिहाजा", "मेरा मानना", "मेरी मान्यता",
        "मेरी राय", "मुझे लगता", "हमारा मानना", "मैं मानता", "मैं मानती", "यही कारण",
        "इसी कारण", "इसके कारण", "जिसके कारण", "उदाहरण के लिए", "उदाहरणार्थ", "फलस्वरूप",
        "इसके चलते", "ताकि", "इसका मतलब", "इससे साफ"]
CUE_RE = re.compile("|".join(re.escape(norm(c)) for c in CUES))


def cue_count(text):
    return len(CUE_RE.findall(norm(text)))


# Sentence splitting. Lok Sabha text is OCR'd with a hard line break at every printed line, so for
# source == "lok" single newlines are joined (a line break is NOT a sentence boundary) and we split
# only on sentence punctuation and before list markers such as "(ख)". Web text (Sangraha) keeps
# newlines as boundaries.
SPLIT = re.compile(r'(?<=[।?!])\s+|(?<=[^\s0-9]\.)\s+(?=[\'"\u201c\u2018(]?[\u0900-\u097F0-9])|\n+')
SPLIT_JOINED = re.compile(r'(?<=[।?!])\s+|(?<=[^\s0-9]\.)\s+(?=[\'"\u201c\u2018(]?[\u0900-\u097F0-9])|\s+(?=\([\u0915\u0916\u0917\u0918\u0919]\)\s)')
END_PUNCT = re.compile(r'[।?!.;:,"\u201d\u2019]\s*$')


def sentences(text, join_lines=False):
    if join_lines:
        text = re.sub(r'\s*\n\s*', ' ', text)
        parts = SPLIT_JOINED.split(text)
    else:
        parts = SPLIT.split(text)
    out = []
    for s in parts:
        s = re.sub(r'\s+', ' ', s).strip()
        if s:
            out.append(s)
    return out


def dev_ratio(s):
    dev = sum('ऀ' <= c <= 'ॿ' for c in s)
    asc = sum(c.isascii() and c.isalnum() for c in s)
    return dev / max(1, dev + asc)


def valid_sent(s):
    n = len(s.split())
    return 3 <= n <= 120 and dev_ratio(s) >= 0.5


def valid_windows(sents, w):
    ok = [valid_sent(s) for s in sents]
    pu = [1 if END_PUNCT.search(s) else 0 for s in sents]
    return [i for i in range(0, len(sents) - w + 1)
            if all(ok[i:i + w]) and sum(pu[i:i + w]) >= 0.7 * w]


def make_doc(doc_id, source, text, meta, w):
    sents = sentences(text[:MAX_CHARS], join_lines=(source == "lok"))[:MAX_SENTS]
    if not valid_windows(sents, w):
        return None
    return {"id": doc_id, "source": source, "meta": meta, "sents": sents}


# ---------------------------------------------------------------- pool builders
def pool_sang(a):
    os.makedirs(OUT, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(DEDUP, "shard_*.jsonl")))
    if a.limit_shards:
        shards = shards[:a.limit_shards]
    st = collections.Counter()
    dst = os.path.join(OUT, "pool_sang.jsonl")
    with open(dst, "w", encoding="utf-8") as fo:
        for k, sp in enumerate(shards):
            with open(sp, encoding="utf-8") as f:
                for line in f:
                    st["lines"] += 1
                    m = MD5_TAIL.search(line[-80:])
                    try:
                        r = json.loads(line)
                    except ValueError:
                        st["bad_json"] += 1
                        continue
                    md5 = m.group(1) if m else r.get("text_md5", "")
                    if not md5 or int(md5[:8], 16) % 100 >= 1:
                        continue
                    st["val_bucket"] += 1
                    if r.get("pib_like"):
                        st["pib_like"] += 1
                        continue
                    if not (r["distinct_strong"] >= CORE_DISTINCT and r["political_score"] >= CORE_SCORE):
                        continue
                    st["val_core"] += 1
                    d = make_doc(md5, "sang", r["text"], {"shard": os.path.basename(sp)}, a.window)
                    if d is None:
                        st["no_window"] += 1
                        continue
                    st["pooled"] += 1
                    fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(shards)} shards  {dict(st)}", flush=True)
    print("pool_sang done:", dict(st), flush=True)
    json.dump(dict(st), open(os.path.join(OUT, "pool_sang_stats.json"), "w"))


def pick_text(row):
    if isinstance(row.get("text"), str):
        return row["text"]
    best = ""
    for v in row.values():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best


def pool_lok(a):
    os.makedirs(OUT, exist_ok=True)
    if a.lok_local:                       # testing only
        rows = (json.loads(l) for l in open(a.lok_local, encoding="utf-8"))
    else:
        from datasets import load_dataset
        rows = load_dataset(REPO, "default", split="validation", streaming=True)
    st = collections.Counter()
    dst = os.path.join(OUT, "pool_lok.jsonl")
    with open(dst, "w", encoding="utf-8") as fo:
        for i, row in enumerate(rows):
            st["rows"] += 1
            t = pick_text(row)
            if len(t) < 200:
                st["short"] += 1
                continue
            meta = {k: v for k, v in row.items() if isinstance(v, str) and v is not t and len(v) <= 100}
            d = make_doc(f"lok_val_{i}", "lok", t, meta, a.window)
            if d is None:
                st["no_window"] += 1
                continue
            st["pooled"] += 1
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("pool_lok done:", dict(st), flush=True)
    json.dump(dict(st), open(os.path.join(OUT, "pool_lok_stats.json"), "w"))


# ---------------------------------------------------------------- selection
def read_pool(source):
    p = os.path.join(OUT, f"pool_{source}.jsonl")
    seen, docs = set(), []
    for l in open(p, encoding="utf-8"):
        d = json.loads(l)
        key = hashlib.md5(re.sub(r"\W+", "", d["sents"][0] + d["sents"][-1])[:300].encode()).hexdigest()
        if key in seen:
            continue                      # near-exact duplicate document
        seen.add(key)
        docs.append(d)
    return docs


def select(a):
    rnd = random.Random(a.seed)
    w = a.window
    targets = {"sang": {"random": a.sang_random, "cue": a.sang_cue},
               "lok": {"random": a.lok_random, "cue": a.lok_cue}}
    picked, report = [], {}
    for source in ("sang", "lok"):
        docs = read_pool(source)
        report[source] = {"pool_docs": len(docs)}
        for d in docs:
            ws = valid_windows(d["sents"], w)
            cues = [cue_count(" ".join(d["sents"][i:i + w])) for i in ws]
            m = max(cues)
            best = [i for i, c in zip(ws, cues) if c == m]
            d["_ws"], d["_best_start"], d["_best_cue"] = ws, rnd.choice(best), m
        used = set()
        # cue stratum
        elig = sorted([d for d in docs if d["_best_cue"] >= 2], key=lambda d: -d["_best_cue"])
        top = elig[:max(targets[source]["cue"] * 3, int(len(elig) * 0.4))]
        n = min(targets[source]["cue"], len(top))
        if n < targets[source]["cue"]:
            print(f"WARNING {source} cue: only {n} eligible docs", file=sys.stderr)
        for d in rnd.sample(top, n):
            used.add(d["id"])
            picked.append((f"{source}_cue", d, d["_best_start"]))
        # random stratum
        rest = [d for d in docs if d["id"] not in used]
        n = min(targets[source]["random"], len(rest))
        if n < targets[source]["random"]:
            print(f"WARNING {source} random: only {n} docs", file=sys.stderr)
        for d in rnd.sample(rest, n):
            picked.append((f"{source}_random", d, rnd.choice(d["_ws"])))
        report[source]["cue_eligible"] = len(elig)
    # sets: stratified, per stratum
    by_stratum = collections.defaultdict(list)
    for p in picked:
        by_stratum[p[0]].append(p)
    ex = []
    for s, lst in by_stratum.items():
        rnd.shuffle(lst)
        n = len(lst)
        n_cal, n_a = round(n * 0.08), round(n * 0.20)
        for k, p in enumerate(lst):
            ex.append((("CAL" if k < n_cal else "A" if k < n_cal + n_a else "B"), p))
    rnd.shuffle(ex)
    os.makedirs(OUT, exist_ok=True)
    seg_rows, ex_rows, sheets = [], [], collections.defaultdict(list)
    for idx, (sset, (stratum, d, start)) in enumerate(ex, 1):
        eid = f"E{idx:03d}"
        segs = d["sents"][start:start + w]
        ex_rows.append({"excerpt_id": eid, "set": sset, "stratum": stratum, "source": d["source"],
                        "doc_id": d["id"], "meta": d["meta"], "start": start,
                        "cue_count": cue_count(" ".join(segs)), "segments": segs})
        for k, s in enumerate(segs, 1):
            seg_rows.append([eid, f"{eid}_{k:02d}", sset, stratum, d["source"], k, s])
            sheets[sset].append([eid, k, s, ""])
    with open(os.path.join(OUT, "am_excerpts.jsonl"), "w", encoding="utf-8") as f:
        for r in ex_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "am_segments_FULL.csv"), "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["excerpt_id", "seg_id", "set", "stratum", "source", "seg_no", "segment"])
        wr.writerows(seg_rows)
    for sset, rows in sheets.items():      # annotator sheets: NO stratum / source / pre-labels
        with open(os.path.join(OUT, f"annotate_{sset}.csv"), "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["excerpt_id", "seg_no", "segment", "label"])
            wr.writerows(rows)
    cnt = collections.Counter((r["set"], r["stratum"]) for r in ex_rows)
    cues = collections.defaultdict(list)
    for r in ex_rows:
        cues[r["stratum"]].append(r["cue_count"])
    report["excerpts"] = len(ex_rows)
    report["segments"] = len(seg_rows)
    report["counts"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(cnt.items())}
    report["mean_cues_per_excerpt"] = {k: round(sum(v) / len(v), 2) for k, v in cues.items()}
    report["seed"], report["window"] = a.seed, w
    json.dump(report, open(os.path.join(OUT, "am_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pool-sang", "pool-lok", "select"])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--lok-local", default=None, help="testing only: local jsonl instead of HF")
    for k, v in [("sang_random", 60), ("sang_cue", 60), ("lok_random", 40), ("lok_cue", 40)]:
        ap.add_argument("--" + k.replace("_", "-"), dest=k, type=int, default=v)
    a = ap.parse_args()
    {"pool-sang": pool_sang, "pool-lok": pool_lok, "select": select}[a.cmd](a)


if __name__ == "__main__":
    main()
