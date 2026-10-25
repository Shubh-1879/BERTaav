#!/usr/bin/env python3
"""
Draw a stratified, BLIND evaluation sample for the Sangraha political filter.

Why stratified: political documents are ~18% of the raw data, and the 14M rejected
documents are mostly obviously non-political. A plain random sample would waste almost
every label on easy cases. Instead we sample kept docs per tier, and rejected docs per
"how close to the line" stratum, and record each stratum's population share so the
results can be re-weighted into precision / approximate recall later.

Strata (same thresholds as the real run: >=80 tokens, >=2 distinct strong terms, score>=4.0)
  kept docs  : core / extended      (tier rules as in the HF push)
  rejected   : near       = has >=1 strong term but missed the rules
               weak_only  = no strong term, >=3 weak-term hits
               no_signal  = no strong term, <3 weak hits
               short      = < 80 tokens (counted for weights, never labelled)

Outputs (default: logs/filter_eval/)
  sample_blind.csv   id, preview            -> upload this for labelling (no filter info)
  human40.csv        id, preview, label     -> YOU label these blind (P / B / N)
  sample_key.csv     id, stratum, scores    -> keep private until labels are done
  strata.json        population share of each stratum (from a random raw draw)
  rubric.txt         labelling rubric

Usage:  python scripts/filter_eval_sample.py
"""
import os, sys, re, csv, json, glob, random, argparse, collections

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from sangraha_filter import score  # noqa: E402  (same scorer as the real run)

USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE_BASE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")

MIN_TOKENS, MIN_DISTINCT, MIN_DENSITY = 80, 2, 4.0
CORE_DISTINCT, CORE_SCORE = 3, 8.0
N_TOTAL_RAW = 17_420_932

TARGETS = {"core": 60, "extended": 60, "near": 40, "weak_only": 40, "no_signal": 30}
PREVIEW_CHARS = 700

RUBRIC = """LABELS (judge the document, not the filter):
P = POLITICAL      The text is mainly about politics: politicians, parties, elections, legislatures,
                   political decisions or disputes, or political commentary/analysis (including court
                   cases with clear political stakes).
B = BORDERLINE     Government administration, schemes or announcements, civic information, exam/GK/
                   textbook-style or encyclopedic text that mentions political things but contains no
                   political discussion or contest.
N = NOT POLITICAL  Political words are incidental, or the text is about something else (sports, crime,
                   entertainment, health, religion, ...).
If unsure between two labels, pick the one you would defend if asked.
"""


def classify_raw(sc):
    n, _s, distinct, weak, dens = sc
    if n < MIN_TOKENS:
        return "short"
    if distinct >= MIN_DISTINCT and dens >= MIN_DENSITY:
        return "kept"
    if distinct >= 1:
        return "near"
    return "weak_only" if weak >= 3 else "no_signal"


def tier_of(distinct, dens):
    return "core" if (distinct >= CORE_DISTINCT and dens >= CORE_SCORE) else "extended"


def preview(text):
    return " ".join(text.split())[:PREVIEW_CHARS]


def read_lines_sample(path, k, rng):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    return rng.sample(lines, min(k, len(lines)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "sangraha_hindi_raw"))
    ap.add_argument("--kept-dir", default=os.path.join(STORAGE_BASE, "sangraha_hindi_political_dedup"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "logs", "filter_eval"))
    ap.add_argument("--raw-shards", type=int, default=25)
    ap.add_argument("--per-raw-shard", type=int, default=1000)
    ap.add_argument("--kept-shards", type=int, default=30)
    ap.add_argument("--n-total-raw", type=int, default=N_TOTAL_RAW)
    ap.add_argument("--human-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- 1) random draw from the RAW data: stratum shares + rejected candidates -------------
    raw_shards = sorted(glob.glob(os.path.join(a.raw_dir, "shard_*.jsonl")))
    kept_shards = sorted(glob.glob(os.path.join(a.kept_dir, "shard_*.jsonl")))
    if not raw_shards or not kept_shards:
        sys.exit(f"missing shards: raw={len(raw_shards)} in {a.raw_dir}, kept={len(kept_shards)} in {a.kept_dir}")
    shares = collections.Counter()
    rejected_pool = collections.defaultdict(list)
    n_raw = 0
    for sp in rng.sample(raw_shards, min(a.raw_shards, len(raw_shards))):
        for line in read_lines_sample(sp, a.per_raw_shard, rng):
            try:
                text = json.loads(line)["text"]
            except (ValueError, KeyError):
                continue
            sc = score(text)
            st = classify_raw(sc)
            shares[st] += 1
            n_raw += 1
            if st in ("near", "weak_only", "no_signal"):
                rejected_pool[st].append((text, sc))
    strata = {st: {"share": shares[st] / n_raw, "est_docs": round(shares[st] / n_raw * a.n_total_raw),
                   "sample_count": shares[st]} for st in ("kept", "near", "weak_only", "no_signal", "short")}
    strata["_meta"] = {"raw_docs_sampled": n_raw, "n_total_raw": a.n_total_raw,
                       "thresholds": {"min_tokens": MIN_TOKENS, "min_distinct": MIN_DISTINCT, "min_density": MIN_DENSITY}}

    # ---- 2) kept docs (post-dedup, PIB-like excluded) by tier --------------------------------
    kept_pool = collections.defaultdict(list)
    mismatches = 0
    for sp in rng.sample(kept_shards, min(a.kept_shards, len(kept_shards))):
        with open(sp, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("pib_like"):
                    continue
                t = tier_of(r["distinct_strong"], r["political_score"])
                kept_pool[t].append(r)
    # sanity: recomputing the score on kept docs must reproduce the stored values
    for t in ("core", "extended"):
        for r in rng.sample(kept_pool[t], min(200, len(kept_pool[t]))):
            sc = score(r["text"])
            if abs(sc[4] - r["political_score"]) > 0.02 or sc[2] != r["distinct_strong"]:
                mismatches += 1

    # ---- 3) draw the labelling sample --------------------------------------------------------
    rows = []   # (stratum, tier, text, n, distinct, dens)
    for t in ("core", "extended"):
        for r in rng.sample(kept_pool[t], min(TARGETS[t], len(kept_pool[t]))):
            rows.append((t, t, r["text"], r["n_tokens"], r["distinct_strong"], r["political_score"]))
    for st in ("near", "weak_only", "no_signal"):
        for text, sc in rng.sample(rejected_pool[st], min(TARGETS[st], len(rejected_pool[st]))):
            rows.append((st, "rejected", text, sc[0], sc[2], sc[4]))
    rng.shuffle(rows)

    ids = [f"E{i+1:03d}" for i in range(len(rows))]
    with open(os.path.join(a.out_dir, "sample_blind.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "preview"])
        for i, row in zip(ids, rows):
            w.writerow([i, preview(row[2])])
    with open(os.path.join(a.out_dir, "sample_key.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "stratum", "decision", "n_tokens", "distinct_strong", "political_score"])
        for i, row in zip(ids, rows):
            w.writerow([i, row[0], "kept" if row[0] in ("core", "extended") else "rejected",
                        row[3], row[4], round(row[5], 2)])
    human_idx = sorted(rng.sample(range(len(rows)), min(a.human_n, len(rows))))
    with open(os.path.join(a.out_dir, "human40.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "preview", "label"])
        for k in human_idx:
            w.writerow([ids[k], preview(rows[k][2]), ""])
    with open(os.path.join(a.out_dir, "strata.json"), "w", encoding="utf-8") as f:
        json.dump(strata, f, indent=2)
    with open(os.path.join(a.out_dir, "rubric.txt"), "w", encoding="utf-8") as f:
        f.write(RUBRIC)

    # ---- 4) report ---------------------------------------------------------------------------
    print(f"raw docs scored: {n_raw:,}")
    for st in ("kept", "near", "weak_only", "no_signal", "short"):
        print(f"  {st:10s} {100*strata[st]['share']:6.2f}%   (~{strata[st]['est_docs']:,} docs of {a.n_total_raw:,})")
    print("  (check: 'kept' share should be close to the real run's 18.49% before dedup/PIB drop)")
    print(f"score recompute mismatches on kept docs: {mismatches} (expect 0)")
    c = collections.Counter(r[0] for r in rows)
    print("sample:", dict(c), "total", len(rows), "| human subset:", len(human_idx))
    print("files in", a.out_dir)


if __name__ == "__main__":
    main()
