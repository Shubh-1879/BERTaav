#!/usr/bin/env python3
"""Per-source results: macro-F1 of each model on Lok Sabha vs Sangraha excerpts (mean over seeds),
with a paired bootstrap over excerpts for each model vs base.
  python by_source.py --dir runs_am --map excerpt_source.csv [--ref base]"""
import argparse, glob, os, csv
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--map", required=True)
ap.add_argument("--ref", default="base"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
src = {r["excerpt_id"]: r["source"] for r in csv.DictReader(open(a.map, encoding="utf-8-sig"))}
L = {"N": 0, "C": 1, "P": 2}
def macro(y, p):
    s = 0.0
    for c in (0, 1, 2):
        tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
        s += 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return s / 3
D = {}
for f in sorted(glob.glob(os.path.join(a.dir, "preds_*.csv"))):
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    if not rows: continue
    by = {}
    for r in rows: by.setdefault((r["seg_id"], r["ex"], r["gold"]), {})[int(r["seed"])] = L[r["pred"]]
    seeds = sorted({int(r["seed"]) for r in rows}); keys = sorted(by)
    D[rows[0]["model"]] = (np.array([L[k[2]] for k in keys]), np.array([[by[k].get(s, -1) for s in seeds] for k in keys]),
                           np.array([k[1] for k in keys]), keys)
def score(m, mask):
    y, p, _, _ = D[m]; return float(np.mean([macro(y[mask], p[mask, s]) for s in range(p.shape[1])]))
rng = np.random.default_rng(0)
for s in ("lok", "sang"):
    print(f"\n== {s} excerpts")
    masks = {m: np.array([src.get(e) == s for e in D[m][2]]) for m in D}
    for m in D: print(f"{m:10} macro-F1 {score(m, masks[m]):.3f}  (segments: {int(masks[m].sum())})")
    if a.ref in D:
        exs = sorted({e for e in D[a.ref][2][masks[a.ref]]}); ex_idx = {m: {e: np.where((D[m][2] == e))[0] for e in exs} for m in D}
        for m in D:
            if m == a.ref: continue
            diffs = []
            for _ in range(a.boot):
                pick = rng.choice(exs, len(exs))
                ia = np.concatenate([ex_idx[m][e] for e in pick]); ib = np.concatenate([ex_idx[a.ref][e] for e in pick])
                ya, pa = D[m][0][ia], D[m][1][ia]; yb, pb = D[a.ref][0][ib], D[a.ref][1][ib]
                diffs.append(np.mean([macro(ya, pa[:, k]) for k in range(pa.shape[1])]) - np.mean([macro(yb, pb[:, k]) for k in range(pb.shape[1])]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"  {m} - {a.ref}: diff {score(m, masks[m]) - score(a.ref, masks[a.ref]):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  ({'includes 0' if lo <= 0 <= hi else 'excludes 0'})")
