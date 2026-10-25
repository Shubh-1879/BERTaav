#!/usr/bin/env python3
"""
Summarise preds_*.csv: per model, F1 (macro, claim, premise, neither) averaged over seeds, plus
paired bootstrap 95% CIs over EXCERPTS for the difference of each model vs a reference (default: base).
  python summarize_am.py --dir runs_am --ref base
"""
import argparse, glob, os, csv
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--ref", default="base")
ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
P = {}
for f in sorted(glob.glob(os.path.join(a.dir, "preds_*.csv"))):
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    if rows: P[rows[0]["model"]] = rows
L = {"N": 0, "C": 1, "P": 2}
def f1(y, p, c):
    tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
def metrics(y, p):
    c, pp, n = f1(y, p, 1), f1(y, p, 2), f1(y, p, 0)
    return {"macro": (c + pp + n) / 3, "C": c, "P": pp, "N": n}
# arrays: model -> excerpt -> (y, [pred per seed])
exs = sorted({r["ex"] for rows in P.values() for r in rows})
D = {}
for m, rows in P.items():
    seeds = sorted({int(r["seed"]) for r in rows})
    bykey = {}
    for r in rows: bykey.setdefault((r["seg_id"], r["ex"], r["gold"]), {})[int(r["seed"])] = L[r["pred"]]
    keys = sorted(bykey)
    D[m] = (keys, np.array([L[k[2]] for k in keys]), np.array([[bykey[k].get(s, -1) for s in seeds] for k in keys]), [k[1] for k in keys])
def seedmean(m, sel=None):
    keys, y, p, e = D[m]
    sel = np.arange(len(y)) if sel is None else sel
    outs = [metrics(y[sel], p[sel, s]) for s in range(p.shape[1])]
    return {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}
print(f"{'model':10} {'macro-F1':>9} {'claim F1':>9} {'premise F1':>11} {'neither F1':>11}   seeds")
for m in D:
    r = seedmean(m); print(f"{m:10} {r['macro']:9.3f} {r['C']:9.3f} {r['P']:11.3f} {r['N']:11.3f}   {D[m][2].shape[1]}")
if a.ref in D:
    rng = np.random.default_rng(0)
    for m in D:
        if m == a.ref: continue
        ka, kb = D[a.ref][0], D[m][0]
        assert ka == kb, "models were evaluated on different segments"
        e = np.array(D[a.ref][3]); ue = np.unique(e); idx = {u: np.where(e == u)[0] for u in ue}
        obs = {k: seedmean(m)[k] - seedmean(a.ref)[k] for k in ("macro", "C", "P")}
        bs = {k: [] for k in obs}
        for _ in range(a.boot):
            sel = np.concatenate([idx[u] for u in rng.choice(ue, len(ue))])
            ra, rb = seedmean(a.ref, sel), seedmean(m, sel)
            for k in obs: bs[k].append(rb[k] - ra[k])
        print(f"\n{m} - {a.ref} (paired bootstrap over {len(ue)} excerpts):")
        for k in obs:
            lo, hi = np.percentile(bs[k], [2.5, 97.5])
            print(f"  {k:6} diff {obs[k]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  {'(CI excludes 0)' if lo > 0 or hi < 0 else '(CI includes 0)'}")
