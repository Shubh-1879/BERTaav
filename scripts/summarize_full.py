import os, sys, json, glob, math, csv
RUN = "/storage/shubham.agarwal_phd24/dapt_full/runs/full"
OUT = "/storage/shubham.agarwal_phd24/dapt_full/summary"
os.makedirs(OUT, exist_ok=True)

# 1. what exists in the run folder
print("== run folder ==")
for root, dirs, files in os.walk(RUN):
    depth = root[len(RUN):].count(os.sep)
    if depth > 1: continue
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"{os.path.getsize(p)/1e6:10.1f} MB  {p}")
print("DONE file:", os.path.exists(os.path.join(RUN, "DONE")))

# 2. curve and comparison with base (base numbers = pilot baseline)
BASE = {"lok": 3.035, "sang": 1.422, "generic": 2.046}
rows = list(csv.DictReader(open(os.path.join(RUN, "curve.csv"))))
last = rows[-1]
best = {k: min(rows, key=lambda r: float(r[k])) for k in BASE}
res = {"final_step": int(last["step"]), "tokens_seen": int(last["tokens_seen"]), "sets": {}}
print("\n== final vs base ==")
print(f"{'set':8} {'base':>7} {'final':>7} {'delta':>7} {'ppl base':>9} {'ppl final':>10} {'best@step':>10}")
for k, b in BASE.items():
    f = float(last[k])
    res["sets"][k] = {"base": b, "final": f, "delta": f - b,
                      "ppl_base": math.exp(b), "ppl_final": math.exp(f),
                      "best_loss": float(best[k][k]), "best_step": int(best[k]["step"])}
    print(f"{k:8} {b:7.3f} {f:7.3f} {f-b:7.3f} {math.exp(b):9.1f} {math.exp(f):10.1f} {best[k]['step']:>10}")
json.dump(res, open(os.path.join(OUT, "full_results.json"), "w"), indent=2)

# 3. slope over the last 1000 steps (is it still improving?)
print("\n== slope over last ~1000 steps (loss per 500 steps) ==")
for k in BASE:
    a, z = float(rows[-3][k]), float(rows[-1][k])
    print(k, round(z - a, 4))

# 4. plot
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    steps = [int(r["step"]) for r in rows]
    names = {"lok": "Lok Sabha", "sang": "Sangraha political", "generic": "Generic Sangraha"}
    for a, k in zip(ax, BASE):
        a.plot(steps, [float(r[k]) for r in rows], marker="o", ms=3)
        a.axhline(BASE[k], ls="--", c="grey", label="base model")
        a.set_title(names[k]); a.set_xlabel("step"); a.set_ylabel("MLM loss"); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "full_curves.png"), dpi=160)
    print("\nsaved plot")
except Exception as e:
    print("plot skipped:", e)

# 5. copy small records (not the model) to the summary folder
import shutil
for pat in ("curve.csv", "*.json", "*.txt", "*.log"):
    for p in glob.glob(os.path.join(RUN, pat)):
        shutil.copy(p, OUT)
print("summary in", OUT)
