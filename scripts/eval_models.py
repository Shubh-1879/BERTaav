#!/usr/bin/env python3
"""
Evaluate several models on the SAME fixed-mask validation sets + fill-mask probes (one GPU job).

  python scripts/eval_models.py --out /storage/$USER/dapt_full/eval \
      base=ai4bharat/IndicBERTv2-MLM-only \
      dapt=/storage/$USER/models/bertaav_dapt_300M \
      generic=/storage/$USER/dapt_full/runs/generic/final_model     # add later

Writes <out>/eval_results.json, <out>/eval_table.md, <out>/fillmask_all.txt.
Masks are seeded (eval_loss seed=123), identical to the training-time evaluation.
"""
import os, sys, json, math, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SCRIPT_DIR)
from pilot_dapt import chunk, read_jsonl, eval_loss, fillmask
USER = os.environ.get("USER", "shubham.agarwal_phd24")
PILOT = f"/storage/{USER}/pilot_dapt"
L = 256
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--nchunks", type=int, default=200)
ap.add_argument("models", nargs="+", help="name=path_or_hf_id")
a = ap.parse_args()
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
res, fm = {}, open(os.path.join(a.out, "fillmask_all.txt"), "w", encoding="utf-8")
for spec in a.models:
    name, path = spec.split("=", 1)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path).to(dev)
    vals = {k: chunk(tok, read_jsonl(os.path.join(PILOT, f"{k}_val.jsonl")), L, a.nchunks) for k in ("lok", "sang", "generic")}
    res[name] = {k: eval_loss(model, tok, v, dev, 16) for k, v in vals.items()}
    res[name]["n_blocks"] = {k: len(v) for k, v in vals.items()}
    print(name, res[name], flush=True)
    fillmask(model, tok, dev, name, fm)
    del model; torch.cuda.empty_cache()
fm.close()
json.dump(res, open(os.path.join(a.out, "eval_results.json"), "w"), indent=2)
with open(os.path.join(a.out, "eval_table.md"), "w") as f:
    f.write("| model | Lok Sabha | Sangraha political | Generic |\n|---|---|---|---|\n")
    for n, r in res.items():
        f.write(f"| {n} | " + " | ".join(f"{r[k]:.3f} (ppl {math.exp(r[k]):.1f})" for k in ("lok", "sang", "generic")) + " |\n")
print(open(os.path.join(a.out, "eval_table.md")).read())
