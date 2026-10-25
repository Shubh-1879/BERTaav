#!/usr/bin/env python3
"""
OCR control: does the DAPT gain on Lok Sabha text survive on CLEAN-OCR documents?

Stage 1  (login node, needs internet, CPU):
    python scripts/ocr_control.py prep
  Streams the Lok Sabha VALIDATION split, groups documents by Tesseract mean confidence, and writes
  <OUT>/{clean,noisy,random}.jsonl  (clean: conf >= --hi, noisy: conf <= --lo, random: any, seeded sample).
Stage 2  (GPU job, offline):
    python scripts/ocr_control.py eval base=... dapt=... generic=...
  Evaluates each model on each group with the SAME fixed masks, writes <OUT>/ocr_control.md / .json.
"""
import os, sys, json, math, random, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SCRIPT_DIR)
USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")
OUT = os.path.join(STORAGE, "ocr_control")
L = 256

def group_rows(rows, hi, lo, n_per, seed=0, min_chars=300):
    """rows: iterable of (conf, text). Returns dict group -> list of texts (<= n_per each)."""
    rows = [(c, t) for c, t in rows if c is not None and len(t) >= min_chars]
    rng = random.Random(seed); rng.shuffle(rows)
    g = {"clean": [], "noisy": [], "random": []}
    for c, t in rows:
        if c >= hi and len(g["clean"]) < n_per: g["clean"].append(t[:6000])
        if c <= lo and len(g["noisy"]) < n_per: g["noisy"].append(t[:6000])
        if len(g["random"]) < n_per: g["random"].append(t[:6000])
    return g

def prep(a):
    from datasets import load_dataset
    ds = load_dataset("Shubh1198/BERTaav", "default", split="validation", streaming=True)
    rows = []
    for r in ds:
        try: c = float(r.get("ocr_confidence_mean"))
        except (TypeError, ValueError): c = None
        t = r.get("text") or ""
        rows.append((c, t))
    print(len(rows), "validation documents")
    confs = sorted(c for c, _ in rows if c is not None)
    if confs:
        n = len(confs); print("confidence percentiles 10/25/50/75/90:", [confs[int(n*q)] for q in (.1, .25, .5, .75, .9)])
    g = group_rows(rows, a.hi, a.lo, a.n_docs)
    os.makedirs(OUT, exist_ok=True)
    for k, texts in g.items():
        with open(os.path.join(OUT, f"{k}.jsonl"), "w", encoding="utf-8") as f:
            for t in texts: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print(f"{k}: {len(texts)} documents")

def evaluate(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from pilot_dapt import chunk, read_jsonl, eval_loss
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path); model = AutoModelForMaskedLM.from_pretrained(path).to(dev)
        res[name] = {}
        for k in ("clean", "noisy", "random"):
            blocks = chunk(tok, read_jsonl(os.path.join(OUT, f"{k}.jsonl")), L, a.nchunks)
            res[name][k] = {"loss": eval_loss(model, tok, blocks, dev, 16), "blocks": len(blocks)}
        print(name, {k: round(v["loss"], 3) for k, v in res[name].items()}, flush=True)
        del model
    json.dump(res, open(os.path.join(OUT, "ocr_control.json"), "w"), indent=2)
    with open(os.path.join(OUT, "ocr_control.md"), "w") as f:
        f.write("| model | clean OCR | noisy OCR | random |\n|---|---|---|---|\n")
        for n, r in res.items():
            f.write(f"| {n} | " + " | ".join(f"{r[k]['loss']:.3f} (ppl {math.exp(r[k]['loss']):.1f})" for k in ("clean", "noisy", "random")) + " |\n")
        if "base" in res:
            f.write("\nGain over base (loss decrease; larger = better):\n\n| model | clean | noisy | random | clean gain / random gain |\n|---|---|---|---|---|\n")
            for n, r in res.items():
                if n == "base": continue
                gs = {k: res["base"][k]["loss"] - r[k]["loss"] for k in ("clean", "noisy", "random")}
                ratio = f"{gs['clean']/gs['random']:.2f}" if abs(gs['random']) > 1e-9 else "n/a"
                f.write(f"| {n} | {gs['clean']:.3f} | {gs['noisy']:.3f} | {gs['random']:.3f} | {ratio} |\n")
        f.write("\nblocks per group: " + ", ".join(f"{k}={next(iter(res.values()))[k]['blocks']}" for k in ("clean", "noisy", "random")) + "\n")
    print(open(os.path.join(OUT, "ocr_control.md")).read())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep"); p.add_argument("--hi", type=float, default=90.5); p.add_argument("--lo", type=float, default=84.15)
    p.add_argument("--n-docs", type=int, default=600)
    e = sp.add_parser("eval"); e.add_argument("--nchunks", type=int, default=300); e.add_argument("models", nargs="+")
    a = ap.parse_args(); {"prep": prep, "eval": evaluate}[a.cmd](a)
