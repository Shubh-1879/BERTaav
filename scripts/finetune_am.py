#!/usr/bin/env python3
"""
Downstream argument mining: sentence-level claim / premise / neither (C / P / N).

Input CSV columns: excerpt_id, seg_no, segment, label   (label may carry '*' or '?' flags; they are stripped)
Protocol: 5-fold cross-validation GROUPED by excerpt (no document leaks across folds), several seeds,
          ONE fixed recipe for every model (no per-model tuning, no model selection on the test fold).

  python finetune_am.py --data gold.csv --out runs_am \
      base=ai4bharat/IndicBERTv2-MLM-only dapt=/storage/.../bertaav_dapt_300M generic=/storage/.../bertaav_generic_300M

Writes <out>/preds_<name>.csv (one row per model, seed, segment) and <out>/folds_<name>.csv (metrics per seed x fold).
Resumable: finished (model, seed, fold) combinations are skipped.
"""
import os, sys, csv, json, argparse, random, time
import numpy as np

LAB = {"N": 0, "C": 1, "P": 2}
INV = {v: k for k, v in LAB.items()}

def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    out = []
    for r in rows:
        l = (r.get("label") or "").strip().upper().replace("*", "").replace("?", "")
        if l not in LAB:
            continue
        out.append({"seg_id": f"{r['excerpt_id']}_{int(r['seg_no']):02d}", "ex": r["excerpt_id"], "text": r["segment"], "y": LAB[l]})
    return out

def folds(exs, k, seed=0):
    """Group k-fold with stratification by excerpt-level majority-ish signal: shuffle groups, deal round-robin."""
    ids = sorted(set(exs)); random.Random(seed).shuffle(ids)
    fold_of = {e: i % k for i, e in enumerate(ids)}
    return fold_of

def f1s(y, p):
    res = {}
    for c in (0, 1, 2):
        tp = int(((y == c) & (p == c)).sum()); fp = int(((y != c) & (p == c)).sum()); fn = int(((y == c) & (p != c)).sum())
        res[INV[c]] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    res["macro"] = float(np.mean([res["N"], res["C"], res["P"]]))
    res["acc"] = float((y == p).mean())
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--train-frac", type=float, default=1.0, help="learning-curve runs: fraction of TRAIN excerpts used (test fold unchanged)")
    ap.add_argument("--max-examples", type=int, default=None, help="smoke test")
    ap.add_argument("models", nargs="+", help="name=path_or_hf_id")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(a.data)
    if a.max_examples: data = data[:a.max_examples]
    print(f"{len(data)} labelled segments, {len(set(d['ex'] for d in data))} excerpts; class counts:",
          {INV[c]: sum(d['y'] == c for d in data) for c in (0, 1, 2)}, flush=True)
    fold_of = folds([d["ex"] for d in data], a.k)          # SAME fold split for every model and seed
    os.makedirs(a.out, exist_ok=True)
    for spec in a.models:
        name, path = spec.split("=", 1)
        pfile, ffile = os.path.join(a.out, f"preds_{name}.csv"), os.path.join(a.out, f"folds_{name}.csv")
        done = set()
        if os.path.exists(ffile):
            done = {(int(r["seed"]), int(r["fold"])) for r in csv.DictReader(open(ffile))}
        tok = AutoTokenizer.from_pretrained(path)
        enc = tok([d["text"] for d in data], truncation=True, max_length=a.max_len, padding=False)
        ids = enc["input_ids"]; y_all = np.array([d["y"] for d in data])
        def batch(ix):
            L = max(len(ids[i]) for i in ix)
            x = torch.full((len(ix), L), tok.pad_token_id, dtype=torch.long); m = torch.zeros((len(ix), L), dtype=torch.long)
            for j, i in enumerate(ix):
                x[j, :len(ids[i])] = torch.tensor(ids[i]); m[j, :len(ids[i])] = 1
            return x.to(dev), m.to(dev)
        for seed in range(a.seeds):
            for fold in range(a.k):
                if (seed, fold) in done: continue
                t0 = time.time()
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                tr = [i for i, d in enumerate(data) if fold_of[d["ex"]] != fold]
                te = [i for i, d in enumerate(data) if fold_of[d["ex"]] == fold]
                if a.train_frac < 1.0:      # same subset for every model (depends only on seed and fold)
                    tex = sorted({data[i]["ex"] for i in tr}); keep = set(random.Random(1000 * seed + fold).sample(tex, max(1, round(len(tex) * a.train_frac))))
                    tr = [i for i in tr if data[i]["ex"] in keep]
                model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3).to(dev)
                # class weights from the TRAIN fold only (inverse frequency, normalised)
                cnt = np.bincount(y_all[tr], minlength=3).astype(float) + 1
                w = torch.tensor(cnt.sum() / (3 * cnt), dtype=torch.float, device=dev)
                opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
                steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
                sch = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)
                scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
                g = np.random.default_rng(seed)
                model.train()
                for ep in range(a.epochs):
                    order = g.permutation(tr)
                    for s in range(0, len(order), a.bs):
                        ix = order[s:s + a.bs].tolist(); x, m = batch(ix)
                        yb = torch.tensor(y_all[ix], device=dev)
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                            lg = model(input_ids=x, attention_mask=m).logits
                        loss = torch.nn.functional.cross_entropy(lg.float(), yb, weight=w)
                        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(opt); scaler.update(); sch.step()
                model.eval(); preds = []
                with torch.no_grad():
                    for s in range(0, len(te), 64):
                        ix = te[s:s + 64]; x, m = batch(ix)
                        preds += model(input_ids=x, attention_mask=m).logits.argmax(-1).tolist()
                preds = np.array(preds); r = f1s(y_all[te], preds)       # final epoch only: no selection on the test fold
                new = not os.path.exists(pfile)
                with open(pfile, "a", newline="", encoding="utf-8") as f:
                    wr = csv.writer(f)
                    if new: wr.writerow(["model", "seed", "fold", "seg_id", "ex", "gold", "pred"])
                    for i, p in zip(te, preds): wr.writerow([name, seed, fold, data[i]["seg_id"], data[i]["ex"], INV[y_all[i]], INV[int(p)]])
                new = not os.path.exists(ffile)
                with open(ffile, "a", newline="") as f:
                    wr = csv.writer(f)
                    if new: wr.writerow(["model", "seed", "fold", "macro", "C", "P", "N", "acc"])
                    wr.writerow([name, seed, fold, f"{r['macro']:.4f}", f"{r['C']:.4f}", f"{r['P']:.4f}", f"{r['N']:.4f}", f"{r['acc']:.4f}"])
                print(f"{name} seed {seed} fold {fold}: macro {r['macro']:.3f} C {r['C']:.3f} P {r['P']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                del model, opt; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
