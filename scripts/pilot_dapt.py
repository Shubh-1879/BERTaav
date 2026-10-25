#!/usr/bin/env python3
"""
BERTaav pilot DAPT (small, honest, preliminary).

Two steps, because compute nodes may have no internet:

  1) prep  (login node, needs internet + HF login):  python scripts/pilot_dapt.py prep
       Streams samples from the HF dataset into local jsonl files under
       /storage/<user>/pilot_dapt/ :
         lok_train / lok_val          (Lok Sabha Q&A, config 'default')
         sang_train / sang_val        (Sangraha political, tier=core only)
         generic_val                  (random RAW Sangraha docs with ZERO strong political terms)
       It also pre-downloads the base model into the HF cache.

  2) run   (GPU or CPU node):  python scripts/pilot_dapt.py run --steps 600
       a. measures masked-LM loss of the base model on the three validation sets
       b. continues pre-training on a lok+sangraha mix
       c. re-measures; writes results.json, loss_curve.csv, fillmask.txt

Domain-gap only (no training; works on CPU in minutes):  python scripts/pilot_dapt.py run --steps 0
"""
import os, sys, json, glob, random, time, argparse, csv

USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")
OUT = os.path.join(STORAGE, "pilot_dapt")
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(STORAGE, "hf_cache", "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(STORAGE, "hf_cache", "models"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(STORAGE, "hf_cache", "hub"))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

REPO = "Shubh1198/BERTaav"
MODEL = "ai4bharat/IndicBERTv2-MLM-only"


def pick_text(row):
    if isinstance(row.get("text"), str):
        return row["text"]
    best = ""
    for v in row.values():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best


def dump(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"  wrote {len(texts):,} -> {path}", flush=True)


def stream_texts(config, split, n, keep=None, min_chars=300, skip_every=1):
    from datasets import load_dataset
    ds = load_dataset(REPO, config, split=split, streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i % skip_every:
            continue
        if keep and not keep(row):
            continue
        t = pick_text(row)
        if len(t) >= min_chars:
            out.append(t[:6000])
        if len(out) >= n:
            break
    return out


def prep(a):
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(0)
    print("Lok Sabha ...", flush=True)
    dump(os.path.join(OUT, "lok_train.jsonl"), stream_texts("default", "train", a.n_train))
    dump(os.path.join(OUT, "lok_val.jsonl"), stream_texts("default", "validation", a.n_val))
    print("Sangraha political (core tier) ...", flush=True)
    core = lambda r: r.get("tier") == "core"
    dump(os.path.join(OUT, "sang_train.jsonl"), stream_texts("sangraha_political", "train", a.n_train, keep=core, skip_every=7))
    dump(os.path.join(OUT, "sang_val.jsonl"), stream_texts("sangraha_political", "validation", a.n_val, keep=core))
    print("Generic Sangraha (zero strong political terms) ...", flush=True)
    from sangraha_filter import score
    shards = sorted(glob.glob(os.path.join(ROOT, "data", "sangraha_hindi_raw", "shard_*.jsonl")))
    rng.shuffle(shards)
    gen = []
    for sp in shards[:8]:
        with open(sp, encoding="utf-8") as f:
            for line in f:
                if rng.random() > 0.02:
                    continue
                try:
                    t = json.loads(line)["text"]
                except Exception:
                    continue
                sc = score(t)
                if sc[2] == 0 and len(t) >= 300:
                    gen.append(t[:6000])
                if len(gen) >= a.n_val:
                    break
        if len(gen) >= a.n_val:
            break
    dump(os.path.join(OUT, "generic_val.jsonl"), gen)
    print("Downloading base model ...", flush=True)
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    AutoTokenizer.from_pretrained(a.model)
    AutoModelForMaskedLM.from_pretrained(a.model)
    print("prep done.")


def prep_generic(a):
    """Training pool of generic Sangraha text (no strong political terms) for the control run.
    Uses shards the validation set did NOT use (prep() took the first of the same shuffled order)."""
    os.makedirs(OUT, exist_ok=True)
    from sangraha_filter import score
    rng = random.Random(0)
    shards = sorted(glob.glob(os.path.join(ROOT, "data", "sangraha_hindi_raw", "shard_*.jsonl")))
    rng.shuffle(shards)
    gen = []
    for sp in shards[8:40]:
        with open(sp, encoding="utf-8") as f:
            for line in f:
                if rng.random() > 0.25:
                    continue
                try:
                    t = json.loads(line)["text"]
                except Exception:
                    continue
                if len(t) >= 300 and score(t)[2] == 0:
                    gen.append(t[:6000])
                if len(gen) >= a.n_train:
                    break
        if len(gen) >= a.n_train:
            break
    dump(os.path.join(OUT, "generic_train.jsonl"), gen)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l)["text"] for l in f]


def chunk(tok, texts, L, max_chunks):
    """Tokenise and cut into fixed length blocks (no special tokens inside, added at the end)."""
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id)
    body = L - 2
    blocks = [ids[i:i + body] for i in range(0, len(ids) - body + 1, body)]
    blocks = [[tok.cls_token_id] + b + [tok.sep_token_id] for b in blocks]
    return blocks[:max_chunks]


def mask_batch(tok, batch, gen, p=0.15):
    import torch
    x = torch.tensor(batch)
    labels = x.clone()
    special = (x == tok.cls_token_id) | (x == tok.sep_token_id) | (x == tok.pad_token_id)
    prob = torch.full(x.shape, p)
    prob[special] = 0
    m = torch.bernoulli(prob, generator=gen).bool()
    labels[~m] = -100
    r = torch.rand(x.shape, generator=gen)
    x = x.clone()
    x[m & (r < 0.8)] = tok.mask_token_id
    rnd = m & (r >= 0.8) & (r < 0.9)
    x[rnd] = torch.randint(len(tok), x.shape, generator=gen)[rnd]
    return x, labels


def eval_loss(model, tok, blocks, dev, bs, seed=123):
    import torch
    model.eval()
    gen = torch.Generator().manual_seed(seed)            # same masks every time -> fair before/after
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(blocks), bs):
            x, y = mask_batch(tok, blocks[i:i + bs], gen)
            out = model(input_ids=x.to(dev), attention_mask=torch.ones_like(x).to(dev), labels=y.to(dev))
            k = (y != -100).sum().item()
            tot += out.loss.item() * k
            n += k
    model.train()
    return tot / max(n, 1)


FILL = ["लोकसभा में प्रधानमंत्री ने विपक्ष पर {m} साधा।",
        "सरकार ने संसद में यह {m} रखा कि बिल पारित किया जाए।",
        "चुनाव आयोग ने विधानसभा चुनाव की {m} की घोषणा की।",
        "मंत्री महोदय बताएँगे कि इस संबंध में सरकार ने क्या {m} उठाए हैं?",
        # harder, parliament-specific probes
        "माननीय अध्यक्ष महोदय, मैं इस गंभीर विषय की ओर सरकार का ध्यान {m} करना चाहता हूँ।",
        "यह एक {m} प्रश्न है, जिसका उत्तर सदन की मेज पर रखा जाएगा।",
        "विपक्ष ने सदन में सरकार के विरुद्ध अविश्वास {m} पेश किया।",
        "सदस्यों ने अध्यक्ष से अनुरोध किया कि इस मुद्दे पर चर्चा के लिए {m} स्थगन किया जाए।"]


def fillmask(model, tok, dev, tag, f):
    import torch
    model.eval()
    f.write(f"\n=== {tag} ===\n")
    for s in FILL:
        text = s.replace("{m}", tok.mask_token)
        enc = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = model(**enc).logits
        pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0].item()
        top = lg[0, pos].topk(5).indices.tolist()
        f.write(text + "\n   -> " + " | ".join(tok.decode([t]).strip() for t in top) + "\n")
    model.train()


def run(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  model={a.model}  steps={a.steps}", flush=True)
    if dev == "cpu":
        torch.set_num_threads(int(os.environ.get("NCPUS", "8")))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model).to(dev)
    L = a.max_len
    vals = {k: chunk(tok, read_jsonl(os.path.join(OUT, f"{k}_val.jsonl")), L, a.n_eval)
            for k in ("lok", "sang", "generic")}
    print({k: len(v) for k, v in vals.items()}, "eval blocks", flush=True)
    res = {"model": a.model, "device": dev, "max_len": L, "steps": a.steps, "eval_blocks": {k: len(v) for k, v in vals.items()}}
    odir = os.path.join(OUT, "runs", a.tag) if a.tag else OUT
    os.makedirs(odir, exist_ok=True)
    res["sources"] = a.sources; res["tag"] = a.tag
    fm = open(os.path.join(odir, "fillmask.txt"), "w", encoding="utf-8")
    curve = []

    def evaluate(tag, step):
        r = {k: eval_loss(model, tok, v, dev, a.eval_bs) for k, v in vals.items()}
        import math
        print(f"[{tag}] step {step}  " + "  ".join(f"{k}: loss {v:.3f} (ppl {math.exp(v):.1f})" for k, v in r.items()), flush=True)
        curve.append({"step": step, **r})
        return r

    res["before"] = evaluate("before", 0)
    fillmask(model, tok, dev, "base model", fm)

    if a.steps > 0:
        train_texts = []
        for s in a.sources.split(","):
            train_texts += read_jsonl(os.path.join(OUT, f"{s.strip()}_train.jsonl"))
        random.Random(1).shuffle(train_texts)
        need = a.steps * a.bs
        blocks = chunk(tok, train_texts, L, need)
        print(f"sources={a.sources} train blocks: {len(blocks):,} (need {need:,})", flush=True)
        if len(blocks) < need:
            print("WARNING: fewer blocks than steps*bs; data will be reused (wraps around)", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * a.steps))) * max(0.0, 1 - s / a.steps))
        gen = torch.Generator().manual_seed(7)
        model.train()
        t0 = time.time()
        for step in range(1, a.steps + 1):
            i = ((step - 1) * a.bs) % max(1, len(blocks) - a.bs)
            x, y = mask_batch(tok, blocks[i:i + a.bs], gen)
            out = model(input_ids=x.to(dev), attention_mask=torch.ones_like(x).to(dev), labels=y.to(dev))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 20 == 0:
                print(f"step {step}/{a.steps} train_loss {out.loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
            if step % a.eval_every == 0 and step != a.steps:
                evaluate("mid", step)
        res["after"] = evaluate("after", a.steps)
        fillmask(model, tok, dev, f"after {a.steps} steps", fm)
        res["train_seconds"] = round(time.time() - t0)
        mdir = os.path.join(odir, f"model_dapt_{a.steps}steps")
        model.save_pretrained(mdir); tok.save_pretrained(mdir)
        print("saved adapted model to", mdir, flush=True)
    fm.close()
    with open(os.path.join(odir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(odir, "loss_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lok", "sang", "generic"]); w.writeheader(); w.writerows(curve)
    print("wrote results.json, loss_curve.csv, fillmask.txt in", odir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-train", type=int, default=30000)
    p.add_argument("--n-val", type=int, default=600)
    g = sp.add_parser("prep-generic")
    g.add_argument("--n-train", type=int, default=30000)
    r = sp.add_parser("run")
    r.add_argument("--sources", default="lok,sang", help="comma list of: lok, sang, generic")
    r.add_argument("--tag", default="", help="writes to <OUT>/runs/<tag>/ instead of <OUT>/")
    r.add_argument("--model", default=MODEL)
    r.add_argument("--steps", type=int, default=600)
    r.add_argument("--bs", type=int, default=16)
    r.add_argument("--max-len", type=int, default=256)
    r.add_argument("--lr", type=float, default=5e-5)
    r.add_argument("--n-eval", type=int, default=200)
    r.add_argument("--eval-bs", type=int, default=16)
    r.add_argument("--eval-every", type=int, default=200)
    a = ap.parse_args()
    {"prep": prep, "prep-generic": prep_generic, "run": run}[a.cmd](a)
