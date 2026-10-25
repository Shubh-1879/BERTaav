#!/usr/bin/env python3
"""
BERTaav DAPT, full-scale trainer (continued masked-LM pre-training of IndicBERTv2-MLM-only).

Built for a shared GPU, a walltime limit and multi-day runs:
  * fp16 autocast (V100 tensor cores) with a GradScaler
  * vocabulary logits computed ONLY at masked positions (about 7x less memory/compute than the
    default Hugging Face loss, which scores all 256 positions against a 250k vocabulary)
  * gradient accumulation (effective batch = mbs * accum blocks of 256 tokens)
  * checkpoint + automatic resume (chain several jobs with qsub -W depend=afterany: each one
    resumes; once finished, later jobs exit immediately)
  * stops cleanly before the walltime (--max-hours) after saving a checkpoint
  * data = pre-tokenised .npy streams from pretokenize_dapt.py; Lok Sabha is up-weighted (--p-lok)
  * evaluation on the SAME three validation sets / masks as the pilot, so numbers are comparable

Benchmark first (no data needed, ~2 minutes):   python scripts/dapt_full.py --bench 30
Quick real-data smoke test:                      python scripts/dapt_full.py --tag smoke --max-steps 20 --eval-every 10 --ckpt-every 10
Real run:                                        python scripts/dapt_full.py --tag full --total-tokens 300000000
"""
import os, sys, json, glob, time, math, argparse, shutil
import numpy as np

USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(STORAGE, "hf_cache", "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(STORAGE, "hf_cache", "models"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from pilot_dapt import chunk, read_jsonl, eval_loss, fillmask   # same eval + probes as the pilot

PILOT = os.path.join(STORAGE, "pilot_dapt")
ROOT_OUT = os.path.join(STORAGE, "dapt_full")
MODEL = "ai4bharat/IndicBERTv2-MLM-only"
L = 256
BODY = L - 2


def load_source(prefix):
    files = sorted(glob.glob(os.path.join(ROOT_OUT, "tokens", f"{prefix}_*.npy")))
    arrs = [np.load(f, mmap_mode="r") for f in files]
    arrs = [a for a in arrs if len(a) > 4 * BODY]
    sizes = np.array([len(a) for a in arrs], dtype=np.float64)
    return arrs, (sizes / sizes.sum() if len(arrs) else sizes), int(sizes.sum())


def sample_blocks(src, rng, n):
    arrs, p, _ = src
    out = np.empty((n, BODY), dtype=np.int64)
    for i in range(n):
        a = arrs[rng.choice(len(arrs), p=p)]
        off = int(rng.integers(0, len(a) - BODY))
        out[i] = a[off:off + BODY]
    return out


def make_batch(body, tok, seed_int, vocab):
    import torch
    gen = torch.Generator().manual_seed(seed_int)
    x = torch.from_numpy(body).long()
    B = x.shape[0]
    x = torch.cat([torch.full((B, 1), tok.cls_token_id), x, torch.full((B, 1), tok.sep_token_id)], 1)
    labels = x.clone()
    prob = torch.full(x.shape, 0.15)
    prob[:, 0] = 0; prob[:, -1] = 0
    m = torch.bernoulli(prob, generator=gen).bool()
    labels[~m] = -100
    r = torch.rand(x.shape, generator=gen)
    x[m & (r < 0.8)] = tok.mask_token_id
    rnd = m & (r >= 0.8) & (r < 0.9)
    x[rnd] = torch.randint(vocab, x.shape, generator=gen)[rnd]
    return x, labels


def fast_loss(model, x, labels):
    """Masked-LM loss with vocabulary logits computed only at masked positions."""
    import torch, torch.nn.functional as F
    h = model.bert(input_ids=x).last_hidden_state
    m = labels != -100
    logits = model.cls(h[m])
    return F.cross_entropy(logits.float(), labels[m])


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--total-tokens", type=float, default=3e8)
    ap.add_argument("--mbs", type=int, default=32, help="micro-batch (blocks)")
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--p-lok", type=float, default=0.15, help="share of blocks drawn from Lok Sabha")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--max-hours", type=float, default=11.5)
    ap.add_argument("--max-steps", type=int, default=None, help="cap total steps (smoke tests)")
    ap.add_argument("--bench", type=int, default=0, help="benchmark N steps on random tokens and exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sang-prefix", default="sang", help="token-file prefix for the non-Lok stream (use 'gen' for the generic control)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = dev == "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model).to(dev)
    assert hasattr(model, "bert") and hasattr(model, "cls"), "unexpected architecture; fast_loss needs .bert/.cls"
    vocab = len(tok)
    tokens_per_step = a.mbs * a.accum * L
    total_steps = int(math.ceil(a.total_tokens / tokens_per_step))
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    print(f"device={dev} fp16={use_fp16} mbs={a.mbs} accum={a.accum} -> {tokens_per_step:,} tokens/step; "
          f"total_steps={total_steps:,} (~{total_steps*tokens_per_step/1e6:.0f}M tokens)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    def set_lr(step):
        f = min(1.0, (step + 1) / max(1, a.warmup)) * max(0.0, 1.0 - step / max(1, total_steps))
        for g in opt.param_groups:
            g["lr"] = a.lr * f

    def train_step(step, batches):
        set_lr(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for x, y in batches:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                loss = fast_loss(model, x.to(dev), y.to(dev)) / a.accum
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        return tot

    model.train()
    # ---------------- benchmark mode ----------------
    if a.bench:
        rng = np.random.default_rng(0)
        t0 = None
        for s in range(a.bench):
            if s == 3:
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.time()
            batches = [make_batch(rng.integers(5, vocab, (a.mbs, BODY)), tok, s * 100 + i, vocab) for i in range(a.accum)]
            l = train_step(s, batches)
            print(f"bench step {s} loss {l:.2f}", flush=True)
        if dev == "cuda": torch.cuda.synchronize()
        dt = (time.time() - t0) / (a.bench - 3)
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0
        print(f"\nBENCH: {dt:.2f} s/step, {tokens_per_step/dt:,.0f} tokens/s, peak GPU memory {mem:.1f} GiB", flush=True)
        print(f"-> 300M tokens would take about {3e8/(tokens_per_step/dt)/3600:.1f} hours at this speed", flush=True)
        return

    # ---------------- data ----------------
    lok, sang = load_source("lok"), load_source(a.sang_prefix)
    print(f"data: lok {lok[2]:,} tokens in {len(lok[0])} files; {a.sang_prefix} {sang[2]:,} tokens in {len(sang[0])} files", flush=True)
    assert len(sang[0]) and (a.p_lok == 0 or len(lok[0])), "no token files: run pretokenize first"
    if a.sang_prefix != "sang":
        assert sang[2] >= a.total_tokens, f"{a.sang_prefix} stream has only {sang[2]:,} tokens, need >= {a.total_tokens:,.0f} for a matched run"
    vals = {k: chunk(tok, read_jsonl(os.path.join(PILOT, f"{k}_val.jsonl")), L, 200) for k in ("lok", "sang", "generic")}

    odir = os.path.join(ROOT_OUT, "runs", a.tag)
    os.makedirs(odir, exist_ok=True)
    ckpt = os.path.join(odir, "ckpt")
    if os.path.exists(os.path.join(odir, "DONE")):
        print("already finished (DONE file exists); exiting."); return

    start = 0
    if os.path.exists(os.path.join(ckpt, "state.pt")):
        st = torch.load(os.path.join(ckpt, "state.pt"), map_location="cpu")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        start = st["step"]; del st
        print(f"RESUMED from step {start}", flush=True)

    def save_ckpt(step):
        tmp = ckpt + ".tmp"; old = ckpt + ".old"
        shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step},
                   os.path.join(tmp, "state.pt"))
        shutil.rmtree(old, ignore_errors=True)
        if os.path.exists(ckpt): os.rename(ckpt, old)
        os.rename(tmp, ckpt); shutil.rmtree(old, ignore_errors=True)
        print(f"checkpoint saved at step {step}", flush=True)

    def evaluate(step):
        r = {k: eval_loss(model, tok, v, dev, 16) for k, v in vals.items()}
        print(f"[eval] step {step}  " + "  ".join(f"{k}: {v:.3f} (ppl {math.exp(v):.1f})" for k, v in r.items()), flush=True)
        newfile = not os.path.exists(os.path.join(odir, "curve.csv"))
        with open(os.path.join(odir, "curve.csv"), "a") as f:
            if newfile: f.write("step,tokens_seen,lok,sang,generic\n")
            f.write(f"{step},{step*tokens_per_step},{r['lok']:.4f},{r['sang']:.4f},{r['generic']:.4f}\n")
        model.train()
        return r

    if start == 0:
        evaluate(0)
    t_start = time.time(); t_log = time.time(); loss_ema = None
    for step in range(start, total_steps):
        batches = []
        for mi in range(a.accum):
            rng = np.random.default_rng([a.seed, step, mi])
            n_lok = int(rng.binomial(a.mbs, a.p_lok))
            parts = []
            if n_lok: parts.append(sample_blocks(lok, rng, n_lok))
            if a.mbs - n_lok: parts.append(sample_blocks(sang, rng, a.mbs - n_lok))
            batches.append(make_batch(np.concatenate(parts), tok, (a.seed * 1_000_003 + step) * 101 + mi, vocab))
        l = train_step(step, batches)
        loss_ema = l if loss_ema is None else 0.98 * loss_ema + 0.02 * l
        done = step + 1
        if done % 20 == 0:
            el = time.time() - t_log
            print(f"step {done}/{total_steps} loss {l:.3f} (ema {loss_ema:.3f}) lr {opt.param_groups[0]['lr']:.2e} "
                  f"{20*tokens_per_step/el:,.0f} tok/s", flush=True)
            t_log = time.time()
        if done % a.eval_every == 0 and done != total_steps:
            evaluate(done)
        if done % a.ckpt_every == 0 and done != total_steps:
            save_ckpt(done)
        if (time.time() - t_start) > a.max_hours * 3600 and done != total_steps:
            save_ckpt(done); print("time budget reached; exiting so the next chained job can resume", flush=True)
            return

    res = evaluate(total_steps)
    with open(os.path.join(odir, "fillmask.txt"), "w", encoding="utf-8") as f:
        fillmask(model, tok, dev, f"after {total_steps} steps", f)
    mdir = os.path.join(odir, "final_model")
    model.save_pretrained(mdir); tok.save_pretrained(mdir)
    with open(os.path.join(odir, "results.json"), "w") as f:
        json.dump({"steps": total_steps, "tokens": total_steps * tokens_per_step, "final": res, "args": vars(a)}, f, indent=2)
    open(os.path.join(odir, "DONE"), "w").write("ok")
    shutil.rmtree(ckpt, ignore_errors=True)
    print("DONE. final model in", mdir, flush=True)


if __name__ == "__main__":
    main()
