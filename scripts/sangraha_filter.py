#!/usr/bin/env python3
"""
Filter the raw Sangraha Hindi shards down to politically relevant documents.

Subcommands (run from anywhere; paths are anchored to the project root):
  selftest   sanity-check tokenizer + scoring on built-in sentences (no data needed)
  calibrate  sample real shards, print keep-rates for a grid of thresholds
  sample     write a spot-check file (kept / near-miss / rejected docs) to read by eye
  run        filter shards (chunkable for PBS array jobs, resumable)
  summary    aggregate the per-chunk stats after a run

Score = 100 * (2*strong_hits + weak_hits) / n_tokens   (first MAX_CHARS characters)
Keep a doc if: n_tokens >= min_tokens AND distinct_strong >= min_distinct
               AND score >= min_density
"""
import os, sys, re, json, glob, random, argparse, statistics, hashlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from political_terms import STRONG, WEAK  # noqa: E402

USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE_BASE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")
DEFAULT_RAW = os.path.join(ROOT, "data", "sangraha_hindi_raw")       # symlink -> /storage
DEFAULT_OUT = os.path.join(STORAGE_BASE, "sangraha_hindi_political")
LOGS_DIR = os.path.join(ROOT, "logs")

# Devanagari block minus danda (U+0964) and double danda (U+0965).
TOKEN_RE = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")
MAX_CHARS = 20000


# ----------------------------------------------------------------- scoring
def tokenize(text):
    return TOKEN_RE.findall(text)


def score(text):
    toks = tokenize(text[:MAX_CHARS])
    n = len(toks)
    if n == 0:
        return (0, 0, 0, 0, 0.0)
    strong = weak = 0
    seen = set()
    for t in toks:
        if t in STRONG:
            strong += 1
            seen.add(t)
        elif t in WEAK:
            weak += 1
    density = 100.0 * (2 * strong + weak) / n
    return (n, strong, len(seen), weak, density)


def keep(sc, a):
    n, _strong, distinct, _weak, density = sc
    return n >= a.min_tokens and distinct >= a.min_distinct and density >= a.min_density


# ----------------------------------------------------------------- io helpers
def list_shards(raw_dir):
    shards = sorted(glob.glob(os.path.join(raw_dir, "shard_*.jsonl")))
    if not shards:
        sys.exit(f"No shards found in {raw_dir}. Check the data/sangraha_hindi_raw "
                 f"symlink (ls -la data/) or pass --raw-dir.")
    return shards


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)["text"]
            except (ValueError, KeyError):
                continue


def sample_docs(a, n_shards, per_shard, seed):
    """Random docs from random shards, so we don't only see the start of the crawl."""
    rng = random.Random(seed)
    shards = list_shards(a.raw_dir)
    picked = rng.sample(shards, min(n_shards, len(shards)))
    docs = []
    for sp in picked:
        texts = list(read_texts(sp))
        docs.extend(rng.sample(texts, min(per_shard, len(texts))))
    return docs, len(picked)


# ----------------------------------------------------------------- commands
def cmd_selftest(a):
    assert tokenize("सरकार ने कहा।") == ["सरकार", "ने", "कहा"], "danda not split off"
    assert tokenize("कार्यान्वयन") == ["कार्यान्वयन"], "matras/conjuncts broken"

    political = ("प्रधानमंत्री ने आज लोकसभा में विपक्ष पर हमला बोला। सरकार ने कहा कि विधेयक "
                 "पर चर्चा होगी और चुनाव आयोग ने मतदान की तारीख घोषित की। भाजपा और कांग्रेस "
                 "के सांसदों ने सदन में नारेबाजी की। ") * 4
    cricket = ("भारतीय क्रिकेट टीम ने कल रात खेले गए मैच में शानदार जीत दर्ज की। कप्तान ने "
               "शतक बनाया और गेंदबाजों ने विकेट चटकाए। दर्शकों ने स्टेडियम में खूब तालियां बजाईं। ") * 5
    phone = ("कंपनी ने नया फोन लॉन्च किया जिसमें बड़ी बैटरी और तेज कैमरा है। सरकार ने टैक्स "
             "बढ़ाया है लेकिन कीमत नहीं बदली गई। ग्राहकों को यह फोन अगले हफ्ते से दुकानों में मिलेगा। ") * 5

    results = {}
    for name, text in [("political", political), ("cricket", cricket), ("phone_mentions_govt", phone)]:
        sc = score(text)
        results[name] = keep(sc, a)
        n, strong, distinct, weak, dens = sc
        print(f"{name:22s} tokens={n:4d} strong={strong:3d} distinct={distinct:2d} "
              f"weak={weak:2d} density={dens:6.2f} -> {'KEEP' if results[name] else 'drop'}")

    assert results["political"], "political sample should be kept at current thresholds"
    assert not results["cricket"], "cricket sample should be dropped"
    assert not results["phone_mentions_govt"], "single passing mention of सरकार should be dropped"
    print("\nSELFTEST PASSED")


def cmd_calibrate(a):
    docs, ns = sample_docs(a, a.shards, a.docs_per_shard, a.seed)
    scs = [score(d) for d in docs]
    print(f"Sampled {len(scs)} docs from {ns} shards")
    ns_tok = [s[0] for s in scs]
    print(f"median tokens/doc: {statistics.median(ns_tok):.0f}   "
          f"docs >= {a.min_tokens} tokens: {100*sum(x >= a.min_tokens for x in ns_tok)/len(scs):.1f}%")

    dens_q = sorted(s[4] for s in scs if s[0] >= a.min_tokens and s[2] >= 2)
    if dens_q:
        print("\ndensity percentiles among docs with >=2 distinct strong terms:")
        for p in (0.25, 0.5, 0.75, 0.9):
            print(f"  p{int(p*100)}: {dens_q[min(int(p*len(dens_q)), len(dens_q)-1)]:.2f}")

    print("\nkeep-rate (% of ALL sampled docs) by min_distinct x min_density")
    dens_grid = (1, 2, 3, 5, 8, 12)
    print("distinct\\density " + "".join(f"{d:>8}" for d in dens_grid))
    for md in (1, 2, 3, 4):
        row = []
        for dg in dens_grid:
            k = sum(1 for s in scs if s[0] >= a.min_tokens and s[2] >= md and s[4] >= dg)
            row.append(f"{100*k/len(scs):7.2f}%")
        print(f"{md:>16} " + "".join(row))
    print("\nNext: python scripts/sangraha_filter.py sample --min-distinct X --min-density Y")


def cmd_sample(a):
    docs, ns = sample_docs(a, a.shards, a.docs_per_shard, a.seed + 1)
    rows = [(d, score(d)) for d in docs]
    kept = [r for r in rows if keep(r[1], a)]
    near = [r for r in rows if not keep(r[1], a) and r[1][0] >= a.min_tokens
            and r[1][2] >= max(1, a.min_distinct - 1) and r[1][4] >= 0.5 * a.min_density]
    rest = [r for r in rows if not keep(r[1], a)]
    rng = random.Random(a.seed)
    groups = [("KEPT", rng.sample(kept, min(30, len(kept)))),
              ("NEAR-MISS (just rejected)", rng.sample(near, min(15, len(near)))),
              ("RANDOM REJECTED", rng.sample(rest, min(10, len(rest))))]

    os.makedirs(LOGS_DIR, exist_ok=True)
    out = os.path.join(LOGS_DIR, "sangraha_spotcheck.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"sampled {len(rows)} docs from {ns} shards; kept {len(kept)} "
                f"({100*len(kept)/len(rows):.2f}%)  thresholds: min_tokens={a.min_tokens} "
                f"min_distinct={a.min_distinct} min_density={a.min_density}\n\n")
        for title, grp in groups:
            f.write(f"{'='*20} {title} {'='*20}\n")
            for i, (d, (n, strong, distinct, weak, dens)) in enumerate(grp):
                f.write(f"--- [{title.split()[0]} {i}] tokens={n} strong={strong} "
                        f"distinct={distinct} weak={weak} density={dens:.2f}\n")
                f.write(d[:500].replace("\n", " ") + "\n\n")
    print(f"Wrote {out}  (kept {len(kept)}/{len(rows)} = {100*len(kept)/len(rows):.2f}%)")
    print("Read it. Ask: are KEPT docs really political? are NEAR-MISS docs ones you'd want?")


def cmd_run(a):
    shards = list_shards(a.raw_dir)[a.chunk_id::a.num_chunks]
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    stats_path = os.path.join(LOGS_DIR, f"sangraha_filter_stats_chunk{a.chunk_id}.tsv")
    print(f"chunk {a.chunk_id}/{a.num_chunks}: {len(shards)} shards -> {a.out_dir}", flush=True)

    for sp in shards:
        name = os.path.basename(sp)
        final = os.path.join(a.out_dir, name)
        if os.path.exists(final):
            print(f"skip {name} (already done)", flush=True)
            continue
        tmp = final + ".tmp"
        seen = kept = kept_bytes = 0
        with open(tmp, "w", encoding="utf-8") as fo:
            for text in read_texts(sp):
                seen += 1
                sc = score(text)
                if keep(sc, a):
                    rec = {"text": text, "source": "sangraha_hindi", "n_tokens": sc[0],
                           "distinct_strong": sc[2], "political_score": round(sc[4], 2),
                           "pib_like": text.lstrip().startswith("Posted On"),
                           "text_md5": hashlib.md5(" ".join(text.split()).encode("utf-8")).hexdigest()}
                    line = json.dumps(rec, ensure_ascii=False)
                    fo.write(line + "\n")
                    kept += 1
                    kept_bytes += len(text.encode("utf-8"))
        os.replace(tmp, final)  # atomic: a half-written shard never looks "done"
        with open(stats_path, "a") as fs:
            fs.write(f"{name}\t{seen}\t{kept}\t{kept_bytes}\n")
        print(f"{name}: seen={seen} kept={kept} ({100*kept/max(seen,1):.2f}%)", flush=True)


def cmd_summary(a):
    total_shards = len(list_shards(a.raw_dir))
    done, seen, kept, kbytes = set(), 0, 0, 0
    for p in glob.glob(os.path.join(LOGS_DIR, "sangraha_filter_stats_chunk*.tsv")):
        with open(p) as f:
            for line in f:
                name, s, k, b = line.rstrip("\n").split("\t")
                if name in done:
                    continue
                done.add(name)
                seen += int(s); kept += int(k); kbytes += int(b)
    print(f"shards done: {len(done)}/{total_shards}")
    print(f"docs seen:   {seen:,}")
    print(f"docs kept:   {kept:,} ({100*kept/max(seen,1):.2f}%)")
    print(f"kept text:   {kbytes/1e9:.2f} GB")


# ----------------------------------------------------------------- cli
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--raw-dir", default=DEFAULT_RAW)
    common.add_argument("--out-dir", default=DEFAULT_OUT)
    common.add_argument("--min-tokens", type=int, default=80)
    common.add_argument("--min-distinct", type=int, default=2,
                        help="min DISTINCT strong terms in the doc")
    common.add_argument("--min-density", type=float, default=3.0,
                        help="weighted political hits per 100 tokens (PLACEHOLDER: calibrate!)")
    common.add_argument("--shards", type=int, default=15, help="shards to sample")
    common.add_argument("--docs-per-shard", type=int, default=500)
    common.add_argument("--seed", type=int, default=42)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("selftest", cmd_selftest), ("calibrate", cmd_calibrate),
                     ("sample", cmd_sample), ("summary", cmd_summary)]:
        sub.add_parser(name, parents=[common]).set_defaults(fn=fn)
    r = sub.add_parser("run", parents=[common])
    r.add_argument("--chunk-id", type=int, default=0)
    r.add_argument("--num-chunks", type=int, default=1)
    r.set_defaults(fn=cmd_run)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
