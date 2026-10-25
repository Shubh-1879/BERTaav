#!/usr/bin/env python3
"""
Tokenizer fertility on our own Hindi text: average sub-word tokens per whitespace word.
Lower = the tokenizer fits Hindi better (shorter sequences, less fragmenting of words).
Run on the LOGIN node (needs internet to fetch the tokenizers; no GPU needed, takes ~1-2 min):

    python scripts/tokenizer_fertility.py
"""
import os, json
USER = os.environ.get("USER", "shubham.agarwal_phd24")
STORAGE = os.environ.get("BERTAAV_STORAGE", f"/storage/{USER}")
os.environ.setdefault("HF_HUB_CACHE", os.path.join(STORAGE, "hf_cache", "hub"))
D = os.path.join(STORAGE, "pilot_dapt")
MODELS = ["ai4bharat/IndicBERTv2-MLM-only", "google/muril-base-cased",
          "xlm-roberta-base", "bert-base-multilingual-cased"]
SETS = ["lok_val", "sang_val", "generic_val"]

from transformers import AutoTokenizer
texts = {}
for s in SETS:
    with open(os.path.join(D, f"{s}.jsonl"), encoding="utf-8") as f:
        texts[s] = [json.loads(l)["text"] for l in f][:300]
print(f"{'model':36s} " + " ".join(f"{s:>12s}" for s in SETS) + "   vocab")
for m in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(m)
    except Exception as e:
        print(f"{m:36s} FAILED to load ({str(e)[:60]})")
        continue
    row = []
    for s in SETS:
        n_tok = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts[s])
        n_words = sum(len(t.split()) for t in texts[s])
        row.append(n_tok / n_words)
    print(f"{m:36s} " + " ".join(f"{v:12.2f}" for v in row) + f"   {len(tok):,}")
