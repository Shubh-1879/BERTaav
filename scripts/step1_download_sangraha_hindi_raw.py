# step1_download_sangraha_hindi_raw.py
from datasets import load_dataset
import json, os

OUT_DIR = "data/sangraha_hindi_raw"
os.makedirs(OUT_DIR, exist_ok=True)

ds = load_dataset("ai4bharat/sangraha", data_dir="verified/hin", split="train", streaming=True)

shard_idx, buffer, shard_size = 0, [], 20000
total_docs, total_bytes = 0, 0

for row in ds:
    buffer.append({"text": row["text"]})
    total_docs += 1
    total_bytes += len(row["text"].encode("utf-8"))
    if len(buffer) >= shard_size:
        with open(f"{OUT_DIR}/shard_{shard_idx:05d}.jsonl", "w", encoding="utf-8") as f:
            for r in buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"shard {shard_idx}: docs={total_docs} size={total_bytes/1e9:.2f}GB")
        shard_idx += 1
        buffer = []

if buffer:
    with open(f"{OUT_DIR}/shard_{shard_idx:05d}.jsonl", "w", encoding="utf-8") as f:
        for r in buffer:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"DONE: {total_docs} docs, {total_bytes/1e9:.2f} GB total")
