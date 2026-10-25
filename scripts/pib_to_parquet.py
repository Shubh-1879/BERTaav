import json, os
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
repo = "Shubh1198/BERTaav"; api = HfApi()
d = "/storage/shubham.agarwal_phd24/pib_clean"; out = f"{d}/parquet"
schema = pa.schema([("id", pa.string()), ("title", pa.string()), ("date", pa.string()), ("ministry", pa.string()),
                    ("prid", pa.int64()), ("url", pa.string()), ("text", pa.string())])
cols = [f.name for f in schema]
for lang in ("hi", "mr"):
    os.makedirs(f"{out}/pib_{lang}", exist_ok=True)
    dst = f"{out}/pib_{lang}/train-00000-of-00001.parquet"
    w = pq.ParquetWriter(dst, schema, compression="zstd"); buf = []; n = 0
    def flush():
        global buf
        if buf: w.write_table(pa.Table.from_pylist(buf, schema=schema)); buf = []
    for line in open(f"{d}/by_lang/pib_{lang}.jsonl", encoding="utf-8"):
        r = json.loads(line); buf.append({c: r.get(c) for c in cols}); n += 1
        if len(buf) >= 5000: flush()
    flush(); w.close(); print(lang, n, "rows ->", dst, os.path.getsize(dst) // 2**20, "MB", flush=True)
    api.upload_file(path_or_fileobj=dst, path_in_repo=f"pib_{lang}/train-00000-of-00001.parquet", repo_id=repo,
                    repo_type="dataset", commit_message=f"Add pib_{lang} parquet")
    api.delete_file(f"pib/pib_{lang}.jsonl", repo_id=repo, repo_type="dataset", commit_message=f"Remove pib_{lang}.jsonl")
p = hf_hub_download(repo, "README.md", repo_type="dataset", force_download=True)
t = open(p, encoding="utf-8").read()
t = t.replace("path: pib/pib_hi.jsonl", "path: pib_hi/train-*").replace("path: pib/pib_mr.jsonl", "path: pib_mr/train-*")
api.upload_file(path_or_fileobj=t.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                commit_message="PIB configs point to parquet")
print("README fixed")
