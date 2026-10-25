from huggingface_hub import HfApi
api = HfApi(); repo = "Shubh1198/BERTaav"
d = "/storage/shubham.agarwal_phd24/pib_clean"
for src, dst in [("by_lang/pib_hi.jsonl", "pib/pib_hi.jsonl"),
                 ("by_lang/pib_mr.jsonl", "pib/pib_mr.jsonl"),
                 ("pib_dedup_stats.json", "pib/pib_dedup_stats.json"),
                 ("by_lang/pib_lang_split.json", "pib/pib_lang_split.json")]:
    api.upload_file(path_or_fileobj=f"{d}/{src}", path_in_repo=dst, repo_id=repo,
                    repo_type="dataset", commit_message=f"Add {dst}")
    print("done", dst, flush=True)
