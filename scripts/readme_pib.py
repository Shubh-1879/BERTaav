from huggingface_hub import HfApi, hf_hub_download
repo = "Shubh1198/BERTaav"
p = hf_hub_download(repo, "README.md", repo_type="dataset", force_download=True)
t = open(p, encoding="utf-8").read()
assert "pib_hi" not in t, "PIB already in README"
cfg = ("- config_name: pib_hi\n  data_files:\n  - split: train\n    path: pib/pib_hi.jsonl\n"
       "- config_name: pib_mr\n  data_files:\n  - split: train\n    path: pib/pib_mr.jsonl\n")
assert t.count("\ndataset_info:\n") == 1
t = t.replace("\ndataset_info:\n", "\n" + cfg + "dataset_info:\n", 1)
t = t.rstrip() + """

## PIB press releases (`pib_hi`, `pib_mr`)

Press Information Bureau releases, cleaned and deduplicated. Fields: `id`, `title`, `date`, `ministry`, `prid`, `url`, `text`.

- Source: 176,278 scraped files. Repeated page text and PIB boilerplate lines were removed (about half of the raw characters).
- Dropped: 170 too-short files, 10,039 files with mostly non-Devanagari text, 768 exact duplicates, and 5,756 near-duplicates (MinHash on 5-word shingles, estimated Jaccard >= 0.8).
- Kept: 159,545 releases, split by a marker-word heuristic into `pib_hi` (106,378 Hindi), `pib_mr` (51,499 Marathi) and 1,668 unclassified releases (not published). The source `language` flag labels Marathi releases as Hindi, so it was not used.
- The language split is heuristic, not hand-verified. These releases were **not** used for DAPT.
"""
HfApi().upload_file(path_or_fileobj=t.encode("utf-8"), path_in_repo="README.md", repo_id=repo,
                    repo_type="dataset", commit_message="Add PIB configs and description")
print("README updated")
