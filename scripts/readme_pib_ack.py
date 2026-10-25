from huggingface_hub import HfApi, hf_hub_download
repo = "Shubh1198/BERTaav"
p = hf_hub_download(repo, "README.md", repo_type="dataset", force_download=True)
t = open(p, encoding="utf-8").read()
assert "Source and attribution" not in t, "already added"
assert "These releases were **not** used for DAPT." in t, "PIB section not found"
t = t.rstrip() + """

### Source and attribution

The `pib_hi` and `pib_mr` configs contain press releases published by the Press Information Bureau, Government of India (https://pib.gov.in). PIB's copyright policy permits reproduction of its material free of charge, without prior approval, provided it is reproduced accurately, not in a misleading context, and with the source acknowledged. The policy does not extend to third-party material. The text here has been cleaned and deduplicated (boilerplate and repeated page text removed), and the original wording of each release is otherwise unchanged. The underlying texts of all components of this dataset remain subject to the terms of their original publishers; the licence declared in this repository's metadata should be read with that in mind.
"""
HfApi().upload_file(path_or_fileobj=t.encode("utf-8"), path_in_repo="README.md", repo_id=repo,
                    repo_type="dataset", commit_message="Add PIB source and attribution")
print("README updated")
