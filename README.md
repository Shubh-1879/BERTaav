# BERTaav

Code for a Hindi political-discourse corpus, domain-adaptive pre-training (DAPT) of IndicBERTv2 on it, and a small Hindi argument-mining benchmark (claim / premise / neither).

Status: work in progress. A written report is being finalised, and parts of this repository will change.

## What this project asks

Does continuing the pre-training of an Indic masked language model on Hindi political text (parliamentary question-and-answer text, a political subset of Sangraha) make it a better starting point for argument mining than the original model, or than a model trained for the same budget on general Hindi text?

The design follows RooseBERT (Dore, Cabrio and Villata, 2025) for English political debate, with a different language, a smaller training budget and a fixed Indic base model (`ai4bharat/IndicBERTv2-MLM-only`).

## Main results so far

Held-out masked-language-model loss (perplexity in brackets), single run unless noted:

| Model | Lok Sabha | Sangraha political | General Hindi |
| --- | --- | --- | --- |
| Base (IndicBERTv2-MLM-only) | 3.035 (20.8) | 1.422 (4.1) | 2.046 (7.7) |
| DAPT, 300M tokens | 1.805 (6.1) | 1.147 (3.1) | 1.776 (5.9) |
| Generic-Hindi control, 300M tokens | 2.750 (15.6) | 1.400 (4.1) | 1.568 (4.8) |

Two more DAPT seeds agree with the first to about 0.002 loss on Lok Sabha. The gain also holds on pages with clean OCR, and a near-duplicate check found no Lok Sabha validation document with a close copy in the training split.

Downstream, sentence-level argument mining (2,000 gold segments, 5-fold CV grouped by excerpt, 5 seeds), macro-F1:

| Model | Macro-F1 |
| --- | --- |
| Base | 0.676 |
| DAPT | 0.677 |
| Generic control | 0.668 |

DAPT minus base is +0.001 (95% CI -0.011 to +0.013), so the adaptation gives no detectable benefit on this task. The report discusses why, and the limits of this evidence (one task, one base model, one annotator for most of the gold set).

## Data and models

Hosted on Hugging Face (currently private; they will be made public once licences are settled):

- Dataset `Shubh1198/BERTaav` with configs `default` (Lok Sabha Q&A text from OCR), `sangraha_political`, `pib_hi` and `pib_mr` (cleaned Press Information Bureau releases, Hindi and Marathi).
- Models `Shubh1198/BERTaav-dapt-300M` and `Shubh1198/BERTaav-generic-300M`.

PIB releases are published by the Press Information Bureau, Government of India (pib.gov.in), and are reproduced under its copyright policy, which requires accurate reproduction and acknowledgement of the source. All underlying texts remain subject to the terms of their original publishers.

The gold-annotated argument segments are not in this repository.

## Repository layout

- `Code/`: earlier collection pipeline (scrapers, PDF-to-image conversion, OCR, structuring and post-processing).
- `scripts/`: corpus filtering, DAPT, evaluation and downstream scripts (below).
- `jobs/`: PBS job scripts used on the cluster, one per run. Paths in them are specific to the author's cluster account and need editing before reuse.

### Scripts

| Script | Purpose |
| --- | --- |
| `sangraha_filter.py`, `political_terms.py` | Select political documents from Sangraha Hindi with a strong/weak term score |
| `dedup_sangraha.py` | Exact deduplication of the political subset |
| `pretokenize_dapt.py`, `pretokenize_generic.py` | Tokenise the DAPT and control streams |
| `pilot_dapt.py`, `dapt_full.py` | Pilot and full continued pre-training |
| `eval_models.py` | Masked-LM loss of several models on the fixed validation sets |
| `ocr_control.py` | Evaluate on clean, noisy and random OCR groups |
| `neardup.py` | PIB cleaning and near-duplicate removal, Hindi/Marathi split, train/validation leakage check |
| `am_sample.py` | Draw the gold-set excerpts from held-out text |
| `finetune_am.py`, `summarize_am.py`, `by_source.py` | Argument-mining fine-tuning, summary with bootstrap intervals, per-source breakdown |
| `filter_eval_sample.py` | Draw the sample used to check the political filter |

## Reproducing the pipeline (outline)

1. Filter and deduplicate Sangraha Hindi (`sangraha_filter.py`, `dedup_sangraha.py`).
2. Tokenise (`pretokenize_dapt.py`, `pretokenize_generic.py`).
3. Train: `dapt_full.py` (the DAPT run uses `--p-lok 0.15`; the control uses `--sang-prefix gen --p-lok 0`).
4. Evaluate: `eval_models.py`, `ocr_control.py`, `neardup.py leak`.
5. Fine-tune and summarise: `finetune_am.py`, then `summarize_am.py --ref base`.

The runs used one NVIDIA V100 (32 GB) under PBS Pro. Settings: 300M tokens, 256-token blocks, 65,536 tokens per step, learning rate 3e-4, 200 warmup steps, linear decay, fp16.

## Use of AI tools

AI assistance was used for writing code, for first-pass labelling of the gold-set segments (the final gold labels are human labels), and for drafting the report.

## Licence and citation

Licence to be decided. A citation entry will be added when the report is released.

## References

- Gururangan et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. ACL.
- Dore, Cabrio and Villata (2025). RooseBERT: A New Deal For Political Language Modelling. arXiv:2508.03250.
- Doddapaneni et al. (2023). Towards Leaving No Indic Language Behind. ACL.
- Khan et al. (2024). IndicLLMSuite. ACL.
