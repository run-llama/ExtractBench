# Gemini CLI — Gemini 4 Argon, medium thinking

Submission on the 100-document public release of [ExtractBench Pro](https://huggingface.co/datasets/llamaindex/ExtractBenchPro/tree/4ae1ffe48696297a8f6c9153034b3914fc7763d0).

- **30% perfect · 42.44 Value F1 · 33.90 word grounding F1 · 35.81 page grounding F1.**
- The recorded runs allowed one hour per document. For this submission, all 26 successful outputs completed after 1,800 seconds score zero, alongside the 25 extraction failures. The remaining 49 outputs are included in `predictions.jsonl.gz`.
- Value F1 and perfect-document percentage use all 100 documents; both grounding metrics use the 81 documents with verified boxes. Scores are recomputed with this repository's evaluator at IoU 0.5. No partial output from failed or late attempts is scored.
- Two OOM attempts and six runner-interrupted attempts are excluded; their replacement attempts determine those documents' results. Three outputs were reused only after matching public PDF and schema hashes.

`config.json` and `prompt.txt` record the Gemini CLI 0.59.0 configuration and prompt. The model may use CLI helper models; their reported usage is included in cost. `documents.csv` records every document's scores, elapsed time, cost sample, original workflow/source revision, trace filename, and input hashes. Source workflow IDs identify the original attempt, including carried-forward outputs.

## Cost estimate

The estimated sample costs total **$165.8092313**: 93 original measurements ($158.3381986) plus seven fresh measurements ($7.4710327) replacing documents whose original attempts never reported usage. The replacements change cost only; original quality outcomes remain zero. Unknown failed-request usage is assumed to incur no additional charge. This is not a complete historical billing total: the seven original unmeasured attempts are excluded, as are infrastructure failures and diagnostic runs.

`Cost_Per_Page` follows the leaderboard's document-average convention: `mean(cost_usd / pages)` = **$0.181650/page** overall. For comparison, pooled cost divided by all 2,622 pages is $0.063238/page. Full measured attempt costs are retained, including spend after the 30-minute scoring cutoff; this is not a measured 30-minute spending total. The seven replacement measurements use Bifrost; original quality runs used Google's API directly. The replacement runs ended in five request-size rejections, one CLI error, and one timeout; their reported usage is retained.

## Offline verification

No inference or paid API calls are made. Supply the pinned Hugging Face snapshot (the six JSONL files are sufficient):

```bash
hf download llamaindex/ExtractBenchPro --repo-type dataset \
  --revision 4ae1ffe48696297a8f6c9153034b3914fc7763d0 \
  --include '*.jsonl' --local-dir data/gemini-argon-pro
uv run python submissions/gemini-4-argon/verify.py --snapshot data/gemini-argon-pro
uv run python scripts/update_readme.py --check
```

The verifier checks dataset hashes, replays the 49 predictions against public ground truth, enforces failure/deadline zeroing, and reconciles all seven leaderboard rows and the cost total. It does not validate unreported billing or reproduce model inference.
