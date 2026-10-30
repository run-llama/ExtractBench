# Using the Harbor datasets

Run coding agents on [ExtractBench](https://huggingface.co/datasets/llamaindex/ExtractBench)
(370 tasks) or [ExtractBench Pro](https://huggingface.co/datasets/llamaindex/ExtractBenchPro)
(100 tasks) using their published Harbor tasks. Each task provides a PDF and a
JSON schema. The agent writes `output.json`; a separate verifier scores it.

## Download and run

Install Harbor 0.24.0 or later, Git with Git LFS, and start Docker. Configure your
model provider's API credentials for Harbor, then run:

```bash
uv tool install harbor
git lfs install

harbor run --repo https://huggingface.co/datasets/llamaindex/ExtractBench -p tasks \
  -a codex -m <model_name> --allow-agent-host <model_api_host>

harbor run --repo https://huggingface.co/datasets/llamaindex/ExtractBenchPro -p tasks \
  -a codex -m <model_name> --allow-agent-host <model_api_host>
```

Harbor downloads the repository's tasks and runs the selected agent/model. Add
`-l 1 -n 1` to run one task before evaluating the full dataset. Model runs use
your provider's API and are billed by that provider.

Replace `<model_name>` with your model identifier and `<model_api_host>` with the model
endpoint's hostname, such as `api.openai.com`. Repeat `--allow-agent-host` only
for additional required model API hosts. During extraction, other network
destinations are blocked, including public dataset hosts. Setup can install the
coding agent before this restriction takes effect. The separate verifier runs
without networking. Harbor rejects backends that cannot enforce the policy.

## Download for local use

To download the data without starting an evaluation:

```bash
git lfs install
git clone https://huggingface.co/datasets/llamaindex/ExtractBench extractbench-harbor
git clone https://huggingface.co/datasets/llamaindex/ExtractBenchPro extractbenchpro-harbor
```

Run the downloaded tasks by replacing `--repo ... -p tasks` with the local path:

```bash
harbor run -p extractbench-harbor/tasks \
  -a codex -m <model_name> --allow-agent-host <model_api_host>
```

For a one-task check without model calls, use the reference solution:

```bash
harbor run -p extractbench-harbor/tasks -a oracle -l 1 -n 1
```

The oracle checks that packaging and grading work; it does not measure extraction
ability. For reproducible evaluations, pin remote repositories with
`--repo https://huggingface.co/datasets/llamaindex/ExtractBench@<commit>` and retain
the Harbor job configuration and dataset's `harbor-manifest.json`.

## Scoring and outputs

Tasks use the shared Claude Code/Codex prompt from
[ExtractBench Appendix C.1](https://arxiv.org/html/2607.29677v1#A3.SS1), with the
staged document filename standardized. The schema is included in full.

The agent image contains only the document, schema, PDF/OCR tools and Python.
`tests/` and `solution/` are outside its Docker build context. The separate
verifier receives only the declared `/app/output.json` artifact; its reference
and grading code are in its own image. The verifier calls
`ExtractEvaluator.compute_metrics`, including its existing normalization,
acceptable-value and row-matching logic.

`/logs/verifier/reward.txt` contains `extract_unified_value_f1` in [0, 1].
`metrics.json` contains the value precision/recall/F1 and perfect-document flag.
Average Value F1 across all cases gives macro Value F1; average the perfect flags
to obtain the perfect-document fraction. Keep failed trials in those denominators.
Malformed, missing or non-object outputs score zero. A reference or evaluator
error fails grading visibly and does not produce a reward.

`feedback.json` attributes that same reward to scored fields and table cells.
It records `reference_path`, `output_path`, `outcome`, and `f1_credit` for each
cell. The outcomes are `matched`, `value_differs`, `reference_only`, and
`prediction_only`. One-sided cells have a null path on the absent side; an
unmatched row produces an entry for each of its scored leaves. Primitive
arrays remain single cells, following the evaluator. Reference values are not
included in this feedback.

For example, when table rows are reordered, a matching cell may be reported as:

```json
{
  "reference_path": "items[0].amount",
  "output_path": "items[3].amount",
  "outcome": "matched",
  "f1_credit": 0.02
}
```

The scorer emits these records during the same assignment that computes Value
F1. Each match receives `2 / (expected_cells + predicted_cells)` credit; other
cells receive zero. The credits sum to the document reward, within floating
point precision. `counts` includes the expected, predicted, and matched totals;
one-sided cells contribute to the denominator. Credit explains the scored
submission, not the counterfactual reward change from editing a field: an edit
can change both the denominator and row assignment. This is output attribution,
not credit for reasoning steps or tokens.

Paths address the **normalized scoring inputs**: schema defaults, list
normalization, and EOB layout reconciliation can change the representation from
the submitted JSON. A defaulted field can therefore receive credit even if
omitted from the submitted file. Acceptable values and normalizers use the
existing evaluator's comparison rules. The report preserves `value_incomplete`
when the scorer limits an oversized assignment; feedback is then an explanation
of that lower-bound score. Invalid submissions produce an empty cell list and
an error status. A failed grader removes stale feedback and produces no reward.

The published prompt does not ask for citations, so this lane does not report
grounding quality. It does not impose a new schema-validation penalty beyond
the existing evaluator. Both original ExtractBench and Pro allow 1,800 seconds
per document for the agent. Verifier time is separate from agent time.

For regenerating or publishing task exports, see the
[maintainer guide](harbor-maintenance.md).

## ExtractBench Pro challenge names

The [challenge table](extractbench-pro.md#challenges) pairs the public challenge names with their published split identifiers. Harbor task names derive from the document identifier without the split; the final challenge rename leaves those document-based names unchanged. The renamed release does not provide aliases for old Harbor task/package names. PDFs, schemas, reference answers, prompt, scorer, and reward remain unchanged; validating the export requires no model inference.
