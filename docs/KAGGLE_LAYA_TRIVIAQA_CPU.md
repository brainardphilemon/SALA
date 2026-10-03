# LAYA on the saved TriviaQA generations (CPU-only)

This experiment is independent of the SALA continuation run. It consumes the
already-completed TriviaQA generation and BLEURT artifacts from
`nvthaai/sala-g14-6-source-triviaqa-version-6-artifacts` and does not load Qwen,
regenerate answers, or recompute BLEURT.

LAYA receives only:

- the TriviaQA question (`input_text`); and
- Qwen's generated answer (`output_text`).

The reference and BLEURT-20 score are joined after inference. The evaluation
labels are:

```text
ground_truth_hallucination_label = int(bleurt_score < 0.5)
predicted_label = int(laya_confidence_score > 0.5)
```

This deliberately uses strict `>` for the requested LAYA threshold. A score of
exactly `0.5` is therefore `not_hallucinated`.

## Kaggle

Import `kaggle/laya_triviaqa_cpu.ipynb` into a new notebook, attach the private
artifact dataset, enable Internet, and set the accelerator to **None**. The
notebook installs LAYA from its GitHub main branch and pins the SALA checkout to
the integration commit.

Outputs are written to `/kaggle/working/laya_triviaqa_cpu`:

- `laya_triviaqa_rows.jsonl`: append-only resumable checkpoint;
- `results.csv`: one row per TriviaQA prediction; and
- `summary.json`: accuracy, precision, recall, F1, and confusion matrix against
  BLEURT-derived hallucination labels.

The result log includes the input text, Qwen output, selected reference,
BLEURT score, BLEURT-derived hallucination label, LAYA probability/confidence,
and predicted label.
