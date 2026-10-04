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

## Updated factuality definition (GPU notebook)

`kaggle/laya_triviaqa_gpu_multijudge.ipynb` runs the revised LAYA task on a Kaggle
GPU. Its state keys are `question` and `assistant_response`. The task definition
uses the supplied factual hallucination criteria and assistant policy, and asks
LAYA for three judgments in one batch: `hallucination` (`noul`),
`hallucination_type` (`choice`), and `severity` (`score`). The notebook uses a new
output directory, `/kaggle/working/laya_triviaqa_gpu_multijudge`, so results
from the earlier definition cannot be mixed into this run.

The JSONL and CSV log include the `noul` probability, predicted label (strictly
greater than 0.5), issue type and its option probabilities, severity's expected
level index, its probabilities and legend, plus the original BLEURT fields.
Severity runs from 0 (no hallucination) to 3 (major) and may be fractional.
These three LAYA judgments are independent outputs; the predicted binary label
comes only from the `noul` probability. The model does not read the reference
answer or BLEURT score.

`laya_triviaqa.py` supports `--device auto` (default), `--device cpu`, and
`--device cuda`; requesting CUDA without an available GPU fails immediately.
An existing checkpoint created under a different task definition is rejected.
