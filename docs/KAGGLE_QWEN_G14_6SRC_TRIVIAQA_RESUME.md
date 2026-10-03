# Resume the six-source G14 TriviaQA run from Kaggle Version 6

Kaggle Version 6 (`nvthaai/multisalagithub`, version 6) reached the 12-hour runtime
limit after completing:

- Qwen2.5-7B-Instruct generation and hidden-state extraction for all seven domains;
- BLEURT-20 scoring for all source and TriviaQA records; and
- 21 of 29 LODO projection layers.

The Version 6 code did not persist projection-layer checkpoints, so its first 21
projection layers cannot be recovered. The continuation reuses the expensive,
completed generation and BLEURT artifacts and reruns only the projection/final-MLP
training stage. This is much shorter than rerunning the full pipeline.

Use [`kaggle/qwen25_7b_sala_g14_6src_triviaqa_resume.ipynb`](../kaggle/qwen25_7b_sala_g14_6src_triviaqa_resume.ipynb).
The notebook pins code commit `7d09466`, attaches the exact Version 6 output with
`kagglehub`, and runs:

```bash
python sala_kaggle.py \
  --stage train \
  --input-artifact-dir <version-6-artifact-directory> \
  --output-dir /kaggle/working/sala_qwen25_g14_6src_triviaqa_resume \
  --settings G14_6SRC \
  --samples-per-domain 0 \
  --use-lodo-dim \
  --classifier-loss bce
```

The continuation keeps the Version 1 experimental configuration: full datasets,
LODO dimension selection, BCE final-MLP loss, BLEURT cutoff 0.5, and source-validation
threshold calibration. No generation or BLEURT scoring is repeated.

The updated trainer saves every completed projection layer atomically under
`results/G14_6SRC/projection_checkpoints/`. If training is interrupted within the
same writable output directory, rerunning the command resumes at the first missing
layer.

Expected final files include:

- `results.csv`
- `results.jsonl`
- `summary.json`
- `results/G14_6SRC/results.csv`
- `results/G14_6SRC/results.jsonl`
- `results/G14_6SRC/summary.json`

