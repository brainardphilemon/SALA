# Exact Kaggle V1 reproduction: SALA G14 on TriviaQA

This workflow reproduces Kaggle V1 (`scriptVersionId=353143016`) without changing
its loss, model, datasets, splits, projection search, or decision rule.

## Exact configuration

| Component | V1 value |
|---|---|
| SALA setting | G14 |
| Source training domains | TruthfulQA + NQ-Open + SciQ |
| Unseen target domain | TriviaQA `rc.nocontext` validation questions |
| Dataset size | Full splits (`--samples-per-domain 0`) |
| Model | Qwen2.5-7B-Instruct, pinned revision, NF4 4-bit |
| Correctness evaluator | BLEURT-20, maximum over reference aliases |
| Label rule | `int(bleurt_score >= 0.5)` |
| Projection dimensions | LODO selection (`--use-lodo-dim`) |
| Final MLP loss | Ordinary binary cross-entropy with logits (BCE) |
| Prediction threshold | Calibrated on pooled source-validation examples |
| Target supervision | None for fitting, selection, or threshold calibration |
| Random seed | 42 |

The reproduction pins repository commit `5c2c021`, which is the G14 implementation
used before the later logit-adjustment experiment. Therefore it does not rely on
new defaults or compatibility switches.

## Run on Kaggle

Enable Internet and a GPU accelerator. V1 used `GPU T4 x2` and completed in about
3 hours 49 minutes; actual availability and runtime can differ.

Import `kaggle/qwen25_7b_sala_g14_triviaqa_v1_repro.ipynb` and choose **Save Version**
with **Save & Run All**. The equivalent commands are:

```python
!git clone https://github.com/brainardphilemon/SALA.git /kaggle/working/SALA
%cd /kaggle/working/SALA
!git checkout 5c2c021
!pip install -q -r requirements-kaggle.txt
```

```python
!python sala_kaggle.py \
    --output-dir /kaggle/working/sala_qwen25_g14_triviaqa_v1_repro \
    --settings G14 \
    --samples-per-domain 0 \
    --use-lodo-dim
```

Expected source counts are 817 TruthfulQA, 3,610 NQ-Open, and 1,000 SciQ examples.
TriviaQA is deduplicated by `question_id`, producing 9,960 target questions in V1.
The final output is:

```text
/kaggle/working/sala_qwen25_g14_triviaqa_v1_repro/results.csv
```

Generation and BLEURT caches are written incrementally, but a committed Kaggle run
must finish before its working directory becomes a reusable version output.
