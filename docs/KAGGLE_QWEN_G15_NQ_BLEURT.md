# SALA G15 with Qwen2.5-7B and full NQ-Open on Kaggle

This workflow runs the repository's official `G15` domain mapping:

| Setting | Source training domains | Unseen target domain |
|---|---|---|
| G15 | TruthfulQA + SciQ + TriviaQA | NQ-Open |

`--samples-per-domain 0` uses every example in each selected benchmark split. This
includes the complete NQ-Open validation split as the unseen target and the complete
TriviaQA `rc.nocontext` validation question set as a source domain. Repeated TriviaQA
`question_id` rows are deduplicated because the context-free prompt is identical;
this is not random subsampling.

## What the run does

1. Loads the pinned Qwen2.5-7B-Instruct revision in NF4 4-bit mode.
2. Generates a short answer for every source and target question with greedy decoding.
3. Stores the final non-EOS answer-token representation from the embedding output and
   all 28 transformer layers.
4. Scores every answer/reference-alias pair with BLEURT-20 and uses the maximum score.
5. Creates `ground_truth_label = int(bleurt_score >= 0.5)`.
6. Splits each source domain independently into 67.5% train, 7.5% validation and 25%
   source test partitions, preserving the official multi-environment G15 structure.
7. Trains SALA across the three source-domain IDs and evaluates every NQ-Open target
   example without using NQ-Open labels for training, scaling, projection learning,
   early stopping, or threshold calibration.

BLEURT-20 is the only reference-based evaluator. The label is an automatic
pseudo-label because BLEURT does not define an official binary correctness cutoff.

## Run on Kaggle

Enable a GPU accelerator and Internet in **Notebook options**, then import
`kaggle/qwen25_7b_sala_g15_nq_bleurt.ipynb` and choose **Run All**.

The equivalent commands are:

```python
!git clone https://github.com/brainardphilemon/SALA.git /kaggle/working/SALA
%cd /kaggle/working/SALA
!pip install -q -r requirements-kaggle.txt
```

```python
!python sala_kaggle.py \
    --output-dir /kaggle/working/sala_qwen25_g15_nq_bleurt \
    --settings G15 \
    --samples-per-domain 0
```

The notebook defaults to a fixed 32-dimensional projection per layer. This is the
practical full-data Kaggle configuration. To reproduce the repository's much more
expensive leave-one-source-domain-out dimension search, set `USE_LODO_DIM = True` in
the notebook or add:

```text
--use-lodo-dim
```

LODO evaluates the configured candidate dimensions independently at all 29 layer
outputs and can substantially increase runtime. It is valid for G15 because G15 has
three source domains.

## Logged result columns

The compact CSV contains one row per NQ-Open target question:

- `input_text`
- `output_text`
- `ground_truth_text` (highest-scoring reference alias)
- `ground_truth_label`
- `bleurt_score`
- `sala_probability`
- `predicted_label`
- `setting`, `source_domain`, `target_domain`, `split`
- `sala_threshold`
- `ground_truth_answers`

`sala_probability` is the sigmoid probability assigned to class `1 =
BLEURT-correct/non-hallucinated`. `predicted_label` applies a threshold calibrated
only on the combined source validation partitions.

Results are written to:

```text
/kaggle/working/sala_qwen25_g15_nq_bleurt/
├── generation/{tqa,sciq,triviaqa,nq_open}/
├── bleurt/{tqa,sciq,triviaqa,nq_open}/
├── results/G15/results.csv
├── results/G15/results.jsonl
├── results/G15/summary.json
├── results.csv
├── results.jsonl
└── summary.json
```

Generation features and BLEURT scores are flushed after each completed example. If a
Kaggle session ends, persist the output directory in a saved notebook version and run
the same command against the restored directory to resume.

For a quick end-to-end check, use a fresh output directory and
`--samples-per-domain 100 --epochs-proj 5 --epochs-erm 5` before launching the full
run.
