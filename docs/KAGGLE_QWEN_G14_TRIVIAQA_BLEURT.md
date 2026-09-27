# SALA G14 with Qwen2.5-7B and full TriviaQA on Kaggle

This workflow runs the repository's official `G14` domain mapping:

| Setting | Source training domains | Unseen target domain |
|---|---|---|
| G14 | TruthfulQA + NQ-Open + SciQ | TriviaQA |

`--samples-per-domain 0` uses every example in each selected benchmark split. For
TriviaQA this means the complete `rc.nocontext` validation question set after
deduplicating repeated `question_id` rows. Deduplication avoids generating the same
context-free question more than once; it does not take a smaller random sample.

## What the run does

1. Loads the pinned Qwen2.5-7B-Instruct revision in NF4 4-bit mode.
2. Generates a short answer for every source and target question with greedy decoding.
3. Stores the final non-EOS answer-token representation from the embedding output and
   all 28 transformer layers.
4. Scores every answer/reference-alias pair with BLEURT-20 and uses the maximum score.
5. Creates `ground_truth_label = int(bleurt_score >= 0.5)`.
6. Splits each source domain independently into 67.5% train, 7.5% validation and 25%
   source test partitions, preserving the official multi-environment G14 structure.
7. Trains SALA across the three source-domain IDs and evaluates every TriviaQA target
   example without using TriviaQA labels for training, scaling, projection learning,
   early stopping, or threshold calibration.

BLEURT-20 is the only reference-based evaluator. The label is an automatic
pseudo-label because BLEURT does not define an official binary correctness cutoff.

## Run on Kaggle

Enable a GPU accelerator and Internet in **Notebook options**, then import
`kaggle/qwen25_7b_sala_g14_triviaqa_bleurt.ipynb` and choose **Run All**.

The equivalent commands are:

```python
!git clone https://github.com/brainardphilemon/SALA.git /kaggle/working/SALA
%cd /kaggle/working/SALA
!pip install -q -r requirements-kaggle.txt
```

```python
!python sala_kaggle.py \
    --output-dir /kaggle/working/sala_qwen25_g14_triviaqa_bleurt \
    --settings G14 \
    --samples-per-domain 0
```

The notebook defaults to a fixed 32-dimensional projection per layer. This is the
practical full-data Kaggle configuration. To reproduce the repository's much more
expensive leave-one-source-domain-out dimension search, add:

```text
--use-lodo-dim
```

LODO evaluates the configured candidate dimensions independently at all 29 layer
outputs and can substantially increase runtime. It is valid for G14 because G14 has
three source domains; it is not valid for the single-source G3/G5 settings.

## Logged result columns

The compact CSV contains one row per TriviaQA target question:

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
/kaggle/working/sala_qwen25_g14_triviaqa_bleurt/
├── generation/{tqa,nq_open,sciq,triviaqa}/
├── bleurt/{tqa,nq_open,sciq,triviaqa}/
├── results/G14/results.csv
├── results/G14/results.jsonl
├── results/G14/summary.json
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
