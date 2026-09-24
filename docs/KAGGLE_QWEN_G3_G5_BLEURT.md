# SALA G3/G5 with Qwen2.5-7B and BLEURT-20 on Kaggle

This workflow prepares SALA's missing feature caches from scratch and then runs the
official `G3` and `G5` settings:

| Setting | Source training domain | Unseen target domain |
|---|---|---|
| G3 | TruthfulQA | TriviaQA |
| G5 | SciQ | NQ-Open |

G3 is not an NQ-Open setting in the original repository. G5 supplies the requested
NQ-Open evaluation; changing G3's target to NQ-Open would be a custom experiment and
is therefore not mislabeled as official G3 here.

## What is logged

The compact CSV and detailed JSONL preserve the same fields requested for the HIDE
workflow, with HIDE's detector score replaced by SALA's classifier probability:

- `input_text`
- `output_text`
- `ground_truth_text` (the highest-scoring reference alias)
- `ground_truth_label`
- `bleurt_score`
- `sala_probability`
- `predicted_label`
- `split` (`target_evaluation` for every reported row)
- `sala_threshold`
- `ground_truth_answers`
- setting and source/target domain identifiers

`sala_probability` is `sigmoid(logit)`: the probability assigned by the source-trained
SALA classifier to `1 = BLEURT-correct/non-hallucinated`. `predicted_label` is obtained
by applying a threshold calibrated only on source validation examples. Target-domain
labels are never used for feature scaling, projection learning, classifier training,
early stopping, or threshold calibration.

## Correctness labels

BLEURT-20 is the **only reference-based correctness evaluator** in this workflow. No
cosine similarity, ROUGE, or exact-match score makes the labels. Every answer alias is
scored, and the maximum BLEURT-20 score becomes `bleurt_score`.

The default requested label is:

```text
ground_truth_label = int(bleurt_score >= 0.5)
```

BLEURT does not define an official binary correctness cutoff, so `0.5` is an explicit,
configurable operating point (`--bleurt-threshold`). These are automatic pseudo-labels,
not human correctness annotations.

## Feature compatibility

For each generated answer, the script replays the exact prompt and non-EOS answer
tokens, then stores the final answer token's hidden state at the embedding output and
all 28 Qwen transformer layers. This produces `[examples, 29, 3584]`, the layer-wise
format consumed by SALA and the same last-token extraction convention used by the
feature-generation lineage behind the repository's expected filenames.

Qwen uses NF4 4-bit weights by default so generation and hidden-state extraction fit
on a 16 GB Kaggle GPU. Use `--no-4bit` on a larger GPU for FP16/BF16 weights.

## Run on Kaggle

1. Create a Kaggle notebook.
2. Enable a GPU accelerator and Internet in **Notebook options**.
3. Import `kaggle/qwen25_7b_sala_g3_g5_bleurt.ipynb` and choose **Run All**, or run:

```python
!git clone https://github.com/brainardphilemon/SALA.git /kaggle/working/SALA
%cd /kaggle/working/SALA
!pip install -q -r requirements-kaggle.txt
```

```python
!python sala_kaggle.py --output-dir /kaggle/working/sala_qwen25_g3_g5_bleurt --settings G3 G5
```

The full run generates four datasets and can take a long time. Run a complete small
pipeline first with a new output directory:

```python
!python sala_kaggle.py \
    --output-dir /kaggle/working/sala_pilot \
    --settings G3 G5 \
    --samples-per-domain 100 \
    --epochs-proj 5 \
    --epochs-erm 5
```

Generation features and BLEURT rows are flushed after every completed example, so an
interrupted command can resume when the same output directory and settings are used.
Save a Kaggle notebook version to persist `/kaggle/working` between sessions.

## Outputs

```text
OUTPUT_DIR/
├── generation/DOMAIN/
│   ├── features.npy       # resumable float16 layer-wise feature array
│   ├── rows.jsonl         # input, output, aliases and feature row
│   └── manifest.json
├── bleurt/DOMAIN/
│   ├── scores.jsonl       # resumable max-alias BLEURT scores
│   └── manifest.json
├── results/G3/
│   ├── results.csv
│   ├── results.jsonl
│   └── summary.json
├── results/G5/
│   ├── results.csv
│   ├── results.jsonl
│   └── summary.json
├── results.csv            # combined G3 + G5 target rows
├── results.jsonl
└── summary.json
```

## Staged and smaller runs

```text
--stage generate             Qwen answers plus layer-wise features only
--stage bleurt               resume BLEURT scoring only
--stage train                train SALA from completed feature/BLEURT caches
--settings G5                run only SciQ -> NQ-Open
--samples-per-domain 100     deterministic pilot size per domain; 0 means full
--bleurt-batch-size 8        reduce if BLEURT exhausts GPU memory
--proj-dim 32                fixed layer-wise projected dimension
--epochs-proj 40             SALA projection epochs per layer
--epochs-erm 40              final SALA classifier epochs
```

The original `--use_lodo_dim` selector is intentionally not used for G3/G5 because
each has one source domain; leave-one-domain-out selection is undefined with a single
domain. The documented fixed 32-dimensional projection is used instead.
