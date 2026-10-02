# SALA with six source datasets and unseen TriviaQA

This custom `G14_6SRC` experiment changes only the set of source-training domains.
It preserves the Kaggle V1 model, feature extraction, BLEURT labels, data splitting,
LODO projection search, final-MLP BCE loss, and source-validation threshold rule.

## Dataset configuration

| Role | Domain | Pinned Hugging Face split | Examples |
|---|---|---|---:|
| Original source | TruthfulQA | `validation` | 817 |
| Original source | NQ-Open | `validation` | 3,610 |
| Original source | SciQ | `validation` | 1,000 |
| New source | PopQA | `test` | 14,267 |
| New source | WebQuestions | `train` | 3,778 |
| New source | HotpotQA | `distractor/validation` | 7,405 |
| Unseen target | TriviaQA | `rc.nocontext/validation`, deduplicated by question ID | 9,960 |

The six sources contain 30,877 examples before their independent stratified splits.
PopQA aliases and WebQuestions answer lists are all retained for BLEURT scoring.
HotpotQA contributes its question and gold answer only; its passages are intentionally
not inserted into the prompt because V1 is a closed-book, question-only experiment.

## V1 settings preserved

- Qwen2.5-7B-Instruct with the same pinned model revision and NF4 loading.
- Greedy generation with the same prompt and token limits.
- BLEURT-20 maximum over aliases and label cutoff `0.5`.
- Independent 67.5%/7.5%/25% source train/validation/test splits.
- LODO projection-dimension selection over all six source-domain IDs.
- Ordinary BCE-with-logits for the final MLP (`--classifier-loss bce`).
- Prediction threshold selected from pooled source-validation examples.
- No TriviaQA labels used for fitting, scaling, projection selection, early stopping,
  or threshold selection.

This is not an official SALA table setting; `G14_6SRC` is a clearly named extension
of V1 for this six-source experiment.

## Run on Kaggle

Enable Internet and `GPU T4 x2`, import
`kaggle/qwen25_7b_sala_g14_6src_triviaqa_v1.ipynb`, and select **Save Version** →
**Save & Run All**. The notebook pins code commit `78d2ac4`.

Equivalent commands:

```python
!git clone https://github.com/brainardphilemon/SALA.git /kaggle/working/SALA
%cd /kaggle/working/SALA
!git checkout 78d2ac4
!pip install -q -r requirements-kaggle.txt
```

```python
!python sala_kaggle.py \
    --output-dir /kaggle/working/sala_qwen25_g14_6src_triviaqa_v1 \
    --settings G14_6SRC \
    --samples-per-domain 0 \
    --use-lodo-dim \
    --classifier-loss bce
```

The final combined log is written to:

```text
/kaggle/working/sala_qwen25_g14_6src_triviaqa_v1/results.csv
```

This run processes roughly twice as many questions as V1 and performs six-domain
LODO folds, so it can take substantially longer than the original four-hour run.
