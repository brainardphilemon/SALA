# Raw-concatenation MLP baseline for the six-source SALA run

Use [`kaggle/qwen25_7b_g14_6src_raw_concat.ipynb`](../kaggle/qwen25_7b_g14_6src_raw_concat.ipynb)
in a **separate Kaggle notebook**. Attach the private dataset
`nvthaai/sala-g14-6-source-triviaqa-version-6-artifacts`, enable Internet and
GPU T4 x2, then Save & Run All.
The notebook pins implementation commit `b7f8f4b`.

The notebook consumes the exact saved Qwen2.5-7B-Instruct question/answer rows,
answer-token hidden states, and BLEURT-20 scores for all six training domains
and held-out TriviaQA. It does not regenerate answers or recompute BLEURT.
The source domains, splits, BLEURT cutoff (0.5), 40-epoch BCE training, Mish
MLP (1024 hidden units, 0.2 dropout), AdamW (learning rate 1e-3, weight decay
1e-4), seed 42, and source-validation threshold rule match the completed SALA
continuation. Only the feature transform changes: no projections or scaling;
all 29 raw 3584-dimensional states are flattened into a 103,936-dimensional
MLP input. TriviaQA labels are used only for reporting metrics after inference.

The raw input is about 18 times wider than the projected 5,768-dimensional
SALA input, so training can use substantially more GPU memory and time. The
script streams saved feature files in batches and writes restart checkpoints
after each epoch. If Kaggle's time limit interrupts a committed run, retain
the output files and resume from the checkpoint in a new notebook version;
the checkpoint is not part of the original SALA model.

Outputs under `/kaggle/working/sala_g14_6src_raw_concat/`:
`results.csv`, `results.jsonl`, `summary.json`, `best_mlp.pt`, and
`training_checkpoint.pt`. The row logs contain input/output text, ground
truth text and label, BLEURT score, raw logit, predicted probability, and
predicted label. The probability means *BLEURT-correct/non-hallucinated*;
it is not a hallucination probability.
