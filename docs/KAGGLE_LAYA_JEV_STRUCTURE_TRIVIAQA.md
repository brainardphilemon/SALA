# LAYA with the Jev-run structure on TriviaQA

Import [`kaggle/laya_triviaqa_jev_structure.ipynb`](../kaggle/laya_triviaqa_jev_structure.ipynb) into a **new** Kaggle notebook. Attach notebook output `nvthaai/laya-triviaqa-gpu-multijudge`, enable Internet and GPU T4 x2, then Save & Run All. No Jev API key or Jev service is used.

This run uses the prior output only to recover the original question, saved Qwen answer, reference answer, and BLEURT-20 score. It discards the previous LAYA predictions. The new `state` is exactly the Jev run's `assistant_policy` list and `user_message` string (`Question: ...\n\nPredicted Answer: ...`). The three question definitions are identical to the Jev run. References and BLEURT scores are not sent to LAYA.

The new output directory is `/kaggle/working/laya_triviaqa_jev_structure`. Its `results.csv` logs input, output, reference, BLEURT score/label, LAYA `noul`, primary issue type, severity, and binary predicted label. `summary.json` contains aggregate binary-label metrics, and `laya_triviaqa_rows.jsonl` checkpoints progress. The prediction is `hallucinated` iff LAYA's `noul > 0.5`; the ground-truth label is `hallucinated` iff BLEURT-20 `< 0.5`.

Changing the state or question wording can change LAYA's answers. This is a new run, not a continuation of the earlier LAYA checkpoint; its separate output directory prevents mixing the two configurations.
