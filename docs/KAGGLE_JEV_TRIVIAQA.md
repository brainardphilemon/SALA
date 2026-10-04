# Jev on saved TriviaQA answers

Use the separate notebook [`kaggle/jev_triviaqa_multijudge.ipynb`](../kaggle/jev_triviaqa_multijudge.ipynb). It calls the hosted Jev model through the [official TypeSafe Python SDK](https://github.com/typesafe-ai/typesafe-sdk-python); it does not run Qwen or BLEURT again.

1. Create a new Kaggle notebook and upload/import the `.ipynb` file.
2. Attach Kaggle dataset `nvthaai/sala-g14-6-source-triviaqa-version-6-artifacts` as input.
3. Enable Internet. GPU is unnecessary for this hosted API call.
4. Add a Kaggle notebook secret named `TYPESAFE_API_KEY` and grant the notebook access. Do not paste the key into notebook code, logs, or GitHub.
5. Run the notebook. `MAX_SAMPLES = 0` means all saved TriviaQA rows; set it to `1` first if you want to check one potentially billable Jev request.

The `state` sent for every example contains exactly the supplied `assistant_policy` list and `user_message` formatted as `Question: ...\n\nPredicted Answer: ...`. The three supplied question definitions (`hallucination`, `hallucination_type`, and `severity`) are sent unchanged. Ground-truth answers and BLEURT scores are joined only after the Jev response; they are never sent to Jev.

The output directory is `/kaggle/working/jev_triviaqa_multijudge`. `jev_triviaqa_rows.jsonl` is the per-row checkpoint; `results.csv` contains the question, predicted answer, reference answer, BLEURT score/label, Jev `noul`, binary predicted label, issue type, severity, and token usage; `summary.json` contains aggregate binary-label metrics. If a Kaggle session stops, attach the previous output as an input dataset and copy the checkpoint into the new output directory before rerunning to resume without repeating completed API calls. A normal Kaggle restart without preserving `/kaggle/working` does not preserve the checkpoint.

Jev's `noul` is a yes/true probability, not a literal boolean ([SDK schema](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/src/typesafe_sdk/_schemas/models.py)). The requested binary prediction is `hallucinated` iff `noul > 0.5`; there is no separate confidence-metric evaluation. The reference label remains `hallucinated` iff BLEURT-20 `< 0.5`. These label definitions measure different notions of factuality, so disagreements are expected. Jev calls may incur charges; check your account's limits and pricing before processing all rows.
