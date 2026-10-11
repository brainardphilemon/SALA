"""Replay saved question/answer pairs through Qwen2.5-7B and save per-token features.

For every row the saved chat prompt (`model_input_text`) and generated answer (`output_text`) are
re-run through the same 4-bit Qwen model used for generation (teacher forcing, no sampling). For
each question token, the last prompt token and every answer token we keep:

* hidden states at a few layers, standardized and reduced with per-layer PCA fitted on
  source-train rows only (`--stage pca`), and
* uncertainty features from the model's own next-token distributions: log-probability of the
  token, entropy / max probability / top-2 margin of the distribution that produced it, whether
  it was the argmax, the entropy of the next-token distribution, and P(<|im_end|>) next.

`--stage extract --shard k --num-shards n` processes every n-th row so two GPUs can run in parallel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sala_kaggle import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    SETTING_MAP,
    load_labels,
    source_split_indices,
)
from sala_raw_concat import SETTING


QUESTION, PROMPT_END, ANSWER = 1, 2, 3
TOKEN_FEATURES = ("logprob", "entropy_prev", "maxprob_prev", "margin_prev", "is_argmax",
                  "entropy_next", "p_end_next")
ROW_FEATURES = ("answer_mean_logprob", "answer_min_logprob", "answer_mean_entropy",
                "answer_p_end", "question_mean_logprob", "answer_tokens")


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {"revision": args.model_revision, "torch_dtype": torch.float16}
    if args.no_4bit:
        kwargs["device_map"] = {"": args.device}
    else:
        # Same quantization as the generation stage in sala_kaggle.py.
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True), device_map={"": args.device})
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    return tokenizer, model


def prepare(row: dict, tokenizer, args) -> dict:
    """Token ids plus the positions whose features are kept."""
    prompt = row["model_input_text"]
    encoded = tokenizer(prompt, truncation=True, max_length=args.max_input_tokens,
                        return_offsets_mapping=True)
    prompt_ids = encoded["input_ids"]
    start = prompt.rfind(row["input_text"])
    question_positions = []
    if start >= 0 and row["input_text"]:
        end = start + len(row["input_text"])
        question_positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"])
                              if b > a and a >= start and b <= end]
    question_positions = question_positions[-args.max_question_tokens:]
    answer_ids = tokenizer(row["output_text"], add_special_tokens=False)["input_ids"][:args.max_answer_tokens]
    p = len(prompt_ids)
    positions = question_positions + [p - 1] + list(range(p, p + len(answer_ids)))
    segments = [QUESTION] * len(question_positions) + [PROMPT_END] + [ANSWER] * len(answer_ids)
    return dict(ids=prompt_ids + answer_ids, positions=positions, segments=segments,
                answer_tokens=len(answer_ids), retokenized_match=len(answer_ids) == row.get("output_length"))


def replay(model, tokenizer, items: list[dict], layers: list[int], device, pca=None):
    """Yield (item, states[n, L, H or K], token_features[n, F]) for each prepared item."""
    import torch

    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    width = max(len(item["ids"]) for item in items)
    input_ids = torch.full((len(items), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for b, item in enumerate(items):
        input_ids[b, :len(item["ids"])] = torch.tensor(item["ids"])
        mask[b, :len(item["ids"])] = 1
    with torch.inference_mode():
        out = model.model(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                          output_hidden_states=True, use_cache=False)
        for b, item in enumerate(items):
            pos = torch.tensor(item["positions"], device=device)
            prev = (pos - 1).clamp(min=0)
            final = out.last_hidden_state[b]
            logp_prev = torch.log_softmax(model.lm_head(final[prev]).float(), dim=-1)
            logp_next = torch.log_softmax(model.lm_head(final[pos]).float(), dim=-1)
            tokens = input_ids[b, pos.cpu()].to(device)
            p_prev = logp_prev.exp()
            top2 = p_prev.topk(2, dim=-1).values
            features = torch.stack([
                logp_prev.gather(1, tokens[:, None]).squeeze(1),
                -(p_prev * logp_prev).sum(-1),
                top2[:, 0],
                top2[:, 0] - top2[:, 1],
                (logp_prev.argmax(-1) == tokens).float(),
                -(logp_next.exp() * logp_next).sum(-1),
                logp_next[:, end_id].exp(),
            ], dim=1)
            features[pos == 0, :5] = 0.0  # no distribution precedes the very first token
            states = torch.stack([out.hidden_states[layer][b, pos].float() for layer in layers], dim=1)
            if pca is not None:
                states = torch.einsum("nlh,lkh->nlk", (states - pca["mean"]) / pca["std"], pca["components"])
            yield item, states.half().cpu().numpy(), features.cpu().numpy()


def row_summary(features: np.ndarray, segments: np.ndarray) -> np.ndarray:
    answer = features[segments == ANSWER]
    question = features[segments == QUESTION]
    end = features[segments == PROMPT_END][-1] if (segments == PROMPT_END).any() else None
    last = answer[-1] if len(answer) else end
    return np.asarray([
        answer[:, 0].mean() if len(answer) else 0.0,
        answer[:, 0].min() if len(answer) else 0.0,
        answer[:, 1].mean() if len(answer) else 0.0,
        last[6] if last is not None else 0.0,
        question[:, 0].mean() if len(question) else 0.0,
        float(len(answer)),
    ], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pca", "extract"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,28")
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--pca-rows", type=int, default=1200)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-question-tokens", type=int, default=48)
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    layers = [int(value) for value in args.layers.split(",")]
    artifact_dir = args.input_artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    domains = list(SETTING_MAP[SETTING]["source"]) + [SETTING_MAP[SETTING]["target"]]
    tokenizer, model = load_model(args)
    device = torch.device(args.device)

    if args.stage == "pca":
        # Fit per-layer standardization + PCA on token states from source-train rows only.
        rng = np.random.default_rng(args.seed)
        pool = []
        for name in SETTING_MAP[SETTING]["source"]:
            rows, labels = load_labels(name, args, artifact_dir)
            train_idx, _, _ = source_split_indices(labels, SETTING, name)
            pool.extend(rows[i] for i in train_idx)
        picked = [pool[i] for i in rng.choice(len(pool), size=min(args.pca_rows, len(pool)), replace=False)]
        collected = []
        for start in range(0, len(picked), args.batch_size):
            items = [prepare(row, tokenizer, args) for row in picked[start:start + args.batch_size]]
            collected.extend(states for _, states, _ in replay(model, tokenizer, items, layers, device))
        states = np.concatenate(collected).astype(np.float32)  # tokens x layers x hidden
        from sklearn.decomposition import PCA

        mean = states.mean(0)
        std = np.maximum(states.std(0), 1e-3)
        components, explained = [], []
        for layer in range(len(layers)):
            pca = PCA(n_components=args.pca_dim, svd_solver="randomized", random_state=args.seed)
            pca.fit((states[:, layer] - mean[layer]) / std[layer])
            components.append(pca.components_.astype(np.float32))
            explained.append(float(pca.explained_variance_ratio_.sum()))
        np.savez(output_dir / "pca.npz", mean=mean, std=std, components=np.stack(components),
                 layers=np.asarray(layers))
        json.dump({"layers": layers, "pca_dim": args.pca_dim, "fit_rows": len(picked),
                   "fit_tokens": int(len(states)), "explained_variance": explained},
                  open(output_dir / "pca.json", "w"), indent=2)
        print(f"Fitted PCA on {len(states)} tokens from {len(picked)} source-train rows; "
              f"explained variance per layer={[round(v, 3) for v in explained]}", flush=True)
        return

    saved = np.load(output_dir / "pca.npz")
    if list(saved["layers"]) != layers:
        raise ValueError("--layers does not match the fitted PCA")
    pca = {key: torch.from_numpy(saved[key]).to(device) for key in ("mean", "std", "components")}
    for name in domains:
        rows, _ = load_labels(name, args, artifact_dir)
        row_ids = np.arange(args.shard, len(rows), args.num_shards)
        items = [prepare(rows[i], tokenizer, args) | {"row_id": int(i)} for i in row_ids]
        order = sorted(range(len(items)), key=lambda k: len(items[k]["ids"]))
        results = {}
        for done, start in enumerate(range(0, len(order), args.batch_size)):
            batch = [items[k] for k in order[start:start + args.batch_size]]
            for item, states, features in replay(model, tokenizer, batch, layers, device, pca):
                results[item["row_id"]] = (item, states, features)
            if done % 100 == 0:
                print(f"shard {args.shard} {name}: {min(start + args.batch_size, len(order))}/{len(order)}", flush=True)
        kept = [results[i] for i in row_ids]
        lengths = np.asarray([len(item["positions"]) for item, _, _ in kept], dtype=np.int64)
        segments = np.concatenate([np.asarray(item["segments"], dtype=np.int8) for item, _, _ in kept])
        features = np.concatenate([f for _, _, f in kept]).astype(np.float32)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        summaries = np.stack([row_summary(features[o:o + n], segments[o:o + n])
                              for o, n in zip(offsets, lengths)])
        np.savez(output_dir / f"{name}_shard{args.shard}.npz", row_ids=row_ids, lengths=lengths,
                 offsets=offsets, segments=segments, features=features, summaries=summaries,
                 states=np.concatenate([s for _, s, _ in kept]))
        matched = sum(bool(item["retokenized_match"]) for item, _, _ in kept)
        print(f"shard {args.shard} {name}: saved {len(kept)} rows, {int(lengths.sum())} tokens; "
              f"answer retokenization matched generation length for {matched}/{len(kept)}", flush=True)


if __name__ == "__main__":
    main()
