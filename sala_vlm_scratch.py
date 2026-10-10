"""Six-source SALA ablation: a small vision-language model trained from scratch.

The log-sum-exp pooled Qwen layer states are the "image": each of the 15 pooled layer vectors is
a patch. A vision encoder (patch embedding + layer-position embedding + transformer) encodes them,
an MLP projector maps them into the multimodal width, and a single-stream fusion transformer
attends jointly over [CLS], the projected layer tokens, the question tokens and the generated
answer tokens. A linear head on [CLS] predicts whether the answer is BLEURT-correct
(non-hallucinated). Every weight is randomly initialized; only the Qwen tokenizer's vocabulary is
reused, remapped to the token ids that occur in source-train text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from sala_kaggle import (
    SETTING_MAP,
    binary_metrics,
    calibrate_threshold,
    csv_dump_atomic,
    json_dump_atomic,
    jsonl_dump_atomic,
    load_labels,
    read_jsonl,
    source_split_indices,
)
from sala_raw_concat import SETTING, sigmoid_focal_loss
from sala_vlm_fusion import logsumexp_pool


TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PAD, UNK, CLS, SEP = 0, 1, 2, 3
# Token types: pooled-layer patch, question text, answer text.
PATCH_TYPE, QUESTION_TYPE, ANSWER_TYPE = 0, 1, 2
# Tokenization happens before DataLoader workers fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    parser.add_argument("--epochs-erm", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vision-layers", type=int, default=2)
    parser.add_argument("--fusion-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-question-tokens", type=int, default=96)
    parser.add_argument("--max-answer-tokens", type=int, default=48)
    parser.add_argument("--feature-clip", type=float, default=10.0,
                        help="Clamp standardized pooled states to +/- this value")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--text-mode", choices=("full", "none"), default="full",
                        help="'none' trains on the pooled-layer tokens only (text ablation)")
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()
    if args.model_dim % args.heads:
        parser.error("--model-dim must be divisible by --heads")

    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer

    from utils.helpers import seed_everything

    artifact_dir = args.input_artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists() and (output_dir / "results.csv").exists():
        print("Completed output already exists; refusing to overwrite it", flush=True)
        return

    source_names = list(SETTING_MAP[SETTING]["source"])
    target_name = SETTING_MAP[SETTING]["target"]
    target_rows, target_labels = load_labels(target_name, args, artifact_dir)
    target_features = np.load(artifact_dir / "generation" / target_name / "features.npy", mmap_mode="r")
    if len(target_features) != len(target_rows):
        raise ValueError("TriviaQA feature count mismatch")
    bundles = []
    for name in source_names:
        rows, labels = load_labels(name, args, artifact_dir)
        features = np.load(artifact_dir / "generation" / name / "features.npy", mmap_mode="r")
        if features.shape[1:] != target_features.shape[1:] or len(features) != len(rows):
            raise ValueError(f"Feature shape/count mismatch in {name}")
        train_idx, val_idx, test_idx = source_split_indices(labels, SETTING, name)
        bundles.append(dict(name=name, rows=rows, labels=labels, features=features,
                            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
    target_bundle = dict(name=target_name, rows=target_rows, labels=target_labels,
                         features=target_features, all_idx=np.arange(len(target_rows)))
    n_layers, hidden_size = target_features.shape[1:]
    pooled_layers = (n_layers + 1) // 2

    # Text: Qwen BPE pieces, remapped to a compact vocabulary built from source-train rows only.
    text_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    all_bundles = (*bundles, target_bundle)
    for bundle in all_bundles:
        rows = bundle["rows"]
        bundle["question_bpe"] = tokenizer([row["input_text"] for row in rows],
                                           add_special_tokens=False)["input_ids"]
        bundle["answer_bpe"] = tokenizer([row["output_text"] for row in rows],
                                         add_special_tokens=False)["input_ids"]
    counts = Counter()
    for bundle in bundles:
        for row_id in bundle["train_idx"]:
            counts.update(bundle["question_bpe"][row_id])
            counts.update(bundle["answer_bpe"][row_id])
    vocab = {bpe_id: index + 4 for index, bpe_id in enumerate(sorted(counts))}
    truncated = 0
    for bundle in all_bundles:
        bundle["text"] = []
        for question, answer in zip(bundle["question_bpe"], bundle["answer_bpe"]):
            truncated += (len(question) > args.max_question_tokens
                          or len(answer) > args.max_answer_tokens)
            question = [vocab.get(i, UNK) for i in question[:args.max_question_tokens]]
            answer = [vocab.get(i, UNK) for i in answer[:args.max_answer_tokens]]
            if args.text_mode == "none":
                question, answer = [], []
            ids = question + [SEP] + answer
            types = [QUESTION_TYPE] * (len(question) + 1) + [ANSWER_TYPE] * len(answer)
            bundle["text"].append((ids, types))
    max_text_len = args.max_question_tokens + 1 + args.max_answer_tokens
    text_lengths = [len(ids) for bundle in bundles for row_id in bundle["train_idx"]
                    for ids in (bundle["text"][row_id][0],)]
    example = bundles[0]
    example_id = int(example["train_idx"][0])
    print(f"Text mode={args.text_mode}; tokenized {sum(len(b['rows']) for b in all_bundles)} rows in "
          f"{time.perf_counter() - text_start:.1f}s; source-train text tokens per row: "
          f"mean={np.mean(text_lengths):.1f} max={max(text_lengths)}", flush=True)
    print(f"Example {example['name']} train row: question={example['rows'][example_id]['input_text']!r} "
          f"answer={example['rows'][example_id]['output_text']!r} -> "
          f"{len(example['text'][example_id][0])} text tokens", flush=True)

    class VLMDataset(Dataset):
        def __init__(self, split_bundles: list[dict], split: str, mask_text: bool = False):
            self.bundles = split_bundles
            self.mask_text = mask_text
            self.entries = [
                (bundle_id, int(row_id), int(bundle["labels"][row_id]))
                for bundle_id, bundle in enumerate(split_bundles)
                for row_id in bundle[f"{split}_idx"]
            ]

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, position):
            bundle_id, row_id, label = self.entries[position]
            bundle = self.bundles[bundle_id]
            states = np.asarray(bundle["features"][row_id], dtype=np.float16).copy()
            ids, types = ([SEP], [QUESTION_TYPE]) if self.mask_text else bundle["text"][row_id]
            return states, ids, types, np.float32(label)

    def collate(batch):
        states = torch.from_numpy(np.stack([item[0] for item in batch]))
        width = max(len(item[1]) for item in batch)
        token_ids = torch.full((len(batch), width), PAD, dtype=torch.long)
        token_types = torch.full((len(batch), width), ANSWER_TYPE, dtype=torch.long)
        for position, item in enumerate(batch):
            token_ids[position, :len(item[1])] = torch.tensor(item[1], dtype=torch.long)
            token_types[position, :len(item[2])] = torch.tensor(item[2], dtype=torch.long)
        labels = torch.tensor([item[3] for item in batch])
        return states, token_ids, token_types, labels

    train_ds = VLMDataset(bundles, "train")
    val_ds = VLMDataset(bundles, "val")
    test_ds = VLMDataset(bundles, "test")
    target_ds = VLMDataset([target_bundle], "all")
    y_train = np.asarray([entry[2] for entry in train_ds.entries], dtype=np.int64)
    y_val = np.asarray([entry[2] for entry in val_ds.entries], dtype=np.int64)
    y_test = np.asarray([entry[2] for entry in test_ds.entries], dtype=np.int64)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Source splits must contain both BLEURT classes")
    print(f"Sources={source_names}; target={target_name}; source train/val/test="
          f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}; target={len(target_ds)}; "
          f"patches={pooled_layers}x{hidden_size}; text vocab={len(vocab) + 4}; "
          f"rows with truncated text={truncated}", flush=True)

    seed_everything(args.seed)
    device = torch.device(args.train_device if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        amp_dtype = torch.bfloat16
    elif device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    loader_kwargs = dict(collate_fn=collate, pin_memory=pin, num_workers=args.num_workers,
                         persistent_workers=args.num_workers > 0)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, **loader_kwargs)
    target_loader = DataLoader(target_ds, batch_size=256, shuffle=False, **loader_kwargs)

    # Per-coordinate standardization of pooled states, fitted on source-train rows only.
    total = torch.zeros(pooled_layers, hidden_size, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    with torch.inference_mode():
        for states, _, _, _ in DataLoader(train_ds, batch_size=256, shuffle=False, **loader_kwargs):
            pooled = logsumexp_pool(states.to(device).float()).cpu().double()
            total += pooled.sum(0)
            total_sq += pooled.square().sum(0)
    feature_mean = (total / len(train_ds)).float()
    feature_std = (total_sq / len(train_ds) - (total / len(train_ds)).square()).clamp(min=0).sqrt()
    feature_std = feature_std.float().clamp(min=1e-4)
    print("Fitted pooled-state standardization on source-train rows", flush=True)

    def encoder(layers: int):
        layer = nn.TransformerEncoderLayer(args.model_dim, args.heads, 4 * args.model_dim,
                                           args.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(args.model_dim),
                                     enable_nested_tensor=False)

    class ScratchVLM(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.model_dim
            self.register_buffer("feature_mean", feature_mean.clone())
            self.register_buffer("feature_std", feature_std.clone())
            # Vision encoder over the pooled layer "patches".
            self.patch_embed = nn.Linear(hidden_size, d)
            self.patch_pos = nn.Parameter(torch.zeros(1, pooled_layers, d))
            self.vision = encoder(args.vision_layers)
            # LLaVA-style MLP projector into the multimodal space.
            self.projector = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            # Text embeddings and the single-stream fusion transformer.
            self.token_embed = nn.Embedding(len(vocab) + 4, d, padding_idx=PAD)
            self.text_pos = nn.Parameter(torch.zeros(1, max_text_len, d))
            self.type_embed = nn.Embedding(3, d)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.fusion = encoder(args.fusion_layers)
            self.head = nn.Linear(d, 1)
            for parameter in (self.patch_pos, self.text_pos, self.cls):
                nn.init.trunc_normal_(parameter, std=0.02)

        def forward(self, states, token_ids, token_types):
            pooled = logsumexp_pool(states.float())
            pooled = ((pooled - self.feature_mean) / self.feature_std).clamp(-args.feature_clip,
                                                                             args.feature_clip)
            patches = self.vision(self.patch_embed(pooled) + self.patch_pos)
            patches = self.projector(patches) + self.type_embed.weight[PATCH_TYPE]
            text = (self.token_embed(token_ids) + self.text_pos[:, :token_ids.shape[1]]
                    + self.type_embed(token_types))
            batch = states.shape[0]
            sequence = torch.cat((self.cls.expand(batch, -1, -1), patches.to(text.dtype), text), dim=1)
            fixed = torch.zeros(batch, 1 + pooled_layers, dtype=torch.bool, device=states.device)
            padding = torch.cat((fixed, token_ids == PAD), dim=1)
            hidden = self.fusion(sequence, src_key_padding_mask=padding)
            return self.head(hidden[:, 0]).squeeze(-1).float()

    model = ScratchVLM().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or name.endswith(("_pos", "cls")) else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs_erm
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    if args.loss == "focal":
        def loss_fn(logits, targets):
            return sigmoid_focal_loss(logits, targets, args.focal_alpha, args.focal_gamma)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    def autocast():
        return torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    def predict(loader):
        model.eval()
        logits = []
        with torch.inference_mode(), autocast():
            for states, token_ids, token_types, _ in loader:
                logits.append(model(states.to(device, non_blocking=True),
                                    token_ids.to(device, non_blocking=True),
                                    token_types.to(device, non_blocking=True)).float().cpu().numpy())
        return np.concatenate(logits).reshape(-1)

    print(f"ScratchVLM parameters={parameter_count:,}; amp={amp_dtype}; "
          f"steps/epoch={len(train_loader)}; epochs={args.epochs_erm}", flush=True)
    best_path = output_dir / "best_vlm.pt"
    best_auc, best_epoch = 0.0, None
    for epoch in range(args.epochs_erm):
        model.train()
        total_loss = 0.0
        for states, token_ids, token_types, labels in train_loader:
            states = states.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)
            token_types = token_types.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                logits = model(states, token_ids, token_types)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item()) * len(labels)
        val_auc = float(roc_auc_score(y_val, predict(val_loader)))
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
            torch.save({"model": model.state_dict(), "vocab": vocab, "best_epoch": best_epoch,
                        "best_auc": best_auc, "args": vars(args) | {"input_artifact_dir": str(artifact_dir),
                                                                    "output_dir": str(output_dir)}},
                       best_path)
        print(f"epoch {epoch + 1}/{args.epochs_erm} loss={total_loss / len(train_ds):.5f} "
              f"source_val_auc={val_auc:.5f} best={best_auc:.5f}", flush=True)
    if best_epoch is None:
        raise RuntimeError("No valid best model was saved")
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])

    val_scores = 1 / (1 + np.exp(-np.clip(predict(val_loader), -80, 80)))
    test_scores = 1 / (1 + np.exp(-np.clip(predict(test_loader), -80, 80)))
    target_logits = predict(target_loader)
    target_scores = 1 / (1 + np.exp(-np.clip(target_logits, -80, 80)))
    threshold, calibration = calibrate_threshold(val_scores.tolist(), y_val.tolist())
    # Reliance check: score the trained model with the question/answer text blanked out.
    masked_val_auc = masked_target_auc = None
    if args.text_mode == "full":
        masked_val_auc = float(roc_auc_score(y_val, predict(DataLoader(
            VLMDataset(bundles, "val", mask_text=True), batch_size=256, shuffle=False, **loader_kwargs))))
        masked_target_auc = float(roc_auc_score(target_labels, predict(DataLoader(
            VLMDataset([target_bundle], "all", mask_text=True), batch_size=256, shuffle=False,
            **loader_kwargs))))
        print(f"Text-masked evaluation of the trained model: source_val_auc={masked_val_auc:.5f} "
              f"target_auc={masked_target_auc:.5f}", flush=True)
    test_predictions = (test_scores >= threshold).astype(np.int64)
    target_predictions = (target_scores >= threshold).astype(np.int64)
    bleurt = {str(row["id"]): row for row in
              read_jsonl(artifact_dir / "bleurt" / target_name / "scores.jsonl")}
    results = []
    for row, label, logit, score, predicted in zip(
        target_rows, target_labels, target_logits, target_scores, target_predictions
    ):
        judged = bleurt[str(row["id"])]
        results.append({**row, "setting": "G14_6SRC_SCRATCH_VLM", "source_domains": source_names,
                        "pooling_method": "logsumexp1d", "pooled_layers": pooled_layers,
                        "target_domain": target_name, "ground_truth_text": judged["ground_truth_text"],
                        "ground_truth_label": int(label), "bleurt_score": float(judged["bleurt_score"]),
                        "raw_logit": float(logit), "sala_probability": float(score),
                        "predicted_label": int(predicted), "classifier_loss": args.loss,
                        "sala_threshold": threshold, "split": "target_evaluation"})
    summary = {
        "setting": "G14_6SRC_SCRATCH_VLM", "source_domains": source_names,
        "target_domain": target_name, "model": "Qwen/Qwen2.5-7B-Instruct",
        "feature_policy": ("log-sum-exp pool saved answer-token hidden states over 3-layer, stride-2 "
                           "windows; standardize per coordinate with source-train statistics; each "
                           "pooled layer is one image patch"),
        "architecture": ("vision encoder (patch embedding + layer positions + transformer) -> MLP "
                         "projector -> single-stream fusion transformer over [CLS] + layer tokens + "
                         "question + [SEP] + answer with modality/segment embeddings -> linear head "
                         "on [CLS]; all weights randomly initialized"),
        "text_tokenizer": f"{TOKENIZER}@{TOKENIZER_REVISION} BPE remapped to source-train vocabulary",
        "text_mode": args.text_mode,
        "text_vocab_size": len(vocab) + 4, "text_rows_truncated": truncated,
        "text_masked_source_val_auroc": masked_val_auc,
        "text_masked_target_auroc": masked_target_auc,
        "max_question_tokens": args.max_question_tokens, "max_answer_tokens": args.max_answer_tokens,
        "model_dim": args.model_dim, "heads": args.heads, "vision_layers": args.vision_layers,
        "fusion_layers": args.fusion_layers, "dropout": args.dropout,
        "parameters": parameter_count,
        "num_layers_including_embedding_output": n_layers, "hidden_size": hidden_size,
        "pooling_method": "logsumexp1d", "pooled_layers": pooled_layers,
        "pool_kernel_size": 3, "pool_stride": 2, "feature_clip": args.feature_clip,
        "epochs_erm": args.epochs_erm, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "lr_schedule": "linear warmup then cosine decay",
        "grad_clip_norm": 1.0, "mixed_precision": str(amp_dtype), "classifier_loss": args.loss,
        "focal_alpha": args.focal_alpha if args.loss == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None, "seed": args.seed,
        "bleurt_threshold": args.bleurt_threshold,
        "label_definition": "1 = BLEURT-correct/non-hallucinated; 0 = otherwise",
        "probability_definition": "sigmoid(raw VLM logit) for BLEURT-correct/non-hallucinated",
        "sala_threshold": threshold, "threshold_calibration": calibration,
        "best_source_validation_auroc_during_training": best_auc, "best_epoch": best_epoch,
        "source_test_metrics": binary_metrics(y_test.tolist(), test_predictions.tolist(), test_scores.tolist()),
        "target_metrics": binary_metrics(target_labels.tolist(), target_predictions.tolist(), target_scores.tolist()),
        "target_supervision_used": False,
        "source_counts": {bundle["name"]: {"total": len(bundle["labels"]),
                                          "train": len(bundle["train_idx"]),
                                          "validation": len(bundle["val_idx"]),
                                          "test": len(bundle["test_idx"])} for bundle in bundles},
    }
    jsonl_dump_atomic(output_dir / "results.jsonl", results)
    csv_dump_atomic(output_dir / "results.csv", results)
    json_dump_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {len(results)} held-out TriviaQA rows to {output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
