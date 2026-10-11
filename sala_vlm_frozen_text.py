"""Six-source SALA ablation: fuse Qwen hidden-state layer tokens with frozen pretrained text embeddings.

Image side: the saved answer-token hidden states become one token per layer (15 log-sum-exp pooled
windows by default, or all 29 raw layers), standardized per coordinate with source-train statistics
and projected by a shared linear map plus a learned layer-position embedding.

Text side: a frozen pretrained encoder (BGE-base) reads the question and the generated answer as
one sentence pair; its token states are computed once and projected into the fusion width.

A small fusion transformer trained from scratch attends over [CLS] + layer tokens + text tokens
with modality embeddings, and a linear head on [CLS] predicts whether the answer is BLEURT-correct
(non-hallucinated). Text tokens are dropped for a fraction of training samples (modality dropout)
so the hidden-state branch stays useful on its own. `--mode image` / `--mode text` train the same
model on one modality for comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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


TEXT_ENCODER = "BAAI/bge-base-en-v1.5"
TEXT_ENCODER_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
# Text embedding happens before DataLoader workers fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def encode_text(bundles: list[dict], args, device) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Run the frozen encoder over every (question, answer) pair once.

    Returns one contiguous float16 array of token states plus per-bundle offsets and lengths.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    cache = Path(args.text_cache) if args.text_cache else None
    key = f"{args.text_encoder}@{args.text_encoder_revision}:{args.max_text_tokens}"
    if cache is not None and cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if str(saved["key"]) == key:
            print(f"Loaded cached text embeddings from {cache}", flush=True)
            return (saved["states"], [saved[f"offsets_{i}"] for i in range(len(bundles))],
                    [saved[f"lengths_{i}"] for i in range(len(bundles))])
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, revision=args.text_encoder_revision)
    encoder = AutoModel.from_pretrained(args.text_encoder, revision=args.text_encoder_revision,
                                        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    encoder.to(device).eval()
    chunks, offsets, lengths, total = [], [], [], 0
    with torch.inference_mode():
        for bundle in bundles:
            questions = [row["input_text"] for row in bundle["rows"]]
            answers = [row["output_text"] for row in bundle["rows"]]
            bundle_lengths = np.zeros(len(questions), dtype=np.int64)
            bundle_offsets = np.zeros(len(questions), dtype=np.int64)
            for start in range(0, len(questions), 256):
                encoded = tokenizer(questions[start:start + 256], answers[start:start + 256],
                                    truncation=True, max_length=args.max_text_tokens,
                                    padding=True, return_tensors="pt").to(device)
                hidden = encoder(**encoded).last_hidden_state.float().cpu().numpy().astype(np.float16)
                for row, (states, mask) in enumerate(zip(hidden, encoded["attention_mask"].cpu().numpy())):
                    length = int(mask.sum())
                    chunks.append(states[:length])
                    bundle_offsets[start + row] = total
                    bundle_lengths[start + row] = length
                    total += length
            offsets.append(bundle_offsets)
            lengths.append(bundle_lengths)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    states = np.concatenate(chunks)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, key=np.asarray(key), states=states,
                 **{f"offsets_{i}": o for i, o in enumerate(offsets)},
                 **{f"lengths_{i}": l for i, l in enumerate(lengths)})
    return states, offsets, lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    parser.add_argument("--epochs-erm", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--mode", choices=("fused", "image", "text"), default="fused")
    parser.add_argument("--image", choices=("logsumexp", "raw"), default="logsumexp",
                        help="logsumexp: 15 pooled layer-window tokens; raw: all 29 layer tokens")
    parser.add_argument("--text-encoder", default=TEXT_ENCODER)
    parser.add_argument("--text-encoder-revision", default=TEXT_ENCODER_REVISION)
    parser.add_argument("--text-cache", default="/tmp/sala_text_embeddings.npz",
                        help="Reuse frozen text embeddings across runs ('' disables)")
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--text-dropout", type=float, default=0.3,
                        help="Fraction of fused training samples whose text tokens are dropped")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feature-clip", type=float, default=10.0,
                        help="Clamp standardized hidden states to +/- this value")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()
    if args.model_dim % args.heads:
        parser.error("--model-dim must be divisible by --heads")
    if not 0.0 <= args.text_dropout < 1.0:
        parser.error("--text-dropout must be in [0, 1)")
    use_image, use_text = args.mode in ("fused", "image"), args.mode in ("fused", "text")

    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

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
    all_bundles = [*bundles, target_bundle]
    n_layers, hidden_size = target_features.shape[1:]
    image_tokens = n_layers if args.image == "raw" else (n_layers + 1) // 2

    seed_everything(args.seed)
    device = torch.device(args.train_device if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    text_states, text_width = None, 0
    if use_text:
        text_states, offsets, lengths = encode_text(all_bundles, args, device)
        for bundle, bundle_offsets, bundle_lengths in zip(all_bundles, offsets, lengths):
            bundle["text_offsets"], bundle["text_lengths"] = bundle_offsets, bundle_lengths
        text_width = text_states.shape[1]
        train_lengths = np.concatenate([b["text_lengths"][b["train_idx"]] for b in bundles])
        print(f"Frozen text encoder {args.text_encoder}: {len(text_states)} token states x {text_width}; "
              f"source-train text tokens per row mean={train_lengths.mean():.1f} max={train_lengths.max()}",
              flush=True)

    class FusionDataset(Dataset):
        def __init__(self, split_bundles: list[dict], split: str):
            self.bundles = split_bundles
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
            states = (np.asarray(bundle["features"][row_id], dtype=np.float16).copy() if use_image
                      else np.zeros((0,), dtype=np.float16))
            if use_text:
                start = int(bundle["text_offsets"][row_id])
                text = text_states[start:start + int(bundle["text_lengths"][row_id])].copy()
            else:
                text = np.zeros((0, 0), dtype=np.float16)
            return states, text, np.float32(label)

    def collate(batch):
        states = torch.from_numpy(np.stack([item[0] for item in batch]))
        lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
        text = torch.zeros(len(batch), int(lengths.max()) if use_text else 0, text_width,
                           dtype=torch.float16)
        for position, item in enumerate(batch):
            if use_text:
                text[position, :len(item[1])] = torch.from_numpy(item[1])
        labels = torch.tensor([item[2] for item in batch])
        return states, text, lengths, labels

    train_ds = FusionDataset(bundles, "train")
    val_ds = FusionDataset(bundles, "val")
    test_ds = FusionDataset(bundles, "test")
    target_ds = FusionDataset([target_bundle], "all")
    y_train = np.asarray([entry[2] for entry in train_ds.entries], dtype=np.int64)
    y_val = np.asarray([entry[2] for entry in val_ds.entries], dtype=np.int64)
    y_test = np.asarray([entry[2] for entry in test_ds.entries], dtype=np.int64)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Source splits must contain both BLEURT classes")
    print(f"Mode={args.mode}; sources={source_names}; target={target_name}; source train/val/test="
          f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}; target={len(target_ds)}; "
          f"image={args.image} {image_tokens} layer tokens x {hidden_size}" if use_image else
          f"Mode={args.mode}; source train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}; "
          f"target={len(target_ds)}", flush=True)

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

    def make_image(states):
        states = states.float()
        return states if args.image == "raw" else logsumexp_pool(states)

    feature_mean = feature_std = None
    if use_image:
        # Per-coordinate standardization of layer tokens, fitted on source-train rows only.
        total = torch.zeros(image_tokens, hidden_size, dtype=torch.float64)
        total_sq = torch.zeros_like(total)
        with torch.inference_mode():
            for states, _, _, _ in DataLoader(train_ds, batch_size=256, shuffle=False, **loader_kwargs):
                image = make_image(states.to(device)).cpu().double()
                total += image.sum(0)
                total_sq += image.square().sum(0)
        feature_mean = (total / len(train_ds)).float()
        feature_std = (total_sq / len(train_ds) - (total / len(train_ds)).square()).clamp(min=0).sqrt()
        feature_std = feature_std.float().clamp(min=1e-4)
        print("Fitted layer-token standardization on source-train rows", flush=True)

    class FrozenTextFusion(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.model_dim
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.type_embed = nn.Embedding(2, d)  # 0 = layer token, 1 = text token
            if use_image:
                self.register_buffer("feature_mean", feature_mean.clone())
                self.register_buffer("feature_std", feature_std.clone())
                self.layer_proj = nn.Linear(hidden_size, d)
                self.layer_pos = nn.Parameter(torch.zeros(1, image_tokens, d))
                nn.init.trunc_normal_(self.layer_pos, std=0.02)
            if use_text:
                self.text_proj = nn.Sequential(nn.LayerNorm(text_width), nn.Linear(text_width, d))
            self.input_dropout = nn.Dropout(args.dropout)
            layer = nn.TransformerEncoderLayer(d, args.heads, 4 * d, args.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.fusion = nn.TransformerEncoder(layer, args.fusion_layers, norm=nn.LayerNorm(d),
                                                enable_nested_tensor=False)
            self.head = nn.Linear(d, 1)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward(self, states, text, lengths, drop_text=None, drop_image=False):
            batch = lengths.shape[0]
            parts = [self.cls.expand(batch, -1, -1)]
            masks = [torch.zeros(batch, 1, dtype=torch.bool, device=lengths.device)]
            if use_image:
                image = ((make_image(states) - self.feature_mean) / self.feature_std).clamp(
                    -args.feature_clip, args.feature_clip)
                parts.append(self.layer_proj(image) + self.layer_pos + self.type_embed.weight[0])
                masks.append(torch.full((batch, image_tokens), drop_image, dtype=torch.bool,
                                        device=lengths.device))
            if use_text:
                positions = torch.arange(text.shape[1], device=lengths.device)
                text_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
                if drop_text is not None:
                    text_mask = text_mask | drop_text.unsqueeze(1)
                parts.append(self.text_proj(text.float()) + self.type_embed.weight[1])
                masks.append(text_mask)
            sequence = self.input_dropout(torch.cat([p.float() for p in parts], dim=1))
            hidden = self.fusion(sequence, src_key_padding_mask=torch.cat(masks, dim=1))
            return self.head(hidden[:, 0]).squeeze(-1).float()

    model = FrozenTextFusion().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or name in ("cls", "layer_pos") else decay).append(parameter)
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

    def predict(loader, mask_text=False, mask_image=False):
        model.eval()
        logits = []
        with torch.inference_mode(), autocast():
            for states, text, lengths, _ in loader:
                lengths = lengths.to(device, non_blocking=True)
                drop = torch.full_like(lengths, mask_text, dtype=torch.bool) if use_text else None
                logits.append(model(states.to(device, non_blocking=True), text.to(device, non_blocking=True),
                                    lengths, drop_text=drop, drop_image=mask_image).float().cpu().numpy())
        return np.concatenate(logits).reshape(-1)

    print(f"FrozenTextFusion trainable parameters={parameter_count:,}; amp={amp_dtype}; "
          f"steps/epoch={len(train_loader)}; epochs={args.epochs_erm}", flush=True)
    best_path = output_dir / "best_fusion.pt"
    best_auc, best_epoch = 0.0, None
    text_dropout = args.text_dropout if args.mode == "fused" else 0.0
    for epoch in range(args.epochs_erm):
        model.train()
        total_loss = 0.0
        for states, text, lengths, labels in train_loader:
            states = states.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            drop = (torch.rand(len(labels), device=device) < text_dropout) if use_text else None
            with autocast():
                logits = model(states, text, lengths, drop_text=drop)
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
            torch.save({"model": model.state_dict(), "best_epoch": best_epoch, "best_auc": best_auc},
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
    test_predictions = (test_scores >= threshold).astype(np.int64)
    target_predictions = (target_scores >= threshold).astype(np.int64)
    # Reliance checks for the fused model: blank one modality at evaluation time.
    reliance = {}
    if args.mode == "fused":
        for name, kwargs in (("text_masked", dict(mask_text=True)), ("image_masked", dict(mask_image=True))):
            reliance[f"{name}_source_val_auroc"] = float(roc_auc_score(y_val, predict(val_loader, **kwargs)))
            reliance[f"{name}_target_auroc"] = float(roc_auc_score(target_labels,
                                                                   predict(target_loader, **kwargs)))
        print("Reliance checks: " + "; ".join(f"{k}={v:.5f}" for k, v in reliance.items()), flush=True)
    bleurt = {str(row["id"]): row for row in
              read_jsonl(artifact_dir / "bleurt" / target_name / "scores.jsonl")}
    results = []
    for row, label, logit, score, predicted in zip(
        target_rows, target_labels, target_logits, target_scores, target_predictions
    ):
        judged = bleurt[str(row["id"])]
        results.append({**row, "setting": "G14_6SRC_FROZEN_TEXT_FUSION", "mode": args.mode,
                        "source_domains": source_names, "image": args.image,
                        "target_domain": target_name, "ground_truth_text": judged["ground_truth_text"],
                        "ground_truth_label": int(label), "bleurt_score": float(judged["bleurt_score"]),
                        "raw_logit": float(logit), "sala_probability": float(score),
                        "predicted_label": int(predicted), "classifier_loss": args.loss,
                        "sala_threshold": threshold, "split": "target_evaluation"})
    summary = {
        "setting": "G14_6SRC_FROZEN_TEXT_FUSION", "mode": args.mode, "source_domains": source_names,
        "target_domain": target_name, "model": "Qwen/Qwen2.5-7B-Instruct",
        "architecture": ("[CLS] + hidden-state layer tokens (shared linear projection + layer "
                         "positions) + frozen pretrained text-encoder token states (LayerNorm + "
                         "linear) with modality embeddings -> fusion transformer trained from "
                         "scratch -> linear head on [CLS]"),
        "image": args.image if use_image else None, "image_tokens": image_tokens if use_image else None,
        "text_encoder": f"{args.text_encoder}@{args.text_encoder_revision}" if use_text else None,
        "text_input": "frozen encoder over the (question, generated answer) sentence pair" if use_text else None,
        "max_text_tokens": args.max_text_tokens, "text_dropout": text_dropout,
        "model_dim": args.model_dim, "heads": args.heads, "fusion_layers": args.fusion_layers,
        "dropout": args.dropout, "trainable_parameters": parameter_count,
        "num_layers_including_embedding_output": n_layers, "hidden_size": hidden_size,
        "feature_clip": args.feature_clip, "epochs_erm": args.epochs_erm, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "lr_schedule": "linear warmup then cosine decay",
        "grad_clip_norm": 1.0, "mixed_precision": str(amp_dtype), "classifier_loss": args.loss,
        "focal_alpha": args.focal_alpha if args.loss == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None, "seed": args.seed,
        "bleurt_threshold": args.bleurt_threshold,
        "label_definition": "1 = BLEURT-correct/non-hallucinated; 0 = otherwise",
        "probability_definition": "sigmoid(raw fusion logit) for BLEURT-correct/non-hallucinated",
        "sala_threshold": threshold, "threshold_calibration": calibration,
        "best_source_validation_auroc_during_training": best_auc, "best_epoch": best_epoch,
        **reliance,
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
