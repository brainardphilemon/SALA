"""Six-source SALA ablation: fuse layer tokens with Qwen's own per-token view of the text.

Inputs per example:
* 15 log-sum-exp pooled layer tokens from the saved last-answer-token hidden states (as in
  sala_vlm_frozen_text.py),
* one token per question token, last prompt token and answer token, built from Qwen's hidden
  states at several layers (PCA-reduced, from sala_token_features.py) plus Qwen's uncertainty
  for that token (log-probability, entropies, margins, P(end)), with segment and position
  embeddings,
* one summary token of answer-level uncertainty.

A small fusion transformer trained from scratch attends over [CLS] + all of these; a linear head
on [CLS] predicts BLEURT-correct (non-hallucinated). Text-side tokens are dropped for a fraction
of training samples. `--mode layers` / `--mode text` train on one side only.
"""

from __future__ import annotations

import argparse
import json
import math
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
from sala_token_features import ROW_FEATURES, TOKEN_FEATURES
from sala_vlm_fusion import logsumexp_pool


def load_token_features(token_dir: Path, name: str, n_rows: int) -> dict:
    """Merge extraction shards into per-row (shard, offset, length) lookups."""
    shards = sorted(token_dir.glob(f"{name}_shard*.npz"))
    if not shards:
        raise FileNotFoundError(f"No token features for {name} in {token_dir}")
    arrays, where, summaries = [], np.full((n_rows, 3), -1, dtype=np.int64), np.zeros((n_rows, len(ROW_FEATURES)), np.float32)
    for shard_index, path in enumerate(shards):
        data = np.load(path)
        arrays.append({key: data[key] for key in ("states", "features", "segments")})
        where[data["row_ids"], 0] = shard_index
        where[data["row_ids"], 1] = data["offsets"]
        where[data["row_ids"], 2] = data["lengths"]
        summaries[data["row_ids"]] = data["summaries"]
    if (where[:, 0] < 0).any():
        raise ValueError(f"Token features missing for {int((where[:, 0] < 0).sum())} {name} rows")
    return dict(arrays=arrays, where=where, summaries=summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact-dir", type=Path, required=True)
    parser.add_argument("--token-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    parser.add_argument("--epochs-erm", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--mode", choices=("fused", "layers", "text"), default="fused")
    parser.add_argument("--text-dropout", type=float, default=0.2,
                        help="Fraction of fused training samples whose text-side tokens are dropped")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feature-clip", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()
    if args.model_dim % args.heads:
        parser.error("--model-dim must be divisible by --heads")
    use_layers, use_text = args.mode in ("fused", "layers"), args.mode in ("fused", "text")

    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from utils.helpers import seed_everything

    artifact_dir = args.input_artifact_dir.resolve()
    token_dir = args.token_feature_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "summary.json").exists() and (output_dir / "results.csv").exists():
        print("Completed output already exists; refusing to overwrite it", flush=True)
        return

    source_names = list(SETTING_MAP[SETTING]["source"])
    target_name = SETTING_MAP[SETTING]["target"]
    target_rows, target_labels = load_labels(target_name, args, artifact_dir)
    target_features = np.load(artifact_dir / "generation" / target_name / "features.npy", mmap_mode="r")
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
    layer_tokens = (n_layers + 1) // 2
    for bundle in all_bundles:
        bundle["tokens"] = load_token_features(token_dir, bundle["name"], len(bundle["rows"]))
    sample = all_bundles[0]["tokens"]["arrays"][0]["states"]
    state_layers, state_dim = sample.shape[1:]
    max_text_tokens = int(max(b["tokens"]["where"][:, 2].max() for b in all_bundles))

    # Source-train statistics for the scalar uncertainty features.
    train_tok = np.concatenate([
        b["tokens"]["arrays"][s]["features"][o:o + n]
        for b in bundles for s, o, n in b["tokens"]["where"][b["train_idx"]]])
    tok_mean, tok_std = train_tok.mean(0), np.maximum(train_tok.std(0), 1e-4)
    train_sum = np.concatenate([b["tokens"]["summaries"][b["train_idx"]] for b in bundles])
    sum_mean, sum_std = train_sum.mean(0), np.maximum(train_sum.std(0), 1e-4)

    # Zero-training baselines: each answer-level uncertainty feature used directly as a score.
    def raw_auroc(bundle_list, split):
        y = np.concatenate([b["labels"][b[f"{split}_idx"]] for b in bundle_list])
        x = np.concatenate([b["tokens"]["summaries"][b[f"{split}_idx"]] for b in bundle_list])
        return {name: float(roc_auc_score(y, x[:, k])) for k, name in enumerate(ROW_FEATURES)}

    baselines = {"source_test": raw_auroc(bundles, "test"), "target": raw_auroc([target_bundle], "all")}
    print("Raw uncertainty-feature AUROC (higher value = predicted correct): "
          + json.dumps({k: {n: round(v, 4) for n, v in d.items()} for k, d in baselines.items()}), flush=True)

    class TokenMapDataset(Dataset):
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
            states = (np.asarray(bundle["features"][row_id], dtype=np.float16).copy() if use_layers
                      else np.zeros((0,), dtype=np.float16))
            shard, offset, length = bundle["tokens"]["where"][row_id]
            arrays = bundle["tokens"]["arrays"][shard]
            return (states, arrays["states"][offset:offset + length],
                    (arrays["features"][offset:offset + length] - tok_mean) / tok_std,
                    arrays["segments"][offset:offset + length],
                    (bundle["tokens"]["summaries"][row_id] - sum_mean) / sum_std, np.float32(label))

    def collate(batch):
        states = torch.from_numpy(np.stack([item[0] for item in batch]))
        lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
        width = int(lengths.max())
        tok_states = torch.zeros(len(batch), width, state_layers, state_dim, dtype=torch.float16)
        tok_feats = torch.zeros(len(batch), width, len(TOKEN_FEATURES))
        segments = torch.zeros(len(batch), width, dtype=torch.long)
        for b, item in enumerate(batch):
            n = len(item[1])
            tok_states[b, :n] = torch.from_numpy(item[1])
            tok_feats[b, :n] = torch.from_numpy(item[2].astype(np.float32))
            segments[b, :n] = torch.from_numpy(item[3].astype(np.int64))
        summaries = torch.from_numpy(np.stack([item[4] for item in batch]).astype(np.float32))
        labels = torch.tensor([item[5] for item in batch])
        return states, tok_states, tok_feats, segments, lengths, summaries, labels

    train_ds = TokenMapDataset(bundles, "train")
    val_ds = TokenMapDataset(bundles, "val")
    test_ds = TokenMapDataset(bundles, "test")
    target_ds = TokenMapDataset([target_bundle], "all")
    y_train = np.asarray([entry[2] for entry in train_ds.entries], dtype=np.int64)
    y_val = np.asarray([entry[2] for entry in val_ds.entries], dtype=np.int64)
    y_test = np.asarray([entry[2] for entry in test_ds.entries], dtype=np.int64)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Source splits must contain both BLEURT classes")
    print(f"Mode={args.mode}; seed={args.seed}; source train/val/test={len(train_ds)}/{len(val_ds)}/"
          f"{len(test_ds)}; target={len(target_ds)}; text tokens per row up to {max_text_tokens} "
          f"({state_layers} layers x {state_dim} PCA dims + {len(TOKEN_FEATURES)} uncertainty features)",
          flush=True)

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

    feature_mean = feature_std = None
    if use_layers:
        total = torch.zeros(layer_tokens, hidden_size, dtype=torch.float64)
        total_sq = torch.zeros_like(total)
        with torch.inference_mode():
            for batch in DataLoader(train_ds, batch_size=256, shuffle=False, **loader_kwargs):
                image = logsumexp_pool(batch[0].to(device).float()).cpu().double()
                total += image.sum(0)
                total_sq += image.square().sum(0)
        feature_mean = (total / len(train_ds)).float()
        feature_std = (total_sq / len(train_ds) - (total / len(train_ds)).square()).clamp(min=0).sqrt()
        feature_std = feature_std.float().clamp(min=1e-4)

    class TokenMapFusion(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.model_dim
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.type_embed = nn.Embedding(3, d)  # 0 = layer token, 1 = text token, 2 = summary
            if use_layers:
                self.register_buffer("feature_mean", feature_mean.clone())
                self.register_buffer("feature_std", feature_std.clone())
                self.layer_proj = nn.Linear(hidden_size, d)
                self.layer_pos = nn.Parameter(torch.zeros(1, layer_tokens, d))
                nn.init.trunc_normal_(self.layer_pos, std=0.02)
            if use_text:
                self.state_proj = nn.Sequential(nn.LayerNorm(state_layers * state_dim),
                                                nn.Linear(state_layers * state_dim, d))
                self.feature_proj = nn.Linear(len(TOKEN_FEATURES), d)
                self.segment_embed = nn.Embedding(4, d)  # pad, question, prompt end, answer
                self.text_pos = nn.Parameter(torch.zeros(1, max_text_tokens, d))
                self.summary_proj = nn.Linear(len(ROW_FEATURES), d)
                nn.init.trunc_normal_(self.text_pos, std=0.02)
            self.input_dropout = nn.Dropout(args.dropout)
            layer = nn.TransformerEncoderLayer(d, args.heads, 4 * d, args.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.fusion = nn.TransformerEncoder(layer, args.fusion_layers, norm=nn.LayerNorm(d),
                                                enable_nested_tensor=False)
            self.head = nn.Linear(d, 1)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def forward(self, states, tok_states, tok_feats, segments, lengths, summaries,
                    drop_text=None, drop_layers=False):
            batch = lengths.shape[0]
            parts = [self.cls.expand(batch, -1, -1)]
            masks = [torch.zeros(batch, 1, dtype=torch.bool, device=lengths.device)]
            if use_layers:
                image = ((logsumexp_pool(states.float()) - self.feature_mean) / self.feature_std).clamp(
                    -args.feature_clip, args.feature_clip)
                parts.append(self.layer_proj(image) + self.layer_pos + self.type_embed.weight[0])
                masks.append(torch.full((batch, layer_tokens), drop_layers, dtype=torch.bool,
                                        device=lengths.device))
            if use_text:
                dropped = (drop_text if drop_text is not None
                           else torch.zeros(batch, dtype=torch.bool, device=lengths.device))
                parts.append((self.summary_proj(summaries) + self.type_embed.weight[2]).unsqueeze(1))
                masks.append(dropped.unsqueeze(1))
                width = tok_states.shape[1]
                text = (self.state_proj(tok_states.flatten(2).float()) + self.feature_proj(tok_feats)
                        + self.segment_embed(segments) + self.text_pos[:, :width] + self.type_embed.weight[1])
                positions = torch.arange(width, device=lengths.device)
                parts.append(text)
                masks.append((positions.unsqueeze(0) >= lengths.unsqueeze(1)) | dropped.unsqueeze(1))
            sequence = self.input_dropout(torch.cat([p.float() for p in parts], dim=1))
            hidden = self.fusion(sequence, src_key_padding_mask=torch.cat(masks, dim=1))
            return self.head(hidden[:, 0]).squeeze(-1).float()

    model = TokenMapFusion().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or name in ("cls", "layer_pos", "text_pos") else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.learning_rate)
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

    def to_device(batch):
        return [tensor.to(device, non_blocking=True) for tensor in batch]

    def predict(loader, mask_text=False, mask_layers=False):
        model.eval()
        logits = []
        with torch.inference_mode(), autocast():
            for batch in loader:
                states, tok_states, tok_feats, segments, lengths, summaries, _ = to_device(batch)
                drop = torch.full_like(lengths, mask_text, dtype=torch.bool)
                logits.append(model(states, tok_states, tok_feats, segments, lengths, summaries,
                                    drop_text=drop, drop_layers=mask_layers).float().cpu().numpy())
        return np.concatenate(logits).reshape(-1)

    print(f"TokenMapFusion parameters={parameter_count:,}; amp={amp_dtype}; "
          f"steps/epoch={len(train_loader)}; epochs={args.epochs_erm}", flush=True)
    best_path = output_dir / "best_fusion.pt"
    best_auc, best_epoch = 0.0, None
    text_dropout = args.text_dropout if args.mode == "fused" else 0.0
    for epoch in range(args.epochs_erm):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            states, tok_states, tok_feats, segments, lengths, summaries, labels = to_device(batch)
            drop = torch.rand(len(labels), device=device) < text_dropout
            with autocast():
                logits = model(states, tok_states, tok_feats, segments, lengths, summaries, drop_text=drop)
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
            torch.save({"model": model.state_dict(), "best_epoch": best_epoch, "best_auc": best_auc}, best_path)
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
    reliance = {}
    if args.mode == "fused":
        for name, kwargs in (("text_masked", dict(mask_text=True)), ("layers_masked", dict(mask_layers=True))):
            reliance[f"{name}_source_val_auroc"] = float(roc_auc_score(y_val, predict(val_loader, **kwargs)))
            reliance[f"{name}_target_auroc"] = float(roc_auc_score(target_labels, predict(target_loader, **kwargs)))
        print("Reliance checks: " + "; ".join(f"{k}={v:.5f}" for k, v in reliance.items()), flush=True)
    bleurt = {str(row["id"]): row for row in read_jsonl(artifact_dir / "bleurt" / target_name / "scores.jsonl")}
    results = []
    for row, label, logit, score, predicted in zip(
        target_rows, target_labels, target_logits, target_scores, target_predictions
    ):
        judged = bleurt[str(row["id"])]
        results.append({**row, "setting": "G14_6SRC_TOKEN_MAP_FUSION", "mode": args.mode, "seed": args.seed,
                        "source_domains": source_names, "target_domain": target_name,
                        "ground_truth_text": judged["ground_truth_text"], "ground_truth_label": int(label),
                        "bleurt_score": float(judged["bleurt_score"]), "raw_logit": float(logit),
                        "sala_probability": float(score), "predicted_label": int(predicted),
                        "classifier_loss": args.loss, "sala_threshold": threshold, "split": "target_evaluation"})
    summary = {
        "setting": "G14_6SRC_TOKEN_MAP_FUSION", "mode": args.mode, "source_domains": source_names,
        "target_domain": target_name, "model": "Qwen/Qwen2.5-7B-Instruct",
        "architecture": ("[CLS] + 15 log-sum-exp layer tokens + answer-uncertainty summary token + one "
                         "token per question/prompt-end/answer token (Qwen multi-layer PCA states + "
                         "uncertainty features + segment/position embeddings) -> fusion transformer "
                         "trained from scratch -> linear head on [CLS]"),
        "token_feature_dir": str(token_dir), "token_state_layers": int(state_layers),
        "token_state_dim": int(state_dim), "token_features": list(TOKEN_FEATURES),
        "row_features": list(ROW_FEATURES), "max_text_tokens": max_text_tokens,
        "text_dropout": text_dropout, "model_dim": args.model_dim, "heads": args.heads,
        "fusion_layers": args.fusion_layers, "dropout": args.dropout, "parameters": parameter_count,
        "epochs_erm": args.epochs_erm, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
        "lr_schedule": "linear warmup then cosine decay", "grad_clip_norm": 1.0,
        "mixed_precision": str(amp_dtype), "classifier_loss": args.loss, "seed": args.seed,
        "bleurt_threshold": args.bleurt_threshold,
        "label_definition": "1 = BLEURT-correct/non-hallucinated; 0 = otherwise",
        "sala_threshold": threshold, "threshold_calibration": calibration,
        "best_source_validation_auroc_during_training": best_auc, "best_epoch": best_epoch,
        **reliance, "raw_uncertainty_feature_auroc": baselines,
        "source_test_metrics": binary_metrics(y_test.tolist(), test_predictions.tolist(), test_scores.tolist()),
        "target_metrics": binary_metrics(target_labels.tolist(), target_predictions.tolist(), target_scores.tolist()),
        "target_supervision_used": False,
    }
    jsonl_dump_atomic(output_dir / "results.jsonl", results)
    csv_dump_atomic(output_dir / "results.csv", results)
    json_dump_atomic(output_dir / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("mode", "seed", "best_epoch",
                      "best_source_validation_auroc_during_training", *reliance)}, indent=2), flush=True)
    print(f"target_auroc={summary['target_metrics']['auroc']:.5f} "
          f"source_test_auroc={summary['source_test_metrics']['auroc']:.5f}", flush=True)


if __name__ == "__main__":
    main()
