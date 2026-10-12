"""Six-source SALA ablation: polysemantic-token transformer over log-sum-exp pooled Qwen states.

The saved answer-token states (29 layers x hidden size) are log-sum-exp pooled over 3-layer,
stride-2 windows exactly as in `sala_raw_concat.py --pooling logsumexp1d`, giving a 15 x hidden
matrix. Each hidden unit then becomes one "polysemantic token": its 15-value trajectory across the
pooled layers, standardized with source-train statistics, embedded by a shared Linear(15 -> d), plus
an optional learned per-neuron positional embedding and an optional embedding of data-derived neuron
descriptors (source-train mean, spread and label correlation per pooled layer).

`--encoder perceiver` (default): learned latent tokens cross-attend to the neuron tokens (cost linear
in the hidden size), pre-LN self-attention blocks run over the latents, and their mean is the
constant embedding. `--encoder full`: a [CLS] token plus all neuron tokens go through pre-LN
self-attention blocks and [CLS] is projected to the embedding. Either embedding feeds the same
residual Mish MLP head as the 4 x 128 raw-concat runs; split, loss, batch size, checkpoint selection
and threshold calibration match sala_raw_concat.py.
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
from sala_raw_concat import SETTING, RawStateDataset, save_checkpoint, sigmoid_focal_loss
from sala_vlm_fusion import logsumexp_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    parser.add_argument("--epochs-erm", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--encoder", choices=("perceiver", "full"), default="perceiver")
    parser.add_argument("--neuron-positions", choices=("learned", "none"), default="learned")
    parser.add_argument("--no-descriptors", dest="descriptors", action="store_false",
                        help="Disable the data-derived neuron descriptor embedding")
    parser.add_argument("--latents", type=int, default=64, help="Perceiver latent tokens")
    parser.add_argument("--latent-heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine",
                        help="cosine = linear warmup then cosine decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--model-dim", type=int, default=128, help="Neuron-token width")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=512,
                        help="Constant output embedding fed to the MLP head (perceiver latent width)")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4,
                        help="Hidden linear layers in the MLP head, as in the 4 x 128 runs")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feature-clip", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()
    if args.model_dim % args.heads or args.embedding_dim % args.latent_heads:
        parser.error("Model widths must be divisible by their head counts")

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
    bundles = []
    for name in source_names:
        rows, labels = load_labels(name, args, artifact_dir)
        features = np.load(artifact_dir / "generation" / name / "features.npy", mmap_mode="r")
        if features.shape[1:] != target_features.shape[1:] or len(features) != len(rows):
            raise ValueError(f"Feature shape/count mismatch in {name}")
        train_idx, val_idx, test_idx = source_split_indices(labels, SETTING, name)
        bundles.append(dict(name=name, rows=rows, labels=labels, features=features,
                            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx))
    if len(target_features) != len(target_rows):
        raise ValueError("TriviaQA feature count mismatch")
    n_layers, hidden_size = target_features.shape[1:]
    pooled_layers = (n_layers + 1) // 2

    class TargetDataset(Dataset):
        def __init__(self, features):
            self.features = features

        def __len__(self):
            return len(self.features)

        def __getitem__(self, position):
            return np.asarray(self.features[position], dtype=np.float32).reshape(-1).copy(), np.float32(0)

    train_ds = RawStateDataset(bundles, "train")
    val_ds = RawStateDataset(bundles, "val")
    test_ds = RawStateDataset(bundles, "test")
    y_train = np.asarray([entry[2] for entry in train_ds.entries], dtype=np.int64)
    y_val = np.asarray([entry[2] for entry in val_ds.entries], dtype=np.int64)
    y_test = np.asarray([entry[2] for entry in test_ds.entries], dtype=np.int64)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Source splits must contain both BLEURT classes")
    print(f"Sources={source_names}; target={target_name}; source train/val/test="
          f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}; target={len(target_rows)}; "
          f"pooled matrix {pooled_layers} layers x {hidden_size} hidden -> {hidden_size} polysemantic "
          f"tokens of length {pooled_layers}", flush=True)

    seed_everything(args.seed)
    device = torch.device(args.train_device if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        amp_dtype = torch.bfloat16
    elif device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    loader_kwargs = dict(pin_memory=pin, num_workers=args.num_workers,
                         persistent_workers=args.num_workers > 0)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, **loader_kwargs)
    target_loader = DataLoader(TargetDataset(target_features), batch_size=256, shuffle=False, **loader_kwargs)

    def pool(flat):
        return logsumexp_pool(flat.float().reshape(-1, n_layers, hidden_size))

    # Per-coordinate (layer window x neuron) statistics, fitted on source-train rows only.
    total = torch.zeros(pooled_layers, hidden_size, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    total_xy = torch.zeros_like(total)
    with torch.inference_mode():
        for xb, yb in DataLoader(train_ds, batch_size=256, shuffle=False, **loader_kwargs):
            pooled = pool(xb.to(device)).cpu().double()
            total += pooled.sum(0)
            total_sq += pooled.square().sum(0)
            total_xy += (pooled * yb.double()[:, None, None]).sum(0)
    n = len(train_ds)
    mean64 = total / n
    std64 = (total_sq / n - mean64.square()).clamp(min=0).sqrt()
    feature_mean = mean64.float()
    feature_std = std64.float().clamp(min=1e-4)
    # Neuron descriptors (hidden x 3*layers): signed-log mean, log spread and point-biserial label
    # correlation per pooled layer, standardized across neurons.
    p = float(y_train.mean())
    correlation = ((total_xy / n - mean64 * p) / (std64.clamp(min=1e-8) * math.sqrt(p * (1 - p)))).clamp(-1, 1)
    descriptors = torch.cat((mean64.sign() * mean64.abs().log1p(), std64.clamp(min=1e-8).log(),
                             correlation), dim=0).T.float()
    descriptors = (descriptors - descriptors.mean(0)) / descriptors.std(0).clamp(min=1e-6)
    print(f"Fitted per-coordinate standardization and {descriptors.shape[1]} neuron descriptors on "
          f"source-train rows; max |label correlation|={correlation.abs().max():.3f}", flush=True)

    class PolysemanticTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            d, width = args.model_dim, args.embedding_dim
            self.register_buffer("feature_mean", feature_mean.clone())
            self.register_buffer("feature_std", feature_std.clone())
            # One token per hidden unit: its pooled-layer trajectory (length 15) -> d.
            self.token_embed = nn.Linear(pooled_layers, d)
            self.neuron_pos = (nn.Parameter(torch.zeros(1, hidden_size, d))
                               if args.neuron_positions == "learned" else None)
            if args.descriptors:
                self.register_buffer("descriptors", descriptors.clone())
                self.descriptor_embed = nn.Sequential(nn.Linear(descriptors.shape[1], d), nn.GELU(),
                                                      nn.Linear(d, d))
            self.input_dropout = nn.Dropout(args.dropout)
            if args.encoder == "full":
                self.cls = nn.Parameter(torch.zeros(1, 1, d))
                layer = nn.TransformerEncoderLayer(d, args.heads, args.ffn_dim, args.dropout,
                                                   activation="gelu", batch_first=True, norm_first=True)
                self.blocks = nn.TransformerEncoder(layer, args.blocks, norm=nn.LayerNorm(d),
                                                    enable_nested_tensor=False)
                self.to_embedding = nn.Linear(d, width)
                nn.init.trunc_normal_(self.cls, std=0.02)
            else:
                # Perceiver: latents read the neuron tokens by cross-attention, then self-attend.
                self.token_norm = nn.LayerNorm(d)
                self.latents = nn.Parameter(torch.zeros(1, args.latents, width))
                self.cross_norm = nn.LayerNorm(width)
                self.cross_attn = nn.MultiheadAttention(width, args.latent_heads, dropout=args.dropout,
                                                        kdim=d, vdim=d, batch_first=True)
                self.cross_ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(),
                                               nn.Dropout(args.dropout), nn.Linear(2 * width, width))
                layer = nn.TransformerEncoderLayer(width, args.latent_heads, 2 * width, args.dropout,
                                                   activation="gelu", batch_first=True, norm_first=True)
                self.blocks = nn.TransformerEncoder(layer, args.blocks, norm=nn.LayerNorm(width),
                                                    enable_nested_tensor=False)
                nn.init.trunc_normal_(self.latents, std=0.02)
            # MLP head: same residual pattern as RawConcatMLP in sala_raw_concat.py.
            self.fc_in = nn.Linear(args.embedding_dim, args.hidden_dim)
            self.head_blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(args.hidden_dim), nn.Mish(), nn.Dropout(args.dropout),
                              nn.Linear(args.hidden_dim, args.hidden_dim))
                for _ in range(args.hidden_layers - 1)
            ])
            self.out = nn.Linear(args.hidden_dim, 1)
            if self.neuron_pos is not None:
                nn.init.trunc_normal_(self.neuron_pos, std=0.02)

        def neuron_tokens(self, flat):
            pooled = ((pool(flat) - self.feature_mean) / self.feature_std).clamp(
                -args.feature_clip, args.feature_clip)
            tokens = self.token_embed(pooled.transpose(1, 2))
            if self.neuron_pos is not None:
                tokens = tokens + self.neuron_pos
            if args.descriptors:
                tokens = tokens + self.descriptor_embed(self.descriptors)
            return self.input_dropout(tokens)

        def embed(self, flat):
            """Constant-size embedding: (batch, n_layers * hidden) -> (batch, embedding_dim)."""
            tokens = self.neuron_tokens(flat)
            if args.encoder == "full":
                sequence = torch.cat((self.cls.expand(len(tokens), -1, -1), tokens), dim=1)
                return self.to_embedding(self.blocks(sequence)[:, 0])
            tokens = self.token_norm(tokens)
            latents = self.latents.expand(len(tokens), -1, -1)
            attended, _ = self.cross_attn(self.cross_norm(latents), tokens, tokens, need_weights=False)
            latents = latents + attended
            latents = latents + self.cross_ffn(latents)
            return self.blocks(latents).mean(1)

        def forward(self, flat):
            h = self.fc_in(self.embed(flat))
            for block in self.head_blocks:
                h = h + block(h)
            return self.out(h).squeeze(-1).float()

    model = PolysemanticTransformer().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs_erm
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step):
        if args.lr_schedule == "constant":
            return 1.0
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
            for xb, _ in loader:
                logits.append(model(xb.to(device, non_blocking=True)).float().cpu().numpy())
        return np.concatenate(logits).reshape(-1)

    print(f"PolysemanticTransformer encoder={args.encoder}; positions={args.neuron_positions}; "
          f"descriptors={args.descriptors}; parameters={parameter_count:,}; neuron tokens={hidden_size}; "
          f"amp={amp_dtype}; steps/epoch={len(train_loader)}; epochs={args.epochs_erm}", flush=True)
    best_path = output_dir / "best_polysemantic.pt"
    best_auc, best_epoch = 0.0, None
    for epoch in range(args.epochs_erm):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast():
                logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item()) * len(yb)
        val_auc = float(roc_auc_score(y_val, predict(val_loader)))
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
            save_checkpoint(best_path, {"model": model.state_dict(), "best_epoch": best_epoch,
                                        "best_auc": best_auc, "hidden_size": hidden_size,
                                        "pooled_layers": pooled_layers})
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
    bleurt = {str(row["id"]): row for row in read_jsonl(artifact_dir / "bleurt" / target_name / "scores.jsonl")}
    results = []
    for row, label, logit, score, predicted in zip(
        target_rows, target_labels, target_logits, target_scores, target_predictions
    ):
        judged = bleurt[str(row["id"])]
        results.append({**row, "setting": "G14_6SRC_POLYSEMANTIC", "source_domains": source_names,
                        "pooling_method": "logsumexp1d", "pooled_layers": pooled_layers,
                        "target_domain": target_name, "ground_truth_text": judged["ground_truth_text"],
                        "ground_truth_label": int(label), "bleurt_score": float(judged["bleurt_score"]),
                        "raw_logit": float(logit), "sala_probability": float(score),
                        "predicted_label": int(predicted), "classifier_loss": args.loss,
                        "sala_threshold": threshold, "split": "target_evaluation"})
    summary = {
        "setting": "G14_6SRC_POLYSEMANTIC", "source_domains": source_names,
        "target_domain": target_name, "model": "Qwen/Qwen2.5-7B-Instruct",
        "feature_policy": ("log-sum-exp pool saved answer-token hidden states over 3-layer, stride-2 windows; "
                           "standardize per coordinate with source-train statistics; one token per hidden unit"),
        "architecture": ("hidden-unit tokens (pooled-layer trajectory -> Linear(15, d) [+ per-neuron position] "
                         "[+ neuron-descriptor embedding]) -> " + (
                             "[CLS] + pre-LN self-attention blocks -> [CLS] -> Linear(d, 512)"
                             if args.encoder == "full" else
                             "Perceiver: latents cross-attend to neuron tokens, pre-LN latent self-attention "
                             "blocks, mean over latents") + " -> residual Mish MLP head"),
        "encoder": args.encoder, "latents": args.latents if args.encoder == "perceiver" else None,
        "latent_heads": args.latent_heads if args.encoder == "perceiver" else None,
        "neuron_descriptors": (["signed-log mean", "log std", "label correlation"]
                               if args.descriptors else None),
        "num_layers_including_embedding_output": n_layers, "hidden_size": hidden_size,
        "pooling_method": "logsumexp1d", "pooled_layers": pooled_layers, "pool_kernel_size": 3,
        "pool_stride": 2, "polysemantic_tokens": hidden_size, "token_length": pooled_layers,
        "positional_embedding": ("learned per neuron" if args.neuron_positions == "learned" else None),
        "model_dim": args.model_dim, "heads": args.heads,
        "transformer_blocks": args.blocks, "ffn_dim": args.ffn_dim, "embedding_dim": args.embedding_dim,
        "readout": "[CLS]" if args.encoder == "full" else "mean over latents",
        "mlp_hidden_dimension": args.hidden_dim, "mlp_hidden_layers": args.hidden_layers,
        "mlp_activation": "mish", "dropout": args.dropout, "feature_clip": args.feature_clip,
        "parameters": parameter_count, "epochs_erm": args.epochs_erm, "batch_size": 128,
        "learning_rate": args.learning_rate, "weight_decay": 1e-4,
        "lr_schedule": ("constant" if args.lr_schedule == "constant" else
                        f"linear warmup ({args.warmup_ratio:.0%}) then cosine decay"), "grad_clip_norm": 1.0,
        "mixed_precision": str(amp_dtype), "classifier_loss": args.loss, "seed": args.seed,
        "bleurt_threshold": args.bleurt_threshold,
        "label_definition": "1 = BLEURT-correct/non-hallucinated; 0 = otherwise",
        "probability_definition": "sigmoid(raw logit) for BLEURT-correct/non-hallucinated",
        "sala_threshold": threshold, "threshold_calibration": calibration,
        "best_source_validation_auroc_during_training": best_auc, "best_epoch": best_epoch,
        "source_test_metrics": binary_metrics(y_test.tolist(), test_predictions.tolist(), test_scores.tolist()),
        "target_metrics": binary_metrics(target_labels.tolist(), target_predictions.tolist(), target_scores.tolist()),
        "target_supervision_used": False,
        "source_counts": {bundle["name"]: {"total": len(bundle["labels"]), "train": len(bundle["train_idx"]),
                                          "validation": len(bundle["val_idx"]), "test": len(bundle["test_idx"])}
                          for bundle in bundles},
    }
    jsonl_dump_atomic(output_dir / "results.jsonl", results)
    csv_dump_atomic(output_dir / "results.csv", results)
    json_dump_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {len(results)} held-out TriviaQA rows to {output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
