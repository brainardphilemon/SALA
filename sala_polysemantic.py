"""Six-source SALA ablation: polysemantic-token transformer over log-sum-exp pooled Qwen states.

The saved answer-token states (29 layers x hidden size) are log-sum-exp pooled over 3-layer,
stride-2 windows exactly as in `sala_raw_concat.py --pooling logsumexp1d`, giving a 15 x hidden
matrix. Each hidden unit then becomes one "polysemantic token": its 15-value trajectory across the
pooled layers, standardized with source-train statistics, embedded by a shared Linear(15 -> d) plus
a learned per-neuron positional embedding. A [CLS] token and the hidden-size neuron tokens pass
through pre-LN transformer blocks; [CLS] is projected to a constant 512-d embedding, which feeds the
same residual Mish MLP head as the 4 x 128 raw-concat runs. Optimizer, batch size, loss, split,
checkpoint selection and threshold calibration match sala_raw_concat.py.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=512,
                        help="Constant output embedding fed to the MLP head")
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
    if args.model_dim % args.heads:
        parser.error("--model-dim must be divisible by --heads")

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

    # Per-coordinate (layer window x neuron) standardization, fitted on source-train rows only.
    total = torch.zeros(pooled_layers, hidden_size, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    with torch.inference_mode():
        for xb, _ in DataLoader(train_ds, batch_size=256, shuffle=False, **loader_kwargs):
            pooled = pool(xb.to(device)).cpu().double()
            total += pooled.sum(0)
            total_sq += pooled.square().sum(0)
    feature_mean = (total / len(train_ds)).float()
    feature_std = (total_sq / len(train_ds) - (total / len(train_ds)).square()).clamp(min=0).sqrt()
    feature_std = feature_std.float().clamp(min=1e-4)
    print("Fitted per-coordinate standardization on source-train rows", flush=True)

    class PolysemanticTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            d = args.model_dim
            self.register_buffer("feature_mean", feature_mean.clone())
            self.register_buffer("feature_std", feature_std.clone())
            # One token per hidden unit: its pooled-layer trajectory (length 15) -> d.
            self.token_embed = nn.Linear(pooled_layers, d)
            self.neuron_pos = nn.Parameter(torch.zeros(1, hidden_size, d))
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.input_dropout = nn.Dropout(args.dropout)
            layer = nn.TransformerEncoderLayer(d, args.heads, args.ffn_dim, args.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.blocks = nn.TransformerEncoder(layer, args.blocks, norm=nn.LayerNorm(d),
                                                enable_nested_tensor=False)
            self.to_embedding = nn.Linear(d, args.embedding_dim)
            # MLP head: same residual pattern as RawConcatMLP in sala_raw_concat.py.
            self.fc_in = nn.Linear(args.embedding_dim, args.hidden_dim)
            self.head_blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(args.hidden_dim), nn.Mish(), nn.Dropout(args.dropout),
                              nn.Linear(args.hidden_dim, args.hidden_dim))
                for _ in range(args.hidden_layers - 1)
            ])
            self.out = nn.Linear(args.hidden_dim, 1)
            nn.init.trunc_normal_(self.neuron_pos, std=0.02)
            nn.init.trunc_normal_(self.cls, std=0.02)

        def embed(self, flat):
            """Constant-size embedding: (batch, n_layers * hidden) -> (batch, embedding_dim)."""
            pooled = ((pool(flat) - self.feature_mean) / self.feature_std).clamp(
                -args.feature_clip, args.feature_clip)
            tokens = self.token_embed(pooled.transpose(1, 2)) + self.neuron_pos
            sequence = torch.cat((self.cls.expand(len(tokens), -1, -1), tokens), dim=1)
            hidden = self.blocks(self.input_dropout(sequence))
            return self.to_embedding(hidden[:, 0])

        def forward(self, flat):
            h = self.fc_in(self.embed(flat))
            for block in self.head_blocks:
                h = h + block(h)
            return self.out(h).squeeze(-1).float()

    model = PolysemanticTransformer().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

    print(f"PolysemanticTransformer parameters={parameter_count:,}; tokens per example={hidden_size + 1}; "
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
        "architecture": ("hidden-unit tokens (pooled-layer trajectory -> Linear(15, d) + learned per-neuron "
                         "position) + [CLS] -> pre-LN transformer blocks -> [CLS] -> Linear(d, 512) -> "
                         "residual Mish MLP head"),
        "num_layers_including_embedding_output": n_layers, "hidden_size": hidden_size,
        "pooling_method": "logsumexp1d", "pooled_layers": pooled_layers, "pool_kernel_size": 3,
        "pool_stride": 2, "polysemantic_tokens": hidden_size, "token_length": pooled_layers,
        "positional_embedding": "learned per neuron", "model_dim": args.model_dim, "heads": args.heads,
        "transformer_blocks": args.blocks, "ffn_dim": args.ffn_dim, "embedding_dim": args.embedding_dim,
        "readout": "[CLS]", "mlp_hidden_dimension": args.hidden_dim, "mlp_hidden_layers": args.hidden_layers,
        "mlp_activation": "mish", "dropout": args.dropout, "feature_clip": args.feature_clip,
        "parameters": parameter_count, "epochs_erm": args.epochs_erm, "batch_size": 128,
        "learning_rate": 1e-3, "weight_decay": 1e-4, "lr_schedule": "constant", "grad_clip_norm": 1.0,
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
