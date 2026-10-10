"""Six-source SALA ablation: train the same MLP on concatenated raw Qwen states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

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


SETTING = "G14_6SRC"


def pool_previous_same_index(states, previous_weight: float = 0.25):
    """Pair each even-indexed layer with its predecessor; keep layer zero intact."""
    pooled = states[:, ::2, :].clone()
    if states.shape[1] > 1:
        pooled[:, 1:, :] = ((1.0 - previous_weight) * pooled[:, 1:, :]
                            + previous_weight * states[:, 1:-1:2, :])
    return pooled


def save_checkpoint(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


class RawStateDataset:
    """Index the saved feature memmaps without materializing a concatenated dataset."""

    def __init__(self, bundles: list[dict], split: str):
        self.features = [bundle["features"] for bundle in bundles]
        self.entries = [
            (domain_id, int(row_id), int(bundle["labels"][row_id]))
            for domain_id, bundle in enumerate(bundles)
            for row_id in bundle[f"{split}_idx"]
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, position: int):
        domain_id, row_id, label = self.entries[position]
        vector = np.asarray(self.features[domain_id][row_id], dtype=np.float32)
        return vector.reshape(-1).copy(), np.float32(label)


def fit_layer_pca(bundles: list[dict], n_layers: int, hidden_size: int,
                  components: int, fit_samples: int, seed: int):
    """Fit PCA across layers using only source-training feature observations."""
    if components > n_layers:
        raise ValueError(f"PCA components ({components}) exceed available layers ({n_layers})")
    domain_ids = np.concatenate([
        np.full(len(bundle["train_idx"]), domain_id, dtype=np.int16)
        for domain_id, bundle in enumerate(bundles)
    ])
    row_ids = np.concatenate([bundle["train_idx"] for bundle in bundles])
    rng = np.random.default_rng(seed)
    picked = rng.integers(len(row_ids), size=fit_samples)
    picked_hidden = rng.integers(hidden_size, size=fit_samples)
    observations = np.empty((fit_samples, n_layers), dtype=np.float32)
    for domain_id, bundle in enumerate(bundles):
        mask = domain_ids[picked] == domain_id
        if not mask.any():
            continue
        rows = row_ids[picked[mask]]
        hidden_ids = picked_hidden[mask]
        observations[mask] = np.asarray(bundle["features"][rows, :, hidden_ids], dtype=np.float32)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(observations)
    return (pca.components_.astype(np.float32), pca.mean_.astype(np.float32),
            pca.explained_variance_ratio_.astype(np.float64))


def predict(model, loader, device):
    import torch

    model.eval()
    logits = []
    with torch.inference_mode():
        for xb, _ in loader:
            logits.append(model(xb.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(logits).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    parser.add_argument("--epochs-erm", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=3,
                        help="Number of width-matched hidden linear layers, excluding the output layer")
    parser.add_argument("--pooling", choices=("none", "dwclp", "max1d", "max_mean", "logsumexp1d", "pca_layers", "prev_same_index"), default="none")
    parser.add_argument("--pool-kernel-size", type=int, default=3)
    parser.add_argument("--pool-stride", type=int, default=2)
    parser.add_argument("--dwclp-radius", type=int, default=2)
    parser.add_argument("--dwclp-sigma", type=float, default=1.0)
    parser.add_argument("--previous-layer-weight", type=float, default=0.25)
    parser.add_argument("--pca-components", type=int, default=15,
                        help="Layer-axis PCA components for pca_layers pooling")
    parser.add_argument("--pca-fit-samples", type=int, default=200_000,
                        help="Number of source-train (example, hidden-index) observations used to fit PCA")
    args = parser.parse_args()
    if args.hidden_dim < 1 or args.hidden_layers < 1:
        parser.error("--hidden-dim and --hidden-layers must be positive")
    if args.pooling in ("dwclp", "max1d", "max_mean", "logsumexp1d") and (args.pool_kernel_size != 3 or args.pool_stride != 2):
        parser.error("Pooling comparison requires a 3-layer window and stride 2")
    if args.pooling == "prev_same_index" and args.pool_stride != 2:
        parser.error("Previous-layer pooling requires stride 2")
    if not 0.0 <= args.previous_layer_weight <= 1.0:
        parser.error("--previous-layer-weight must be between 0 and 1")
    if args.dwclp_radius < 1 or args.dwclp_sigma <= 0:
        parser.error("DWCLP radius and sigma must be positive")
    if args.pca_components < 1 or args.pca_fit_samples < 1:
        parser.error("PCA component and fit-sample counts must be positive")

    import torch
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset

    from utils.helpers import seed_everything

    class RawConcatMLP(nn.Module):
        """StrongMLP's residual pattern with a configurable number of hidden layers."""

        def __init__(self, input_dim: int, hidden_dim: int, hidden_layers: int,
                     n_layers: int, hidden_size: int, pooling: str,
                     pca_components=None, pca_mean=None):
            super().__init__()
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.pooling = pooling
            if pooling == "dwclp":
                offsets = torch.arange(-args.dwclp_radius, args.dwclp_radius + 1,
                                       dtype=torch.float32)
                weights = torch.exp(-0.5 * (offsets / args.dwclp_sigma).square())
                self.register_buffer("distance_kernel", (weights / weights.sum()).view(1, 1, -1))
            if pooling == "pca_layers":
                if pca_components is None or pca_mean is None:
                    raise ValueError("PCA pooling requires fitted layer components and mean")
                self.register_buffer("pca_components", torch.from_numpy(pca_components))
                self.register_buffer("pca_mean", torch.from_numpy(pca_mean))
            self.fc_in = nn.Linear(input_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(hidden_dim), nn.Mish(), nn.Dropout(0.2),
                              nn.Linear(hidden_dim, hidden_dim))
                for _ in range(hidden_layers - 1)
            ])
            self.out = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            if self.pooling != "none":
                states = x.reshape(-1, self.n_layers, self.hidden_size)
                if self.pooling == "max1d":
                    states = F.max_pool1d(states.transpose(1, 2), kernel_size=3,
                                          stride=2, padding=1).transpose(1, 2)
                elif self.pooling == "max_mean":
                    channels_first = states.transpose(1, 2)
                    maximum = F.max_pool1d(channels_first, kernel_size=3,
                                           stride=2, padding=1)
                    # count_include_pad=False avoids artificial zero-valued edge layers.
                    average = F.avg_pool1d(channels_first, kernel_size=3,
                                           stride=2, padding=1, count_include_pad=False)
                    states = torch.cat((maximum, average), dim=1).transpose(1, 2)
                elif self.pooling == "logsumexp1d":
                    # Numerically stable soft maximum over each 3-layer window.
                    # Replication gives every output a full three-layer window at the edges.
                    channels_first = F.pad(states.transpose(1, 2), (1, 1), mode="replicate")
                    windows = channels_first.unfold(dimension=2, size=3, step=2)
                    states = torch.logsumexp(windows, dim=-1).transpose(1, 2)
                elif self.pooling == "pca_layers":
                    # PCA operates only over layers; every hidden coordinate shares the fitted basis.
                    states = ((states.transpose(1, 2) - self.pca_mean)
                              @ self.pca_components.T).transpose(1, 2)
                elif self.pooling == "prev_same_index":
                    states = pool_previous_same_index(states, args.previous_layer_weight)
                else:
                    centers = torch.arange(0, self.n_layers, 2, device=x.device)
                    previous = states.index_select(1, (centers - 1).clamp(min=0))
                    following = states.index_select(1, (centers + 1).clamp(max=self.n_layers - 1))
                    neighbors = torch.stack((previous, following), dim=2)
                    flat = neighbors.reshape(-1, 1, self.hidden_size)
                    flat = F.pad(flat, (args.dwclp_radius, args.dwclp_radius),
                                 mode="replicate")
                    smoothed = F.conv1d(flat, self.distance_kernel)
                    smoothed = smoothed.reshape(states.shape[0], len(centers), 2,
                                                self.hidden_size)
                    states = (states.index_select(1, centers) +
                              0.5 * smoothed[:, :, 0] + 0.5 * smoothed[:, :, 1]) / 2.0
                x = states.reshape(states.shape[0], -1)
            h = self.fc_in(x)
            for block in self.blocks:
                h = h + block(h)
            return self.out(h).squeeze(-1)

    class TargetDataset(Dataset):
        def __init__(self, features):
            self.features = features

        def __len__(self):
            return len(self.features)

        def __getitem__(self, position):
            return np.asarray(self.features[position], dtype=np.float32).reshape(-1).copy(), np.float32(0)

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
    pca_components = pca_mean = pca_variance_ratio = None
    if args.pooling == "pca_layers":
        pca_components, pca_mean, pca_variance_ratio = fit_layer_pca(
            bundles, n_layers, hidden_size, args.pca_components, args.pca_fit_samples, args.seed
        )
        print(f"Fitted layer PCA on {args.pca_fit_samples} source-train observations; "
              f"explained variance={pca_variance_ratio.sum():.6f}", flush=True)
    pooled_layers = (args.pca_components if args.pooling == "pca_layers" else
                     n_layers if args.pooling == "none" else (n_layers + 1) // 2)
    output_channels = 2 if args.pooling == "max_mean" else 1
    input_dim = output_channels * pooled_layers * hidden_size
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
          f"pooling={args.pooling}; MLP input={output_channels}*{pooled_layers}*{hidden_size}={input_dim}", flush=True)

    seed_everything(args.seed)
    device = torch.device(args.train_device if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=pin, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, pin_memory=pin, num_workers=0)
    target_loader = DataLoader(TargetDataset(target_features), batch_size=256, shuffle=False,
                               pin_memory=pin, num_workers=0)
    model = RawConcatMLP(input_dim, args.hidden_dim, args.hidden_layers,
                         n_layers, hidden_size, args.pooling,
                         pca_components=pca_components, pca_mean=pca_mean).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    checkpoint = output_dir / "training_checkpoint.pt"
    start_epoch, best_auc, best_epoch = 0, 0.0, None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved["input_dim"] != input_dim or saved["seed"] != args.seed
                or saved["hidden_dim"] != args.hidden_dim
                or saved["hidden_layers"] != args.hidden_layers
                or saved.get("pooling", "none") != args.pooling
                or saved.get("previous_layer_weight", args.previous_layer_weight)
                != args.previous_layer_weight
                or saved.get("dwclp_radius", args.dwclp_radius) != args.dwclp_radius
                or saved.get("dwclp_sigma", args.dwclp_sigma) != args.dwclp_sigma):
            raise ValueError("Checkpoint configuration mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["next_epoch"]
        best_auc, best_epoch = saved["best_auc"], saved["best_epoch"]
        print(f"Resuming from epoch {start_epoch + 1}", flush=True)
    best_path = output_dir / "best_mlp.pt"

    for epoch in range(start_epoch, args.epochs_erm):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
        val_logits = predict(model, val_loader, device)
        val_auc = float(roc_auc_score(y_val, val_logits))
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
            save_checkpoint(best_path, {"input_dim": input_dim, "hidden_dim": args.hidden_dim,
                                        "hidden_layers": args.hidden_layers, "pooling": args.pooling,
                                        "model": model.state_dict(),
                                        "best_epoch": best_epoch, "best_auc": best_auc})
        save_checkpoint(checkpoint, {"input_dim": input_dim, "seed": args.seed,
                                     "hidden_dim": args.hidden_dim,
                                     "hidden_layers": args.hidden_layers,
                                     "pooling": args.pooling,
                                     "previous_layer_weight": args.previous_layer_weight,
                                     "dwclp_radius": args.dwclp_radius,
                                     "dwclp_sigma": args.dwclp_sigma,
                                     "next_epoch": epoch + 1, "model": model.state_dict(),
                                     "optimizer": optimizer.state_dict(), "best_auc": best_auc,
                                     "best_epoch": best_epoch})
        print(f"epoch {epoch + 1}/{args.epochs_erm} loss={total_loss / len(train_ds):.5f} "
              f"source_val_auc={val_auc:.5f} best={best_auc:.5f}", flush=True)
    if not best_path.exists():
        raise RuntimeError("No valid best model was saved")
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    val_scores = 1 / (1 + np.exp(-np.clip(predict(model, val_loader, device), -80, 80)))
    test_scores = 1 / (1 + np.exp(-np.clip(predict(model, test_loader, device), -80, 80)))
    target_logits = predict(model, target_loader, device)
    target_scores = 1 / (1 + np.exp(-np.clip(target_logits, -80, 80)))
    threshold, calibration = calibrate_threshold(val_scores.tolist(), y_val.tolist())
    test_predictions = (test_scores >= threshold).astype(np.int64)
    target_predictions = (target_scores >= threshold).astype(np.int64)
    bleurt = {str(row["id"]): row for row in
              read_jsonl(artifact_dir / "bleurt" / target_name / "scores.jsonl")}
    results = []
    for row, label, logit, score, predicted in zip(
        target_rows, target_labels, target_logits, target_scores, target_predictions
    ):
        judged = bleurt[str(row["id"])]
        results.append({**row, "setting": "G14_6SRC_RAW_CONCAT", "source_domains": source_names,
                        "pooling_method": args.pooling, "pooled_layers": pooled_layers,
                        "pca_components": args.pca_components if args.pooling == "pca_layers" else None,
                        "mlp_hidden_dimension": args.hidden_dim,
                        "mlp_hidden_layers": args.hidden_layers,
                        "target_domain": target_name, "ground_truth_text": judged["ground_truth_text"],
                        "ground_truth_label": int(label), "bleurt_score": float(judged["bleurt_score"]),
                        "raw_logit": float(logit), "sala_probability": float(score),
                        "predicted_label": int(predicted), "classifier_loss": "bce",
                        "sala_threshold": threshold, "split": "target_evaluation"})
    summary = {
        "setting": "G14_6SRC_RAW_CONCAT", "source_domains": source_names,
        "target_domain": target_name, "model": "Qwen/Qwen2.5-7B-Instruct",
        "feature_policy": (
            "flatten and concatenate all saved answer-token hidden states; no projection or scaling"
            if args.pooling == "none" else
            ("fit source-train-only PCA across the layer axis, transform every hidden coordinate, then flatten"
             if args.pooling == "pca_layers" else
             "pool saved answer-token hidden states across layers, then flatten; no learned projection")
        ),
        "num_layers_including_embedding_output": n_layers, "hidden_size": hidden_size,
        "pooling_method": args.pooling, "pooled_layers": pooled_layers,
        "pooling_output_channels": output_channels,
        "pool_kernel_size": (2 if args.pooling == "prev_same_index" else
                             args.pool_kernel_size if args.pooling != "none" else None),
        "pool_stride": args.pool_stride if args.pooling != "none" else None,
        "dwclp_radius": args.dwclp_radius if args.pooling == "dwclp" else None,
        "dwclp_sigma": args.dwclp_sigma if args.pooling == "dwclp" else None,
        "dwclp_neighbor_weight": 0.5 if args.pooling == "dwclp" else None,
        "previous_layer_weight": args.previous_layer_weight if args.pooling == "prev_same_index" else None,
        "pca_fit_samples": args.pca_fit_samples if args.pooling == "pca_layers" else None,
        "pca_explained_variance_ratio": (pca_variance_ratio.tolist()
                                          if args.pooling == "pca_layers" else None),
        "pca_explained_variance_total": (float(pca_variance_ratio.sum())
                                           if args.pooling == "pca_layers" else None),
        "mlp_input_dimension": input_dim, "mlp_hidden_dimension": args.hidden_dim,
        "mlp_hidden_layers": args.hidden_layers,
        "mlp_activation": "mish", "mlp_dropout": 0.2,
        "epochs_erm": args.epochs_erm, "batch_size": 128, "learning_rate": 1e-3,
        "weight_decay": 1e-4, "classifier_loss": "bce", "seed": args.seed,
        "bleurt_threshold": args.bleurt_threshold,
        "label_definition": "1 = BLEURT-correct/non-hallucinated; 0 = otherwise",
        "probability_definition": "sigmoid(raw MLP logit) for BLEURT-correct/non-hallucinated",
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
