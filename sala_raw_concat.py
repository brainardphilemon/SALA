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
    args = parser.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from models.classifiers import StrongMLP
    from utils.helpers import seed_everything

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
    input_dim = n_layers * hidden_size
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
          f"raw MLP input={n_layers}*{hidden_size}={input_dim}", flush=True)

    seed_everything(args.seed)
    device = torch.device(args.train_device if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, pin_memory=pin, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, pin_memory=pin, num_workers=0)
    target_loader = DataLoader(TargetDataset(target_features), batch_size=256, shuffle=False,
                               pin_memory=pin, num_workers=0)
    model = StrongMLP(input_dim=input_dim, hidden_dim=1024, dropout=0.2, act="mish").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    checkpoint = output_dir / "training_checkpoint.pt"
    start_epoch, best_auc, best_epoch = 0, 0.0, None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["input_dim"] != input_dim or saved["seed"] != args.seed:
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
            save_checkpoint(best_path, {"input_dim": input_dim, "model": model.state_dict(),
                                        "best_epoch": best_epoch, "best_auc": best_auc})
        save_checkpoint(checkpoint, {"input_dim": input_dim, "seed": args.seed,
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
                        "target_domain": target_name, "ground_truth_text": judged["ground_truth_text"],
                        "ground_truth_label": int(label), "bleurt_score": float(judged["bleurt_score"]),
                        "raw_logit": float(logit), "sala_probability": float(score),
                        "predicted_label": int(predicted), "classifier_loss": "bce",
                        "sala_threshold": threshold, "split": "target_evaluation"})
    summary = {
        "setting": "G14_6SRC_RAW_CONCAT", "source_domains": source_names,
        "target_domain": target_name, "model": "Qwen/Qwen2.5-7B-Instruct",
        "feature_policy": "flatten and concatenate all saved answer-token hidden states; no projection or scaling",
        "num_layers_including_embedding_output": n_layers, "hidden_size": hidden_size,
        "mlp_input_dimension": input_dim, "mlp_hidden_dimension": 1024,
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
