"""Six-source SALA ablation: LLaVA-style fusion of pooled Qwen states with question/answer text.

The log-sum-exp pooled layer states play the role of image patches: each pooled layer vector is
standardized with source-train statistics, mapped by a two-layer MLP projector into the embedding
space of a small pretrained language model, and placed in the sequence ahead of the question and
generated answer. The language model is fine-tuned end to end and a linear head on the final
token's hidden state predicts whether the answer is BLEURT-correct (non-hallucinated).
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


DEFAULT_LM = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LM_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PREFIX_TEXT = "Hidden states of the answering model:"
# Tokenization happens before DataLoader workers fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def logsumexp_pool(states):
    """Same pooling as sala_raw_concat --pooling logsumexp1d: soft max over 3-layer, stride-2 windows."""
    import torch
    from torch.nn import functional as F

    channels_first = F.pad(states.transpose(1, 2), (1, 1), mode="replicate")
    windows = channels_first.unfold(dimension=2, size=3, step=2)
    return torch.logsumexp(windows, dim=-1).transpose(1, 2)


def suffix_text(row: dict) -> str:
    return (f"\nQuestion: {row['input_text']}\nAnswer: {row['output_text']}\n"
            "Is the answer correct?")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--lm-model", default=DEFAULT_LM)
    parser.add_argument("--lm-revision", default=DEFAULT_LM_REVISION)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lm-lr", type=float, default=2e-5)
    parser.add_argument("--projector-lr", type=float, default=1e-4,
                        help="Learning rate for the projector and classification head")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--feature-clip", type=float, default=10.0,
                        help="Clamp standardized pooled states to +/- this value")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="Stop each epoch after this many steps (0 = full epoch); for smoke tests")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    import torch
    from sklearn.metrics import roc_auc_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

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

    tokenizer = AutoTokenizer.from_pretrained(args.lm_model, revision=args.lm_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prefix_ids = tokenizer(PREFIX_TEXT, add_special_tokens=False)["input_ids"]
    truncated = 0
    for bundle in (*bundles, target_bundle):
        encoded = tokenizer([suffix_text(row) for row in bundle["rows"]], add_special_tokens=False)
        bundle["token_ids"] = []
        for ids in encoded["input_ids"]:
            truncated += len(ids) > args.max_text_tokens
            bundle["token_ids"].append(ids[:args.max_text_tokens])

    class FusionDataset(Dataset):
        """Raw saved states plus tokenized question/answer text for one split."""

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
            states = np.asarray(bundle["features"][row_id], dtype=np.float16).copy()
            return states, bundle["token_ids"][row_id], np.float32(label)

    def collate(batch):
        states = torch.from_numpy(np.stack([item[0] for item in batch]))
        lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
        token_ids = torch.full((len(batch), int(lengths.max())), tokenizer.pad_token_id, dtype=torch.long)
        for position, item in enumerate(batch):
            token_ids[position, :len(item[1])] = torch.tensor(item[1], dtype=torch.long)
        labels = torch.tensor([item[2] for item in batch])
        return states, token_ids, lengths, labels

    train_ds = FusionDataset(bundles, "train")
    val_ds = FusionDataset(bundles, "val")
    test_ds = FusionDataset(bundles, "test")
    target_ds = FusionDataset([target_bundle], "all")
    y_train = np.asarray([entry[2] for entry in train_ds.entries], dtype=np.int64)
    y_val = np.asarray([entry[2] for entry in val_ds.entries], dtype=np.int64)
    y_test = np.asarray([entry[2] for entry in test_ds.entries], dtype=np.int64)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Source splits must contain both BLEURT classes")
    print(f"Sources={source_names}; target={target_name}; source train/val/test="
          f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}; target={len(target_ds)}; "
          f"pooled layer tokens={pooled_layers}x{hidden_size}; text tokens truncated={truncated}",
          flush=True)

    seed_everything(args.seed)
    device = torch.device(args.train_device if torch.cuda.is_available() or args.train_device == "mps"
                          else "cpu")
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
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)
    target_loader = DataLoader(target_ds, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)

    # Per-coordinate standardization of pooled states, fitted on source-train rows only.
    # Accumulate in float64 on the CPU (MPS has no float64).
    total = torch.zeros(pooled_layers, hidden_size, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    with torch.inference_mode():
        for states, _, _, _ in DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False,
                                          **loader_kwargs):
            pooled = logsumexp_pool(states.to(device).float()).cpu().double()
            total += pooled.sum(0)
            total_sq += pooled.square().sum(0)
    feature_mean = (total / len(train_ds)).float().to(device)
    feature_std = (total_sq / len(train_ds) - (total / len(train_ds)).square()).clamp(min=0).sqrt()
    feature_std = feature_std.float().to(device)
    feature_std = feature_std.clamp(min=1e-4)
    print("Fitted pooled-state standardization on source-train rows", flush=True)

    class HiddenStateVLM(nn.Module):
        """Pooled layer states as soft tokens in front of the text, read out at the last token."""

        def __init__(self):
            super().__init__()
            self.lm = AutoModel.from_pretrained(args.lm_model, revision=args.lm_revision,
                                                torch_dtype=torch.float32)
            # Recompute LM activations in backward and keep the vocabulary embeddings frozen so
            # full fine-tuning fits a 16 GB T4.
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.lm.get_input_embeddings().weight.requires_grad_(False)
            width = self.lm.config.hidden_size
            self.register_buffer("feature_mean", feature_mean.clone())
            self.register_buffer("feature_std", feature_std.clone())
            self.register_buffer("prefix_ids", torch.tensor(prefix_ids, dtype=torch.long))
            self.projector = nn.Sequential(nn.Linear(hidden_size, width), nn.GELU(),
                                           nn.Linear(width, width))
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(width, 1))

        def forward(self, states, token_ids, lengths):
            pooled = logsumexp_pool(states.float())
            pooled = ((pooled - self.feature_mean) / self.feature_std).clamp(-args.feature_clip,
                                                                             args.feature_clip)
            embed = self.lm.get_input_embeddings()
            batch = states.shape[0]
            prefix = embed(self.prefix_ids).unsqueeze(0).expand(batch, -1, -1)
            layer_tokens = self.projector(pooled)
            text = embed(token_ids)
            inputs = torch.cat((prefix, layer_tokens.to(text.dtype), text), dim=1)
            fixed = prefix.shape[1] + layer_tokens.shape[1]
            positions = torch.arange(inputs.shape[1], device=inputs.device)
            attention_mask = (positions.unsqueeze(0) < (fixed + lengths).unsqueeze(1)).long()
            hidden = self.lm(inputs_embeds=inputs, attention_mask=attention_mask,
                             use_cache=False).last_hidden_state
            last = hidden[torch.arange(batch, device=hidden.device), fixed + lengths - 1]
            return self.head(last).squeeze(-1).float()

    model = HiddenStateVLM().to(device)
    lm_params = [param for param in model.lm.parameters() if param.requires_grad]
    new_params = list(model.projector.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lm_params, "lr": args.lm_lr},
        {"params": new_params, "lr": args.projector_lr},
    ], weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader) if not args.max_train_steps else min(len(train_loader),
                                                                            args.max_train_steps)
    total_steps = steps_per_epoch * args.epochs
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
            for states, token_ids, lengths, _ in loader:
                logits.append(model(states.to(device, non_blocking=True),
                                    token_ids.to(device, non_blocking=True),
                                    lengths.to(device, non_blocking=True)).float().cpu().numpy())
        return np.concatenate(logits).reshape(-1)

    print(f"LM={args.lm_model}@{args.lm_revision}; amp={amp_dtype}; steps/epoch={steps_per_epoch}; "
          f"epochs={args.epochs}", flush=True)
    best_auc, best_epoch, best_state = 0.0, None, None
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for step, (states, token_ids, lengths, labels) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            states = states.to(device, non_blocking=True)
            token_ids = token_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                logits = model(states, token_ids, lengths)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.item()) * len(labels)
            seen += len(labels)
            if (step + 1) % 50 == 0:
                peak = (f" peak_mem={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB"
                        if device.type == "cuda" else "")
                print(f"epoch {epoch + 1} step {step + 1}/{steps_per_epoch} "
                      f"loss={running / seen:.5f}{peak}", flush=True)
        val_auc = float(roc_auc_score(y_val, predict(val_loader)))
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{args.epochs} loss={running / max(seen, 1):.5f} "
              f"source_val_auc={val_auc:.5f} best={best_auc:.5f}", flush=True)
    if best_state is None:
        raise RuntimeError("No valid best model was found")
    model.load_state_dict(best_state)
    torch.save({"projector": model.projector.state_dict(), "head": model.head.state_dict(),
                "feature_mean": feature_mean.cpu(), "feature_std": feature_std.cpu(),
                "best_epoch": best_epoch, "best_auc": best_auc},
               output_dir / "best_projector_head.pt")

    val_scores = 1 / (1 + np.exp(-np.clip(predict(val_loader), -80, 80)))
    test_scores = 1 / (1 + np.exp(-np.clip(predict(test_loader), -80, 80)))
    target_logits = predict(target_loader)
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
        results.append({**row, "setting": "G14_6SRC_VLM_FUSION", "source_domains": source_names,
                        "pooling_method": "logsumexp1d", "pooled_layers": pooled_layers,
                        "fusion_lm": args.lm_model,
                        "target_domain": target_name, "ground_truth_text": judged["ground_truth_text"],
                        "ground_truth_label": int(label), "bleurt_score": float(judged["bleurt_score"]),
                        "raw_logit": float(logit), "sala_probability": float(score),
                        "predicted_label": int(predicted), "classifier_loss": args.loss,
                        "sala_threshold": threshold, "split": "target_evaluation"})
    summary = {
        "setting": "G14_6SRC_VLM_FUSION", "source_domains": source_names,
        "target_domain": target_name, "model": "Qwen/Qwen2.5-7B-Instruct",
        "feature_policy": ("log-sum-exp pool saved answer-token hidden states over 3-layer, stride-2 "
                           "windows; standardize per coordinate with source-train statistics; project "
                           "each pooled layer to one LM soft token"),
        "architecture": ("LLaVA-style: [prefix text][pooled-layer soft tokens][question + generated "
                         "answer text] -> fine-tuned causal LM -> linear head on last token"),
        "text_template": PREFIX_TEXT + " <layer tokens>" + suffix_text(
            {"input_text": "{input_text}", "output_text": "{output_text}"}),
        "fusion_lm": args.lm_model, "fusion_lm_revision": args.lm_revision,
        "projector": "Linear(hidden_size, lm_width) -> GELU -> Linear(lm_width, lm_width)",
        "num_layers_including_embedding_output": n_layers, "hidden_size": hidden_size,
        "pooling_method": "logsumexp1d", "pooled_layers": pooled_layers,
        "pool_kernel_size": 3, "pool_stride": 2, "feature_clip": args.feature_clip,
        "max_text_tokens": args.max_text_tokens, "text_rows_truncated": truncated,
        "epochs": args.epochs, "steps_per_epoch": steps_per_epoch, "batch_size": args.batch_size,
        "lm_learning_rate": args.lm_lr, "projector_learning_rate": args.projector_lr,
        "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
        "lr_schedule": "linear warmup then cosine decay", "grad_clip_norm": 1.0,
        "lm_gradient_checkpointing": True, "lm_input_embeddings_frozen": True,
        "mixed_precision": str(amp_dtype), "classifier_loss": args.loss,
        "focal_alpha": args.focal_alpha if args.loss == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None, "seed": args.seed,
        "bleurt_threshold": args.bleurt_threshold,
        "label_definition": "1 = BLEURT-correct/non-hallucinated; 0 = otherwise",
        "probability_definition": "sigmoid(raw fusion logit) for BLEURT-correct/non-hallucinated",
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
