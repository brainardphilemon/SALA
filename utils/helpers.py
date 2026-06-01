# utils/helpers.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from models.classifiers import StrongMLP
from models.projections import train_layerwise_invariant_projection, compute_isr_init_W


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def build_layerwise_matrix(embeds: np.ndarray, rep: str = "hidden") -> np.ndarray:
    assert rep in ["hidden", "residual"]
    if rep == "hidden":
        return embeds
    else:
        return embeds[:, 1:, :] - embeds[:, :-1, :]

def _parse_int_list(s: str):
    if s is None: return []
    return [int(chunk.strip()) for chunk in s.replace(";", ",").replace(" ", ",").split(",") if chunk.strip()]

def _parse_proj_hparam_candidates(s: str):
    if s is None: return []
    out = []
    for item in s.replace(";", ",").split(","):
        item = item.strip()
        if not item: continue
        chunks = [c.strip() for c in item.split(":")]
        lam_inv, lam_sep, lam_reg = map(float, chunks)
        out.append({"lam_inv": lam_inv, "lam_sep": lam_sep, "lam_reg": lam_reg})
    return out

def _mean_std_from_folds(folds):
    vals = [float(x) for x in folds if x == x]
    if len(vals) == 0: return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


@torch.no_grad()
def eval_auc_loader(model, loader, device):
    model.eval()
    preds, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float()
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        preds.append(probs)
        labels.append(yb.detach().cpu().numpy().ravel())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    if len(np.unique(labels)) < 2: return float("nan")

    auc1 = roc_auc_score(labels, preds)
    auc2 = roc_auc_score(labels, 1.0 - preds)
    return float(max(auc1, auc2))

def eval_auc_on_numpy(model, X_np, y_np, device, batch_size=256):
    ds = TensorDataset(torch.from_numpy(X_np).float(), torch.from_numpy(y_np).float())
    is_cuda = device.type == "cuda"
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, 
        pin_memory=is_cuda, num_workers=(2 if is_cuda else 0), persistent_workers=is_cuda
    )
    return eval_auc_loader(model, loader, device)

def train_erm(
    Xtr: np.ndarray, y_tr: np.ndarray,
    Xval: np.ndarray, y_val: np.ndarray,
    device, epochs: int = 50, batch_size: int = 128,
    lr: float = 1e-3, wd: float = 1e-4, verbose: bool = True,
):
    Xtr_t = torch.from_numpy(Xtr).float()
    Xval_t = torch.from_numpy(Xval).float()
    ytr_t = torch.from_numpy(y_tr).float()
    yval_t = torch.from_numpy(y_val).float()

    tr_ds = TensorDataset(Xtr_t, ytr_t)
    va_ds = TensorDataset(Xval_t, yval_t)

    is_cuda = device.type == "cuda"
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, pin_memory=is_cuda, num_workers=(4 if is_cuda else 0), persistent_workers=is_cuda)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, pin_memory=is_cuda, num_workers=(2 if is_cuda else 0), persistent_workers=is_cuda)

    model = StrongMLP(input_dim=Xtr.shape[1], act='mish').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()
            loss = bce(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_auc = eval_auc_loader(model, va_loader, device)
        if verbose: print(f"    [ERM] ep {ep+1}/{epochs} | src-val AUC={val_auc:.4f}")
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None: model.load_state_dict(best_state)
    return model, best_auc

def select_best_proj_dim_lodo(
    X_src_tr_std: np.ndarray, y_src_tr: np.ndarray, dom_src_tr: np.ndarray,
    X_src_va_std: np.ndarray, y_src_va: np.ndarray, dom_src_va: np.ndarray,
    proj_dim_candidates, layer, args, device,
    lam_inv=1.0, lam_sep=0.1, lam_reg=0.0,
    epochs_proj: int = 120, epochs_probe: int = 25,
    lr_proj: float = 1e-3, lr_probe: float = 1e-3, wd_probe: float = 1e-4,
):
    stats = {}
    best_dim, best_score = None, -1e9
    lambda_var = 2.0
    uniq = np.unique(np.concatenate([dom_src_tr, dom_src_va], axis=0))

    for k in proj_dim_candidates:
        fold_scores = []
        for e in uniq:
            idx_tr = (dom_src_tr != e)
            idx_va = (dom_src_tr == e)
            if idx_tr.sum() == 0 or idx_va.sum() == 0: continue
            
            seed_everything(args.seed + 1000 + (layer + 1) * 10000)
            W = train_layerwise_invariant_projection(
                X_src_tr_std[idx_tr], y_src_tr[idx_tr], dom_src_tr[idx_tr],
                proj_dim=int(k), n_epochs=epochs_proj, lr=lr_proj,
                init_W=None, device=device, verbose=False,
                lam_inv=lam_inv, lam_sep=lam_sep, lam_reg=lam_reg,
            )

            Z_tr = X_src_tr_std[idx_tr] @ W
            Z_va = X_src_tr_std[idx_va] @ W

            _, auc_e = train_erm(
                Z_tr, y_src_tr[idx_tr], Z_va, y_src_tr[idx_va],
                device=device, epochs=epochs_probe, batch_size=512,
                lr=lr_probe, wd=wd_probe, verbose=False,
            )
            fold_scores.append(float(auc_e))

        mean_auc, std_auc = _mean_std_from_folds(fold_scores)
        worst_auc = float(np.min(fold_scores)) if len(fold_scores) else float("nan")
        score = (mean_auc - lambda_var * std_auc) if (mean_auc == mean_auc and std_auc == std_auc) else float("nan")

        stats[int(k)] = {"mean": mean_auc, "std": std_auc, "worst": worst_auc, "mv_score": score, "folds": fold_scores}

        if score == score and score > best_score:
            best_score = score
            best_dim = int(k)
        elif score == score and best_dim is not None and score == best_score:
            if mean_auc == mean_auc and stats[best_dim]["mean"] == stats[best_dim]["mean"] and mean_auc > stats[best_dim]["mean"]:
                best_dim = int(k)

    return best_dim, best_score, stats