# ==============================================================================
# 1. IMPORTS & CONFIGURATIONS
# ==============================================================================
import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

# --- Local Modules ---
from configs.mappings import generalization_mapping
from utils.helpers import (
    seed_everything, 
    build_layerwise_matrix, 
    _parse_int_list, 
    _parse_proj_hparam_candidates, 
    train_erm, 
    eval_auc_on_numpy, 
    select_best_proj_dim_lodo
)
from models.projections import (
    compute_isr_init_W, 
    train_layerwise_invariant_projection
)

# ==============================================================================
# 2. MAIN EXPERIMENT PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Layer-wise Invariant Subspace Projection for Hallucination Detection")
    
    # --- Core Paths & Settings ---
    parser.add_argument("--data_dir", type=str, default="./data", help="Base directory containing extracted embeddings and labels")
    parser.add_argument("--model_name", type=str, default="Llama3.1-8B")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--rep", type=str, default="hidden", choices=["hidden", "residual"], help="Per-layer representation for ISR/ERM")
    
    # --- Generalization Experiment Settings ---
    parser.add_argument("--gengeneralize_exp", type=str, nargs="+", default=["G13"], help="Generalization setting (e.g., G1 G5 G13)")
    
    # --- Projection & Network Hyperparameters ---
    parser.add_argument("--proj_dim", type=int, default=32, help="Per-layer invariant subspace dim for ISR")
    parser.add_argument("--proj_dim_candidates", type=str, default="8,16,32,64,128,256,512", help="Candidate proj_dim list for LODO")
    parser.add_argument("--proj_hparam_candidates", type=str, default="1.0:0.4:0.1,1.0:0.8:0.0", help="lam_inv:lam_sep:lam_reg candidates")
    parser.add_argument("--proj_epoch_candidates", type=str, default="40", help="n_epochs candidates for training projection")
    
    # --- Training Epochs ---
    parser.add_argument("--epochs_proj_sel", type=int, default=40, help="Projection training epochs during LODO dim selection")
    parser.add_argument("--epochs_probe_sel", type=int, default=40, help="Probe/ERM epochs during LODO dim selection")
    parser.add_argument("--epochs_erm", type=int, default=40, help="ERM epochs per layer")
    
    # --- Switches ---
    g_lodo = parser.add_mutually_exclusive_group()
    g_lodo.add_argument("--use_lodo_dim", dest="use_lodo_dim", action="store_true", help="Select proj_dim via LODO")
    g_lodo.add_argument("--no_lodo_dim", dest="use_lodo_dim", action="store_false", help="Disable LODO dim selection")
    parser.set_defaults(use_lodo_dim=True)

    args = parser.parse_args()

    proj_dim_candidates = _parse_int_list(args.proj_dim_candidates)
    if args.use_lodo_dim and len(proj_dim_candidates) == 0:
        raise ValueError("--use_lodo_dim is on but --proj_dim_candidates is empty")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # ==========================================================================
    # 3. DATA LOADING & SPLITTING
    # ==========================================================================
    for generalize in args.gengeneralize_exp:
        print(f"\n==================================================")
        print(f" Generalization setting: {generalize}")
        print(f"==================================================")
        
        train_list = generalization_mapping[generalize]["train"]
        test_list = generalization_mapping[generalize]["test"]
        print(f"[*] Train envs: {train_list}")
        print(f"[*] Test  envs: {test_list}")

        # --- Load Source Domains ---
        gt_label_test, gt_label_train, gt_label_val = [], [], []
        origin_embed_train, origin_embed_eval, origin_embed_test = [], [], []
        src_dom_train, src_dom_val, src_dom_test = [], [], []
        dom_id_map = {name: i for i, name in enumerate(train_list)}

        for train_data in train_list:
            embed_path = os.path.join(args.data_dir, f"save_for_eval/{train_data}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy")
            label_path = os.path.join(args.data_dir, f"ml_{train_data}_{args.model_name}_bleurt_score.npy")
            
            embed_curr = np.load(embed_path, allow_pickle=True)
            gts = np.load(label_path)

            thres_gt = 0.5 if train_data in ["tqa", "triviaqa"] else 0.2
            gt_label_curr = np.asarray(gts <= thres_gt, dtype=np.int32) if train_data == "tqa" else np.asarray(gts > thres_gt, dtype=np.int32)

            length = len(gt_label_curr)
            idx = np.arange(length)

            sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            trainval_idx, test_idx = next(sss_outer.split(idx, gt_label_curr))

            sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=43)
            inner_train_pos, inner_val_pos = next(sss_inner.split(trainval_idx, gt_label_curr[trainval_idx]))

            train_idx = trainval_idx[inner_train_pos]
            val_idx   = trainval_idx[inner_val_pos]
            
            for i in range(length):
                if i in test_idx:
                    gt_label_test.extend(gt_label_curr[i:i+1])
                    origin_embed_test.extend(embed_curr[i:i+1])
                    src_dom_test.append(dom_id_map[train_data])
                elif i in val_idx:
                    gt_label_val.extend(gt_label_curr[i:i+1])
                    origin_embed_eval.extend(embed_curr[i:i+1])
                    src_dom_val.append(dom_id_map[train_data])
                elif i in train_idx:
                    gt_label_train.extend(gt_label_curr[i:i+1])
                    origin_embed_train.extend(embed_curr[i:i+1])
                    src_dom_train.append(dom_id_map[train_data])

        # --- Load Target Domains ---
        tar_label_test, tar_label_val = [], []
        target_embed_eval, target_embed_test = [], []

        for test_data in test_list:
            tar_embed_path = os.path.join(args.data_dir, f"save_for_eval/{test_data}_hal_det/most_likely_{args.model_name}_gene_embeddings_layer_wise.npy")
            tar_label_path = os.path.join(args.data_dir, f"ml_{test_data}_{args.model_name}_bleurt_score.npy")
            
            tar_embed_curr = np.load(tar_embed_path, allow_pickle=True)
            gts = np.load(tar_label_path)

            thres_gt = 0.5 if test_data in ["tqa", "triviaqa"] else 0.2
            tar_label_curr = np.asarray(gts > thres_gt, dtype=np.int32)
            
            length = len(tar_label_curr)
            idx = np.arange(length)

            tar_sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
            tar_test_idx, tar_val_idx = next(tar_sss_outer.split(idx, tar_label_curr))

            for i in range(length):
                if i in tar_test_idx:
                    tar_label_test.extend(tar_label_curr[i:i+1])
                    target_embed_test.extend(tar_embed_curr[i:i+1])
                elif i in tar_val_idx:
                    tar_label_val.extend(tar_label_curr[i:i+1])
                    target_embed_eval.extend(tar_embed_curr[i:i+1])

        # --- Convert & Format Features ---
        origin_embed_train = build_layerwise_matrix(np.array(origin_embed_train), rep=args.rep)
        origin_embed_eval  = build_layerwise_matrix(np.array(origin_embed_eval), rep=args.rep)
        origin_embed_test  = build_layerwise_matrix(np.array(origin_embed_test), rep=args.rep)
        tar_embed_eval     = build_layerwise_matrix(np.array(target_embed_eval), rep=args.rep)
        tar_embed_test     = build_layerwise_matrix(np.array(target_embed_test), rep=args.rep)

        gt_label_train, gt_label_val, gt_label_test = np.asarray(gt_label_train), np.asarray(gt_label_val), np.asarray(gt_label_test)
        tar_label_val, tar_label_test = np.asarray(tar_label_val), np.asarray(tar_label_test)
        src_dom_train, src_dom_val = np.array(src_dom_train), np.array(src_dom_val)

        num_layers, d = origin_embed_train.shape[1], origin_embed_train.shape[2]
        print(f"[*] src-train={origin_embed_train.shape}, src-val={origin_embed_eval.shape}, src-test={origin_embed_test.shape}")
        print(f"[*] tgt-val={tar_embed_eval.shape}, tgt-test={tar_embed_test.shape}")
        print(f"[*] num_layers={num_layers}, per-layer dim={d}")

        # ==========================================================================
        # 4. LAYER-WISE INVARIANT PROJECTION
        # ==========================================================================
        Z_src_tr_list, Z_src_va_list, Z_src_te_list = [], [], []
        Z_tgt_va_list, Z_tgt_te_list = [], []
        chosen_proj_dims, chosen_proj_epochs, chosen_proj_hparams = [], [], []

        seed_everything(args.seed)
        for l in range(num_layers):
            print(f"\n{'-'*20} Layer {l} {'-'*20}")

            X_src_tr, X_src_va, X_src_te = origin_embed_train[:, l, :], origin_embed_eval[:, l, :], origin_embed_test[:, l, :]
            X_tgt_va, X_tgt_te = tar_embed_eval[:, l, :], tar_embed_test[:, l, :]

            # --- Standardization ---
            mu = X_src_tr.mean(axis=0)
            std = X_src_tr.std(axis=0) + 1e-8
            X_src_tr_std, X_src_va_std, X_src_te_std = (X_src_tr - mu)/std, (X_src_va - mu)/std, (X_src_te - mu)/std
            X_tgt_va_std, X_tgt_te_std = (X_tgt_va - mu)/std, (X_tgt_te - mu)/std

            # ---------------------------------------------------------
            # Baseline-MLP (Raw features) Evaluation
            # ---------------------------------------------------------
            print("  [Baseline-MLP] training on raw std features ...")
            seed_everything(args.seed)
            mdl_raw, best_val_auc_raw = train_erm(
                X_src_tr_std, gt_label_train, X_src_va_std, gt_label_val,
                device=device, epochs=args.epochs_erm, batch_size=512, lr=1e-3, wd=1e-4, verbose=False
            )
            src_test_auc_raw = eval_auc_on_numpy(mdl_raw, X_src_te_std, gt_label_test, device)
            tgt_test_auc_raw = eval_auc_on_numpy(mdl_raw, X_tgt_te_std, tar_label_test, device)
            print(f"  [Baseline-MLP] src-val AUC={best_val_auc_raw:.4f} | src-test AUC={src_test_auc_raw:.4f} | tgt-test AUC={tgt_test_auc_raw:.4f}")

            # --- Projection Hyperparameters ---
            proj_hp_cands = _parse_proj_hparam_candidates(args.proj_hparam_candidates) or [{"lam_inv": 1.0, "lam_sep": 0.4, "lam_reg": 0.1}]
            best_hp = proj_hp_cands[0]
            proj_epoch_cands = _parse_int_list(args.proj_epoch_candidates) or [200]
            dim_inv_l, best_proj_epochs = int(args.proj_dim), int(max(proj_epoch_cands))

            # --- LODO Dimension Selection ---
            if args.use_lodo_dim:
                proj_dim_cands = _parse_int_list(args.proj_dim_candidates) or [args.proj_dim]
                dim_inv_l, mv_score, _ = select_best_proj_dim_lodo(
                    X_src_tr_std, gt_label_train, src_dom_train,
                    X_src_va_std, gt_label_val, src_dom_val,
                    proj_dim_candidates=proj_dim_cands, layer=l, args=args, device=device,
                    lam_inv=best_hp["lam_inv"], lam_sep=best_hp["lam_sep"], lam_reg=best_hp["lam_reg"],
                    epochs_proj=args.epochs_proj_sel, epochs_probe=args.epochs_probe_sel,
                )
                print(f"  [LODO] best (proj_dim)={dim_inv_l} | mv_score={mv_score:.4f}")

            chosen_proj_dims.append(dim_inv_l)
            chosen_proj_epochs.append(best_proj_epochs)
            chosen_proj_hparams.append(best_hp)

            # --- Train Subspace Projection ---
            print(f"  [InvSub] training projection W_l with proj_dim={dim_inv_l} ...")
            seed_everything(args.seed + 1000 + (l + 1) * 10000)
            init_W_isr = compute_isr_init_W(X_src_tr_std, gt_label_train, src_dom_train, proj_dim=dim_inv_l, device=device)
            
            W_l = train_layerwise_invariant_projection(
                X_src_tr_std, gt_label_train, src_dom_train,
                init_W=init_W_isr, proj_dim=dim_inv_l, n_epochs=args.epochs_proj_sel,
                lr=1e-3, lam_inv=best_hp["lam_inv"], lam_sep=best_hp["lam_sep"], lam_reg=best_hp["lam_reg"],
                device=device, verbose=False
            )

            # --- Project Features ---
            Z_src_tr = X_src_tr_std @ W_l
            Z_src_va = X_src_va_std @ W_l
            Z_src_te = X_src_te_std @ W_l
            Z_tgt_va = X_tgt_va_std @ W_l
            Z_tgt_te = X_tgt_te_std @ W_l

            # ---------------------------------------------------------
            # InvSub-MLP (Projected features) Evaluation
            # ---------------------------------------------------------
            print("  [InvSub-MLP] training on projected features ...")
            mdl_inv, best_val_auc_inv = train_erm(
                Z_src_tr, gt_label_train, Z_src_va, gt_label_val,
                device=device, epochs=args.epochs_erm, batch_size=512, lr=1e-3, wd=1e-4, verbose=False
            )
            src_test_auc_inv = eval_auc_on_numpy(mdl_inv, Z_src_te, gt_label_test, device)
            tgt_test_auc_inv = eval_auc_on_numpy(mdl_inv, Z_tgt_te, tar_label_test, device)
            print(f"  [InvSub-MLP] src-val AUC={best_val_auc_inv:.4f} | src-test AUC={src_test_auc_inv:.4f} | tgt-test AUC={tgt_test_auc_inv:.4f}")

            Z_src_tr_list.append(Z_src_tr)
            Z_src_va_list.append(Z_src_va)
            Z_src_te_list.append(Z_src_te)
            Z_tgt_va_list.append(Z_tgt_va)
            Z_tgt_te_list.append(Z_tgt_te)

        # ==========================================================================
        # 5. FEATURE CONCATENATION & FINAL CLASSIFICATION
        # ==========================================================================
        selected_layers = list(range(len(Z_src_tr_list)))

        # Block-wise standardize
        for li in selected_layers:
            mu_z = Z_src_tr_list[li].mean(axis=0, keepdims=True)
            sig_z = Z_src_tr_list[li].std(axis=0, keepdims=True) + 1e-6
            Z_src_tr_list[li] = (Z_src_tr_list[li] - mu_z) / sig_z
            Z_src_va_list[li] = (Z_src_va_list[li] - mu_z) / sig_z
            Z_src_te_list[li] = (Z_src_te_list[li] - mu_z) / sig_z
            Z_tgt_va_list[li] = (Z_tgt_va_list[li] - mu_z) / sig_z
            Z_tgt_te_list[li] = (Z_tgt_te_list[li] - mu_z) / sig_z

        Z_src_tr_concat = np.concatenate([Z_src_tr_list[i] for i in selected_layers], axis=1)
        Z_src_va_concat = np.concatenate([Z_src_va_list[i] for i in selected_layers], axis=1)
        Z_src_te_concat = np.concatenate([Z_src_te_list[i] for i in selected_layers], axis=1)
        Z_tgt_va_concat = np.concatenate([Z_tgt_va_list[i] for i in selected_layers], axis=1)
        Z_tgt_te_concat = np.concatenate([Z_tgt_te_list[i] for i in selected_layers], axis=1)

        print(f"\n[Concat-DICA] final dimension: {Z_src_tr_concat.shape[1]}")

        # --- Final ERM Evaluation ---
        seed_everything(args.seed)
        print("[*] Training final ERM on concatenated ISR features...")
        mdl_concat, best_val_auc_concat = train_erm(
            Z_src_tr_concat, gt_label_train, Z_src_va_concat, gt_label_val,
            device=device, epochs=args.epochs_erm, batch_size=128, lr=1e-3, wd=1e-4
        )

        src_test_auc_concat = eval_auc_on_numpy(mdl_concat, Z_src_te_concat, gt_label_test, device)
        tgt_test_auc_concat = eval_auc_on_numpy(mdl_concat, Z_tgt_te_concat, tar_label_test, device)

        print(f"\n==================================================")
        print(f" FINAL RESULTS FOR {generalize}")
        print(f"==================================================")
        print(f"  [Src] Val AUC:  {best_val_auc_concat:.4f}")
        print(f"  [Src] Test AUC: {src_test_auc_concat:.4f}")
        print(f"  [Tgt] Test AUC: {tgt_test_auc_concat:.4f}")
        print(f"==================================================\n")

if __name__ == "__main__":
    main()