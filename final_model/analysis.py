import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def analyze_gating_weights(model, loader, device, no_static=False, no_text=False):
    """
    Extracts and visualizes the gating weights from the model.
    """
    print("\n[Analysis] Analyzing Gating Weights...")
    model.eval()
    
    all_gates = []
    
    with torch.no_grad():
        for batch in loader:
            static = batch['static'].to(device)
            dynamic = batch['dynamic'].to(device)
            text = batch['text'].to(device)
            mask = batch['mask'].to(device)
            lengths = batch['lengths']
            
            # Ablation
            if no_static:
                static = torch.zeros_like(static)
            if no_text:
                text = torch.zeros_like(text)
            
            _, gates = model(static, dynamic, text, lengths) # (B, L, 3)
            
            # We only care about valid time steps
            B, L, _ = gates.shape
            for i in range(B):
                valid_len = lengths[i]
                if valid_len > 0:
                    valid_gates = gates[i, :valid_len, :].cpu().numpy()
                    all_gates.append(valid_gates)
                    
    if len(all_gates) > 0:
        all_gates = np.concatenate(all_gates, axis=0) # (Total_Valid_Steps, 3)
        mean_gates = np.mean(all_gates, axis=0)
        std_gates = np.std(all_gates, axis=0)
        
        print(f"Mean Gating Weights: Static={mean_gates[0]:.4f}, Dynamic={mean_gates[1]:.4f}, Text={mean_gates[2]:.4f}")
        
        # Visualization
        modalities = ['Static', 'Dynamic', 'Text']
        plt.figure(figsize=(8, 6))
        plt.bar(modalities, mean_gates, yerr=std_gates, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        plt.ylabel("Average Gating Weight")
        plt.title("Modality Importance (Gating Weights)")
        plt.ylim(0, 1.1)
        for i, v in enumerate(mean_gates):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
            
        plt.savefig('final_model/gating_weights.png')
        print("Saved Gating Weights plot to final_model/gating_weights.png")
    else:
        print("No valid gates found.")

def run_permfit(model, loader, scaler, static_names, dynamic_names, device):
    """
    PermFIT-inspired Permutation Feature Importance.
    Measures the increase in Loss (MSE) when a feature is permuted.
    Uses batched inference to avoid OOM.
    """
    print("\n[PermFIT] Starting Permutation-based Feature Importance Analysis...")
    model.eval()
    criterion = nn.MSELoss()
    
    # 1. Collect all data
    all_static = []
    all_dynamic = []
    all_text = []
    all_target = []
    all_mask = []
    all_lengths = []
    
    with torch.no_grad():
        for batch in loader:
            all_static.append(batch['static'])
            all_dynamic.append(batch['dynamic'])
            all_text.append(batch['text'])
            all_target.append(batch['target'])
            all_mask.append(batch['mask'])
            all_lengths.append(batch['lengths'])
            
    # Handle variable sequence lengths by padding to global max length
    max_len = max([d.size(1) for d in all_dynamic])
    
    import torch.nn.functional as F
    
    padded_dynamic = []
    padded_target = []
    padded_mask = []
    
    for d, t, m in zip(all_dynamic, all_target, all_mask):
        # Pad dynamic: (B, L, F) -> (B, MaxL, F)
        pad_len = max_len - d.size(1)
        if pad_len > 0:
            d_padded = F.pad(d, (0, 0, 0, pad_len), value=0)
            t_padded = F.pad(t, (0, pad_len), value=-999)
            m_padded = F.pad(m, (0, pad_len), value=False)
        else:
            d_padded = d
            t_padded = t
            m_padded = m
            
        padded_dynamic.append(d_padded)
        padded_target.append(t_padded)
        padded_mask.append(m_padded)
            
    static = torch.cat(all_static, dim=0) # CPU
    dynamic = torch.cat(padded_dynamic, dim=0) # CPU
    text = torch.cat(all_text, dim=0) # CPU
    target = torch.cat(padded_target, dim=0) # CPU
    mask = torch.cat(padded_mask, dim=0) # CPU
    lengths = torch.cat(all_lengths, dim=0) # CPU
    
    BATCH_SIZE = 256
    
    def get_score_batched(s, d, t):
        """
        Runs inference in batches.
        s, d, t are on CPU.
        """
        total_loss = 0
        total_samples = 0
        
        num_samples = s.size(0)
        num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, num_samples)
                
                b_s = s[start_idx:end_idx].to(device)
                b_d = d[start_idx:end_idx].to(device)
                b_t = t[start_idx:end_idx].to(device)
                b_target = target[start_idx:end_idx].to(device)
                b_mask = mask[start_idx:end_idx].to(device)
                b_lengths = lengths[start_idx:end_idx]
                
                preds, _ = model(b_s, b_d, b_t, b_lengths)
                
                # Compute loss only on masked elements
                # preds: [B, MaxL, 1] -> squeeze -> [B, MaxL]
                # b_target: [B, MaxL]
                # b_mask: [B, MaxL]
                
                p = preds.squeeze(-1)
                
                # Flatten for loss calculation
                p_flat = p[b_mask]
                t_flat = b_target[b_mask]
                
                if len(p_flat) > 0:
                    loss = criterion(p_flat, t_flat)
                    total_loss += loss.item() * len(p_flat)
                    total_samples += len(p_flat)
                
        return total_loss / total_samples if total_samples > 0 else 0
    
    baseline_loss = get_score_batched(static, dynamic, text)
    print(f"Baseline Loss: {baseline_loss:.4f}")
    
    importances_mean = {}
    importances_std = {}
    p_values = {}
    
    from scipy.stats import ttest_1samp
    import json
    
    # Static Features
    for i, name in enumerate(tqdm(static_names, desc="Permuting Static")):
        scores = []
        for _ in range(n_repeats):
            s_perm = static.clone()
            idx = torch.randperm(s_perm.size(0))
            s_perm[:, i] = s_perm[idx, i] # Shuffle column i
            scores.append(get_score_batched(s_perm, dynamic, text))
        
        # Importance = Permuted Loss - Baseline Loss
        imp_scores = np.array(scores) - baseline_loss
        importances_mean[name] = np.mean(imp_scores)
        importances_std[name] = np.std(imp_scores)
        
        # T-test: H0: mean <= 0, H1: mean > 0
        if n_repeats > 1:
            _, p_val = ttest_1samp(imp_scores, 0, alternative='greater')
            p_values[name] = p_val
        else:
            p_values[name] = 1.0 # Cannot compute p-value with 1 repeat
        
    # Dynamic Features
    for i, name in enumerate(tqdm(dynamic_names, desc="Permuting Dynamic")):
        scores = []
        for _ in range(n_repeats):
            d_perm = dynamic.clone()
            idx = torch.randperm(d_perm.size(0))
            d_perm[:, :, i] = d_perm[idx, :, i] # Shuffle channel i across batch
            scores.append(get_score_batched(static, d_perm, text))
            
        imp_scores = np.array(scores) - baseline_loss
        importances_mean[name] = np.mean(imp_scores)
        importances_std[name] = np.std(imp_scores)
        
        if n_repeats > 1:
            _, p_val = ttest_1samp(imp_scores, 0, alternative='greater')
            p_values[name] = p_val
        else:
            p_values[name] = 1.0
        
    # Text
    scores = []
    for _ in range(n_repeats):
        t_perm = text.clone()
        idx = torch.randperm(t_perm.size(0))
        t_perm = t_perm[idx] # Shuffle entire text embedding
        scores.append(get_score_batched(static, dynamic, t_perm))
        
    imp_scores = np.array(scores) - baseline_loss
    importances_mean['Text_Embeddings'] = np.mean(imp_scores)
    importances_std['Text_Embeddings'] = np.std(imp_scores)
    
    if n_repeats > 1:
        _, p_val = ttest_1samp(imp_scores, 0, alternative='greater')
        p_values['Text_Embeddings'] = p_val
    else:
        p_values['Text_Embeddings'] = 1.0
    
    # Visualization
    features = list(importances_mean.keys())
    means = [importances_mean[f] for f in features]
    stds = [importances_std[f] for f in features]
    pvals = [p_values[f] for f in features]
    
    # Create DataFrame
    df_imp = pd.DataFrame({
        'feature': features, 
        'importance_mean': means,
        'importance_std': stds,
        'p_value': pvals
    })
    
    # Filter out total_opioid_mme as per user request
    df_imp = df_imp[df_imp['feature'] != 'total_opioid_mme']
    
    df_imp = df_imp.sort_values('importance_mean', ascending=True) # Sort ascending for barh
    
    # Save full CSV
    df_imp.to_csv('final_model/permfit_results.csv', index=False)
    print("Saved PermFIT results to final_model/permfit_results.csv")
    
    # Extract Significant Features (P < 0.05)
    sig_features = df_imp[df_imp['p_value'] < 0.05]['feature'].tolist()
    
    # Save significant features
    with open('final_model/significant_features.json', 'w') as f:
        json.dump(sig_features, f)
    print(f"Saved {len(sig_features)} significant features to final_model/significant_features.json")
    
    # Filter Top 20 for Plot
    top_n = 20
    if len(df_imp) > top_n:
        df_plot = df_imp.tail(top_n)
    else:
        df_plot = df_imp
        
    plt.figure(figsize=(10, 8))
    # Plot with error bars
    plt.barh(df_plot['feature'], df_plot['importance_mean'], xerr=df_plot['importance_std'], capsize=5)
    plt.xlabel("Importance (Increase in MSE)")
    plt.title(f"PermFIT Feature Importance (Top {len(df_plot)})")
    
    # Check if we need log scale
    if df_plot['importance_mean'].max() > 10 * df_plot['importance_mean'].median() and df_plot['importance_mean'].min() > 0:
        plt.xscale('log')
        plt.xlabel("Importance (Increase in MSE)")
        
    plt.tight_layout()
    plt.savefig('final_model/permfit_importance.png')
    print("Saved PermFIT plot to final_model/permfit_importance.png")
