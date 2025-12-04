import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import csv
from datetime import datetime
from dataset import get_dataloaders
from model import GatedFusionModel
from analysis import run_permfit, analyze_gating_weights

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Train GatedFusionModel with Hyperparameters and Ablation')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden Dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout Rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay (L2 Penalty)')
parser.add_argument('--epochs', type=int, default=30, help='Number of Epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')
parser.add_argument('--no_static', action='store_true', help='Ablation: Zero out static features')
parser.add_argument('--no_text', action='store_true', help='Ablation: Zero out text features')
parser.add_argument('--use_significant_features', action='store_true', help='Experiment: Use only significant features (P < 0.05)')
parser.add_argument('--permfit_repeats', type=int, default=1, help='Number of repeats for PermFIT (use >1 for p-value)')
parser.add_argument('--experiment_name', type=str, default='default', help='Name of the experiment for logging')
parser.add_argument('--output_csv', type=str, default='final_model/experiment_results.csv', help='Path to save results CSV')

args = parser.parse_args()

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_PATH = os.path.join(BASE_DIR, 'static_data.csv')
DYNAMIC_PATH = os.path.join(BASE_DIR, 'dynamic_data.csv')
TEXT_DIR = os.path.join(BASE_DIR, 'modernbert_section_group_hierbert_full.pt')

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR = args.lr
HIDDEN_DIM = args.hidden_dim
DROPOUT = args.dropout
WEIGHT_DECAY = args.weight_decay
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running Experiment: {args.experiment_name}")
print(f"Params: LR={LR}, Hidden={HIDDEN_DIM}, Dropout={DROPOUT}, WD={WEIGHT_DECAY}, Epochs={EPOCHS}")
print(f"Ablation: No Static={args.no_static}, No Text={args.no_text}")
print(f"Significant Features Only: {args.use_significant_features}")
print(f"PermFIT Repeats: {args.permfit_repeats}")
print(f"Device: {DEVICE}")

def train_model():
    # Load Significant Features if requested
    selected_features = None
    if args.use_significant_features:
        import json
        try:
            with open('final_model/significant_features.json', 'r') as f:
                selected_features = json.load(f)
            print(f"Loaded {len(selected_features)} significant features.")
        except FileNotFoundError:
            print("Error: final_model/significant_features.json not found. Run PermFIT first.")
            return

    print("Loading Data...")
    train_loader, val_loader, test_loader, scaler, static_names, dynamic_names = get_dataloaders(
        STATIC_PATH, DYNAMIC_PATH, TEXT_DIR, batch_size=BATCH_SIZE, selected_features=selected_features
    )
    
    static_dim = len(static_names)
    dynamic_dim = len(dynamic_names)
    
    # Get Text Dim dynamically from a batch
    sample_batch = next(iter(train_loader))
    text_dim = sample_batch['text'].shape[-1]
    
    print(f"Static Dim: {static_dim}, Dynamic Dim: {dynamic_dim}, Text Dim: {text_dim}")
    
    model = GatedFusionModel(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        text_dim=text_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            static = batch['static'].to(DEVICE)
            dynamic = batch['dynamic'].to(DEVICE)
            text = batch['text'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            lengths = batch['lengths']
            
            # Ablation Logic
            if args.no_static:
                static = torch.zeros_like(static)
            if args.no_text:
                text = torch.zeros_like(text)
            
            optimizer.zero_grad()
            output, _ = model(static, dynamic, text, lengths)
            
            # Mask target
            loss = 0
            valid_samples = 0
            for i in range(len(output)):
                l = lengths[i]
                # Target is aligned such that output[t] predicts target[t]
                # Dynamic input was 0..N-2, Target was 1..N-1
                # Output length matches dynamic input length
                # We need to compute loss only on valid steps
                # output[i] shape: [seq_len] (Already squeezed in model)
                # target[i] shape: [seq_len]
                
                curr_out = output[i][:l]
                curr_target = target[i][:l]
                
                loss += criterion(curr_out, curr_target)
                valid_samples += 1
                
            loss = loss / valid_samples
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                static = batch['static'].to(DEVICE)
                dynamic = batch['dynamic'].to(DEVICE)
                text = batch['text'].to(DEVICE)
                target = batch['target'].to(DEVICE)
                lengths = batch['lengths']
                
                # Ablation Logic
                if args.no_static:
                    static = torch.zeros_like(static)
                if args.no_text:
                    text = torch.zeros_like(text)
                
                output, _ = model(static, dynamic, text, lengths)
                
                for i in range(len(output)):
                    l = lengths[i]
                    curr_out = output[i][:l]
                    curr_target = target[i][:l]
                    
                    val_loss += criterion(curr_out, curr_target).item()
                    
                    val_preds.extend(curr_out.cpu().numpy())
                    val_targets.extend(curr_target.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader.dataset) # Approximation
        
        # R2 Score
        from sklearn.metrics import r2_score
        val_r2 = r2_score(val_targets, val_preds)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val R2: {val_r2:.4f}")
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_val_loss = avg_val_loss
            # Save best model
            save_path = f'final_model/best_model_{args.experiment_name}.pth'
            torch.save(model.state_dict(), save_path)
            
    print(f"Training Complete. Best Val R2: {best_r2:.4f}")
    
    # --- Logging Results ---
    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Experiment', 'LR', 'Hidden', 'Dropout', 'Weight_Decay', 'Epochs', 'No_Static', 'No_Text', 'Best_Val_R2', 'Best_Val_Loss'])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            args.experiment_name,
            LR, HIDDEN_DIM, DROPOUT, WEIGHT_DECAY, EPOCHS,
            args.no_static, args.no_text,
            best_r2, best_val_loss
        ])
        
    # --- Analysis (Only for the best model of this run) ---
    # Load best model
    model.load_state_dict(torch.load(f'final_model/best_model_{args.experiment_name}.pth', weights_only=True))
    model.eval()
    
    # 1. Gating Analysis
    print("Running Gating Analysis...")
    analyze_gating_weights(model, test_loader, DEVICE, args.no_static, args.no_text)
    
    # 2. PermFIT
    # Run PermFIT if requested (e.g., for the Full Model run)
    if args.permfit_repeats > 0 and not args.no_static and not args.no_text:
         print(f"Running PermFIT Analysis with {args.permfit_repeats} repeats...")
         run_permfit(model, test_loader, scaler, static_names, dynamic_names, DEVICE, n_repeats=args.permfit_repeats)

if __name__ == '__main__':
    train_model()
