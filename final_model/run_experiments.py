import subprocess
import itertools
import pandas as pd
import os
import sys

PYTHON_EXEC = sys.executable
TRAIN_SCRIPT = "final_model/train.py"
GRID_RESULTS = "final_model/grid_search_results.csv"
ABLATION_RESULTS = "final_model/ablation_results.csv"

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def grid_search():
    print("=== Starting Grid Search ===")
    lrs = [0.001, 0.0005]
    hidden_dims = [64, 128]
    dropouts = [0.1, 0.3]
    
    # Total 8 combinations
    combinations = list(itertools.product(lrs, hidden_dims, dropouts))
    
    for i, (lr, hidden, dropout) in enumerate(combinations):
        exp_name = f"grid_{i}_lr{lr}_h{hidden}_d{dropout}"
        cmd = (
            f'"{PYTHON_EXEC}" -u {TRAIN_SCRIPT} '
            f'--lr {lr} --hidden_dim {hidden} --dropout {dropout} '
            f'--epochs 10 --experiment_name {exp_name} '
            f'--output_csv {GRID_RESULTS}'
        )
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment {exp_name}: {e}")

def get_best_params():
    if not os.path.exists(GRID_RESULTS):
        print("Grid search results not found!")
        return None
    
    df = pd.read_csv(GRID_RESULTS)
    best_row = df.loc[df['Best_Val_R2'].idxmax()]
    
    print("=== Best Parameters Found ===")
    print(best_row)
    
    return {
        'lr': best_row['LR'],
        'hidden_dim': int(best_row['Hidden']),
        'dropout': best_row['Dropout']
    }

def ablation_study(best_params):
    print("=== Starting Ablation Study ===")
    lr = best_params['lr']
    hidden = best_params['hidden_dim']
    dropout = best_params['dropout']
    
    # 1. Full Model Analysis (with PermFIT P-value calculation)
    print("--- Running Full Model Analysis (with PermFIT) ---")
    exp_name = "full_model_analysis"
    cmd = (
        f'"{PYTHON_EXEC}" -u {TRAIN_SCRIPT} '
        f'--lr {lr} --hidden_dim {hidden} --dropout {dropout} '
        f'--epochs 30 --experiment_name {exp_name} '
        f'--output_csv {ABLATION_RESULTS} '
        f'--permfit_repeats 10' # Enable PermFIT with 10 repeats for P-value
    )
    try:
        run_command(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {exp_name}: {e}")
        return # Stop if full model fails

    # 2. Ablation Scenarios
    scenarios = [
        {'name': 'no_static', 'args': '--no_static'},
        {'name': 'no_text', 'args': '--no_text'},
        {'name': 'dynamic_only', 'args': '--no_static --no_text'}
    ]
    
    for scenario in scenarios:
        exp_name = f"ablation_{scenario['name']}"
        cmd = (
            f'"{PYTHON_EXEC}" -u {TRAIN_SCRIPT} '
            f'--lr {lr} --hidden_dim {hidden} --dropout {dropout} '
            f'--epochs 30 --experiment_name {exp_name} '
            f'--output_csv {ABLATION_RESULTS} '
            f'{scenario["args"]}'
        )
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment {exp_name}: {e}")

    # 3. Significant Features Only
    print("--- Running Significant Features Only Experiment ---")
    exp_name = "significant_features_only"
    cmd = (
        f'"{PYTHON_EXEC}" -u {TRAIN_SCRIPT} '
        f'--lr {lr} --hidden_dim {hidden} --dropout {dropout} '
        f'--epochs 30 --experiment_name {exp_name} '
        f'--output_csv {ABLATION_RESULTS} '
        f'--use_significant_features'
    )
    try:
        run_command(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {exp_name}: {e}")

if __name__ == "__main__":
    # 1. Grid Search
    grid_search()
    
    # 2. Get Best Params
    best_params = get_best_params()
    
    if best_params:
        # 3. Ablation Study
        ablation_study(best_params)
        print("=== All Experiments Completed ===")
    else:
        print("Skipping Ablation Study due to missing grid results.")
