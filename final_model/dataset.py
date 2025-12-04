import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class OpioidDataset(Dataset):
    def __init__(self, static_df, dynamic_df, text_dir, hadm_ids, dynamic_scaler=None, is_train=True, selected_features=None):
        """
        Args:
            static_df: DataFrame with static features.
            dynamic_df: DataFrame with time-series features.
            text_dir: Directory containing .pt files for text embeddings.
            hadm_ids: List of hadm_ids to include in this dataset.
            dynamic_scaler: Fitted StandardScaler for dynamic features (if None, will fit on this data).
            is_train: Boolean, if True and scaler is None, fits the scaler.
            selected_features: List of feature names to keep. If None, keep all.
        """
        self.hadm_ids = hadm_ids
        self.text_dir = text_dir
        
        # 1. Static Data Processing
        self.static_df = static_df[static_df['hadm_id'].isin(hadm_ids)].copy()
        self.static_df = self.static_df.sort_values('hadm_id').reset_index(drop=True)
        
        # Drop non-feature columns
        self.static_features = self.static_df.drop(columns=['subject_id', 'hadm_id'], errors='ignore')
        self.static_features = self.static_features.select_dtypes(include=[np.number]).fillna(0)
        
        self.static_feature_names = self.static_features.columns.tolist()
        
        # 2. Dynamic Data Processing
        self.dynamic_df = dynamic_df[dynamic_df['hadm_id'].isin(hadm_ids)].copy()
        
        # Feature Selection (Initial)
        exclude_cols = ['hadm_id', 'segment_period', 'segment_num'] 
        feature_cols = [c for c in self.dynamic_df.columns if c not in exclude_cols and ('opioid' not in c or c == 'total_opioid_mme')]
        
        self.dynamic_features_df = self.dynamic_df[feature_cols].select_dtypes(include=[np.number])
        self.dynamic_feature_names = self.dynamic_features_df.columns.tolist()
        
        # --- Apply Selected Features Filtering ---
        if selected_features is not None:
            # Filter Static
            self.static_feature_names = [f for f in self.static_feature_names if f in selected_features]
            self.static_features = self.static_features[self.static_feature_names]
            
            # Filter Dynamic
            self.dynamic_feature_names = [f for f in self.dynamic_feature_names if f in selected_features]
            self.dynamic_features_df = self.dynamic_features_df[self.dynamic_feature_names]
            
            print(f"Filtered features. Static: {len(self.static_feature_names)}, Dynamic: {len(self.dynamic_feature_names)}")

        self.static_dim = self.static_features.shape[1]
        self.dynamic_dim = self.dynamic_features_df.shape[1]
        
        # Normalization
        if is_train and dynamic_scaler is None:
            self.dynamic_scaler = StandardScaler()
            self.dynamic_features_normalized = self.dynamic_scaler.fit_transform(self.dynamic_features_df)
        elif dynamic_scaler is not None:
            self.dynamic_scaler = dynamic_scaler
            self.dynamic_features_normalized = self.dynamic_scaler.transform(self.dynamic_features_df)
        else:
            self.dynamic_scaler = StandardScaler()
            self.dynamic_features_normalized = self.dynamic_scaler.fit_transform(self.dynamic_features_df)

        # Assign back normalized values
        self.dynamic_df_norm = pd.DataFrame(self.dynamic_features_normalized, columns=self.dynamic_features_df.columns)
        self.dynamic_df_norm['hadm_id'] = self.dynamic_df['hadm_id'].values
        self.dynamic_df_norm['segment_num'] = self.dynamic_df['segment_num'].values
        self.dynamic_df_norm['target'] = np.log1p(self.dynamic_df['total_opioid_mme'].values) # Log transform target
        
        # Group by hadm_id
        self.grouped_dynamic = self.dynamic_df_norm.groupby('hadm_id')
        
        # Filter valid IDs (must have at least 2 segments for t -> t+1)
        valid_ids = []
        for hid in self.hadm_ids:
            if hid in self.grouped_dynamic.groups:
                if len(self.grouped_dynamic.get_group(hid)) >= 2:
                    valid_ids.append(hid)
        
        # Update hadm_ids to only valid ones
        self.hadm_ids = valid_ids

        # 3. Text Data Loading
        print(f"Loading text embeddings from {text_dir}...")
        try:
            data = torch.load(text_dir, weights_only=False, map_location=torch.device('cpu'))
            
            # 1. Get IDs mapping
            if 'ids' in data and isinstance(data['ids'], torch.Tensor):
                # Assuming ids is [N, 2] where col 1 is hadm_id
                ids_tensor = data['ids']
                ids_np = ids_tensor.numpy()
                # Create map: hadm_id -> index
                self.hadm_id_to_idx = {int(row[1]): i for i, row in enumerate(ids_np)}
                print(f"Created mapping for {len(self.hadm_id_to_idx)} hadm_ids from 'ids' tensor.")
            else:
                print("Warning: 'ids' key not found or not a tensor. Text embeddings will be zeros.")
                self.hadm_id_to_idx = {}

            # 2. Get Embeddings Tensor
            # We prefer 'emb_hier_cls_pca' (64-dim)
            if 'admission' in data and isinstance(data['admission'], dict):
                adm = data['admission']
                if 'emb_hier_cls_pca' in adm:
                    self.text_embeddings = adm['emb_hier_cls_pca']
                    print(f"Using 'emb_hier_cls_pca' embeddings with shape {self.text_embeddings.shape}")
                elif 'emb_from_group_mean_pca' in adm:
                    self.text_embeddings = adm['emb_from_group_mean_pca']
                    print(f"Using 'emb_from_group_mean_pca' embeddings with shape {self.text_embeddings.shape}")
                else:
                    print("Warning: No suitable 64-dim embedding key found in 'admission'.")
                    self.text_embeddings = None
            else:
                print("Warning: 'admission' key not found or not a dict.")
                self.text_embeddings = None
                
        except Exception as e:
            print(f"Error loading text embeddings: {e}")
            self.hadm_id_to_idx = {}
            self.text_embeddings = None
            
    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        
        # 1. Static
        static_row = self.static_features.loc[self.static_df['hadm_id'] == hadm_id]
        if len(static_row) == 0:
            static_tensor = torch.zeros(self.static_dim, dtype=torch.float32)
        else:
            static_tensor = torch.tensor(static_row.values[0], dtype=torch.float32)
            
        # 2. Dynamic (Autoregressive Shift)
        group = self.grouped_dynamic.get_group(hadm_id).sort_values('segment_num')
        
        # Input: t=0 to t=N-2
        # Target: t=1 to t=N-1
        # Drop columns that are not features (hadm_id, target, segment_num)
        # Note: dynamic_features_df already contains only features.
        # But group comes from dynamic_df_norm which has hadm_id, target, segment_num added.
        # We need to select only the feature columns.
        
        features = group[self.dynamic_feature_names].iloc[:-1].values
        targets = group['target'].iloc[1:].values
        
        dynamic_tensor = torch.tensor(features, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        # 3. Text
        text_tensor = torch.zeros(64, dtype=torch.float32)
        if self.text_embeddings is not None and hadm_id in self.hadm_id_to_idx:
            idx = self.hadm_id_to_idx[hadm_id]
            # Retrieve embedding
            emb = self.text_embeddings[idx]
            
            # Ensure it's a tensor
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            
            # If it's on GPU, move to CPU
            if emb.device.type != 'cpu':
                emb = emb.cpu()
                
            text_tensor = emb
            
            # Handle dimensions if needed
            if text_tensor.dim() > 1:
                text_tensor = text_tensor.squeeze()
                
            if text_tensor.shape[0] != 64:
                 # Fallback or resize if unexpected
                 pass 
        
        return {
            'hadm_id': hadm_id,
            'static': static_tensor,
            'dynamic': dynamic_tensor,
            'text': text_tensor,
            'target': target_tensor
        }

def collate_fn(batch):
    hadm_ids = [item['hadm_id'] for item in batch]
    static_data = torch.stack([item['static'] for item in batch])
    text_data = torch.stack([item['text'] for item in batch])
    
    dynamic_list = [item['dynamic'] for item in batch]
    target_list = [item['target'] for item in batch]
    
    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    dynamic_padded = pad_sequence(dynamic_list, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_list, batch_first=True, padding_value=-999) # Mask value
    
    lengths = torch.tensor([len(x) for x in dynamic_list])
    max_len = dynamic_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    
    return {
        'hadm_id': hadm_ids,
        'static': static_data,
        'dynamic': dynamic_padded,
        'text': text_data,
        'target': target_padded,
        'mask': mask,
        'lengths': lengths
    }

def get_dataloaders(static_path, dynamic_path, text_dir, batch_size=32, test_size=0.2, val_size=0.1, selected_features=None):
    static_df = pd.read_csv(static_path)
    # No filtering here, assuming static_path is already filtered/processed for the cohort
        
    dynamic_df = pd.read_csv(dynamic_path)
    
    static_ids = set(static_df['hadm_id'].unique())
    dynamic_ids = set(dynamic_df['hadm_id'].unique())
    common_ids = list(static_ids.intersection(dynamic_ids))
    common_ids.sort()
    
    train_val_ids, test_ids = train_test_split(common_ids, test_size=test_size, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size/(1-test_size), random_state=42)
    
    train_dataset = OpioidDataset(static_df, dynamic_df, text_dir, train_ids, is_train=True, selected_features=selected_features)
    scaler = train_dataset.dynamic_scaler
    
    val_dataset = OpioidDataset(static_df, dynamic_df, text_dir, val_ids, dynamic_scaler=scaler, is_train=False, selected_features=selected_features)
    test_dataset = OpioidDataset(static_df, dynamic_df, text_dir, test_ids, dynamic_scaler=scaler, is_train=False, selected_features=selected_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, scaler, train_dataset.static_feature_names, train_dataset.dynamic_feature_names
