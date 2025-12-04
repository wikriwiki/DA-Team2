import torch
import torch.nn as nn
import torch.nn.functional as F

class StaticEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), # Output hidden_dim
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=dropout if dropout > 0 else 0)
        
    def forward(self, x):
        # x: (Batch, Seq, Feature)
        output, _ = self.gru(x)
        return output # (Batch, Seq, Hidden)

class TextEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), # Output hidden_dim
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class GatedFusionDecoder(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()
        # Gating network: Takes concatenated embeddings and outputs weights for 3 modalities
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3) # 3 weights: Static, Dynamic, Text
        )
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, static_emb, dynamic_emb, text_emb):
        # static_emb: (B, H)
        # dynamic_emb: (B, L, H)
        # text_emb: (B, H)
        
        B, L, H = dynamic_emb.size()
        
        # Expand static and text to match sequence length
        static_expanded = static_emb.unsqueeze(1).expand(B, L, H)
        text_expanded = text_emb.unsqueeze(1).expand(B, L, H)
        
        # Concatenate for gating
        combined = torch.cat([static_expanded, dynamic_emb, text_expanded], dim=-1) # (B, L, 3H)
        
        # Compute gates
        gates = F.softmax(self.gate_net(combined), dim=-1) # (B, L, 3)
        
        # Weighted sum
        fused = (gates[:,:,0:1] * static_expanded + 
                 gates[:,:,1:2] * dynamic_emb + 
                 gates[:,:,2:3] * text_expanded) # (B, L, H)
        
        # Predict
        output = self.head(fused) # (B, L, 1)
        return output.squeeze(-1), gates

class GatedFusionModel(nn.Module):
    def __init__(self, static_dim, dynamic_dim, text_dim=64, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.static_encoder = StaticEncoder(static_dim, hidden_dim, dropout)
        self.ts_encoder = TimeSeriesEncoder(dynamic_dim, hidden_dim, dropout)
        self.text_encoder = TextEncoder(text_dim, hidden_dim, dropout)
        self.decoder = GatedFusionDecoder(hidden_dim, dropout)

    def forward(self, static_data, dynamic_data, text_data, lengths=None):
        static_emb = self.static_encoder(static_data)
        dynamic_emb = self.ts_encoder(dynamic_data)
        text_emb = self.text_encoder(text_data)
        
        prediction, gates = self.decoder(static_emb, dynamic_emb, text_emb)
        return prediction, gates
