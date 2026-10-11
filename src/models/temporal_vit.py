import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class AdvancedTemporalViT(nn.Module):
    """
    Advanced model architecture featuring ViT backbone, difference-feature 
    stream fusion, and multi-head Temporal Self-Attention.
    """
    def __init__(self, sequence_length=8, hidden_size=768, num_heads=8, use_mocks=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        if not use_mocks:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', attn_implementation="eager")
        else:
            config = ViTConfig(hidden_size=hidden_size, num_hidden_layers=2, num_attention_heads=4)
            config._attn_implementation = "eager"
            self.vit = ViTModel(config)
            
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Difference-feature stream: small CNN for optical residuals
        self.residual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_size)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        # Temporal Self-Attention with Positional Encodings
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.temporal_attention = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.temporal_gru = nn.GRU(input_size=hidden_size, hidden_size=256, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, residuals):
        # x: (batch, seq_len, 3, 224, 224)
        # residuals: (batch, seq_len, 3, 224, 224)
        batch, seq_len, C, H, W = x.shape
        x_flat = x.view(batch * seq_len, C, H, W)
        
        vit_outputs = self.vit(pixel_values=x_flat)
        spatial_features = vit_outputs.last_hidden_state[:, 0, :] 
        spatial_features = spatial_features.view(batch, seq_len, self.hidden_size)
        
        res_flat = residuals.view(batch * seq_len, C, H, W)
        res_features = self.residual_encoder(res_flat)
        res_features = res_features.view(batch, seq_len, self.hidden_size)
        
        fused = torch.cat([spatial_features, res_features], dim=-1)
        fused = self.fusion(fused)
        
        fused = self.pos_encoder(fused)
        attended = self.temporal_attention(fused)
        
        gru_out, _ = self.temporal_gru(attended)
        last_step_features = gru_out[:, -1, :]
        
        prob = self.classifier(last_step_features)
        return prob

class TemporalViTBaseline(nn.Module):
    """
    Ablated baseline model (as required by workspace rules).
    Omits the temporal self-attention and difference-feature streams.
    """
    def __init__(self, sequence_length=8, hidden_size=768, use_mocks=False):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        if not use_mocks:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', attn_implementation="eager")
        else:
            config = ViTConfig(hidden_size=hidden_size, num_hidden_layers=2, num_attention_heads=4)
            config._attn_implementation = "eager"
            self.vit = ViTModel(config)
            
        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.temporal_gru = nn.GRU(input_size=hidden_size, hidden_size=256, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, residuals=None):
        batch, seq_len, C, H, W = x.shape
        x_flat = x.view(batch * seq_len, C, H, W)
        
        vit_outputs = self.vit(pixel_values=x_flat)
        spatial_features = vit_outputs.last_hidden_state[:, 0, :] 
        spatial_features = spatial_features.view(batch, seq_len, self.hidden_size)
        
        gru_out, _ = self.temporal_gru(spatial_features)
        last_step_features = gru_out[:, -1, :]
        
        prob = self.classifier(last_step_features)
        return prob
