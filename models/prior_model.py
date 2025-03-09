import torch.nn as nn

def create_prior():
    """Factory for prior model with predefined architecture"""
    return Prior(
        embed_dim=512,
        num_layers=6
    )

class Prior(nn.Module):
    def __init__(self, embed_dim, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=4*embed_dim,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_emb):
        # Add sequence dimension for transformer
        x = text_emb.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_ln(x.squeeze(1))
        return self.output_proj(x)