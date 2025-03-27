import torch
import torch.nn as nn

def create_prior():
    """Factory for prior model with DALL-E-style architecture"""
    return Prior(
        embed_dim=512,
        num_layers=24
    )

class Prior(nn.Module):
    def __init__(self, embed_dim, num_layers=24):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Learnable initial query (acts as "seed" for image embedding)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # DALL-E-style decoder layers with cross-attention
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=4*embed_dim,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.final_ln = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_emb):
        # text_emb shape: (batch_size, embed_dim)
        batch_size = text_emb.size(0)
        
        # Expand learnable query to batch size
        target = self.query.expand(batch_size, -1, -1)  # (B, 1, D)
        
        # Reshape text_emb as memory for cross-attention
        memory = text_emb.unsqueeze(1)  # (B, 1, D)
        
        # Autoregressive decoding with causal masking
        for layer in self.layers:
            target = layer(
                target, 
                memory,
                tgt_mask=self._causal_mask(target)
            )
            
        # Final projection
        x = self.final_ln(target.squeeze(1))
        return self.output_proj(x)

    def _causal_mask(self, x):
        """Optional causal mask for autoregressive generation"""
        seq_len = x.size(1)
        return torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)

# from clip_model import create_text_encoder, CLIPTokenize

# text_encoder = create_text_encoder()
# prior = create_prior()

# input_text = "Input string"
# tokens = CLIPTokenize(input_text)
# encodings = text_encoder(tokens)
# prior_emb = prior(encodings)