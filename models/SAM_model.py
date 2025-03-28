import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=1024, max_frames=32, spatial_size=16):
        super().__init__()
        self.temporal_enc = nn.Embedding(max_frames, embed_dim)
        self.spatial_enc = nn.Parameter(torch.randn(1, spatial_size, spatial_size, embed_dim))
        self.frame_counter = nn.Parameter(torch.arange(max_frames), requires_grad=False)
        self.spatial_size = spatial_size

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        
        # Reshape and interpolate spatial encoding
        spatial = self.spatial_enc.permute(0, 3, 1, 2)  # (1, C, S, S)
        spatial = F.interpolate(spatial, size=(H, W), mode='bilinear', align_corners=False)
        spatial = spatial.permute(0, 2, 3, 1).unsqueeze(1)  # (1, 1, H, W, C)
        spatial = spatial.expand(B, T, -1, -1, -1)  # (B, T, H, W, C)
        
        # Temporal encoding
        time_ids = self.frame_counter[:T].unsqueeze(0).expand(B, -1)
        temporal = self.temporal_enc(time_ids).view(B, T, 1, 1, C)
        
        # Align dimensions and add
        x = x.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        x = x + spatial + temporal
        return x.permute(0, 1, 4, 2, 3)  # Back to (B, T, C, H, W)

class AdaptiveVisionEncoder(nn.Module):
    """Resolution-agnostic image encoder with dynamic spatial processing"""
    def __init__(self, embed_dim=512, base_channels=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Resolution-independent stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        # Resolution-adaptive blocks
        self.blocks = nn.ModuleList([
            self._make_ds_block(base_channels, base_channels*2, stride=2),
            self._make_ds_block(base_channels*2, base_channels*4, stride=2),
            self._make_ds_block(base_channels*4, embed_dim, stride=1),
            
            # Atrous spatial pyramid components
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=2, dilation=2),
                nn.GroupNorm(8, embed_dim),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=4, dilation=4),
                nn.GroupNorm(8, embed_dim),
                nn.GELU()
            )
        ])
        
        # Fix 1: Correct projection layer input channels
        self.projection = nn.Conv2d(1920, embed_dim, 1)  # 128+256+512+512+512=1920
        
        # Dynamic spatial attention
        self.attn = nn.Sequential(
            nn.Conv2d(embed_dim, 1, 1),
            nn.Sigmoid()
        )

    def _make_ds_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 1) if in_c != out_c else nn.Identity()
        )

    def forward(self, x):
        x = self.stem(x)
        spatial_features = []
        
        # Fix 2: Ensure features are collected before concatenation
        for block in self.blocks:
            x = block(x)
            spatial_features.append(x)
        
        # Multi-scale fusion
        fused = torch.cat([
            F.interpolate(f, size=x.shape[-2:], mode='bilinear', align_corners=False)
            for f in spatial_features
        ], dim=1)  # Concatenate along channels
        
        # Fix 3: Apply projection after concatenation
        x = self.projection(fused)  # Now input channels=1920
        
        # Attention-weighted fusion
        attn_weights = self.attn(x)
        return x * attn_weights

class VideoSAM(nn.Module):
    def __init__(self, mem_size=5, num_transformers=6):
        super().__init__()
        self.image_encoder = AdaptiveVisionEncoder()
        self.pos_enc = UnifiedPositionalEncoding(embed_dim=1024)
        self.register_buffer('memory', torch.zeros(1, mem_size, 1024))
        
        # Unified transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1024,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=num_transformers
        )
        
        # Memory-enhanced decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(1024, 256, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.GELU(),
            nn.Conv3d(256, 128, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.GELU(),
            nn.Conv3d(128, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, prior_emb: torch.Tensor):
        # x: (B, [T], C, H, W) -> force 5D
        x = x.unsqueeze(1) if x.ndim == 4 else x  # (B, T, C, H, W)
        B, T = x.shape[:2]
        
        # Encode all frames and preserve spatial dimensions
        encoded_flat = self.image_encoder(x.flatten(0, 1))  # (B*T, C_enc, H_enc, W_enc)
        C_enc, H_enc, W_enc = encoded_flat.shape[1], encoded_flat.shape[2], encoded_flat.shape[3]
        encoded = encoded_flat.view(B, T, C_enc, H_enc, W_enc)  # (B, T, 512, H_enc, W_enc)
        
        # Add prior conditioning with correct spatial dims
        prior = prior_emb.view(B, 1, -1, 1, 1).expand(-1, T, -1, H_enc, W_enc)
        encoded = torch.cat([encoded, prior], dim=2)  # (B, T, 1024, H_enc, W_enc)
        
        # Add unified positional encoding
        encoded = self.pos_enc(encoded)
        
        # Process as spatio-temporal tokens
        tokens = encoded.flatten(3).permute(0, 1, 3, 2)  # (B, T, N, C)
        B, T, N, C = tokens.shape
        tokens = tokens.reshape(B, T*N, C)
        
        # Attend to memory
        memory = self.memory.expand(B, -1, -1)
        tokens = torch.cat([memory, tokens], dim=1)
        
        # Transformer processing
        processed = self.transformer(tokens)
        
        # Decode masks
        masks = processed[:, -T*N:].view(B, T, N, C)
        
        # Reshape to include spatial dimensions (H_enc, W_enc)
        masks = masks.view(B, T, H_enc, W_enc, C)  # (B, T, H_enc, W_enc, C)
        masks = masks.permute(0, 4, 1, 2, 3)       # (B, C, T, H_enc, W_enc)
        
        # Pass to decoder
        masks = self.decoder(masks)  # Output shape: (B, 1, T, H_enc, W_enc)
        
        # Corrected interpolation handling
        B_dec, C_dec, T_dec, H_dec, W_dec = masks.shape
        # Reshape for 2D interpolation (merge batch and temporal dimensions)
        masks = masks.permute(0, 2, 1, 3, 4).reshape(-1, C_dec, H_dec, W_dec)  # (B*T_dec, C_dec, H_dec, W_dec)
        masks = F.interpolate(masks, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Reshape back to original dimensions
        masks = masks.view(B, T_dec, C_dec, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)  # (B, C_dec, T_dec, H, W)
        
        # Update memory
        if self.training:
            with torch.no_grad():  # Disable gradient tracking for memory update
                new_memory = processed[:, :self.memory.size(1)].detach().mean(dim=0, keepdim=True)
                # Clone to avoid in-place modification issues
                updated_memory = torch.cat([new_memory, self.memory], dim=1)[:, :self.memory.size(1)]
                self.memory.copy_(updated_memory)
        
        return torch.sigmoid(masks.squeeze(1))  # Remove channel dim (C_dec=1)
    
# input: (batch, num_frames (remove for images), num_channels, height, width) (height and width have to be greater than 16 and divisible by 8)
# output: (batch, num_frames (remove for images), num_channels, height, width)

# from clip_model import create_text_encoder, CLIPTokenize
# from prior_model import create_prior
# model = VideoSAM()
# text_encoder = create_text_encoder()
# prior = create_prior()
# input_texts = ["Input string","Another input string"]
# tokens = CLIPTokenize(input_texts)
# encodings = text_encoder(tokens)
# prior_emb = prior(encodings)

# # Test with image input
# image = torch.randn(2, 1, 3, 224, 224)  # Batch of 2 images (B, 1, C, H, W)
# masks = model(image, prior_emb)
# print(f"Image test output shape: {masks.size()}")  # Should output: torch.Size([2, 1, 224, 224])

# # Test with video input
# video = torch.randn(2, 8, 3, 224, 224)  # Batch of 2 videos (B, T, C, H, W)
# masks = model(video, prior_emb)
# print(f"Video test output shape {masks.size()}")  # Should output: torch.Size([2, 8, 224, 224])