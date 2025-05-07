import torch
import torch.nn as nn
import torch.nn.functional as F

def iou_loss(pred_masks, true_masks):
    """
    Compute IoU loss for batched inputs.
    pred_masks: Tensor of shape [B, H, W] (logits)
    true_masks: Tensor of shape [B, H, W] (values 0 or 1)
    """
    pred_prob = torch.sigmoid(pred_masks)  # [B, H, W]
    true = true_masks.float()  # [B, H, W]
    
    # Flatten spatial dimensions
    pred_flat = pred_prob.flatten(start_dim=1)  # [B, H*W]
    true_flat = true.flatten(start_dim=1)       # [B, H*W]
    
    intersection = (pred_flat * true_flat).sum(dim=1)        # [B]
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1) - intersection  # [B]
    
    iou = (intersection + 1e-6) / (union + 1e-6)  # [B]
    return 1.0 - iou.mean()

class UnifiedPositionalEncoding(nn.Module):
    def __init__(self, max_frames=32, spatial_size=16, embed_dim=512):
        super().__init__()
        self.temporal_enc = nn.Embedding(max_frames, embed_dim)
        self.spatial_enc = nn.Parameter(torch.randn(1, spatial_size, spatial_size, embed_dim))
        self.max_frames = max_frames
        self.spatial_size = spatial_size
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, t: int):
        C, H, W = x.shape

        spatial = self.spatial_enc.permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial, size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        
        time_id = torch.tensor([t], device=x.device)
        time_id = torch.clamp(time_id, 0, self.max_frames - 1)
        temporal = self.temporal_enc(time_id).view(self.embed_dim, 1, 1).expand(-1, H, W)

        x = x + spatial + temporal
        return x

class AdaptiveVisionEncoder(nn.Module):
    """Resolution-agnostic image encoder with dynamic spatial processing"""
    def __init__(self, embed_dim=512, base_channels=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([
            self._make_ds_block(base_channels, base_channels*2, stride=2),
            self._make_ds_block(base_channels*2, base_channels*4, stride=2),
            self._make_ds_block(base_channels*4, embed_dim, stride=1),
            
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
        
        self.projection = nn.Conv2d(1920, embed_dim, 1)
        
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
        x = x.unsqueeze(0)
        x = self.stem(x)
        spatial_features = []
        
        for block in self.blocks:
            x = block(x)
            spatial_features.append(x)
        
        fused = torch.cat([
            F.interpolate(f, size=x.shape[-2:], mode='bilinear', align_corners=False)
            for f in spatial_features
        ], dim=1)
        
        x = self.projection(fused)
        
        attn_weights = self.attn(x)
        output = x * attn_weights
        return output.squeeze(0)

class FrameTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=512, mem_size=10, num_layers=2, num_heads=8, memory_queue_len=15):
        super().__init__()
        self.mem_size = mem_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.memory_queue_len = memory_queue_len
        self.norm = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, batch_first=True)
            self.layers.append(layer)

        self.mask_prediction = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, memory_queue: torch.Tensor, encoded: torch.Tensor, mask_tokens: torch.Tensor):
        #  Input shapes:
        #  - memory_queue:  [memory_queue_length, mem_size, embed_dim]  (e.g., [15, 10, 512])
        #  - encoded:       [C, H_enc, W_enc]                         (e.g., [512, 14, 14])
        #  - mask_tokens:   [num_mask_tokens, C]                         (e.g., [10, 512])

        C, H_enc, W_enc = encoded.shape
        num_mask_tokens = mask_tokens.shape[0]

        #  Prepare encoded input for the decoder
        encoded_in = encoded.unsqueeze(0).expand(self.memory_queue_len, self.mem_size, C, H_enc, W_enc)
        encoded_in = encoded_in.reshape(self.memory_queue_len * self.mem_size, H_enc * W_enc, C)

        #  Initialize decoder input using mask tokens
        decoder_input = mask_tokens.unsqueeze(0).expand(self.memory_queue_len * self.mem_size, num_mask_tokens, -1)

        #  Transformer Decoder layers
        for layer in self.layers:
            decoder_input = layer(decoder_input, encoded_in)

        decoder_output = self.norm(decoder_input)
        decoder_output_store = decoder_output.reshape(self.memory_queue_len, self.mem_size, num_mask_tokens, self.embed_dim)

        #  Predict masks
        decoder_output = decoder_output_store.mean(dim=0, keepdim=True)
        decoder_output = decoder_output.permute(0, 3, 1, 2)
        mask = torch.einsum("bcmn,chw->bhw", decoder_output, encoded)
        final_mask = self.mask_prediction(mask)

        final_mask = final_mask

        #  Generate the next memory block
        next_memory_block = decoder_output_store.mean(dim=(0, 1))
        next_memory_block = next_memory_block.unsqueeze(0)

        return final_mask, next_memory_block

class VideoSAM(nn.Module):
    def __init__(self, embed_dim=512, patch_size=8, mem_size = 10, memory_queue_len=15):
        super().__init__()
        self.image_encoder = AdaptiveVisionEncoder(embed_dim=embed_dim)
        self.pos_enc = UnifiedPositionalEncoding(embed_dim=embed_dim, spatial_size=32)
        self.mask_tokens = nn.Parameter(torch.zeros(mem_size, embed_dim))  # May be problematic
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.memory_queue_len = memory_queue_len
        
        self.decoder = FrameTransformerDecoder(memory_queue_len=memory_queue_len)

    def forward(self, x: torch.Tensor, prior_emb: torch.Tensor):
        if len(x.shape) == 3: # Image input (C, H, W)
            x = x.unsqueeze(0)

        T, C_in, H, W = x.shape
        H_enc = H // self.patch_size
        W_enc = W // self.patch_size

        masks = []
        
        memory_queue = torch.zeros(self.memory_queue_len, self.decoder.mem_size, self.embed_dim, device=x.device)

        for t in range(T):
            frame = x[t]  # [C, H, W]
            encoded = self.image_encoder(frame)
            
            # Incorporate prior
            prior = prior_emb.view(-1, 1, 1).expand(-1, H_enc, W_enc)
            encoded = torch.cat([encoded, prior], dim=0)
            encoded = self.pos_enc(encoded.mean(dim=0, keepdim=True), t)
            mask, new_memory = self.decoder(memory_queue, encoded, self.mask_tokens)
            masks.append(mask)

            memory_queue = torch.cat([memory_queue[1:], new_memory], dim=0)

        masks = torch.stack(masks, dim=0)
        return masks
    
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