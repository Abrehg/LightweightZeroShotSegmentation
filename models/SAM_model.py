import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def create_SAM(max_memory_length=10, enc_num_layers = 2, dec_num_layers = 2):
    return VideoSAM(
        max_memory_length=max_memory_length,
        enc_num_layers=enc_num_layers,
        dec_num_layers=dec_num_layers
        )

# Loss for overall model (Binary Cross Entropy + IoU Loss)
# pred_masks -> model output
# true_masks -> masks in dataset
def iou_loss(pred_masks, true_masks):
    true_masks = true_masks.float().view_as(pred_masks)
    bce = F.binary_cross_entropy_with_logits(pred_masks, true_masks)

    pred_prob = torch.sigmoid(pred_masks)
    true = true_masks.float()
    
    pred_flat = pred_prob.flatten(start_dim=1)
    true_flat = true.flatten(start_dim=1)
    
    intersection = (pred_flat * true_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return (1.0 - iou.mean()) + bce

class UnifiedPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=512, max_frames=32, initial_spatial_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_frames = max_frames
        
        # Temporal encoding
        self.temporal_enc = nn.Embedding(max_frames, embed_dim)
        
        # Learnable spatial encoding parameter
        self.spatial_enc = nn.Parameter(torch.randn(1, embed_dim, initial_spatial_size, initial_spatial_size))

    def forward(self, x: torch.Tensor, t: int):
        if x.ndim == 3:
            x = x.unsqueeze(0)
            
        C, H, W = x.shape[1:]
        
        # Spatial encoding
        spatial_encoding = F.interpolate(self.spatial_enc, size=(H, W), mode='bilinear', align_corners=False)
        
        # Temporal encoding
        time_id = torch.clamp(torch.tensor([t], device=x.device), 0, self.max_frames - 1)
        temporal_encoding_vec = self.temporal_enc(time_id)
        temporal_encoding = temporal_encoding_vec.view(1, self.embed_dim, 1, 1).expand(x.shape[0], -1, H, W)

        return x + spatial_encoding + temporal_encoding

class PatchVisionEncoder(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, num_layers = 2, num_heads = 8):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class FrameTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_layers=2, num_heads=8, mask_decoder_intermediate_channels=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

        # Upsampling Head for Mask Generation (ViT-style output generation)
        self.mask_upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, mask_decoder_intermediate_channels, kernel_size=2, stride=2),
            nn.GroupNorm(8, mask_decoder_intermediate_channels),
            nn.GELU(),
            nn.ConvTranspose2d(mask_decoder_intermediate_channels, mask_decoder_intermediate_channels // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(mask_decoder_intermediate_channels // 2, 1, kernel_size=1)
        )
        
        self.controller = nn.Linear(embed_dim, embed_dim)

    def forward(self, 
                current_image_features: torch.Tensor, 
                memory_queue: torch.Tensor, 
                object_queries: torch.Tensor,
                original_size: tuple):
        B, C, H, W = current_image_features.shape

        image_flat = current_image_features.flatten(2).transpose(1, 2)
        
        # Concatenate Image Features with Memory Queue along sequence dimension
        if memory_queue is not None and memory_queue.shape[1] > 0:
            transformer_context = torch.cat([image_flat, memory_queue], dim=1)
        else:
            transformer_context = image_flat

        updated_queries = self.transformer(tgt=object_queries, memory=transformer_context)
        updated_queries = self.output_norm(updated_queries)

        # Generate Mask via dynamic filters
        query_filters = self.controller(updated_queries)
        mask_logits = torch.bmm(query_filters, current_image_features.flatten(2))
        mask_logits = mask_logits / (self.embed_dim ** 0.5)
        mask_logits = mask_logits.view(B, -1, H, W)
        
        masks = F.interpolate(
            mask_logits, 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        )

        return masks, updated_queries

class VideoSAM(nn.Module):
    def __init__(self, 
                 embed_dim=768, 
                 prior_dim=768,
                 mem_size=1,
                 max_memory_length=10,
                 fixed_encoder_size=(14, 14),
                 dec_num_layers = 2,
                 enc_num_layers = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.mem_size = mem_size
        self.max_memory_length = max_memory_length
        self.fixed_encoder_size = fixed_encoder_size

        self.image_encoder = PatchVisionEncoder(embed_dim=embed_dim, num_layers=enc_num_layers, num_heads = max(1, embed_dim // 96))
        
        if embed_dim == prior_dim:
            self.prior_proj = nn.Identity()
        else:
            self.prior_proj = nn.Linear(prior_dim, embed_dim)
        
        self.pos_enc = UnifiedPositionalEncoding(embed_dim=embed_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(fixed_encoder_size)

        self.object_queries = nn.Parameter(torch.randn(1, mem_size, embed_dim))
        self.decoder = FrameTransformerDecoder(embed_dim=embed_dim, num_layers=dec_num_layers)

    def init_memory(self, batch_size, device):
        return torch.zeros(batch_size, 0, self.embed_dim, device=device)

    def forward(self, image: torch.Tensor, prior_grid: torch.Tensor, memory_queue=None, frame_idx=0):
        if image.ndim == 5:
            assert image.shape[1] == 1, "Iterate T in the caller; pass one frame at a time."
            image = image.squeeze(1)
            
        B, C, H, W = image.shape
        device = image.device

        if memory_queue is None:
            memory_queue = self.init_memory(B, device)

        # 1. Process Prior grid
        prior_embed = self.prior_proj(prior_grid)
        H_grid, W_grid = self.fixed_encoder_size
        prior_spatial = prior_embed.transpose(1, 2).reshape(B, self.embed_dim, H_grid, W_grid)

        # 2. Process Frame
        img_enc = F.interpolate(image, size=(256,256), mode='bilinear', align_corners=False)
        img_feat = self.image_encoder(img_enc)
        img_feat = self.adaptive_pool(img_feat)
            
        # 3. Combine and generate mask
        conditioned_feat = img_feat + prior_spatial
        encoded_context = self.pos_enc(conditioned_feat, frame_idx)
        mask_logits, updated_query = self.decoder(
            current_image_features=encoded_context,
            memory_queue=memory_queue,
            object_queries=self.object_queries.expand(B, -1, -1),
            original_size=(H, W)
        )

        # 4. Update memory
        if self.max_memory_length > 0:
            memory_queue = torch.cat([memory_queue, updated_query], dim=1)
            cap = self.max_memory_length * self.mem_size
            if memory_queue.shape[1] > cap:
                memory_queue = memory_queue[:, -cap:]
         
        return mask_logits.squeeze(1), memory_queue
    
    def load_weights(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

# # input: (batch, num_frames (1 for images), num_channels, height, width) (height and width have to be greater than 16 and divisible by 8)
# # output: (batch, num_frames (1 for images), height, width)

# from clip_model import create_text_encoder, CLIPTokenize
# from prior_model import create_prior
# model = create_SAM()
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
# video = torch.randn(2, 8, 3, 512, 512)  # Batch of 2 videos (B, T, C, H, W)
# masks = model(video, prior_emb)
# print(f"Video test output shape {masks.size()}")  # Should output: torch.Size([2, 8, 512, 512])

# model.store_weights("/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models", "SAMWeights")

# new_model = create_SAM()
# new_model.load_weights("/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models/SAMWeights")

# # Test with image input
# image = torch.randn(2, 1, 3, 224, 224)  # Batch of 2 images (B, 1, C, H, W)
# new_masks = new_model(image, prior_emb)
# print(f"New Image test output shape: {new_masks.size()}")  # Should output: torch.Size([2, 1, 224, 224])

# # Test with video input
# video = torch.randn(2, 8, 3, 512, 512)  # Batch of 2 videos (B, T, C, H, W)
# new_masks = new_model(video, prior_emb)
# print(f"New Video test output shape {new_masks.size()}")  # Should output: torch.Size([2, 8, 512, 512])