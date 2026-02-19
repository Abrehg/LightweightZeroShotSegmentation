import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def create_SAM():
    return VideoSAM()

# Loss for overall model
# pred_masks -> model output
# true_masks -> masks in dataset
def iou_loss(pred_masks, true_masks):
    pred_prob = torch.sigmoid(pred_masks)
    true = true_masks.float()
    
    pred_flat = pred_prob.flatten(start_dim=1)
    true_flat = true.flatten(start_dim=1)
    
    intersection = (pred_flat * true_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return 1.0 - iou.mean()

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

class AdaptiveVisionEncoder(nn.Module):
    def __init__(self, embed_dim=512, base_channels=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU()
        )
        
        self.block1 = self._make_conv_block(base_channels, base_channels * 2, stride=2)
        self.block2 = self._make_conv_block(base_channels * 2, base_channels * 4, stride=2)
        self.block3 = self._make_conv_block(base_channels * 4, embed_dim, stride=1)
        
        self.feature_projection = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def _make_conv_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(min(16, out_c // (out_c // 16 if out_c >=16 else 1) ), out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(16, out_c // (out_c // 16 if out_c >=16 else 1) ), out_c),
            nn.GELU()
        )

    #Input needs to be of form [B, 3, H, W]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.feature_projection(x)
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
                 fixed_encoder_size=(14, 14)):
        super().__init__()
        self.embed_dim = embed_dim
        self.mem_size = mem_size
        self.max_memory_length = max_memory_length
        self.fixed_encoder_size = fixed_encoder_size

        self.image_encoder = AdaptiveVisionEncoder(embed_dim=embed_dim)
        
        if embed_dim == prior_dim:
            self.prior_proj = nn.Identity()
        else:
            self.prior_proj = nn.Linear(prior_dim, embed_dim)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(fixed_encoder_size)
        self.pos_enc = UnifiedPositionalEncoding(embed_dim=embed_dim)
        
        self.object_queries = nn.Parameter(torch.randn(1, mem_size, embed_dim))
        self.decoder = FrameTransformerDecoder(embed_dim=embed_dim)

    def forward(self, images: torch.Tensor, prior_grid: torch.Tensor):
        if images.ndim == 4:
            images = images.unsqueeze(1) 
            
        B, T, C, H, W = images.shape
        device = images.device

        prior_embed = self.prior_proj(prior_grid)
        
        H_grid, W_grid = self.fixed_encoder_size
        prior_spatial = prior_embed.transpose(1, 2).reshape(B, self.embed_dim, H_grid, W_grid)

        # 2. Initialize Memory
        memory_queue = torch.zeros(B, 0, self.embed_dim, device=device)
        output_masks = []

        # 3. Process Frames
        for t in range(T):
            current_img = images[:, t] 
            
            img_feat = self.image_encoder(current_img) 
            img_feat = self.adaptive_pool(img_feat)    
            
            conditioned_feat = img_feat + prior_spatial
            
            encoded_context = self.pos_enc(conditioned_feat, t)
            
            mask_logits, updated_query = self.decoder(
                current_image_features=encoded_context,
                memory_queue=memory_queue,
                object_queries=self.object_queries.expand(B, -1, -1),
                original_size=(H, W)
            )
            
            output_masks.append(mask_logits)

            if self.max_memory_length > 0:
                current_mem = updated_query 
                memory_queue = torch.cat([memory_queue, current_mem], dim=1)
                
                if memory_queue.shape[1] > self.max_memory_length * self.mem_size:
                    memory_queue = memory_queue[:, -self.max_memory_length * self.mem_size:, :]
        
        final_output = torch.cat(output_masks, dim=1)
            
        return final_output
    
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