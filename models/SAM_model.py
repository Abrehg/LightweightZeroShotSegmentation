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
    """
    Adds learnable temporal and spatial positional encodings.
    Spatial encodings are interpolated to match input feature map size.
    """
    def __init__(self, embed_dim=512, max_frames=32, initial_spatial_size=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_frames = max_frames
        
        # Temporal encoding for each frame
        self.temporal_enc = nn.Embedding(max_frames, embed_dim)
        
        # Learnable spatial encoding parameter, initialized for a certain size
        # It will be interpolated to the actual H, W of the feature map
        self.spatial_enc = nn.Parameter(torch.randn(1, embed_dim, initial_spatial_size, initial_spatial_size))

    def forward(self, x: torch.Tensor, t: int):
        """
        Args:
            x (torch.Tensor): Input features of shape (C, H, W) or (1, C, H, W).
                              C should be embed_dim.
            t (int): Current frame index (0 to T-1).
        Returns:
            torch.Tensor: Features with added positional encodings, same shape as input.
        """
        if x.ndim == 4 and x.shape[0] == 1: # (1, C, H, W)
            x_squeezed = True
            x = x.squeeze(0)
        else:
            x_squeezed = False
        
        C, H, W = x.shape
        if C != self.embed_dim:
            raise ValueError(f"Input feature dimension {C} does not match embed_dim {self.embed_dim}")

        # Interpolate spatial encoding to match input H, W
        # self.spatial_enc is (1, embed_dim, initial_H, initial_W)
        spatial_encoding = F.interpolate(self.spatial_enc, size=(H, W), mode='bilinear', align_corners=False)
        # spatial_encoding is (1, embed_dim, H, W) -> squeeze to (embed_dim, H, W)
        
        # Temporal encoding
        # Clamp t to be within [0, max_frames - 1]
        time_id = torch.clamp(torch.tensor([t], device=x.device), 0, self.max_frames - 1)
        temporal_encoding_vec = self.temporal_enc(time_id) # (1, embed_dim)
        # Expand temporal encoding to match spatial dimensions: (embed_dim, H, W)
        temporal_encoding = temporal_encoding_vec.view(self.embed_dim, 1, 1).expand(-1, H, W)

        x = x + spatial_encoding.squeeze(0) + temporal_encoding
        
        if x_squeezed:
            x = x.unsqueeze(0)
        return x


class AdaptiveVisionEncoder(nn.Module):
    """
    Resolution-agnostic image encoder.
    Outputs features [embed_dim, H_reduced, W_reduced].
    """
    def __init__(self, embed_dim=512, base_channels=64):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=2, padding=1), # H/2, W/2
            nn.GroupNorm(8, base_channels), # Using GroupNorm for stability
            nn.GELU()
        )
        
        # Downsampling blocks
        self.block1 = self._make_conv_block(base_channels, base_channels * 2, stride=2) # H/4, W/4
        self.block2 = self._make_conv_block(base_channels * 2, base_channels * 4, stride=2) # H/8, W/8
        # Further processing blocks (no stride)
        self.block3 = self._make_conv_block(base_channels * 4, embed_dim, stride=1) # H/8, W/8
        
        # Example of feature fusion (can be more sophisticated)
        # This projection is to ensure final output has embed_dim channels
        # If block3 already outputs embed_dim, this could be an Identity or a 1x1 conv for refinement
        self.feature_projection = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)


    def _make_conv_block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(min(16, out_c // (out_c // 16 if out_c >=16 else 1) ), out_c), # Ensure num_groups is valid
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(16, out_c // (out_c // 16 if out_c >=16 else 1) ), out_c),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input image frame [C_in, H, W] or [1, C_in, H, W]
        Returns:
            torch.Tensor: Encoded features [embed_dim, H_reduced, W_reduced]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0) # Add batch dim: [1, C_in, H, W]
        
        x = self.stem(x)    # -> [1, base_channels, H/2, W/2]
        x = self.block1(x)  # -> [1, base_channels*2, H/4, W/4]
        x = self.block2(x)  # -> [1, base_channels*4, H/8, W/8]
        x = self.block3(x)  # -> [1, embed_dim, H/8, W/8]
        
        x = self.feature_projection(x) # Ensure [1, embed_dim, H_reduced, W_reduced]
        return x.squeeze(0) # Return [embed_dim, H_reduced, W_reduced]


class FrameTransformerDecoder(nn.Module):
    def __init__(self, embed_dim=512, mem_size=10, num_layers=2, num_heads=8,
                 fixed_transformer_input_spatial_size=(16,16), # H_enc, W_enc for transformer context
                 mask_decoder_intermediate_channels=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.mem_size = mem_size # Number of slots in a memory unit / number of object queries
        self.fixed_H_enc, self.fixed_W_enc = fixed_transformer_input_spatial_size
        
        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, # Common practice
            batch_first=True # Expects (batch, seq, feature)
        )
        self.transformer_layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

        # Mask Prediction Head
        # This head takes the output of the transformer (next_memory_unit) and
        # the (pooled & processed) image features to predict a mask.
        
        # Projects memory unit queries to a dimension suitable for interacting with image features for mask generation
        self.mask_query_proj = nn.Linear(embed_dim, embed_dim // 2) 
        # Projects (pooled) image features to a compatible dimension
        self.mask_image_proj = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1)
        
        # Upsampling path for the mask
        # Input to this will be [1, 1, H_enc, W_enc] (after mask feature interaction)
        # It will upsample to a larger intermediate size, then F.interpolate to original.
        self.mask_upsample_head = nn.Sequential(
            nn.ConvTranspose2d(1, mask_decoder_intermediate_channels // 4, kernel_size=4, stride=2, padding=1), # H_enc*2, W_enc*2
            nn.ReLU(),
            nn.ConvTranspose2d(mask_decoder_intermediate_channels // 4, mask_decoder_intermediate_channels // 2, kernel_size=4, stride=2, padding=1), # H_enc*4, W_enc*4
            nn.ReLU(),
            nn.Conv2d(mask_decoder_intermediate_channels // 2, 1, kernel_size=3, padding=1) # Output 1 channel logit
        )


    def forward(self, 
                current_memory_queue: torch.Tensor, 
                encoded_image_context: torch.Tensor, 
                object_queries: torch.Tensor,
                unflattened_image_features_for_mask: torch.Tensor,
                original_H: int, 
                original_W: int):
        """
        Args:
            current_memory_queue (torch.Tensor): Shape [max_mem_len, mem_size, embed_dim]
            encoded_image_context (torch.Tensor): Flattened image features for transformer.
                                                  Shape [1, N_img_tokens, embed_dim], 
                                                  where N_img_tokens = fixed_H_enc * fixed_W_enc.
            object_queries (torch.Tensor): Learnable queries. Shape [1, mem_size, embed_dim].
            unflattened_image_features_for_mask (torch.Tensor): Pooled image features for mask head.
                                                               Shape [1, embed_dim, fixed_H_enc, fixed_W_enc].
            original_H (int): Original height of the input frame.
            original_W (int): Original width of the input frame.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - predicted_mask_logits (torch.Tensor): Shape [1, 1, original_H, original_W]
                - next_memory_unit (torch.Tensor): Shape [1, mem_size, embed_dim]
        """
        # 1. Aggregate past memory context
        # current_memory_queue: [max_mem_len, mem_size, embed_dim]
        if current_memory_queue.shape[0] > 0 : # if memory queue is not empty
            aggregated_past_memory = current_memory_queue.mean(dim=0, keepdim=True) # [1, mem_size, embed_dim]
            # Infuse memory into current queries
            transformer_tgt = object_queries + aggregated_past_memory 
        else: # First frame or no memory
            transformer_tgt = object_queries

        # 2. Pass through Transformer Decoder layers
        # transformer_tgt is [1, mem_size, embed_dim] (queries)
        # encoded_image_context is [1, N_img_tokens, embed_dim] (image context)
        transformer_output = self.transformer_layers(tgt=transformer_tgt, memory=encoded_image_context)
        next_memory_unit = self.output_norm(transformer_output) # [1, mem_size, embed_dim]

        # 3. Predict Mask
        # Project queries and image features for mask interaction
        # next_memory_unit is [1, mem_size, embed_dim]
        # unflattened_image_features_for_mask is [1, embed_dim, fixed_H_enc, fixed_W_enc]
        
        projected_queries_for_mask = self.mask_query_proj(next_memory_unit) # [1, mem_size, embed_dim/2]
        projected_image_for_mask = self.mask_image_proj(unflattened_image_features_for_mask) # [1, embed_dim/2, fixed_H_enc, fixed_W_enc]

        # Multiply query embeddings with image features (element-wise product after broadcasting query)
        # Then sum over the mem_size dimension to get a single feature map for the mask
        # This is one way to combine them; others (e.g., attention) are possible.
        # mask_features: [1, embed_dim/2, fixed_H_enc, fixed_W_enc]
        mask_features = torch.einsum('bqc,bchw->bchw', 
                                     projected_queries_for_mask.mean(dim=1, keepdim=True).transpose(1,2), # [1, embed_dim/2, 1] -> effectively [1, embed_dim/2] used for scaling
                                     projected_image_for_mask) 
        # A simpler combination: take the mean of projected queries and use it to scale image features
        # Or, a more common approach:
        # mask_logits_per_query = torch.einsum('bqc,bchw->bqhw', projected_queries_for_mask, projected_image_for_mask)
        # mask_logits_intermediate = mask_logits_per_query.mean(dim=1, keepdim=True) # [1, 1, fixed_H_enc, fixed_W_enc]
        
        # For simplicity, let's use a direct projection from the mean query embedding to a small spatial map
        # then combine with image features. Or even simpler:
        # Use the mean of the transformer output to modulate the image features.
        mean_transformer_output_spatial = next_memory_unit.mean(dim=1).unsqueeze(-1).unsqueeze(-1) # [1, embed_dim, 1, 1]
        # Modulate image features
        modulated_image_features = unflattened_image_features_for_mask * mean_transformer_output_spatial # Element-wise, relies on broadcasting
        
        # Project modulated features to 1 channel for the upsampling head
        single_channel_features = nn.Conv2d(self.embed_dim, 1, kernel_size=1, device=next_memory_unit.device)(modulated_image_features) # [1,1,fixed_H_enc, fixed_W_enc]

        # Upsample using the CNN head
        mask_logits_upsampled = self.mask_upsample_head(single_channel_features) # [1, 1, H_intermed, W_intermed]

        # Interpolate to original frame size
        predicted_mask_logits = F.interpolate(
            mask_logits_upsampled, 
            size=(original_H, original_W), 
            mode='bilinear', 
            align_corners=False
        ) # [1, 1, original_H, original_W]

        return predicted_mask_logits, next_memory_unit


class VideoSAM(nn.Module):
    def __init__(self, embed_dim=512, text_embed_dim=512, # Assuming prior_emb is text encoding
                 mem_size=10, # Number of slots in a memory unit / object queries
                 max_memory_length=15,
                 fixed_encoder_output_spatial_size=(16,16), # H_enc, W_enc for pooled image features
                 decoder_layers=2, decoder_heads=8,
                 pos_enc_max_frames=32, pos_enc_initial_spatial=16):
        super().__init__()
        self.embed_dim = embed_dim # For image features and memory units
        self.text_embed_dim = text_embed_dim
        self.mem_size = mem_size
        self.max_memory_length = max_memory_length
        self.fixed_H_enc, self.fixed_W_enc = fixed_encoder_output_spatial_size

        self.image_encoder = AdaptiveVisionEncoder(embed_dim=embed_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(fixed_encoder_output_spatial_size)
        
        # Positional encoding module
        self.pos_enc = UnifiedPositionalEncoding(
            embed_dim=embed_dim, # Positional encoding added to image features
            max_frames=pos_enc_max_frames,
            initial_spatial_size=pos_enc_initial_spatial 
        )
        
        # Learnable object queries (mask tokens)
        self.object_queries = nn.Parameter(torch.randn(1, mem_size, embed_dim)) # [1, mem_size, embed_dim]
        
        # The Frame Transformer Decoder
        self.decoder = FrameTransformerDecoder(
            embed_dim=embed_dim, 
            mem_size=mem_size, 
            num_layers=decoder_layers, 
            num_heads=decoder_heads,
            fixed_transformer_input_spatial_size=fixed_encoder_output_spatial_size
        )

        # LIFO Memory Queue (stores [1, mem_size, embed_dim] blocks)
        # Initialized with zeros. Shape: [max_memory_length, mem_size, embed_dim]
        # Note: Using unsqueezed version for easier cat with [1, mem_size, embed_dim] units later
        initial_memory_block = torch.zeros(1, mem_size, embed_dim)
        self.register_buffer(
            'memory_queue', 
            initial_memory_block.repeat(max_memory_length, 1, 1), # [max_mem_len, mem_size, embed_dim]
            persistent=False # Re-init in forward if processing separate videos
        )

        # Projection for text prior if its dimension differs from image embed_dim, or for fusion
        # Assuming prior_emb is [1, text_embed_dim]
        if embed_dim != text_embed_dim:
            self.text_prior_projection = nn.Linear(text_embed_dim, embed_dim)
        else:
            self.text_prior_projection = nn.Identity()


    def forward(self, images: torch.Tensor, text_prior_emb: torch.Tensor):
        """
        Args:
            images (torch.Tensor): Input video frames [T, C_img, H, W]. H, W can be variable.
            text_prior_emb (torch.Tensor): Text encoding [1, text_embed_dim].
        Returns:
            torch.Tensor: Predicted masks [T, 1, H, W] (logits).
        """
        if images.ndim == 3: # Single image frame (C,H,W)
            images = images.unsqueeze(0) # Treat as a video of 1 frame [1,C,H,W]

        num_frames = images.shape[0]
        original_H, original_W = images.shape[-2:]
        device = images.device

        # Re-initialize memory queue for each new video sequence for statelessness.
        # If processing continuous streams where state should persist, this might be handled differently.
        if self.max_memory_length > 0:
            self.memory_queue = torch.zeros(
                self.max_memory_length, self.mem_size, self.embed_dim, device=device
            )
        
        # Process text prior once
        # text_prior_emb: [1, text_embed_dim]
        processed_text_prior = self.text_prior_projection(text_prior_emb) # [1, embed_dim]
        # Reshape for broadcasting with image features: [embed_dim, 1, 1]
        text_prior_for_fusion = processed_text_prior.squeeze(0).view(self.embed_dim, 1, 1) 

        output_masks_logits = []

        for t in range(num_frames):
            current_frame = images[t] # [C_img, H, W]
            
            # 1. Encode image frame
            # encoded_frame_raw: [embed_dim, H_raw_reduced, W_raw_reduced]
            encoded_frame_raw = self.image_encoder(current_frame) 
            
            # 2. Pool to fixed spatial size for transformer and memory
            # pooled_frame_features: [embed_dim, fixed_H_enc, fixed_W_enc]
            pooled_frame_features = self.adaptive_pool(encoded_frame_raw.unsqueeze(0)).squeeze(0)

            # 3. Fuse with text prior
            # fused_features: [embed_dim, fixed_H_enc, fixed_W_enc]
            fused_features = pooled_frame_features + text_prior_for_fusion # Broadcasting

            # 4. Add positional encoding
            # positioned_features: [embed_dim, fixed_H_enc, fixed_W_enc]
            positioned_features = self.pos_enc(fused_features, t) 
            
            # 5. Prepare for Transformer Decoder
            # Flatten spatial dimensions and permute for batch_first=True
            # image_context_for_transformer: [1, N_img_tokens, embed_dim]
            # N_img_tokens = fixed_H_enc * fixed_W_enc
            image_context_for_transformer = positioned_features.flatten(start_dim=1).transpose(0,1).unsqueeze(0)
            
            # Unflattened features for mask head (passed to decoder)
            unflattened_img_features_for_mask_head = positioned_features.unsqueeze(0) # [1, embed_dim, fixed_H_enc, fixed_W_enc]

            # 6. Call Frame Transformer Decoder
            # self.object_queries is [1, mem_size, embed_dim]
            # self.memory_queue is [max_mem_len, mem_size, embed_dim]
            current_mask_logits, next_memory_unit = self.decoder(
                self.memory_queue, 
                image_context_for_transformer, 
                self.object_queries.to(device), # Ensure queries are on the same device
                unflattened_img_features_for_mask_head,
                original_H,
                original_W
            )
            output_masks_logits.append(current_mask_logits.squeeze(0)) # Store [1, H, W]

            # 7. Update LIFO Memory Queue
            if self.max_memory_length > 0:
                # next_memory_unit is [1, mem_size, embed_dim]
                # self.memory_queue is [max_len, mem_size, embed_dim]
                self.memory_queue = torch.cat((self.memory_queue[1:], next_memory_unit.detach()), dim=0)
        
        final_masks = torch.stack(output_masks_logits, dim=0) # [T, 1, H, W]
        return final_masks
    
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