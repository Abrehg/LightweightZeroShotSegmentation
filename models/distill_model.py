import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def create_Student(text_transformer_layers=2, max_memory_length=10, output_layers=2):
    return DistilledMemoryStudent(
        text_transformer_layers=text_transformer_layers,
        max_memory_length=max_memory_length,
        num_decoder_layers=output_layers
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

class DistilledMemoryStudent(nn.Module):
    def __init__(self,
                 vocab_size=49408,
                 text_seq_len=77,
                 text_embed_dim=128,
                 text_transformer_layers=2,
                 text_transformer_heads=4,
                 image_input_channels=3,
                 image_feature_dim=128,
                 output_mask_channels=1, 
                 fixed_intermediate_spatial_size=(16, 16),
                 max_memory_length=10,
                 num_decoder_layers=2):
        super().__init__()

        self.text_embed_dim = text_embed_dim
        self.image_feature_dim = image_feature_dim
        self.max_memory_length = max_memory_length
        self.fixed_intermediate_H, self.fixed_intermediate_W = fixed_intermediate_spatial_size
        
        # --- Text Encoder ---
        self.token_embed = nn.Embedding(vocab_size, text_embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, text_seq_len, text_embed_dim)) 
        
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_embed_dim,
            nhead=text_transformer_heads,
            dim_feedforward=text_embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.text_transformer = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=text_transformer_layers
        )

        # --- Image Encoder ---
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32), 
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.Conv2d(64, image_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(16, image_feature_dim // 4 if image_feature_dim >=4 else image_feature_dim), image_feature_dim), 
            nn.GELU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(fixed_intermediate_spatial_size)

        # --- Mask Cross Attention ---
        fusion_dim = image_feature_dim
        num_spatial_tokens = self.fixed_intermediate_H * self.fixed_intermediate_W
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, num_spatial_tokens, fusion_dim) * 0.02
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=fusion_dim,
            nhead=text_transformer_heads,
            dim_feedforward=fusion_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.fusion_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)
 
        # --- Mask Decoder ---
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(fusion_dim, 128, kernel_size=2, stride=2), # 16->32
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # 32->64
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # 64->128
            nn.GELU(),
            nn.Conv2d(32, output_mask_channels, kernel_size=1)
        )
        
    def forward(self, images, text_tokens):
        # Handle single image input by adding time dimension
        if images.ndim == 4: 
            images = images.unsqueeze(1)
        
        B, T, C, H, W = images.shape
        device = images.device

        # --- Process Text ---
        text_embeds = self.token_embed(text_tokens)
        text_embeds = text_embeds + self.pos_embed[:, :text_tokens.size(1), :]
        
        # Transformer encoding
        text_features_seq = self.text_transformer(text_embeds)

        # --- Initialize Memory Queue ---
        fusion_dim = self.image_feature_dim
        
        if self.max_memory_length > 0:
            memory_queue = torch.zeros(
                B, self.max_memory_length, fusion_dim, 
                self.fixed_intermediate_H, self.fixed_intermediate_W
            ).to(device)
        
        output_masks = []

        # --- Process Frames ---
        for t in range(T):
            current_frame = images[:, t]

            img_feats_raw = self.image_encoder(current_frame) 
            img_feats_pooled = self.adaptive_pool(img_feats_raw)

            img_tokens = img_feats_pooled.flatten(2).transpose(1, 2)  # (B, H'*W', img_dim)
            img_tokens = img_tokens + self.spatial_pos_embed
 
            # Cross-attention: image tokens (tgt) attend to text tokens (memory)
            fused_tokens = self.fusion_decoder(
                tgt=img_tokens,
                memory=text_features_seq
            )  # (B, H'*W', fusion_dim)
            fused_tokens = self.fusion_norm(fused_tokens)
 
            # Reshape back to spatial feature map
            fused_spatial = fused_tokens.transpose(1, 2).view(
                B, fusion_dim, self.fixed_intermediate_H, self.fixed_intermediate_W
            )

            if self.max_memory_length > 0:
                memory_queue = torch.cat((memory_queue[:, 1:], fused_spatial.unsqueeze(1)), dim=1)
                features_for_decoder = memory_queue.mean(dim=1)
            else:
                features_for_decoder = fused_spatial

            mask_logits = self.mask_decoder(features_for_decoder)
            mask_final = F.interpolate(
                mask_logits, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            
            output_masks.append(mask_final)

        final_masks = torch.stack(output_masks, dim=1)
        
        return final_masks.squeeze(2)
            
    def compute_distill_loss(self, student_masks, teacher_masks, true_masks):
        # Distillation
        student_prob = torch.sigmoid(student_masks)
        teacher_prob = torch.sigmoid(teacher_masks)
        distill_loss = F.l1_loss(student_prob, teacher_prob)
        
        # Supervised
        supervised_loss = iou_loss(student_masks, true_masks)
        
        return distill_loss + supervised_loss
    
    def load_weights(self, filename):
        if os.path.exists(filename):
            state_dict = torch.load(filename)
            self.load_state_dict(state_dict)

    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

# from SAM_model import VideoSAM
# from clip_model import create_text_encoder, CLIPTokenize
# from prior_model import create_prior

# class TeacherModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.text_encoder = create_text_encoder()
#         self.prior = create_prior()
#         self.sam_decoder = VideoSAM()

#     def forward(self, x, text_tokens):
#         text_emb = self.text_encoder(text_tokens)
#         prior_emb = self.prior(text_emb)
#         return self.sam_decoder(x, prior_emb)

# student = DistilledMemoryStudent()
# teacher = TeacherModel()

# print(student)
# print(teacher)

# # Training step
# student.train()
# video_input = torch.randn(2, 5, 3, 256, 256)

# inputText = ["Test Input", "Another Test Input"]
# text_input = CLIPTokenize(inputText)

# # Forward passes
# student_out = student(video_input, text_input)
# teacher_out = teacher(video_input, text_input)

# print(f"Teacher output shape: {teacher_out.size()}")
# print(f"Student output shape: {student_out.size()}")

# student.store_weights("/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models", "StudentWeights")

# newStudent = create_Student()
# newStudent.load_weights("/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models/StudentWeights")

# student_out = student(video_input, text_input)

# print(f"Student output shape: {student_out.size()}")