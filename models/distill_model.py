import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def create_Student():
    return DistilledMemoryStudent()

def iou_loss(pred_masks, true_masks):
    pred_prob = torch.sigmoid(pred_masks)

    pred_flat = pred_prob.view(pred_prob.size(0), -1)
    true_flat = true_masks.float().view(true_masks.size(0), -1)
    
    intersection = (pred_flat * true_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + true_flat.sum(dim=1) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return 1.0 - iou.mean()

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
                 max_memory_length=10):
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

        # --- Image Encoder (Lightweight) ---
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

        # --- Mask Decoder ---
        combined_feature_dim = image_feature_dim + text_embed_dim
        
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(combined_feature_dim, 128, kernel_size=2, stride=2), # 16->32
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

        # --- Process Text (Once per batch) ---
        text_embeds = self.token_embed(text_tokens)
        text_embeds = text_embeds + self.pos_embed[:, :text_tokens.size(1), :]
        
        # Transformer encoding
        text_features_seq = self.text_transformer(text_embeds)
        text_features_global = text_features_seq.mean(dim=1) 

        # --- Initialize Memory Queue ---
        combined_dim = self.image_feature_dim + self.text_embed_dim
        
        if self.max_memory_length > 0:
            memory_queue = torch.zeros(
                B, self.max_memory_length, combined_dim, 
                self.fixed_intermediate_H, self.fixed_intermediate_W
            ).to(device)
        
        output_masks = []

        # --- Process Frames ---
        for t in range(T):
            current_frame = images[:, t]

            img_feats_raw = self.image_encoder(current_frame) 
            img_feats_pooled = self.adaptive_pool(img_feats_raw)

            text_feats_expanded = text_features_global.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, self.fixed_intermediate_H, self.fixed_intermediate_W
            )
            
            current_combined = torch.cat([img_feats_pooled, text_feats_expanded], dim=1)

            if self.max_memory_length > 0:
                memory_queue = torch.cat((memory_queue[:, 1:], current_combined.unsqueeze(1)), dim=1)
                features_for_decoder = memory_queue.mean(dim=1)
            else:
                features_for_decoder = current_combined

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