import torch
import torch.nn as nn
import torch.nn.functional as F
from .clip_model import CLIPTokenize

def iou_loss(pred_masks, true_masks):
    """
    Compute IoU loss for batched tensors
    pred_masks: (B, 1, H, W) tensor of logits
    true_masks: (B, H, W) tensor of 0/1 values
    """
    pred_prob = torch.sigmoid(pred_masks)
    true = true_masks.unsqueeze(1).float()  # Add channel dimension
    
    intersection = (pred_prob * true).sum(dim=(2,3))
    union = pred_prob.sum(dim=(2,3)) + true.sum(dim=(2,3)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return (1.0 - iou).mean()

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
                 fixed_intermediate_spatial_size=(8, 8), 
                 max_memory_length=15):
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
            # Layer 1: HxW -> H/2 x W/2
            nn.Conv2d(image_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32), 
            nn.GELU(),
            # Layer 2: H/2 x W/2 -> H/4 x W/4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            # Layer 3: H/4 x W/4 -> H/8 x W/8
            nn.Conv2d(64, image_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(16, image_feature_dim // 4 if image_feature_dim >=4 else image_feature_dim), image_feature_dim), 
            nn.GELU()
        )
        
        # Adaptive pooling to resize image features to a fixed spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(fixed_intermediate_spatial_size)

        # --- Mask Decoder ---
        combined_feature_dim = image_feature_dim + text_embed_dim
        decoder_output_H = self.fixed_intermediate_H * (2**3) 
        decoder_output_W = self.fixed_intermediate_W * (2**3)

        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(combined_feature_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(16, 128 // 4), 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(min(16, 64 // 4), 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, output_mask_channels, kernel_size=4, stride=2, padding=1)
        )
        
        # --- Memory Queue ---
        initial_memory_queue = torch.zeros(
            self.max_memory_length,
            combined_feature_dim,
            self.fixed_intermediate_H,
            self.fixed_intermediate_W
        )
        self.register_buffer('memory_queue', initial_memory_queue, persistent=False) # persistent=False if re-init in forward
        
        self.teacher = None

    def forward(self, images, text_tokens):
        """
        Args:
            images (torch.Tensor): Input image frames, shape [T, C, H, W] 
                                   H and W can be variable.
            text_tokens (torch.Tensor): Input text tokens, shape [1, S] (e.g., [1, 77])
        Returns:
            torch.Tensor: Output masks, shape [T, output_mask_channels, H_original, W_original]
        """
        if images.ndim == 3: 
            images = images.unsqueeze(0) 
        
        num_frames = images.shape[0]
        original_H, original_W = images.shape[-2:] 
        current_device = images.device

        print(f"Input image shape: {images.shape}")
        print(f"Input text shape: {text_tokens.shape}")

        #Re-initialize memory queue at the start of each forward call for statelessness between sequences.
        if self.max_memory_length > 0:
            self.memory_queue = torch.zeros(
                self.max_memory_length,
                self.image_feature_dim + self.text_embed_dim,
                self.fixed_intermediate_H,
                self.fixed_intermediate_W,
                device=current_device
            )

        print(f"Input memory shape: {self.memory_queue.shape}")

        # --- Process Text (once for all frames) ---
        text_embeds = self.token_embed(text_tokens)  # [1, S, text_embed_dim]
        text_embeds = text_embeds + self.pos_embed[:, :text_tokens.size(1), :]
        text_features_seq = self.text_transformer(text_embeds) # [1, S, text_embed_dim]
        text_features_global = text_features_seq.mean(dim=1) # [1, text_embed_dim]

        output_masks = []
        for t in range(num_frames):
            current_frame = images[t].unsqueeze(0) # [1, C, H_original, W_original]

            # --- Process Image Frame ---
            image_frame_features_raw = self.image_encoder(current_frame) 
            pooled_image_features = self.adaptive_pool(image_frame_features_raw)

            # --- Fuse Text and Image Features for the current frame ---
            text_features_spatial = text_features_global.unsqueeze(-1).unsqueeze(-1)
            text_features_expanded = text_features_spatial.expand(
                -1, -1, self.fixed_intermediate_H, self.fixed_intermediate_W
            )
            
            # current_combined_features shape: [1, combined_dim, fixed_intermediate_H, fixed_intermediate_W]
            current_combined_features = torch.cat([pooled_image_features, text_features_expanded], dim=1)

            # --- Update Memory Queue (LIFO) ---
            if self.max_memory_length > 0:
                # memory_queue is [max_len, C_mem, H_fixed, W_fixed]
                self.memory_queue = torch.cat((self.memory_queue[1:], current_combined_features), dim=0)
                features_for_decoder = self.memory_queue.mean(dim=0, keepdim=True)
            else: # No memory
                features_for_decoder = current_combined_features

            # --- Decode to Mask (Intermediate Fixed Size) ---
            mask_pred_intermediate = self.mask_decoder(features_for_decoder)
            
            # --- Interpolate to Original Frame Size ---
            mask_pred_final = F.interpolate(
                mask_pred_intermediate, 
                size=(original_H, original_W), 
                mode='bilinear', 
                align_corners=False
            )
            output_masks.append(mask_pred_final.squeeze(0))

        final_masks = torch.stack(output_masks, dim=0)
        return final_masks

    def register_teacher(self, teacher_model):
        """Register teacher model for co-distillation"""
        self.teacher = teacher_model
            
    def compute_distill_loss(self, student_out, teacher_out, true_mask):
        """Combined IoU loss against true masks and teacher outputs"""
        # Loss between student predictions and ground truth
        student_true_loss = iou_loss(student_out, true_mask)
        
        # Align student output to teacher resolution
        student_resized = F.interpolate(
            student_out, 
            size=teacher_out.shape[-2:],
            mode='bilinear', 
            align_corners=False
        )
        
        # Loss between student and teacher predictions
        student_teacher_loss = iou_loss(student_resized, teacher_out)
        
        # Combine losses
        return student_true_loss + student_teacher_loss

    def compute_distill_loss(self, student_masks, teacher_masks, true_masks):
        """
        Compute a combined loss for distillation and mask prediction.
        """
        distill_loss = F.l1_loss(student_masks, teacher_masks)
        iou_l = iou_loss(student_masks, true_masks)
        return distill_loss + iou_l

def load_student(weights_path):
    """Load trained student model"""
    student = DistilledMemoryStudent()
    state_dict = torch.load(weights_path)
    student.load_state_dict(state_dict)
    return student.eval()

def process_video(student, frames, prompt, update_memory=False):
    """Process video with optional memory updates"""
    student.eval()
    
    with torch.no_grad():
        text_ids = CLIPTokenize(prompt)
        text_feats = student.text_encoder(text_ids)
        
        masks = []
        for frame in frames:
            frame = frame.unsqueeze(0).to(text_feats.device)
            mask = student(frame.unsqueeze(1), text_feats)  # Add temporal dim
            
            if update_memory:
                student.update_memory(
                    student.image_encoder(frame),
                    mask
                )
                
            masks.append(mask.squeeze().cpu())
    
    return torch.stack(masks)

# from .SAM_model import VideoSAM
# from .clip_model import create_text_encoder
# from .prior_model import create_prior

# class TeacherModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Text processing components
#         self.text_encoder = create_text_encoder()
#         self.prior = create_prior()
#         self.sam_decoder = VideoSAM()

#     def forward(self, x, text_tokens):
#         """
#         Args:
#             x: Input video/image tensor (B, T, C, H, W)
#             text_tokens: Tokenized text indices (B, seq_len)
#         """
#         # Text processing
#         text_emb = self.text_encoder(text_tokens)
#         prior_emb = self.prior(text_emb)
        
#         # VideoSAM processing
#         return self.sam_decoder(x, prior_emb)


# # Initialize with teacher for distillation
# student = DistilledMemoryStudent()
# teacher = TeacherModel()
# student.register_teacher(teacher)

# # Training step
# student.train()
# video_input = torch.randn(2, 5, 3, 256, 256)

# inputText = ["Test Input", "Another Test Input"]
# text_input = CLIPTokenize(inputText)

# # Forward passes
# student_out = student(video_input, text_input)
# with torch.no_grad():
#     teacher_out = teacher(video_input, text_input)

# print("Completed sequence")