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
                 text_embed_dim=256,
                 image_embed_dim=128,
                 mem_size=5,
                 num_layers=4):
        super().__init__()

        self.token_embed = nn.Embedding(49408, text_embed_dim)
        self.pos_embed = nn.Embedding(512, text_embed_dim)  # Increased position capacity
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_embed_dim,
                nhead=8,
                dim_feedforward=4*text_embed_dim,
                batch_first=True
            ),
            num_layers=4
        )
        self.text_ln = nn.LayerNorm(text_embed_dim)
        
        # Resolution-adaptive image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((None, None)),  # Preserve spatial dims
            nn.Conv2d(128, image_embed_dim, 3, padding=1)
        )
        
        # Memory buffers with FIFO update
        self.register_buffer('memory', torch.zeros(mem_size, image_embed_dim, 64, 64))
        self.register_buffer('mask_memory', torch.zeros(mem_size, 1, 64, 64))
        self.mem_size = mem_size
        self.mem_ptr = 0
        
        # Temporal fusion components
        self.temporal_fuser = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,1,1),
            padding=(1,0,0)
        )
        
        # Unified fusion layers
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(384 if i == 0 else 128, 128, 1),
                nn.GroupNorm(8, 128),
                nn.GELU()
            ) for i in range(num_layers)
        ])
        
        # Mask decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, 1)
        )

    def update_memory(self, features, masks):
        """FIFO memory update with spatial features"""
        with torch.no_grad():
            # Concatenate features and masks
            new_entry = torch.cat([
                F.adaptive_avg_pool2d(features, (64, 64)),
                F.interpolate(masks, size=(64, 64))
            ], dim=1)  # Shape: (B*T, 128+1, 64, 64)
        
            # Aggregate batch dimension
            aggregated_entry = new_entry.mean(dim=0, keepdim=True)  # (1, 129, 64, 64)
        
            if self.mem_ptr < self.mem_size:
                # Store aggregated features and masks
                self.memory[self.mem_ptr] = aggregated_entry[:, :self.memory.size(1)]  # (1, 128, 64, 64)
                self.mask_memory[self.mem_ptr] = aggregated_entry[:, -1:]  # (1, 1, 64, 64)
                self.mem_ptr += 1
            else:
                # FIFO update with aggregated entry
                self.memory = torch.cat([self.memory[1:], aggregated_entry[:, :self.memory.size(1)]])
                self.mask_memory = torch.cat([self.mask_memory[1:], aggregated_entry[:, -1:]])

    def forward(self, x, text_tokens):
        # Handle video/image inputs
        is_video = x.ndim == 5
        B = x.size(0)
        T = x.size(1) if is_video else 1

        # Encode frames
        x = x.flatten(0, 1) if is_video else x.unsqueeze(1)
        img_feats = self.image_encoder(x)  # (B*T, 128, H', W')

        # Text processing
        B_text, seq_len = text_tokens.shape
        positions = torch.arange(seq_len, device=text_tokens.device).expand(B_text, -1)
        token_emb = self.token_embed(text_tokens)
        pos_emb = self.pos_embed(positions)
        text_feats = self.text_ln(self.transformer(token_emb + pos_emb)).mean(dim=1)
    
        # Expand text features to match spatial dimensions
        text_feats = text_feats.view(B, 1, -1).expand(-1, T, -1)  # (B, T, 256)
        text_feats = text_feats.reshape(B*T, -1)[:, :, None, None]  # (B*T, 256, 1, 1)
        text_feats = text_feats.expand(-1, -1, *img_feats.shape[-2:])  # (B*T, 256, H', W')

        # Temporal fusion (if video)
        if is_video:
            img_feats = img_feats.view(B, T, *img_feats.shape[1:])
            img_feats = img_feats.permute(0, 2, 1, 3, 4)  # (B, 128, T, H', W')
            img_feats = self.temporal_fuser(img_feats)
            img_feats = img_feats.permute(0, 2, 1, 3, 4).flatten(0, 1)  # (B*T, 128, H', W')

        # Concatenate along channels
        fused = torch.cat([img_feats, text_feats], dim=1)  # (B*T, 384, H', W')
    
        for layer in self.fusion:
            fused = layer(fused)
        
        # Decode masks
        masks = self.decoder(fused)
        
        # Update memory during training
        if self.training:
            self.update_memory(img_feats, masks)
        
        return torch.sigmoid(masks)

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

from .SAM_model import VideoSAM
from .clip_model import create_text_encoder
from .prior_model import create_prior

class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Text processing components
        self.text_encoder = create_text_encoder()
        self.prior = create_prior()
        self.sam_decoder = VideoSAM()
        
        # Freeze all components
        for module in [self.text_encoder, self.prior, self.sam_decoder]:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, x, text_tokens):
        """
        Args:
            x: Input video/image tensor (B, T, C, H, W)
            text_tokens: Tokenized text indices (B, seq_len)
        """
        # Text processing
        text_emb = self.text_encoder(text_tokens)
        prior_emb = self.prior(text_emb)
        
        # VideoSAM processing
        return self.sam_decoder(x, prior_emb)


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

# # Compute loss
# loss = student.compute_distill_loss(student_out, teacher_out)
# loss.backward()

# print("Completed sequence")

# # Inference
# student.eval()
# frame_sequence = [torch.randn(3, 512, 512) for _ in range(10)]
# masks = process_video(student, frame_sequence, "A dancing person", update_memory=True)