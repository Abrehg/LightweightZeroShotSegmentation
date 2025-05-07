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
                 num_layers=4,
                 memory_queue_len=15):  # Added memory_queue_len
        super().__init__()

        self.token_embed = nn.Embedding(49408, text_embed_dim)
        self.pos_embed = nn.Embedding(512, text_embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_embed_dim,
                nhead=8,
                dim_feedforward=4*text_embed_dim
            ),
            num_layers=num_layers
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, image_embed_dim, kernel_size=3, stride=2, padding=1),
        )
        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(text_embed_dim + image_embed_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        )
        self.memory_queue_len = memory_queue_len
        self.memory_queue = torch.zeros(memory_queue_len, 1, text_embed_dim + image_embed_dim) # Initialize with zeros
        self.teacher = None

    def forward(self, images, text):
        if images.ndim == 3:  # Image input (C, H, W)
            images = images.unsqueeze(0)
        
        T, C, H, W = images.shape
        masks = []

        text_embeds = self.token_embed(text)
        text_embeds = text_embeds + self.pos_embed(torch.arange(text_embeds.size(1), device=text_embeds.device))
        text_embeds = self.transformer(text_embeds)

        for t in range(T):
            image = images[t]
            image_embeds = self.image_encoder(image)
            image_embeds = F.interpolate(image_embeds, size=text_embeds.shape[1], mode='bilinear', align_corners=False)
            
            # Concatenate text and image embeddings
            frame_embeds = torch.cat([text_embeds, image_embeds], dim=0)
            
            # Use memory queue
            decoder_input = torch.cat([frame_embeds.unsqueeze(0), self.memory_queue], dim=0).permute(1, 0, 2) # Prepend current frame embeds

            mask = self.mask_decoder(decoder_input[:, 0]).unsqueeze(0)
            masks.append(mask)

            # Update memory queue
            self.memory_queue = torch.cat([self.memory_queue[1:], frame_embeds.unsqueeze(0)], dim=0)

        return torch.stack(masks, dim=0).unsqueeze(1)

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