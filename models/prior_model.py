import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import CLIPVisionModel

# Prior model factory
def create_prior():
    return Prior()

# Input encoding shape: (1, seq_len, 768) (max seq_len is 77)
# Output shape: (1, 196, 768)
class Prior(nn.Module):
    def __init__(self, text_dim=768, vision_dim=768, num_patches=196, num_layers=12):
        super().__init__()
        self.num_patches = num_patches
        self.vision_dim = vision_dim

        self.text_proj = nn.Linear(text_dim, vision_dim)
        self.query = nn.Parameter(torch.randn(1, num_patches, vision_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, vision_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=vision_dim,
            nhead=8,
            dim_feedforward=4*vision_dim,
            batch_first=True,
            norm_first=True 
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.final_ln = nn.LayerNorm(vision_dim)
        self.output_proj = nn.Linear(vision_dim, vision_dim)

    def forward(self, text_emb):
        B, _, _ = text_emb.shape
        
        memory = self.text_proj(text_emb) 
        target = self.query.expand(B, -1, -1)
        target = target + self.pos_embed

        x = self.layers(target, memory)
        
        x = self.final_ln(x)
        output = self.output_proj(x)
        
        return output
    
    def load_weights(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

class TeacherCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 1:, :] 
            return features

def PriorLoss(predicted_grid, target_grid):
    loss = 1.0 - F.cosine_similarity(
        predicted_grid.reshape(-1, 768), 
        target_grid.reshape(-1, 768), 
        dim=-1
    ).mean()
    return loss

# from clip_model import create_text_encoder, CLIPTokenize

# text_encoder = create_text_encoder()
# prior = create_prior()

# input_text = ["Input string"]
# tokens = CLIPTokenize(input_text)
# encodings = text_encoder(tokens)
# prior_emb = prior(encodings)
# print(prior_emb.size())

# input_text = ["Input string", "Another input string"]
# tokens = CLIPTokenize(input_text)
# encodings = text_encoder(tokens)
# prior_emb = prior(encodings)
# print(prior_emb.size())


# input_text = ["A photo of a cat", "A photo of a dog"] 
# text_tokens = CLIPTokenize(input_text)
# real_images = torch.randn(2, 3, 224, 224)

# tokens = CLIPTokenize(input_text)
# encodings = text_encoder(tokens)
# prior_emb = prior(encodings)
# teacher = TeacherCLIP()
# target_grid = teacher.get_feature_grid(real_images)
# print(f"prior size: {prior_emb.size()}")
# print(f"target size: {prior_emb.size()}")

# prior.store_weights("/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models", "PriorWeights")

# newPrior = create_prior()
# newPrior.load_weights("/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models/PriorWeights")

# input_text = ["Input string"]
# tokens = CLIPTokenize(input_text)
# encodings = text_encoder(tokens)
# new_prior_emb = newPrior(encodings)
# print(new_prior_emb.size())