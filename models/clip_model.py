import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer

MAXSEQLENGTH = 77

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def CLIPTokenize(inputText):
    tokens = tokenizer(inputText, padding="max_length", truncation=True, max_length=MAXSEQLENGTH, return_tensors="pt")
    # tokens shape = (numSequences, 77)
    return tokens['input_ids']

def VecToText(vector):
    return tokenizer.convert_ids_to_tokens(vector)

# Text encoder factory
def create_text_encoder(num_layers=11):
    return TextEncoder(
        vocab_size=49408, 
        max_seq_len=MAXSEQLENGTH,
        embed_dim=768, 
        num_layers=num_layers
    )

# Image encoder factory
def create_image_encoder(num_layers=6):
    return ImageEncoder(
        embed_dim=768,
        input_channels=3,
        num_layers=num_layers
    )

# Contrastive loss from CLIP paper
def clip_contrastive_loss(logits_per_image, logits_per_text):
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2

# Text input: (1, seq_len) (max seq_len is 77)
# Output shape: (1, seq_len, 768)
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_layers=6):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8,
            dim_feedforward=4*embed_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, text):
        positions = torch.arange(text.size(1), device=text.device).expand(text.size(0), -1)
        x = self.token_embedding(text) + self.positional_embedding(positions)
        return self.transformer(x)
    
    def load_weights(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

# Image input shape: (1, 3, Height, Width) (Height and Width can be any size greater than 16)
# Output shape: (1, 768)
ENCODER_INPUT_SIZE = (224, 224)
ENCODER_PATCH_SIZE = 16
 
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=768, input_channels=3, num_layers=4, num_heads=8):
        super().__init__()
        num_patches = (ENCODER_INPUT_SIZE[0] // ENCODER_PATCH_SIZE) ** 2 
        
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim,
            kernel_size=ENCODER_PATCH_SIZE, stride=ENCODER_PATCH_SIZE
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.spatial_norm = nn.LayerNorm(embed_dim)
 
        # Global pooling head — mean pool + project → single vector for contrastive loss
        self.pool_norm = nn.LayerNorm(embed_dim)
        self.pool_proj = nn.Linear(embed_dim, embed_dim)
 
    def forward(self, image):
        # Resize to fixed resolution so patch count is always 196
        if image.shape[-2:] != ENCODER_INPUT_SIZE:
            image = F.interpolate(image.float(), size=ENCODER_INPUT_SIZE,
                                  mode='bilinear', align_corners=False)
 
        x = self.patch_embed(image)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        spatial_grid = self.spatial_norm(x)
 
        global_vec = self.pool_proj(
            self.pool_norm(spatial_grid.mean(dim=1))
        )
        return spatial_grid, global_vec
 
    def load_weights(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)
 
    def store_weights(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, filename))

# Helper class for training
class CLIPWrapper(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super().__init__()
        self.text_encoder:TextEncoder = text_encoder
        self.image_encoder:ImageEncoder = image_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())
        self.final_ln = nn.LayerNorm(768)
        self.projection = nn.Linear(768, 768)

    def forward(self, text, images):
        x = self.text_encoder(text)
        x = x.mean(dim=1)
        x = self.final_ln(x)
        text_features = self.projection(x)
        _, global_vec = self.image_encoder(images)
        return text_features, global_vec, self.logit_scale.exp()
    
    def load_weights(self, wrapper_filename, img_filename, txt_filename):
        self.image_encoder.load_weights(img_filename)
        self.text_encoder.load_weights(txt_filename)
        if wrapper_filename:
            state_dict = torch.load(wrapper_filename)
            wrapper_keys = {k: v for k, v in state_dict.items() 
                            if not k.startswith('text_encoder.') 
                            and not k.startswith('image_encoder.')}
            self.load_state_dict(wrapper_keys, strict=False)

    def store_weights(self, path, txt_filename, img_filename, wrapper_filename):
        self.image_encoder.store_weights(path, img_filename)
        self.text_encoder.store_weights(path, txt_filename)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        state_dict = self.state_dict()
        wrapper_keys = {k: v for k, v in state_dict.items() 
                        if not k.startswith('text_encoder.') 
                        and not k.startswith('image_encoder.')}
        
        torch.save(wrapper_keys, os.path.join(path, wrapper_filename))

# text_encoder = create_text_encoder()

# input_text = "Input string"
# tokens = CLIPTokenize(input_text)
# encodings = text_encoder(tokens)
# print(encodings.size())

# input_text = ["Input string", "Another input string"]
# tokens = CLIPTokenize(input_text)
# encodings = text_encoder(tokens)
# print(encodings.size())

# img_encoder = create_image_encoder()
# wrapper = CLIPWrapper(text_encoder, img_encoder)

# wrapper.store_weights("/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models", "txtEncWeights", "imgEncWeights", "CLIPWrapperWeights")

# text_encoder_new = create_text_encoder()
# img_encoder_new = create_image_encoder()
# wrapper_new = CLIPWrapper(text_encoder_new, img_encoder_new)

# wrapper_new.load_weights("/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models/CLIPWrapperWeights", "/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models/imgEncWeights", "/Users/adityaasuratkal/Downloads/GitHub/ImgResearch/models/txtEncWeights")