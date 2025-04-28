import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer

MAXSEQLENGTH = 77

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def create_text_encoder():
    """Factory for text encoder with predefined architecture"""
    return TextEncoder(
        vocab_size=49408, 
        embed_dim=512, 
        max_seq_len=MAXSEQLENGTH,
        num_layers=12
    )

def create_image_encoder():
    """Factory for image encoder with predefined architecture"""
    return ImageEncoder(
        embed_dim=512,
        input_channels=3
    )

def clip_contrastive_loss(logits_per_image, logits_per_text):
    # Contrastive loss from CLIP paper
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_layers=6):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8,
            dim_feedforward=4*embed_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.final_ln = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, text):
        # text shape: (batch_size, seq_len)
        positions = torch.arange(text.size(1), device=text.device).expand(text.size(0), -1)
        x = self.token_embedding(text) + self.positional_embedding(positions)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Average pooling over sequence
        x = self.final_ln(x)
        return self.projection(x)
    
# Text input: (1, seq_len) (max seq_len is 77)
# Output shape: (1, 512)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, input_channels=3):
        super().__init__()
        self.cnn = nn.Sequential(
            # Stem
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            # ResNet-style blocks
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 512, stride=2),
            
            # Final pooling and projection
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(512),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, image):
        return self.cnn(image)

# Image input shape: (1, 3, Height, Width) (Height and Width can be any size greater than 16)
# Output shape: (1, 512)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.GroupNorm(8, out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)

# Helper class for training
class CLIPWrapper(nn.Module):
    """Utility class for training CLIP components together"""
    def __init__(self, text_encoder, image_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, text, images):
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(images)
        return text_features, image_features, self.logit_scale.exp()

def CLIPTokenize(inputText):
    tokens = tokenizer(inputText, padding="max_length", truncation=True, max_length=MAXSEQLENGTH, return_tensors="pt")

    # tokens shape = (numSequences, 77)
    return tokens['input_ids']

def VecToText(vector):
    return tokenizer.convert_ids_to_tokens(vector)

class CLIPWrapper(torch.nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * torch.tensor(1/0.07).log())

    def forward(self, text, images):
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(images)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return text_features, image_features, self.logit_scale.exp()


# text_encoder = create_text_encoder()

# input_text = "Input string"
# tokens = CLIPTokenize(input_text)
# encodings = text_encoder(tokens)
# print(encodings.size())