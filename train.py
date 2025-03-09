# train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
import torch.utils.data.distributed
from models.clip_model import create_text_encoder, create_image_encoder
from models.prior_model import create_prior
import torchvision.transforms as T
from data.dataset import build_datasets

# Add these transforms and tokenizer
def get_transforms():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                    (0.26862954, 0.26130258, 0.27577711))
    ])

def clip_loss(logits_per_text, logits_per_image, temperature):
    labels = torch.arange(logits_per_text.size(0), device=logits_per_text.device)
    loss_text = nn.CrossEntropyLoss()(logits_per_text, labels)
    loss_image = nn.CrossEntropyLoss()(logits_per_image, labels)
    return (loss_text + loss_image) / 2

def train_clip(
    text_encoder,
    image_encoder,
    dataloader,
    num_epochs=10,
    lr=1e-4,
    temperature=0.07,
    device="cuda"
):
    text_encoder = text_encoder.to(device)
    image_encoder = image_encoder.to(device)
    
    optimizer = optim.Adam(
        list(text_encoder.parameters()) + list(image_encoder.parameters()),
        lr=lr
    )
    
    for epoch in range(num_epochs):
        for images, texts in dataloader:
            images = images.to(device)
            texts = texts.to(device)
            
            # Forward pass
            text_features = text_encoder(texts)
            image_features = image_encoder(images)
            
            # Normalize features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Compute similarity matrix
            logit_scale = torch.tensor(1/temperature).exp().to(device)
            logits_per_text = logit_scale * text_features @ image_features.t()
            logits_per_image = logits_per_text.t()
            
            # Compute loss
            loss = clip_loss(logits_per_text, logits_per_image, temperature)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")

def train_prior(
    prior,
    text_encoder,
    image_encoder,
    dataloader,
    num_epochs=5,
    lr=1e-4,
    device="cuda"
):
    prior = prior.to(device)
    text_encoder = text_encoder.to(device).eval()
    image_encoder = image_encoder.to(device).eval()
    
    optimizer = optim.Adam(prior.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    with torch.no_grad():  # Freeze CLIP components
        for epoch in range(num_epochs):
            for images, texts in dataloader:
                images = images.to(device)
                texts = texts.to(device)
                
                # Get target embeddings
                image_features = image_encoder(images)
                text_features = text_encoder(texts)
                
                # Prior prediction
                pred_features = prior(text_features)
                
                # Compute loss
                loss = criterion(pred_features, image_features)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            print(f"Prior Epoch {epoch+1}/{num_epochs} Loss: {loss.item():.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models
    text_encoder = create_text_encoder()
    image_encoder = create_image_encoder()
    prior = create_prior()

    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-path', type=str)
    parser.add_argument('--cc3m-path', type=str)
    parser.add_argument('--custom400m-path', type=str)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # Initialize components
    transform = get_transforms()
    
    # Build datasets
    config = {
        'coco_path': args.coco_path,
        'cc3m_path': args.cc3m_path,
        'custom400m_path': args.custom400m_path
    }
    
    full_dataset = build_datasets(config, transform)
    
    # Create distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        full_dataset,
        num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
        rank=int(os.environ.get('RANK', 0))
    )

    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=True
    )

    # Train CLIP
    train_clip(
        text_encoder,
        image_encoder,
        train_loader,
        num_epochs=args.epochs
    )
    
    # Save CLIP components
    torch.save(text_encoder.state_dict(), "text_encoder.pth")
    torch.save(image_encoder.state_dict(), "image_encoder.pth")
    
    # Train Prior
    train_prior(
        prior,
        text_encoder,
        image_encoder,
        train_loader,
        num_epochs=5,
        device=device
    )
    
    # Save Prior
    torch.save(prior.state_dict(), "prior.pth")

if __name__ == "__main__":
    main()