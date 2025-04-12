# test_pipeline.py
import torch
import os
from torch.utils.data import DataLoader
from data.custom400m import get_laion_streaming_dataset, adaptive_collate
import argparse

# ======== Hyperparameters & Setup ========
HYPERPARAMS = {
    "CLIP_EPOCHS": 1,#10,
    "PRIOR_EPOCHS": 1,#10,
    "CLIP_LR": 0.0001,
    "PRIOR_LR": 0.0001,
    "BATCH_SIZE": 64,
    "CHECKPOINT_DIR": "weights"  # Will be created automatically
}

# ======== CLIP Training ========
def train_clip(hf_token):
    print("\n=== Training CLIP ===")
    from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss
    
    # Initialize components
    text_encoder = create_text_encoder()
    image_encoder = create_image_encoder()
    clip_model = CLIPWrapper(text_encoder, image_encoder)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=HYPERPARAMS["CLIP_LR"])
    
    # Streaming dataset
    train_dataset = get_laion_streaming_dataset(
        HUGGINGFACE = hf_token, 
        text_processor = CLIPTokenize
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["BATCH_SIZE"],
        collate_fn=adaptive_collate,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(HYPERPARAMS["CLIP_EPOCHS"]):
        clip_model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            images, texts = batch
            optimizer.zero_grad()
            
            # Forward pass
            text_features, image_features, logit_scale = clip_model(texts, images)
            
            # Contrastive loss
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()
            loss = clip_contrastive_loss(logits_per_image, logits_per_text)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{HYPERPARAMS['CLIP_EPOCHS']} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        # Save CLIP checkpoint
        text_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_text_epoch_{epoch+1}.pt")
        image_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_image_epoch_{epoch+1}.pt")
        torch.save(text_encoder.state_dict(), text_ckpt)
        torch.save(image_encoder.state_dict(), image_ckpt)
        print(f"Saved CLIP checkpoints:\n- {text_ckpt}\n- {image_ckpt}")
    
    print("CLIP training completed.\n")

# ======== Prior Training ========
def train_prior(hf_token):
    print("\n=== Training Prior ===")
    from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize
    from models.prior_model import create_prior
    
    # Load frozen CLIP
    text_encoder = create_text_encoder()
    image_encoder = create_image_encoder()
    
    # Load latest CLIP checkpoint
    ckpt_files = [f for f in os.listdir(HYPERPARAMS['CHECKPOINT_DIR']) 
                 if f.startswith("clip_text_epoch_")]
    if not ckpt_files:
        raise FileNotFoundError("No CLIP checkpoints found")
    
    latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in ckpt_files])
    text_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_text_epoch_{latest_epoch}.pt")
    image_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_image_epoch_{latest_epoch}.pt")
    
    text_encoder.load_state_dict(torch.load(text_ckpt))
    image_encoder.load_state_dict(torch.load(image_ckpt))
    
    # Freeze CLIP
    for param in text_encoder.parameters():
        param.requires_grad_(False)
    for param in image_encoder.parameters():
        param.requires_grad_(False)
    
    # Initialize Prior
    prior = create_prior()
    optimizer = torch.optim.Adam(prior.parameters(), lr=HYPERPARAMS["PRIOR_LR"])
    
    # Streaming dataset (same as CLIP)
    train_dataset = get_laion_streaming_dataset(
        HUGGINGFACE=hf_token, 
        text_processor=CLIPTokenize
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["BATCH_SIZE"],
        collate_fn=adaptive_collate,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(HYPERPARAMS["PRIOR_EPOCHS"]):
        prior.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            images, texts = batch
            optimizer.zero_grad()
            
            # Get frozen CLIP embeddings
            with torch.no_grad():
                text_emb = text_encoder(texts)
                image_emb = image_encoder(images)
            
            # Prior forward
            prior_emb = prior(text_emb)
            loss = torch.mean((prior_emb - image_emb)**2)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{HYPERPARAMS['PRIOR_EPOCHS']} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        # Save Prior checkpoint
        ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"prior_epoch_{epoch+1}.pt")
        torch.save(prior.state_dict(), ckpt_path)
        print(f"Saved Prior checkpoint to {ckpt_path}")
    
    print("Prior training completed.\n")

def main(hf_token):
    """Central training orchestration function"""
    # Create checkpoint directory
    os.makedirs(HYPERPARAMS["CHECKPOINT_DIR"], exist_ok=True)
    
    # Phase 1: CLIP training
    print("\n🚀 Starting CLIP Training Phase")
    train_clip(hf_token)
    
    # Phase 2: Prior training
    print("\n🚀 Starting Prior Training Phase")
    train_prior(hf_token)
    
    print("\n✅ All training completed!")

# ======== Main ========
if __name__ == "__main__":
    # Configure command line interface
    parser = argparse.ArgumentParser(description="Train CLIP and Prior models")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    args = parser.parse_args()
    
    # Setup torch environment
    torch.multiprocessing.freeze_support()
    torch.manual_seed(42)
    
    # Start training pipeline
    main(args.token)

