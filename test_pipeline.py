# test_pipeline.py
import torch
from torch.utils.data import DataLoader
from data.custom400m import get_laion_test_dataset, adaptive_collate
import argparse

parser = argparse.ArgumentParser(description="Load LAION-400M dataset from Hugging Face.")
parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")

args = parser.parse_args()
HUGGINGFACE_TOKEN = args.token

def clip_contrastive_loss(logits_per_image, logits_per_text):
    # Contrastive loss from CLIP paper
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_img + loss_txt) / 2

def main():
    # Main Test ###################################################################
    
    # Test CLIP training
    print("\n[CLIP Test] Initializing CLIP components...")
    from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize,CLIPWrapper
    text_encoder = create_text_encoder()
    image_encoder = create_image_encoder()
    clip_model = CLIPWrapper(text_encoder, image_encoder)
    
    print("[CLIP Test] Creating test dataset...")
    test_dataset = get_laion_test_dataset(HUGGINGFACE_TOKEN, 5, text_processor=CLIPTokenize)

    print(f"[CLIP Test] Dataset size: {len(test_dataset)} samples")

    test_loader = DataLoader(
        test_dataset, 
        batch_size=2,
        num_workers=1,
        collate_fn=adaptive_collate,
        persistent_workers=True, 
        pin_memory=True
    )
    
    print("[CLIP Test] Starting training step...")
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-4)
    
    for batch in test_loader:
        if batch is None:
            print(f"Skipping empty batch")
            continue
        optimizer.zero_grad()
        images, texts = batch

        if isinstance(texts, (list, tuple)):
            texts = torch.stack(texts, dim=0)

        print(texts.shape)
        
        # Forward pass
        text_features, image_features, logit_scale = clip_model(texts, images)

        print(f"Text features shape: {text_features.shape}")
        print(f"Image features shape: {image_features.shape}")
        
        # Calculate similarity matrix
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        
        # Compute loss
        loss = clip_contrastive_loss(logits_per_image, logits_per_text)
        
        loss.backward()
        optimizer.step()
        print(f"CLIP Training Step: ✓ (Loss: {loss.item():.4f})")
        break

    # Test Prior training
    print("\n[Prior Test] Initializing Prior model...")
    from models.prior_model import create_prior
    prior = create_prior()
    
    print("[Prior Test] Freezing CLIP encoders...")
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    print("[Prior Test] Starting training loop...")
    optimizer = torch.optim.Adam(prior.parameters(), lr=1e-4)
    for batch_idx, batch in enumerate(test_loader):
        if batch is None:
            print(f"Skipping empty batch {batch_idx+1}")
            continue
        print(f"\n[Prior Test] Processing batch {batch_idx+1}")
        images, texts = batch
        with torch.no_grad():
            print("[Prior Test] Generating CLIP embeddings...")
            text_emb = text_encoder(texts)
            image_emb = image_encoder(images)
        
        optimizer.zero_grad()
        print("[Prior Test] Running prior model...")
        prior_emb = prior(text_emb)

        print(f"Text embeddings shape: {text_emb.shape}")
        print(f"Image embeddings shape: {image_emb.shape}")
        print(f"Prior embeddings shape: {prior_emb.shape}")

        loss = torch.mean((prior_emb - image_emb)**2)
        loss.backward()
        optimizer.step()
        print(f"[Prior Test] Batch {batch_idx+1} completed - Loss: {loss.item():.4f}")
        break
    
    # coco_loader = DataLoader(
    #     build_pretrain_datasets(TEST_DATA_DIR, "2017", "training", None),
    #     batch_size=2
    # )
    # test_training_step(clip_model, coco_loader, "CLIP")

    # Test Segmentation training
    # print("\nTesting Segmentation training:")
    # from models import VideoSAM
    # teacher_model = VideoSAM(
    #     text_encoder=text_encoder,
    #     image_encoder=image_encoder,
    #     mem_size=2
    # )
    
    # sa1b_loader = DataLoader(
    #     SegmentationDataset(TEST_DATA_DIR, "sa1b"),
    #     batch_size=2
    # )
    # test_training_step(teacher_model, sa1b_loader, "Teacher")

    # Cleanup

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Add this line
    torch.manual_seed(42)
    main()