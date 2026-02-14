# test_pipeline.py
import torch
from torch.utils.data import DataLoader
from data.custom400m import adaptive_collate, get_laion_test_dataset
from data.segmentation import SAM_adaptive_collate, SA1BDataset, SAVDataset
import argparse
import time
import os

parser = argparse.ArgumentParser(description="Load LAION-400M dataset from Hugging Face.")
parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")

args = parser.parse_args()
HUGGINGFACE_TOKEN = args.token

def main():
    # Main Test ###################################################################
    
    # Test CLIP training
    print("\n[CLIP Test] Initializing CLIP components...")
    from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize,CLIPWrapper, clip_contrastive_loss
    text_encoder = create_text_encoder()
    image_encoder = create_image_encoder()
    clip_model = CLIPWrapper(text_encoder, image_encoder)
    
    print("[CLIP Test] Creating test dataset...")
    test_dataset = get_laion_test_dataset(HUGGINGFACE_TOKEN, num_samples=5, val_boundary=5, text_processor=CLIPTokenize, split="train")
    val_dataset = get_laion_test_dataset(HUGGINGFACE_TOKEN, num_samples=5, val_boundary=5, text_processor=CLIPTokenize, split="val")
    print(f"[CLIP Test] Dataset size: {len(test_dataset)} samples")
    print(f"[CLIP Val] Dataset size: {len(val_dataset)} samples")

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

    # Test Segmentation training
    print("\n[Teacher Training] Initializing Teacher Model...")
    from models.clip_model import create_text_encoder
    from models.prior_model import create_prior
    from models.SAM_model import VideoSAM, iou_loss
    
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"\nUsing device: {device}\n")

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = create_text_encoder().to(device)
            self.prior = create_prior().to(device)
            self.sam_decoder = VideoSAM().to(device)

        def forward(self, x, text_tokens):
            text_emb = self.text_encoder(text_tokens)
            prior_emb = self.prior(text_emb)
            return self.sam_decoder(x, prior_emb)

    teacher = TeacherModel()

    # Freeze text_encoder and prior
    for param in teacher.text_encoder.parameters():
        param.requires_grad_(False)
    for param in teacher.prior.parameters():
        param.requires_grad_(False)

    # Create datasets and dataloader
    def load_file_list(file_path):
        with open(file_path) as f:
            return [line.strip().split('\t') for line in f.readlines()[1:]]

    print("Initializing SA-1B dataset")
    sa1b_files = load_file_list("data/Datasets/SA-1B_dataset_copy.txt")

    print("Initializing SA-V dataset")
    sav_files = load_file_list("data/Datasets/SA-V_dataset_copy.txt")

    CACHE_PATH = "img_dataset_cache.pth"

    # if os.path.exists(CACHE_PATH):
    #     os.remove(CACHE_PATH)

    # Try loading cached dataset FIRST
    if os.path.exists(CACHE_PATH):
        print("\nLoading cached dataset...")
        start = time.time()
        cache = torch.load(CACHE_PATH)
        
        # Create dataset WITHOUT building index
        sa1b_dataset = SA1BDataset(
            root_dir="./data", 
            file_list=sa1b_files,
            build_index=False,
            verify_files=False
        )
        
        # Restore cached state
        sa1b_dataset.samples = cache['samples']
        sa1b_dataset.available_files = cache['available_files']
        print(f"Loaded cached dataset in {time.time()-start:.1f}s")
    else:
        # Create dataset WITH index building
        sa1b_dataset = SA1BDataset(
            root_dir="./data",
            file_list=sa1b_files,
            build_index=True,
            verify_files=True
        )
        # Save cache
        print("\nCaching dataset...")
        torch.save({
            'samples': sa1b_dataset.samples,
            'available_files': sa1b_dataset.available_files
        }, CACHE_PATH)

    VIDEO_CACHE_PATH = "video_dataset_cache.pth"

    # if os.path.exists(VIDEO_CACHE_PATH):
    #     os.remove(VIDEO_CACHE_PATH)

    if os.path.exists(VIDEO_CACHE_PATH):
        print("\nLoading cached dataset...")
        start = time.time()
        cache = torch.load(VIDEO_CACHE_PATH)
        
        # Create dataset WITHOUT building index
        sav_dataset = SAVDataset(
            root_dir="./data",
            file_list=sav_files,
            build_index=False,
            verify_files=False
        )
        
        # Restore cached state
        sav_dataset.samples = cache['samples']
        sav_dataset.available_files = cache['available_files']
        print(f"Loaded cached dataset in {time.time()-start:.1f}s")
    else:
        # Create dataset WITH index building
        sav_dataset = SAVDataset(
            root_dir="./data", 
            file_list=sav_files,
            build_index=True,
            verify_files=True
        )
        # Save cache
        print("\nCaching dataset...")
        torch.save({
            'samples': sav_dataset.samples,
            'available_files': sav_dataset.available_files
        }, VIDEO_CACHE_PATH)

    combined_dataset = torch.utils.data.ConcatDataset([sa1b_dataset, sav_dataset])
    dataloader = DataLoader(sav_dataset, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=1, 
                            collate_fn=SAM_adaptive_collate, 
                            pin_memory=True)

    # Train teacher only
    print("[Teacher Training] Training SAM decoder...")
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=0.001)

    for batch in dataloader:
        if batch is None:
            continue

        images, true_masks, texts = batch

        target_size = 256
        vid_target_size = 64

        losses = []
        import torchvision.transforms.functional as F

        optimizer_teacher.zero_grad()

        for img, mask, txt in zip(images, true_masks, texts):
            # Image resizing
            # Get original dimensions
            print(f"Image shape: {img.shape}")
            # C, H, W = img.shape[-3], img.shape[-2], img.shape[-1]
        
            # # Calculate new size while preserving aspect ratio
            # scale = target_size / max(H, W)
            # new_H, new_W = int(H * scale), int(W * scale)
        
            # # Resize image with bilinear interpolation
            # img = F.resize(img, [new_H, new_W], interpolation=F.InterpolationMode.BILINEAR)
        
            # # Resize mask with nearest neighbor (preserve integer labels)
            # mask = F.resize(mask, [new_H, new_W], interpolation=F.InterpolationMode.NEAREST)

            # # Video resizing
            # C, H, W = img.shape[-3], img.shape[-2], img.shape[-1]
        
            # # Calculate new size while preserving aspect ratio
            # scale = vid_target_size / max(H, W)
            # new_H, new_W = int(H * scale), int(W * scale)
            # resized_img = []
            # resized_mask = []
            # for i in range(img.shape[1]):
            #     resized_img.append(F.resize(img[i], [new_H, new_W]))  # [T, C, H, W]
            #     resized_mask.append(F.resize(mask[i], [new_H, new_W],
            #                     interpolation=F.InterpolationMode.NEAREST))  # [T, 1, H, W]

            # img = torch.stack(resized_img, dim=0)
            # mask = torch.stack(resized_mask, dim=0)

            mask = mask.to(device).float()
            img = img.to(device)
            txt = txt.to(device)

            print(f"Resized image: {img.shape}")
            print(f"Resized mask: {mask.shape}")

            with torch.autograd.detect_anomaly():
                # Forward pass
                text_emb = teacher.text_encoder(txt)
                prior_emb = teacher.prior(text_emb)
                pred_mask = teacher.sam_decoder(img, prior_emb)

                print(f"Output mask: {mask.shape}")

                loss = iou_loss(pred_mask, mask)
                loss.backward()
                losses.append(loss.item())
        
        optimizer_teacher.step()

        print(f"Teacher Training Loss: {sum(losses)/len(losses):.4f}")
        break

    from models.distill_model import DistilledMemoryStudent

    print("\n[Joint Training] Initializing Student...")
    student = DistilledMemoryStudent().to(device)
    student.register_teacher(teacher)

    # Separate optimizers
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=0.0001)  # Teacher LR
    optimizer_student = torch.optim.Adam(student.parameters(), lr=0.001)  # Student LR

    print("[Joint Training] Starting joint training...")
    
    for batch in dataloader:
        if batch is None:
            continue

        images, true_masks, texts = batch

        target_size = 256
        vid_target_size = 64

        teacher_losses = []
        student_losses = []

        import torchvision.transforms.functional as F

        optimizer_teacher.zero_grad()
        optimizer_student.zero_grad()

        for img, mask, txt in zip(images, true_masks, texts):
            # Image resizing
            # Get original dimensions
            # C, H, W = img.shape[-3], img.shape[-2], img.shape[-1]
        
            # # Calculate new size while preserving aspect ratio
            # scale = target_size / max(H, W)
            # new_H, new_W = int(H * scale), int(W * scale)
        
            # # Resize image with bilinear interpolation
            # img = F.resize(img, [new_H, new_W], interpolation=F.InterpolationMode.BILINEAR)
        
            # # Resize mask with nearest neighbor (preserve integer labels)
            # mask = F.resize(mask, [new_H, new_W], interpolation=F.InterpolationMode.NEAREST)
        

            # # Video resizing
            # C, H, W = img.shape[-3], img.shape[-2], img.shape[-1]
        
            # # Calculate new size while preserving aspect ratio
            # scale = vid_target_size / max(H, W)
            # new_H, new_W = int(H * scale), int(W * scale)
            # resized_img = []
            # resized_mask = []
            # for i in range(img.shape[1]):
            #     resized_img.append(F.resize(img[i], [new_H, new_W]))  # [T, C, H, W]
            #     resized_mask.append(F.resize(mask[i], [new_H, new_W],
            #                     interpolation=F.InterpolationMode.NEAREST))  # [T, 1, H, W]

            mask = mask.to(device).float()
            img = img.to(device)
            txt = txt.to(device)

            print(f"Resized image: {img.shape}")
            print(f"Resized mask: {mask.shape}")

            with torch.autograd.detect_anomaly():
                # Forward pass
                teacher_out = teacher(img, txt)
                print(f"Teacher Output mask: {teacher_out.shape}")
                student_out = student(img, txt)
                print(f"Student Output mask: {student_out.shape}")

                # Compute losses
                teacher_loss = iou_loss(teacher_out, mask)
                student_loss = student.compute_distill_loss(student_out, teacher_out, mask)

                # Backward passes
                teacher_loss.backward(retain_graph=True)
                student_loss.backward()

                teacher_losses.append(teacher_loss.item())
                student_losses.append(student_loss.item())

        # Update parameters
        optimizer_teacher.step()
        optimizer_student.step()

        print(f"Joint Step - Teacher Loss: {sum(teacher_losses)/len(teacher_losses):.4f}, Student Loss: {sum(student_losses)/len(student_losses):.4f}")
        break

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    torch.manual_seed(42)
    main()