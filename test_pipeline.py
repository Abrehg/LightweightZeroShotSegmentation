# test_pipeline.py
import torch
from torch.utils.data import DataLoader
from data.custom400m import get_laion_test_dataset, adaptive_collate
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
    
    # # Test CLIP training
    # print("\n[CLIP Test] Initializing CLIP components...")
    # from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize,CLIPWrapper, clip_contrastive_loss
    # text_encoder = create_text_encoder()
    # image_encoder = create_image_encoder()
    # clip_model = CLIPWrapper(text_encoder, image_encoder)
    
    # print("[CLIP Test] Creating test dataset...")
    # test_dataset = get_laion_test_dataset(HUGGINGFACE_TOKEN, 5, text_processor=CLIPTokenize)

    # print(f"[CLIP Test] Dataset size: {len(test_dataset)} samples")

    # test_loader = DataLoader(
    #     test_dataset, 
    #     batch_size=2,
    #     num_workers=1,
    #     collate_fn=adaptive_collate,
    #     persistent_workers=True, 
    #     pin_memory=True
    # )
    
    # print("[CLIP Test] Starting training step...")
    # optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-4)
    
    # for batch in test_loader:
    #     if batch is None:
    #         print(f"Skipping empty batch")
    #         continue
    #     optimizer.zero_grad()
    #     images, texts = batch

    #     if isinstance(texts, (list, tuple)):
    #         texts = torch.stack(texts, dim=0)

    #     print(texts.shape)
        
    #     # Forward pass
    #     text_features, image_features, logit_scale = clip_model(texts, images)

    #     print(f"Text features shape: {text_features.shape}")
    #     print(f"Image features shape: {image_features.shape}")
        
    #     # Calculate similarity matrix
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logit_scale * text_features @ image_features.t()
        
    #     # Compute loss
    #     loss = clip_contrastive_loss(logits_per_image, logits_per_text)
        
    #     loss.backward()
    #     optimizer.step()
    #     print(f"CLIP Training Step: ✓ (Loss: {loss.item():.4f})")
    #     break

    # # Test Prior training
    # print("\n[Prior Test] Initializing Prior model...")
    # from models.prior_model import create_prior
    # prior = create_prior()
    
    # print("[Prior Test] Freezing CLIP encoders...")
    # for param in clip_model.parameters():
    #     param.requires_grad_(False)
    
    # print("[Prior Test] Starting training loop...")
    # optimizer = torch.optim.Adam(prior.parameters(), lr=1e-4)
    # for batch_idx, batch in enumerate(test_loader):
    #     if batch is None:
    #         print(f"Skipping empty batch {batch_idx+1}")
    #         continue
    #     print(f"\n[Prior Test] Processing batch {batch_idx+1}")
    #     images, texts = batch
    #     with torch.no_grad():
    #         print("[Prior Test] Generating CLIP embeddings...")
    #         text_emb = text_encoder(texts)
    #         image_emb = image_encoder(images)
        
    #     optimizer.zero_grad()
    #     print("[Prior Test] Running prior model...")
    #     prior_emb = prior(text_emb)

    #     print(f"Text embeddings shape: {text_emb.shape}")
    #     print(f"Image embeddings shape: {image_emb.shape}")
    #     print(f"Prior embeddings shape: {prior_emb.shape}")

    #     loss = torch.mean((prior_emb - image_emb)**2)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"[Prior Test] Batch {batch_idx+1} completed - Loss: {loss.item():.4f}")
    #     break

    # Test Segmentation training
    print("\n[Teacher Training] Initializing Teacher Model...")
    from models.clip_model import create_text_encoder, CLIPTokenize
    from models.prior_model import create_prior
    from models.SAM_model import VideoSAM, iou_loss
    
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"\nUsing device: {device}\n")

    if torch.backends.mps.is_available():
        from torch.mps import empty_cache
        empty_cache()

    import psutil
    print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

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
    
    print("[DEBUG] SAMDecoder parameters require_grad status:")
    for name, param in teacher.sam_decoder.named_parameters():
        print(f"{name}: {param.requires_grad}")

    # Create datasets and dataloader
    def load_file_list(file_path):
        with open(file_path) as f:
            return [line.strip().split('\t') for line in f.readlines()[1:]]


    print("Initializing SA-1B dataset")
    sa1b_files = load_file_list("data/Datasets/SA-1B_dataset_copy.txt")

    #sav_files = load_file_list("data/Datasets/SA-V_dataset_copy.txt")

    CACHE_PATH = "dataset_cache.pth"

    # Try loading cached dataset FIRST
    if os.path.exists(CACHE_PATH):
        print("\nLoading cached dataset...")
        start = time.time()
        cache = torch.load(CACHE_PATH)
        
        # Create dataset WITHOUT building index
        sa1b_dataset = SA1BDataset(
            root_dir="./data", 
            text_processor=CLIPTokenize, 
            file_list=sa1b_files,
            build_index=False,  # Skip index building
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
            text_processor=CLIPTokenize, 
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
    
    # sav_dataset = SAVDataset(root_dir="./data", 
    #                          text_processor=CLIPTokenize, 
    #                          file_list=sav_files)

    #combined_dataset = torch.utils.data.ConcatDataset([sa1b_dataset, sav_dataset])
    dataloader = DataLoader(sa1b_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=SAM_adaptive_collate, pin_memory=True)

    # Train teacher only
    print("[Teacher Training] Training SAM decoder...")
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=0.001)

    for batch in dataloader:
        if batch is None:
            continue

        images, true_masks, texts = batch

        target_size = 256

        losses = []
        from torchvision.transforms import Resize
        import torchvision.transforms.functional as F

        optimizer_teacher.zero_grad()

        for img, mask, txt in zip(images, true_masks, texts):
            # Get original dimensions
            C, H, W = img.shape[-3], img.shape[-2], img.shape[-1]
        
            # Calculate new size while preserving aspect ratio
            scale = target_size / max(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
        
            # Resize image with bilinear interpolation
            resized_img = F.resize(img, [new_H, new_W], interpolation=F.InterpolationMode.BILINEAR)
        
            # Resize mask with nearest neighbor (preserve integer labels)
            resized_mask = F.resize(mask.unsqueeze(0), [new_H, new_W], 
                               interpolation=F.InterpolationMode.NEAREST).unsqueeze(1)
            
            resized_mask = resized_mask.to(device).float()
            
            txt = txt.squeeze(1).to(device)
        
            # Process individual samples but maintain gradients
            img = resized_img.unsqueeze(0).to(device)  # Add batch dimension
            #txt = txt.unsqueeze(0).to(device)
    
            with torch.autograd.detect_anomaly():
                # Forward pass
                print("Starting text encoder for texts input")
                print(txt.size())
                text_emb = teacher.text_encoder(txt)
                print(f"Finished processing text imput with shape {text_emb.size()}")
                print("Starting prior model processing")

                prior_emb = teacher.prior(text_emb)
                print(f"Finished prior model processing with size {prior_emb.size()}")
                print(f"Starting SAM decoder processing")

                pred_mask = teacher.sam_decoder(img, prior_emb)
                print(f"Finished SAM decoder processing with size {pred_mask.size()}")
                print(resized_mask.size())
                print(f"Device: pred_mask={pred_mask.device}, mask={resized_mask.device}")
                print(f"Dtype: pred_mask={pred_mask.dtype}, mask={resized_mask.dtype}")
                print(f"pred_mask.requires_grad: {pred_mask.requires_grad}")  # Debugging line
                print(f"pred_mask grad_fn: {pred_mask.grad_fn}")

                loss = iou_loss(pred_mask, resized_mask.to(device))
                print(f"loss.requires_grad: {loss.requires_grad}")  # Debugging line
                print(f"loss grad_fn: {loss.grad_fn}")

                loss.backward()
        
                # Accumulate gradients
                #loss.backward()
                losses.append(loss.item())
        
        optimizer_teacher.step()

        print(f"Teacher Training Loss: {sum(losses)/len(losses):.4f}")
        break


    from models.distill_model import DistilledMemoryStudent

    print("\n[Joint Training] Initializing Student...")
    student = DistilledMemoryStudent().to(device)
    student.register_teacher(teacher)  # Pass trained teacher

    if torch.backends.mps.is_available():
        from torch.mps import empty_cache
        empty_cache()
    
    import psutil
    print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

    # Separate optimizers
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=0.0001)  # Teacher LR
    optimizer_student = torch.optim.Adam(student.parameters(), lr=0.001)  # Student LR

    print("[Joint Training] Starting joint training...")
    
    
    for batch in dataloader:
        if batch is None:
            continue

        images, texts, true_masks = batch

        target_size = 256

        teacher_losses = []
        student_losses = []

        from torchvision.transforms import Resize
        import torchvision.transforms.functional as F

        optimizer_teacher.zero_grad()
        optimizer_student.zero_grad()

        for img, mask, txt in zip(images, true_masks, texts):
            # Get original dimensions
            C, H, W = img.shape[-3], img.shape[-2], img.shape[-1]
        
            # Calculate new size while preserving aspect ratio
            scale = target_size / max(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
        
            # Resize image with bilinear interpolation
            resized_img = F.resize(img, [new_H, new_W], interpolation=F.InterpolationMode.BILINEAR)
        
            # Resize mask with nearest neighbor (preserve integer labels)
            resized_mask = F.resize(mask.unsqueeze(0), [new_H, new_W], 
                               interpolation=F.InterpolationMode.NEAREST).squeeze(0)
            
            txt = txt.squeeze(1)
        
            # Process individual samples but maintain gradients
            img = resized_img.unsqueeze(0).to(device)  # Add batch dimension
            txt = txt.unsqueeze(0).to(device)
    
            # Forward pass

            teacher_out = teacher(resized_img, txt)
            student_out = student(resized_img, txt)

            # Compute losses
            teacher_loss = iou_loss([teacher_out], [resized_mask])
            student_loss = student.compute_distill_loss([student_out], [teacher_out], [resized_mask])

            # Backward passes
            teacher_loss.backward()
            student_loss.backward()

            teacher_losses.append(teacher_loss.item())
            student_losses.append(student_loss.item())

        # Update parameters
        optimizer_teacher.step()
        optimizer_student.step()

        print(f"Teacher Training Loss: {sum(losses)/len(losses):.4f}")
        print(f"Joint Step - Teacher Loss: {sum(teacher_losses)/len(teacher_losses):.4f}, Student Loss: {sum(student_losses)/len(student_losses):.4f}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Add this line
    torch.manual_seed(42)
    main()