# test_pipeline.py
import torch
from torch.utils.data import DataLoader
from data.custom400m import get_laion_test_dataset, adaptive_collate
from data.segmentation import SAM_adaptive_collate, SA1BDataset, SAVDataset
import argparse

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
    

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = create_text_encoder()
            self.prior = create_prior()
            self.sam_decoder = VideoSAM()

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

    #sav_files = load_file_list("data/Datasets/SA-V_dataset_copy.txt")

    sa1b_dataset = SA1BDataset(root_dir="./data", 
                               text_processor=CLIPTokenize, 
                               file_list=sa1b_files)
    
    # sav_dataset = SAVDataset(root_dir="./data", 
    #                          text_processor=CLIPTokenize, 
    #                          file_list=sav_files)

    #combined_dataset = torch.utils.data.ConcatDataset([sa1b_dataset, sav_dataset])
    dataloader = DataLoader(sa1b_dataset, batch_size=8, shuffle=True, num_workers=1, collate_fn=SAM_adaptive_collate, pin_memory=True)

    # Train teacher only
    print("[Teacher Training] Training SAM decoder...")
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=0.001)

    for batch in dataloader:
        if batch is None:
            continue

        images, texts, true_masks = batch

        print(f"Images size: {images.size()}")
        print(f"Texts size: {texts.size()}")
        print(f"Masks size: {true_masks.size()}")

        if isinstance(texts, (list, tuple)):
            texts = torch.stack(texts, dim=0)
        
        optimizer_teacher.zero_grad()

        text_emb = teacher.text_encoder(texts)
        prior_emb = teacher.prior(text_emb)
        pred_masks = teacher.sam_decoder(images, prior_emb)
        
        loss = iou_loss(pred_masks, true_masks)
        loss.backward()
        optimizer_teacher.step()

        print(f"Teacher Training Loss: {loss.item():.4f}")
        break


    from models.distill_model import DistilledMemoryStudent

    print("\n[Joint Training] Initializing Student...")
    student = DistilledMemoryStudent()
    student.register_teacher(teacher)  # Pass trained teacher
    
    # Separate optimizers
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=0.0001)  # Teacher LR
    optimizer_student = torch.optim.Adam(student.parameters(), lr=0.001)  # Student LR

    print("[Joint Training] Starting joint training...")
    for batch in dataloader:
        if batch is None:
            continue
        images, texts, true_masks = batch
        if isinstance(texts, (list, tuple)):
            texts = torch.stack(texts, dim=0)
        
        # Zero gradients
        optimizer_teacher.zero_grad()
        optimizer_student.zero_grad()

        # Forward passes
        teacher_out = teacher(images, texts)
        student_out = student(images, texts)

        # Compute losses
        teacher_loss = iou_loss(teacher_out, true_masks)
        student_loss = student.compute_distill_loss(student_out, teacher_out, true_masks)

        # Backward passes
        teacher_loss.backward()
        student_loss.backward()

        # Update parameters
        optimizer_teacher.step()
        optimizer_student.step()

        print(f"Joint Step - Teacher Loss: {teacher_loss.item():.4f}, Student Loss: {student_loss.item():.4f}")

    # Cleanup

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # Add this line
    torch.manual_seed(42)
    main()