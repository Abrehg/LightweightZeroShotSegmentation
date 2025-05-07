# test_pipeline.py
import torch
import os
from torch.utils.data import DataLoader
from data.custom400m import get_laion_streaming_dataset, adaptive_collate
import argparse
from models.prior_model import create_prior
from models.SAM_model import VideoSAM, iou_loss
from models.distill_model import DistilledMemoryStudent
import time
from data.segmentation import SAM_adaptive_collate, SA1BDataset, SAVDataset
from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss

# ======== Hyperparameters & Setup ========
HYPERPARAMS = {
    "CLIP_EPOCHS": 1,#10,
    "PRIOR_EPOCHS": 1,#10,
    "TEACHER_STUDENT_EPOCHS": 1,#10,
    "CLIP_LR": 0.0001,
    "PRIOR_LR": 0.0001,
    "DECODER_LR": 0.0001,
    "TEACHER_LR": 0.00001,
    "STUDENT_LR": 0.0001,
    "BATCH_SIZE": 64,
    "CHECKPOINT_DIR": "weights"  # Will be created automatically
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('conda')

# ======== CLIP Training ========
def train_clip(hf_token):
    print("\n=== Training CLIP ===")
    
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

def train_SAM_decoder(dataloader):
    print("\n[Teacher Training] Initializing Teacher Model...")
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

    ckpt_files = [f for f in os.listdir(HYPERPARAMS['CHECKPOINT_DIR']) 
                 if f.startswith("clip_text_epoch_")]
    if not ckpt_files:
        raise FileNotFoundError("No CLIP checkpoints found")

    latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in ckpt_files])
    text_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_text_epoch_{latest_epoch}.pt")
    prior_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_image_epoch_{latest_epoch}.pt")
    
    teacher.text_encoder.load_state_dict(torch.load(text_ckpt))
    teacher.prior.load_state_dict(torch.load(prior_ckpt))

    # Freeze text_encoder and prior
    for param in teacher.text_encoder.parameters():
        param.requires_grad_(False)
    for param in teacher.prior.parameters():
        param.requires_grad_(False)

    # Train teacher only
    print("[Teacher Training] Training SAM decoder...")
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["DECODER_LR"])

    for batch in dataloader:
        if batch is None:
            continue

        images, true_masks, texts = batch

        losses = []

        optimizer_teacher.zero_grad()

        for img, mask, txt in zip(images, true_masks, texts):
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

        # Save Prior checkpoint
        ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"Decoder.pt")
        torch.save(teacher.sam_decoder.state_dict(), ckpt_path)
        print(f"Saved SAM Decoder checkpoint to {ckpt_path}")

def train_student(dataloader):
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

    ckpt_files = [f for f in os.listdir(HYPERPARAMS['CHECKPOINT_DIR']) 
                 if f.startswith("clip_text_epoch_")]
    if not ckpt_files:
        raise FileNotFoundError("No CLIP checkpoints found")

    latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in ckpt_files])
    text_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_text_epoch_{latest_epoch}.pt")
    prior_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_image_epoch_{latest_epoch}.pt")
    
    teacher.text_encoder.load_state_dict(torch.load(text_ckpt))
    teacher.prior.load_state_dict(torch.load(prior_ckpt))

    # Freeze text_encoder and prior
    for param in teacher.text_encoder.parameters():
        param.requires_grad_(False)
    for param in teacher.prior.parameters():
        param.requires_grad_(False)

    print("\n[Joint Training] Initializing Student...")
    student = DistilledMemoryStudent().to(device)
    student.register_teacher(teacher)  # Pass trained teacher

    # Separate optimizers
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["TEACHER_LR"])  # Teacher LR
    # optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=HYPERPARAMS["TEACHER_LR"])  # Teacher LR
    optimizer_student = torch.optim.Adam(student.parameters(), lr=HYPERPARAMS["STUDENT_LR"])  # Student LR

    print("[Joint Training] Starting joint training...")
    for i in range(HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]):
        for batch in dataloader:
            if batch is None:
                continue

            images, true_masks, texts = batch

            teacher_losses = []
            student_losses = []

            optimizer_teacher.zero_grad()
            optimizer_student.zero_grad()

            for img, mask, txt in zip(images, true_masks, texts):

                mask = mask.to(device).float()
                img = img.to(device)
                txt = txt.to(device)

                with torch.autograd.detect_anomaly():
                    # Forward pass
                    teacher_out = teacher(img, txt)
                    student_out = student(img, txt)

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

        ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"Teacher_Text_Encoder.pt")
        torch.save(teacher.text_encoder.state_dict(), ckpt_path)
        print(f"Saved Teacher Text Encoder checkpoint to {ckpt_path}")

        ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"Teacher_Prior_Model.pt")
        torch.save(teacher.prior.state_dict(), ckpt_path)
        print(f"Saved Teacher Prior Model checkpoint to {ckpt_path}")

        ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"Teacher_SAM_Decoder.pt")
        torch.save(teacher.sam_decoder.state_dict(), ckpt_path)
        print(f"Saved Teacher SAM Decoder checkpoint to {ckpt_path}")

        ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"Student.pt")
        torch.save(student.state_dict(), ckpt_path)
        print(f"Saved Student Model checkpoint to {ckpt_path}")

def main(hf_token):
    """Central training orchestration function"""
    # Create checkpoint directory
    os.makedirs(HYPERPARAMS["CHECKPOINT_DIR"], exist_ok=True)

    # Phase 1: Mixed Dataset loading
    def load_file_list(file_path):
        with open(file_path) as f:
            return [line.strip().split('\t') for line in f.readlines()[1:]]

    print("Initializing SA-1B dataset")
    sa1b_files = load_file_list("data/Datasets/SA-1B_dataset_copy.txt")

    print("Initializing SA-V dataset")
    sav_files = load_file_list("data/Datasets/SA-V_dataset_copy.txt")

    CACHE_PATH = "img_dataset_cache.pth"

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
    dataloader = DataLoader(combined_dataset, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=1, 
                            collate_fn=SAM_adaptive_collate, 
                            pin_memory=True)
    
    # Phase 2: CLIP training
    print("\n🚀 Starting CLIP Training Phase")
    train_clip(hf_token)
    
    # Phase 3: Prior training
    print("\n🚀 Starting Prior Training Phase")
    train_prior(hf_token)

    # Phase 4: SAM Decoder training
    print("\n🚀 Starting SAM Decoder Training Phase")
    train_SAM_decoder(dataloader)

    # Phase 5: Student training
    print("\n🚀 Starting Student Training Phase")
    train_student(dataloader)
    
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

