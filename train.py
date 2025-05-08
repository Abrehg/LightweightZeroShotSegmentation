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
import re
import glob
import wandb
from data.segmentation import SAM_adaptive_collate, SA1BDataset, SAVDataset
from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss

# ======== Hyperparameters & Setup ========
HYPERPARAMS = {
    "CLIP_EPOCHS": 1, #10,
    "PRIOR_EPOCHS": 1, #10,
    "SAM_DECODER_EPOCHS": 1, # Added for dedicated SAM decoder training
    "TEACHER_STUDENT_EPOCHS": 1, #10,
    "CLIP_LR": 0.0001,
    "PRIOR_LR": 0.0001,
    "DECODER_LR": 0.0001, # For SAM Decoder training
    "TEACHER_LR": 0.00001, # For teacher fine-tuning during student training
    "STUDENT_LR": 0.0001,
    "LAION_BATCH_SIZE": 64,
    "SAM_BATCH_SIZE": 512,
    "CHECKPOINT_DIR": "weights",
    "WANDB_PROJECT_NAME": "Zero Shot Segmentation", # Replace with your project name
    "WANDB_ENTITY_NAME": "adityaasuratkal-rensselaer-polytechnic-institute" # Replace with your entity/team name if applicable, else None
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== Helper Function for Checkpoints ========
def get_latest_epoch_checkpoint(directory, prefix):
    """
    Finds the checkpoint file with the highest epoch number for a given prefix.
    Example: prefix_epoch_1.pt, prefix_epoch_2.pt -> returns path to prefix_epoch_2.pt and epoch 2.
    """
    if not os.path.isdir(directory):
        return None, -1
    
    files = glob.glob(os.path.join(directory, f"{prefix}_epoch_*.pt"))
    if not files:
        return None, -1

    latest_epoch = -1
    latest_file = None
    
    for f_path in files:
        filename = os.path.basename(f_path)
        match = re.search(rf"{prefix}_epoch_(\d+)\.pt", filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = f_path
                
    return latest_file, latest_epoch

# ======== CLIP Training ========
def train_clip(hf_token, run: wandb, start_epoch = 0):
    print("\n=== Training CLIP ===")
    
    # Initialize components
    text_encoder = create_text_encoder().to(device)
    image_encoder = create_image_encoder().to(device)
    clip_model = CLIPWrapper(text_encoder, image_encoder).to(device)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=HYPERPARAMS["CLIP_LR"])
    
    if start_epoch > 0:
        text_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_text_epoch_{start_epoch}.pt")
        image_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_image_epoch_{start_epoch}.pt")
        if os.path.exists(text_ckpt_to_load) and os.path.exists(image_ckpt_to_load):
            print(f"Resuming CLIP training from epoch {start_epoch}")
            text_encoder.load_state_dict(torch.load(text_ckpt_to_load, map_location=device))
            image_encoder.load_state_dict(torch.load(image_ckpt_to_load, map_location=device))
        else:
            print(f"Warning: Checkpoint for epoch {start_epoch} not found. Starting CLIP from scratch.")
            start_epoch = 0 # Reset if checkpoint not found

    # Streaming dataset
    train_dataset = get_laion_streaming_dataset(
        HUGGINGFACE = hf_token, 
        text_processor = CLIPTokenize
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["LAION_BATCH_SIZE"],
        collate_fn=adaptive_collate,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(start_epoch, HYPERPARAMS["CLIP_EPOCHS"]):
        clip_model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            images, texts = batch
            images = images.to(device)
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
                print(f"CLIP Epoch {epoch+1}/{HYPERPARAMS['CLIP_EPOCHS']} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                run.log({
                    "clip_batch_loss": loss.item(), 
                    "clip_epoch": epoch + 1,
                    "clip_batch_idx": batch_idx
                })
        
        avg_epoch_loss = total_loss / (batch_idx + 1) if batch_idx > -1 else 0
        print(f"CLIP Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        run.log({"clip_epoch_avg_loss": avg_epoch_loss, "clip_epoch": epoch + 1})

        text_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_text_epoch_{epoch+1}.pt")
        image_ckpt = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_image_epoch_{epoch+1}.pt")
        torch.save(text_encoder.state_dict(), text_ckpt)
        torch.save(image_encoder.state_dict(), image_ckpt)
        print(f"Saved CLIP checkpoints for epoch {epoch+1}")
    
    print("CLIP training completed.\n")

# ======== Prior Training ========
def train_prior(hf_token, run: wandb, start_epoch = 0):
    print("\n=== Training Prior ===")
    # Load frozen CLIP
    text_encoder = create_text_encoder().to(device)
    image_encoder = create_image_encoder().to(device)
    
    # Load latest CLIP checkpoint
    latest_clip_text_ckpt, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    latest_clip_image_ckpt, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_image")

    if not latest_clip_text_ckpt or not latest_clip_image_ckpt:
        raise FileNotFoundError("Latest CLIP text or image checkpoints not found. Train CLIP first.")
    
    print(f"Loading CLIP text from: {latest_clip_text_ckpt}")
    text_encoder.load_state_dict(torch.load(latest_clip_text_ckpt, map_location=device))
    print(f"Loading CLIP image from: {latest_clip_image_ckpt}")
    image_encoder.load_state_dict(torch.load(latest_clip_image_ckpt, map_location=device))
    
    # Freeze CLIP
    for param in text_encoder.parameters(): param.requires_grad_(False)
    for param in image_encoder.parameters(): param.requires_grad_(False)
    text_encoder.eval()
    image_encoder.eval()
    
    # Initialize Prior
    prior = create_prior().to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=HYPERPARAMS["PRIOR_LR"])

    if start_epoch > 0:
        prior_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"prior_epoch_{start_epoch}.pt")
        if os.path.exists(prior_ckpt_to_load):
            print(f"Resuming Prior training from epoch {start_epoch}")
            prior.load_state_dict(torch.load(prior_ckpt_to_load, map_location=device))
        else:
            print(f"Warning: Prior checkpoint for epoch {start_epoch} not found. Starting Prior from scratch.")
            start_epoch = 0
    
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
    for epoch in range(start_epoch, HYPERPARAMS["PRIOR_EPOCHS"]):
        prior.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            images, texts = batch
            images = images.to(device)
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
                print(f"Prior Epoch {epoch+1}/{HYPERPARAMS['PRIOR_EPOCHS']} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                run.log({
                    "prior_batch_loss": loss.item(), 
                    "prior_epoch": epoch + 1,
                    "prior_batch_idx": batch_idx
                })

        avg_epoch_loss = total_loss / (batch_idx + 1) if batch_idx > -1 else 0
        print(f"Prior Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        run.log({"prior_epoch_avg_loss": avg_epoch_loss, "prior_epoch": epoch + 1})
        
        ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"prior_epoch_{epoch+1}.pt")
        torch.save(prior.state_dict(), ckpt_path)
        print(f"Saved Prior checkpoint for epoch {epoch+1} to {ckpt_path}")
    
    print("Prior training completed.\n")

def train_SAM_decoder(dataloader, run: wandb, start_epoch = 0):
    print("\n=== Training SAM Decoder (Teacher Component) ===")
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

    # Load latest CLIP Text Encoder
    latest_clip_text_ckpt, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    if not latest_clip_text_ckpt: raise FileNotFoundError("CLIP text checkpoint not found for SAM Decoder training.")
    print(f"Loading CLIP text for SAM Decoder from: {latest_clip_text_ckpt}")
    teacher.text_encoder.load_state_dict(torch.load(latest_clip_text_ckpt, map_location=device))

    # Load latest Prior
    latest_prior_ckpt, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    if not latest_prior_ckpt: raise FileNotFoundError("Prior checkpoint not found for SAM Decoder training.")
    print(f"Loading Prior for SAM Decoder from: {latest_prior_ckpt}")
    teacher.prior.load_state_dict(torch.load(latest_prior_ckpt, map_location=device))

    for param in teacher.text_encoder.parameters(): param.requires_grad_(False)
    for param in teacher.prior.parameters(): param.requires_grad_(False)
    teacher.text_encoder.eval()
    teacher.prior.eval()

    # Load SAM Decoder's own latest checkpoint if resuming
    if start_epoch > 0:
        sam_decoder_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"sam_decoder_epoch_{start_epoch}.pt")
        if os.path.exists(sam_decoder_ckpt_to_load):
            print(f"Resuming SAM Decoder training from epoch {start_epoch}")
            teacher.sam_decoder.load_state_dict(torch.load(sam_decoder_ckpt_to_load, map_location=device))
        else:
            print(f"Warning: SAM Decoder checkpoint for epoch {start_epoch} not found. Starting fresh.")
            start_epoch = 0
            
    print("[Teacher Training] Training SAM decoder...")
    optimizer_sam_decoder = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["DECODER_LR"])

    for epoch in range(start_epoch, HYPERPARAMS["SAM_DECODER_EPOCHS"]):
        teacher.sam_decoder.train()
        total_loss = 0.0
        
        batch_count = 0
        for batch_idx, batch in enumerate(dataloader):
            total_batch_loss = 0.0
            if batch is None:
                continue

            images, true_masks, texts = batch

            current_batch_loss_sum = 0
            num_samples_in_batch = 0

            optimizer_sam_decoder.zero_grad()

            for img, mask, txt in zip(images, true_masks, texts):
                mask = mask.to(device).float()
                img = img.to(device)
                txt = txt.to(device)

                with torch.autograd.detect_anomaly():
                    # Forward pass
                    text_emb = teacher.text_encoder(txt)
                    prior_emb = teacher.prior(text_emb)
                    pred_mask = teacher.sam_decoder(img, prior_emb)

                    loss = iou_loss(pred_mask, mask)
                    loss.backward()
                    current_batch_loss_sum += loss.item()
                    num_samples_in_batch +=1
        
            optimizer_sam_decoder.step()

            if batch_idx % 100 == 0:
                avg_batch_item_loss = current_batch_loss_sum / num_samples_in_batch
                total_loss += avg_batch_item_loss
                print(f"SAM Decoder Epoch {epoch+1}/{HYPERPARAMS['SAM_DECODER_EPOCHS']} | Batch {batch_idx} Avg Item Loss: {avg_batch_item_loss:.4f}")
                run.log({
                    "sam_decoder_batch_avg_item_loss": avg_batch_item_loss,
                    "sam_decoder_epoch": epoch + 1,
                    "sam_decoder_batch_idx": batch_idx
                })
            batch_count = batch_count + 1
            
        avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"SAM Decoder Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        run.log({"sam_decoder_epoch_avg_loss": avg_epoch_loss, "sam_decoder_epoch": epoch + 1})

        ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"sam_decoder_epoch_{epoch+1}.pt")
        torch.save(teacher.sam_decoder.state_dict(), ckpt_path)
        print(f"Saved SAM Decoder checkpoint for epoch {epoch+1} to {ckpt_path}")
    print("SAM Decoder training completed.\n")

def train_student(dataloader, run:wandb, start_epoch = 0):
    print("\n=== Training Student (with Teacher Fine-tuning) ===")

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

    # Load components for Teacher
    latest_clip_text_ckpt, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    latest_prior_ckpt, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    latest_sam_decoder_ckpt, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")

    if not latest_clip_text_ckpt: raise FileNotFoundError("CLIP text ckpt not found for Student training.")
    if not latest_prior_ckpt: raise FileNotFoundError("Prior ckpt not found for Student training.")
    if not latest_sam_decoder_ckpt: raise FileNotFoundError("SAM Decoder ckpt not found for Student training.")

    print(f"Loading for Teacher (Student Phase) - CLIP Text: {latest_clip_text_ckpt}")
    teacher.text_encoder.load_state_dict(torch.load(latest_clip_text_ckpt, map_location=device))
    print(f"Loading for Teacher (Student Phase) - Prior: {latest_prior_ckpt}")
    teacher.prior.load_state_dict(torch.load(latest_prior_ckpt, map_location=device))
    print(f"Loading for Teacher (Student Phase) - SAM Decoder: {latest_sam_decoder_ckpt}")
    teacher.sam_decoder.load_state_dict(torch.load(latest_sam_decoder_ckpt, map_location=device))

    # Freeze text_encoder and prior for teacher during student training
    for param in teacher.text_encoder.parameters(): param.requires_grad_(False)
    for param in teacher.prior.parameters(): param.requires_grad_(False)
    teacher.text_encoder.eval()
    teacher.prior.eval()
    # SAM decoder part of the teacher will be fine-tuned

    student = DistilledMemoryStudent().to(device) # Assuming DistilledMemoryStudent is defined
    student.register_teacher(teacher) 

    if start_epoch > 0:
        student_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"student_phase_student_epoch_{start_epoch}.pt")
        teacher_sam_decoder_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"student_phase_teacher_sam_decoder_epoch_{start_epoch}.pt")

        if os.path.exists(student_ckpt_to_load):
            print(f"Resuming Student training from epoch {start_epoch}")
            student.load_state_dict(torch.load(student_ckpt_to_load, map_location=device))
            if os.path.exists(teacher_sam_decoder_ckpt_to_load): # Also load teacher's SAM decoder state from this phase
                 teacher.sam_decoder.load_state_dict(torch.load(teacher_sam_decoder_ckpt_to_load, map_location=device))
            else:
                print(f"Warning: Student phase teacher SAM decoder checkpoint for epoch {start_epoch} not found. Using SAM decoder from dedicated training.")
        else:
            print(f"Warning: Student checkpoint for epoch {start_epoch} not found. Starting Student training fresh for this phase.")
            start_epoch = 0
            
    optimizer_teacher_finetune = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["TEACHER_LR"])
    optimizer_student = torch.optim.Adam(student.parameters(), lr=HYPERPARAMS["STUDENT_LR"])

    print("[Joint Training] Starting joint training...")
    for epoch in range(start_epoch, HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]):
        teacher.sam_decoder.train()
        student.train()

        total_teacher_loss = 0.0
        total_student_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            images, true_masks, texts = batch

            current_batch_teacher_loss_sum = 0
            current_batch_student_loss_sum = 0
            num_samples_in_batch = 0

            optimizer_teacher_finetune.zero_grad()
            optimizer_student.zero_grad()

            for img, mask, txt in zip(images, true_masks, texts):

                mask = mask.to(device).float()
                img = img.to(device)
                txt = txt.to(device)

                with torch.autograd.detect_anomaly():
                    # Forward pass
                    teacher_out = teacher(img, txt)
                    student_out = student(img, txt)
                    with torch.no_grad(): # Get teacher output for student distillation without tracking its gradients here
                        teacher_out_for_student = teacher(img, txt).detach()

                    # Compute losses
                    teacher_loss = iou_loss(teacher_out, mask)
                    student_loss = student.compute_distill_loss(student_out, teacher_out_for_student, mask)

                    # Backward passes
                    teacher_loss.backward(retain_graph=True)
                    student_loss.backward()

                    current_batch_teacher_loss_sum += teacher_loss.item()
                    current_batch_student_loss_sum += student_loss.item()
                    num_samples_in_batch = num_samples_in_batch + 1
            
            batch_count = batch_count + 1
                    
            if batch_idx % 100 == 0:
                optimizer_teacher_finetune.step()
                optimizer_student.step()

                avg_batch_teacher_loss = current_batch_teacher_loss_sum / num_samples_in_batch
                avg_batch_student_loss = current_batch_student_loss_sum / num_samples_in_batch
                total_teacher_loss += avg_batch_teacher_loss
                total_student_loss += avg_batch_student_loss

                print(f"Student Epoch {epoch+1}/{HYPERPARAMS['TEACHER_STUDENT_EPOCHS']} | Batch {batch_idx} | Teacher Loss: {avg_batch_teacher_loss:.4f}, Student Loss: {avg_batch_student_loss:.4f}")
                run.log({
                    "student_phase_batch_teacher_loss": avg_batch_teacher_loss,
                    "student_phase_batch_student_loss": avg_batch_student_loss,
                    "student_phase_epoch": epoch + 1,
                    "student_phase_batch_idx": batch_idx
                })
            # Update parameters
            optimizer_teacher_finetune.step()
            optimizer_student.step()


        avg_epoch_teacher_loss = total_teacher_loss / batch_count if batch_count > 0 else 0
        avg_epoch_student_loss = total_student_loss / batch_count if batch_count > 0 else 0
        print(f"Student Epoch {epoch+1} Avg Losses - Teacher: {avg_epoch_teacher_loss:.4f}, Student: {avg_epoch_student_loss:.4f}")
        run.log({
            "student_phase_epoch_avg_teacher_loss": avg_epoch_teacher_loss,
            "student_phase_epoch_avg_student_loss": avg_epoch_student_loss,
            "student_phase_epoch": epoch + 1
        })

        # Save checkpoints for this phase
        # Teacher components (text encoder and prior are frozen, but saved if that's the desired package)
        # torch.save(teacher.text_encoder.state_dict(), os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"student_phase_teacher_text_encoder_epoch_{epoch+1}.pt"))
        # torch.save(teacher.prior.state_dict(), os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"student_phase_teacher_prior_epoch_{epoch+1}.pt"))
        torch.save(teacher.sam_decoder.state_dict(), os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"student_phase_teacher_sam_decoder_epoch_{epoch+1}.pt"))
        torch.save(student.state_dict(), os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"student_phase_student_epoch_{epoch+1}.pt"))
        print(f"Saved Student Phase checkpoints for epoch {epoch+1}")

    print("Student training completed.\n")

def main(hf_token):
    """Central training orchestration function"""
    # Create checkpoint directory
    os.makedirs(HYPERPARAMS["CHECKPOINT_DIR"], exist_ok=True)

    run = wandb.init(
        project=HYPERPARAMS["WANDB_PROJECT_NAME"],
        entity=HYPERPARAMS["WANDB_ENTITY_NAME"],
        config=HYPERPARAMS
    )

    # Determine start epochs for each phase by checking for existing checkpoints
    _, clip_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text") # Assuming text/image saved together
    _, prior_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    _, sam_decoder_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")
    _, student_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_student")

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
                            batch_size=HYPERPARAMS["SAM_BATCH_SIZE"], 
                            shuffle=True, 
                            num_workers=10, 
                            collate_fn=SAM_adaptive_collate, 
                            pin_memory=True)
    
    if clip_start_epoch < HYPERPARAMS["CLIP_EPOCHS"]:
        print("\n🚀 Starting CLIP Training Phase")
        train_clip(hf_token, start_epoch=clip_start_epoch + 1, run=run)
    else:
        print("\n✅ CLIP training already completed or up to date.")

    if prior_start_epoch < HYPERPARAMS["PRIOR_EPOCHS"]:
        print("\n🚀 Starting Prior Training Phase")
        train_prior(hf_token, start_epoch=prior_start_epoch + 1, run=run)
    else:
        print("\n✅ Prior training already completed or up to date.")

    if sam_decoder_start_epoch < HYPERPARAMS["SAM_DECODER_EPOCHS"]:
        print("\n🚀 Starting SAM Decoder Training Phase")
        train_SAM_decoder(dataloader, start_epoch=sam_decoder_start_epoch + 1, run=run)
    else:
        print("\n✅ SAM Decoder training already completed or up to date.")

    if student_start_epoch < HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]:
        print("\n🚀 Starting Student Training Phase")
        train_student(dataloader, start_epoch=student_start_epoch + 1, run=run)
    else:
        print("\n✅ Student training already completed or up to date.")
    
    print("\n🏁 All training phases processed!")
    run.finish() # Finish W&B run

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

