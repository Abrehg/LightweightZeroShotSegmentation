# test_pipeline.py
import torch
import os
from torch.utils.data import DataLoader
from data.custom400m import get_laion_streaming_dataset, adaptive_collate
import argparse
from models.prior_model import create_prior, Prior, PriorLoss, TeacherCLIP
from models.SAM_model import VideoSAM, iou_loss
from models.distill_model import DistilledMemoryStudent
import time
import re
import glob
import wandb
from data.segmentation import SAM_adaptive_collate, SA1BDataset, SAVDataset
from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

#Implement custom weight loading/storing functions + validation set accuracy

# ======== Hyperparameters & Setup ========
HYPERPARAMS = {
    "CLIP_EPOCHS": 1, #10,
    "PRIOR_EPOCHS": 1, #10,
    "SAM_DECODER_EPOCHS": 1,
    "TEACHER_STUDENT_EPOCHS": 1, #10,
    "CLIP_LR": 0.0001,
    "PRIOR_LR": 0.0001,
    "DECODER_LR": 0.0001, # For SAM Decoder training
    "TEACHER_LR": 0.00001, # For teacher fine-tuning during student training
    "STUDENT_LR": 0.0001,
    "LAION_BATCH_SIZE": 64,
    "SAM_BATCH_SIZE": 512,
    "CHECKPOINT_DIR": "weights",
    "WANDB_PROJECT_NAME": "Zero Shot Segmentation",
    "WANDB_ENTITY_NAME": "adityaasuratkal-rensselaer-polytechnic-institute"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Change format to store validation weights and then choose weights with lowest error, ignore epochs

# ======== Helper Function for Checkpoints ========
def get_latest_epoch_checkpoint(directory, prefix):
    if not os.path.isdir(directory):
        return None, -1
    
    files = glob.glob(os.path.join(directory, f"{prefix}_epoch_*"))
    if not files:
        return None, -1

    latest_epoch = -1
    latest_file = None
    
    for f_path in files:
        filename = os.path.basename(f_path)
        match = re.search(rf"{prefix}_epoch_(\d+)", filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = f_path
                
    return latest_file, latest_epoch

def get_best_weights_checkpoint(directory, prefix):
    if not os.path.isdir(directory):
        return None, -1
    
    files = glob.glob(os.path.join(directory, f"{prefix}_epoch_*"))
    if not files:
        return None, -1

    latest_epoch = -1
    best_file = None
    
    for f_path in files:
        filename = os.path.basename(f_path)
        match = re.search(rf"{prefix}_epoch_(\d+)", filename)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                best_file = f_path
                
    return best_file, latest_epoch

# ======== CLIP Training ========
def train_clip(hf_token, run: wandb, start_epoch = 0):
    print("\n=== Training CLIP ===")
    
    # Initialize components
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    text_encoder = create_text_encoder().to(device)
    image_encoder = create_image_encoder().to(device)

    clip_model:CLIPWrapper = CLIPWrapper(text_encoder, image_encoder).to(device)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=HYPERPARAMS["CLIP_LR"])
    
    if start_epoch > 0:
        text_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_text_epoch_{start_epoch}")
        image_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_image_epoch_{start_epoch}")
        wrapper_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_wrapper_epoch_{start_epoch}")
        if os.path.exists(text_ckpt_to_load) and os.path.exists(image_ckpt_to_load) and os.path.exists(wrapper_ckpt_to_load):
            print(f"Resuming CLIP training from epoch {start_epoch}")
            clip_model.load_weights(wrapper_ckpt_to_load, image_ckpt_to_load, text_ckpt_to_load)
        else:
            print(f"Warning: Checkpoint for epoch {start_epoch} not found. Starting CLIP from scratch.")
            start_epoch = 0

    clip_model = DDP(clip_model, device_ids=[local_rank])
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
            
            if batch_idx % 100 == 0 and is_main_process():
                print(f"CLIP Epoch {epoch+1}/{HYPERPARAMS['CLIP_EPOCHS']} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                run.log({
                    "clip_batch_loss": loss.item(), 
                    "clip_epoch": epoch + 1,
                    "clip_batch_idx": batch_idx
                })
        
        if is_main_process():
            avg_epoch_loss = total_loss / (batch_idx + 1) if batch_idx > -1 else 0
            print(f"CLIP Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
            run.log({"clip_epoch_avg_loss": avg_epoch_loss, "clip_epoch": epoch + 1})

            #Potentially add validation loss to filename and then implement early stopping

            clip_model.store_weights(HYPERPARAMS['CHECKPOINT_DIR'], f"clip_text_epoch_{epoch+1}", f"clip_image_epoch_{epoch+1}", f"clip_wrapper_epoch_{epoch+1}")
            print(f"Saved CLIP checkpoints for epoch {epoch+1}")
    
    print("CLIP training completed.\n")

# ======== Prior Training ========
def train_prior(hf_token, run: wandb, start_epoch = 0):
    print("\n=== Training Prior ===")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    # Load frozen CLIP
    text_encoder = create_text_encoder().to(device)
    prior_teacher = TeacherCLIP().to(device)
    
    # Load latest CLIP checkpoint
    best_clip_text_ckpt, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")

    if not best_clip_text_ckpt:
        raise FileNotFoundError("Latest CLIP text or image checkpoints not found. Train CLIP first.")
    
    print(f"Loading CLIP text from: {best_clip_text_ckpt}")
    text_encoder.load_weights(best_clip_text_ckpt)
    
    # Freeze CLIP
    for param in text_encoder.parameters(): param.requires_grad_(False)
    text_encoder.eval()
    for param in prior_teacher.parameters(): param.requires_grad_(False)
    prior_teacher.eval()

    # Initialize Prior
    prior_model:Prior = create_prior().to(device)
    
    optimizer = torch.optim.Adam(prior_model.parameters(), lr=HYPERPARAMS["PRIOR_LR"])

    if start_epoch > 0:
        prior_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"prior_epoch_{start_epoch}")
        if os.path.exists(prior_ckpt_to_load):
            print(f"Resuming Prior training from epoch {start_epoch}")
            prior_model.load_weights(prior_ckpt_to_load)
        else:
            print(f"Warning: Prior checkpoint for epoch {start_epoch} not found. Starting Prior from scratch.")
            start_epoch = 0
    
    prior = DDP(prior_model, device_ids=[local_rank])

    # Streaming dataset (same as CLIP)
    train_dataset = get_laion_streaming_dataset(
        HUGGINGFACE=hf_token, 
        text_processor=CLIPTokenize
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["LAION_BATCH_SIZE"],
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
            
            # Get frozen embeddings
            with torch.no_grad():
                text_emb = text_encoder(texts)
                target_grid = prior_teacher(images)
            
            # Prior forward
            prior_grid = prior(text_emb)
            loss = PriorLoss(prior_grid, target_grid)
            
            # Backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0 and is_main_process():
                print(f"Prior Epoch {epoch+1}/{HYPERPARAMS['PRIOR_EPOCHS']} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                run.log({
                    "prior_batch_loss": loss.item(), 
                    "prior_epoch": epoch + 1,
                    "prior_batch_idx": batch_idx
                })

        if is_main_process():
            avg_epoch_loss = total_loss / (batch_idx + 1) if batch_idx > -1 else 0
            print(f"Prior Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
            run.log({"prior_epoch_avg_loss": avg_epoch_loss, "prior_epoch": epoch + 1})
        
            #Potentially add validation loss to filename and then implement early stopping

            prior.module.store_weights(HYPERPARAMS["CHECKPOINT_DIR"], f"prior_epoch_{epoch+1}")
            print(f"Saved Prior checkpoint for epoch {epoch+1}")
    
    print("Prior training completed.\n")

def train_SAM_decoder(dataloader, run: wandb, start_epoch = 0):
    print("\n=== Training SAM Decoder (Teacher Component) ===")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    text_encoder = create_text_encoder().to(device)
    prior = create_prior().to(device)
    sam_decoder_model = VideoSAM().to(device)
    sam_decoder = DDP(sam_decoder_model, device_ids=[local_rank])

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder
            self.prior = prior
            self.sam_decoder = sam_decoder

        def forward(self, x, text_tokens):
            with torch.no_grad():
                text_emb = self.text_encoder(text_tokens)
                prior_emb = self.prior(text_emb)
            result = self.sam_decoder(x, prior_emb)
            return result

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
            teacher.sam_decoder.module.load_state_dict(torch.load(sam_decoder_ckpt_to_load, map_location=device))
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
                    pred_mask = teacher.forward(img, txt)

                    loss = iou_loss(pred_mask, mask)
                    loss.backward()
                    current_batch_loss_sum += loss.item()
                    num_samples_in_batch +=1
        
            optimizer_sam_decoder.step()

            if batch_idx % 100 == 0 and is_main_process():
                avg_batch_item_loss = current_batch_loss_sum / num_samples_in_batch
                total_loss += avg_batch_item_loss
                print(f"SAM Decoder Epoch {epoch+1}/{HYPERPARAMS['SAM_DECODER_EPOCHS']} | Batch {batch_idx} Avg Item Loss: {avg_batch_item_loss:.4f}")
                run.log({
                    "sam_decoder_batch_avg_item_loss": avg_batch_item_loss,
                    "sam_decoder_epoch": epoch + 1,
                    "sam_decoder_batch_idx": batch_idx
                })
            batch_count = batch_count + 1
            
        if is_main_process():
            avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"SAM Decoder Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
            run.log({"sam_decoder_epoch_avg_loss": avg_epoch_loss, "sam_decoder_epoch": epoch + 1})

            ckpt_path = os.path.join(HYPERPARAMS["CHECKPOINT_DIR"], f"sam_decoder_epoch_{epoch+1}.pt")
            torch.save(teacher.sam_decoder.module.state_dict(), ckpt_path)
            print(f"Saved SAM Decoder checkpoint for epoch {epoch+1} to {ckpt_path}")
    print("SAM Decoder training completed.\n")

def train_student(dataloader, run:wandb, start_epoch = 0):
    print("\n=== Training Student (with Teacher Fine-tuning) ===")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    text_encoder = create_text_encoder().to(device)
    prior = create_prior().to(device)
    sam_decoder_model = VideoSAM().to(device)
    sam_decoder = DDP(sam_decoder_model, device_ids=[local_rank])

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder
            self.prior = prior
            self.sam_decoder = sam_decoder

        def forward(self, x, text_tokens):
            with torch.no_grad():
                text_emb = self.text_encoder(text_tokens)
                prior_emb = self.prior(text_emb)
            result = self.sam_decoder(x, prior_emb)
            return result

    teacher = TeacherModel()

    # Load latest CLIP Text Encoder
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
    teacher.sam_decoder.module.load_state_dict(torch.load(latest_sam_decoder_ckpt, map_location=device))

    # Freeze text_encoder and prior for teacher during student training
    for param in teacher.text_encoder.parameters(): param.requires_grad_(False)
    for param in teacher.prior.parameters(): param.requires_grad_(False)
    teacher.text_encoder.eval()
    teacher.prior.eval()

    # Load SAM Decoder's own latest checkpoint if resuming
    if start_epoch > 0:
        sam_decoder_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"sam_decoder_epoch_{start_epoch}.pt")
        if os.path.exists(sam_decoder_ckpt_to_load):
            teacher.sam_decoder.module.load_state_dict(torch.load(sam_decoder_ckpt_to_load, map_location=device))
        else:
            print(f"Warning: SAM Decoder checkpoint for epoch {start_epoch} not found. Starting fresh.")
            start_epoch = 0

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    student = DDP(DistilledMemoryStudent().to(device), device_ids=[local_rank]) # Assuming DistilledMemoryStudent is defined
    student.register_teacher(teacher) 

    if start_epoch > 0:
        student_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"student_phase_student_epoch_{start_epoch}.pt")
        teacher_sam_decoder_ckpt_to_load = os.path.join(HYPERPARAMS['CHECKPOINT_DIR'], f"student_phase_teacher_sam_decoder_epoch_{start_epoch}.pt")

        if os.path.exists(student_ckpt_to_load):
            print(f"Resuming Student training from epoch {start_epoch}")
            student.module.load_state_dict(torch.load(student_ckpt_to_load, map_location=device))
            if os.path.exists(teacher_sam_decoder_ckpt_to_load): # Also load teacher's SAM decoder state from this phase
                teacher.sam_decoder.module.load_state_dict(torch.load(teacher_sam_decoder_ckpt_to_load, map_location=device))
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
                    
            if batch_idx % 100 == 0 and is_main_process():
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

        if is_main_process():
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

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_main_process():
    return dist.get_rank() == 0

def main(hf_token, wandb_key):
    """Central training orchestration function"""
    setup_ddp()

    # Login to W&B on the main process
    if is_main_process():
        try:
            wandb.login(key=wandb_key)
        except wandb.errors.UsageError as e:
            print(f"Failed to login to W&B: {e}")
            # Decide if you want to exit or continue without logging
            return 

    # Create checkpoint directory if it doesn't exist on the main process
    if is_main_process():
        os.makedirs(HYPERPARAMS["CHECKPOINT_DIR"], exist_ok=True)
    
    # All processes will wait here until the directory is created
    dist.barrier()

    # Initialize W&B run (all processes do this, but only main process logs)
    if is_main_process():
        run = wandb.init(
            project=HYPERPARAMS["WANDB_PROJECT_NAME"],
            entity=HYPERPARAMS["WANDB_ENTITY_NAME"],
            config=HYPERPARAMS
        )
    else:
        # Other processes get a mock run object that does nothing
        run = wandb.init(mode="disabled")

    # Determine start epochs for each phase by checking for existing checkpoints
    _, clip_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    _, prior_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    _, sam_decoder_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")
    _, student_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_student")
    
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
    sampler = DistributedSampler(combined_dataset)
    dataloader = DataLoader(combined_dataset, 
                            batch_size=HYPERPARAMS["SAM_BATCH_SIZE"], 
                            shuffle=False, 
                            num_workers=10, 
                            collate_fn=SAM_adaptive_collate, 
                            pin_memory=True,
                            sampler=sampler)

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
    parser = argparse.ArgumentParser(description="Train pipeline for Zero-Shot Segmentation")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API key")
    args = parser.parse_args()
    
    # Setup torch environment
    torch.multiprocessing.freeze_support()
    torch.manual_seed(42)
    
    # Update the main function call
    main(args.token, args.wandb_key)
