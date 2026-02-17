# train.py
import torch
import os
import time
import re
import glob
import wandb
import argparse
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss
from models.prior_model import create_prior, Prior, PriorLoss, TeacherCLIP
from models.SAM_model import VideoSAM, iou_loss, create_SAM
from models.distill_model import DistilledMemoryStudent
from data.custom400m import get_laion_streaming_dataset, adaptive_collate
from data.segmentation import SAM_adaptive_collate, SA1BDataset, SAVDataset

# ======== Hyperparameters & Setup ========
HYPERPARAMS = {
    "CLIP_EPOCHS": 10,
    "PRIOR_EPOCHS": 10,
    "SAM_DECODER_EPOCHS": 3,
    "TEACHER_STUDENT_EPOCHS": 10,
    "CLIP_LR": 0.0001,
    "PRIOR_LR": 0.0001,
    "DECODER_LR": 0.0001, # For SAM Decoder training
    "TEACHER_LR": 0.00001, # For teacher fine-tuning during student training
    "STUDENT_LR": 0.0001,
    "LAION_VAL_SIZE": 10000,
    "LAION_BATCH_SIZE": 64,
    "SA_VAL_SIZE": 10000,
    "SAV_VAL_SIZE": 100,
    "SAM_BATCH_SIZE": 512,
    "CHECKPOINT_DIR": "weights",
    "WANDB_PROJECT_NAME": "Zero Shot Segmentation",
    "WANDB_ENTITY_NAME": "adityaasuratkal-rensselaer-polytechnic-institute"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        match = re.search(rf"{prefix}_epoch_(\d+)_([\d\.]+)", filename)
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

    best_loss = float('inf')
    best_file = None
    best_epoch = -1
    
    for f_path in files:
        filename = os.path.basename(f_path)
        match = re.search(rf"{prefix}_epoch_(\d+)_([\d\.]+)", filename)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            if loss < best_loss:
                best_loss = loss
                best_file = f_path
                best_epoch = epoch
    
    if best_file is None:
        return get_latest_epoch_checkpoint(directory, prefix)
                
    return best_file, best_epoch

# ======== CLIP Training ========
def train_clip(train_loader, val_loader, text_start_weights, img_start_weights, wrapper_start_weights, run: wandb, start_epoch = 0):
    print("\n=== Training CLIP ===")
    
    # Initialize components
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    text_encoder = create_text_encoder().to(device)
    image_encoder = create_image_encoder().to(device)

    clip_model:CLIPWrapper = CLIPWrapper(text_encoder, image_encoder).to(device)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=HYPERPARAMS["CLIP_LR"])
    
    if start_epoch > 0:
        if os.path.exists(text_start_weights) and os.path.exists(img_start_weights) and os.path.exists(wrapper_start_weights):
            print(f"Resuming CLIP training from epoch {start_epoch}")
            clip_model.load_weights(wrapper_start_weights, img_start_weights, text_start_weights)
        else:
            print(f"Warning: Checkpoint for epoch {start_epoch} not found. Starting CLIP from scratch.")
            start_epoch = 0

    clip_model = DDP(clip_model, device_ids=[local_rank])

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
            print(f"CLIP Epoch {epoch+1} Average Train Loss: {avg_epoch_loss:.4f}")

        # --- VALIDATION ---
        clip_model.eval()
        total_val_loss = 0.0
        iterations = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is not None:
                    images, texts = batch
                    images = images.to(device)
                    
                    text_features, image_features, logit_scale = clip_model(texts, images)
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logit_scale * text_features @ image_features.t()
                    loss = clip_contrastive_loss(logits_per_image, logits_per_text)
                    
                    total_val_loss += loss.item()
                    iterations += 1

        avg_val_loss = 999.9
        if iterations > 0:
            avg_val_loss = total_val_loss / iterations

        if is_main_process():
            print(f"CLIP Epoch {epoch+1} Single-Batch Val Loss: {avg_val_loss:.4f}")
            run.log({"clip_epoch_val_loss": avg_val_loss, "clip_epoch": epoch + 1})

            clip_model.module.store_weights(
                HYPERPARAMS['CHECKPOINT_DIR'], 
                f"clip_text_epoch_{epoch+1}_{avg_val_loss:.4f}", 
                f"clip_image_epoch_{epoch+1}_{avg_val_loss:.4f}", 
                f"clip_wrapper_epoch_{epoch+1}_{avg_val_loss:.4f}"
            )
            print(f"Saved CLIP checkpoints for epoch {epoch+1} (Val Loss: {avg_val_loss:.4f})")
    
    print("CLIP training completed.")

# ======== Prior Training ========
def train_prior(hf_token, start_weights, run: wandb, start_epoch = 0):
    print("\n=== Training Prior ===")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    # Load CLIP and teacher
    text_encoder = create_text_encoder().to(device)
    prior_teacher = TeacherCLIP().to(device)
    
    # Load best CLIP checkpoint
    best_clip_text_ckpt, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")

    if not best_clip_text_ckpt:
        raise FileNotFoundError("Latest CLIP text or image checkpoints not found. Train CLIP first.")
    
    print(f"Loading CLIP text from: {best_clip_text_ckpt}")
    text_encoder.load_weights(best_clip_text_ckpt)
    
    # Freeze CLIP and teacher
    for param in text_encoder.parameters(): param.requires_grad_(False)
    text_encoder.eval()
    for param in prior_teacher.parameters(): param.requires_grad_(False)
    prior_teacher.eval()

    # Initialize Prior
    prior_model:Prior = create_prior().to(device)
    
    optimizer = torch.optim.Adam(prior_model.parameters(), lr=HYPERPARAMS["PRIOR_LR"])

    if start_epoch > 0:
        if os.path.exists(start_weights):
            print(f"Resuming Prior training from epoch {start_epoch}")
            prior_model.load_weights(start_weights)
        else:
            print(f"Warning: Prior checkpoint for epoch {start_epoch} not found. Starting Prior from scratch.")
            start_epoch = 0
    
    prior = DDP(prior_model, device_ids=[local_rank])

    # Streaming dataset (same as CLIP)
    train_dataset = get_laion_streaming_dataset(
        HUGGINGFACE = hf_token, 
        text_processor = CLIPTokenize,
        split = "train",
        val_size=HYPERPARAMS["LAION_VAL_SIZE"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=HYPERPARAMS["LAION_BATCH_SIZE"],
        collate_fn=adaptive_collate,
        pin_memory=True
    )
    
    val_dataset = get_laion_streaming_dataset(
        HUGGINGFACE = hf_token, 
        text_processor = CLIPTokenize,
        split = "val",
        val_size=HYPERPARAMS["LAION_VAL_SIZE"]
    )
    val_loader = DataLoader(
        val_dataset,
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

        # --- VALIDATION ---
        prior.eval()
        total_val_loss = 0.0
        iterations = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is not None:
                    images, texts = batch
                    images = images.to(device)
                    
                    text_emb = text_encoder(texts)
                    target_grid = prior_teacher(images)
                    prior_grid = prior(text_emb)
                    loss = PriorLoss(prior_grid, target_grid)
                    
                    total_val_loss += loss.item()
                    iterations += 1
        avg_val_loss = 999.9
        if iterations > 0:
            avg_val_loss = total_val_loss / iterations

        if is_main_process():
            print(f"Prior Epoch {epoch+1} Single-Batch Val Loss: {avg_val_loss:.4f}")
            run.log({"prior_val_loss": avg_val_loss, "prior_epoch": epoch + 1})
        
            prior.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"prior_epoch_{epoch+1}_{avg_val_loss:.4f}"
            )
            print(f"Saved Prior checkpoint for epoch {epoch+1} (Val Loss: {avg_val_loss:.4f})")
    
    print("Prior training completed.\n")

# ======== SAM Teacher Training ========
def train_SAM_decoder(train_dataloader, val_dataloader, start_weights, run: wandb, start_epoch = 0):
    print("\n=== Training SAM Decoder (Teacher Component) ===")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    text_encoder = create_text_encoder().to(device)
    prior = create_prior().to(device)
    sam_decoder = create_SAM().to(device)

    # Load latest CLIP Text Encoder
    latest_clip_text_ckpt, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    if not latest_clip_text_ckpt: raise FileNotFoundError("CLIP text checkpoint not found for SAM Decoder training.")
    print(f"Loading CLIP text for SAM Decoder from: {latest_clip_text_ckpt}")
    text_encoder.load_weights(latest_clip_text_ckpt)

    # Load latest Prior
    latest_prior_ckpt, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    if not latest_prior_ckpt: raise FileNotFoundError("Prior checkpoint not found for SAM Decoder training.")
    print(f"Loading Prior for SAM Decoder from: {latest_prior_ckpt}")
    prior.load_weights(latest_prior_ckpt)

    # Load SAM Decoder's own latest checkpoint if resuming
    if start_epoch > 0:
        if os.path.exists(start_weights):
            sam_decoder.load_weights(start_weights)
        else:
            print(f"Warning: SAM Decoder checkpoint for epoch {start_epoch} not found. Starting fresh.")
            start_epoch = 0

    sam_decoder = DDP(sam_decoder, device_ids=[local_rank])

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

    for param in teacher.text_encoder.parameters(): param.requires_grad_(False)
    for param in teacher.prior.parameters(): param.requires_grad_(False)
    teacher.text_encoder.eval()
    teacher.prior.eval()
            
    print("[Teacher Training] Training SAM decoder...")
    optimizer_sam_decoder = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["DECODER_LR"])

    for epoch in range(start_epoch, HYPERPARAMS["SAM_DECODER_EPOCHS"]):
        teacher.sam_decoder.train()
        total_loss = 0.0
        
        batch_count = 0
        for batch_idx, batch in enumerate(train_dataloader):
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

        # Full Validation Step
        teacher.sam_decoder.eval()
        total_val_loss = 0.0
        iterations = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                if batch is None:
                    continue
                
                images, true_masks, texts = batch

                for img, mask, txt in zip(images, true_masks, texts):
                    mask = mask.to(device).float()
                    img = img.to(device)
                    txt = txt.to(device)

                    pred_mask = teacher.forward(img, txt)

                    loss = iou_loss(pred_mask, mask)
                    total_val_loss += loss.item()

                iterations += 1
        avg_val_loss = 999.9
        if iterations > 0:
            avg_val_loss = total_val_loss / iterations

        if is_main_process():
            print(f"SAM Decoder Epoch {epoch+1} Single-Batch Val Loss: {avg_val_loss:.4f}")
            run.log({"sam_decoder_epoch_avg_loss": avg_val_loss, "sam_decoder_epoch": epoch + 1})
        
            teacher.sam_decoder.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"sam_decoder_epoch_{epoch+1}_{avg_val_loss:.4f}")

            print(f"Saved SAM Decoder checkpoint for epoch {epoch+1} (Val Loss: {avg_val_loss:.4f})")

    print("SAM Decoder training completed.\n")

# ======== SAM Student Training ========
def train_student(train_dataloader, val_dataloader, teacher_start_weights, student_start_weights, run:wandb, start_epoch = 0):
    print("\n=== Training Student (with Teacher Fine-tuning) ===")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    text_encoder = create_text_encoder().to(device)
    prior = create_prior().to(device)
    sam_decoder = create_SAM().to(device)

    # Load latest CLIP Text Encoder
    latest_clip_text_ckpt, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    latest_prior_ckpt, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    latest_sam_decoder_ckpt, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")

    if not latest_clip_text_ckpt: raise FileNotFoundError("CLIP text ckpt not found for Student training.")
    if not latest_prior_ckpt: raise FileNotFoundError("Prior ckpt not found for Student training.")
    if not latest_sam_decoder_ckpt: raise FileNotFoundError("SAM Decoder ckpt not found for Student training.")

    print(f"Loading for Teacher (Student Phase) - CLIP Text: {latest_clip_text_ckpt}")
    text_encoder.load_weights(latest_clip_text_ckpt)
    print(f"Loading for Teacher (Student Phase) - Prior: {latest_prior_ckpt}")
    prior.load_weights(latest_prior_ckpt)
    print(f"Loading for Teacher (Student Phase) - SAM Decoder: {latest_sam_decoder_ckpt}")
    sam_decoder.load_weights(latest_sam_decoder_ckpt)

    # Freeze text_encoder and prior for teacher during student training
    for param in text_encoder.parameters(): param.requires_grad_(False)
    text_encoder.eval()
    for param in prior.parameters(): param.requires_grad_(False)
    prior.eval()

    sam_decoder = DDP(sam_decoder, device_ids=[local_rank])

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

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')

    student = DistilledMemoryStudent().to(device)

    if start_epoch > 0:
        if os.path.exists(student_start_weights):
            print(f"Resuming Student training from epoch {start_epoch}")
            student.load_weights(student_start_weights)
            if os.path.exists(teacher_start_weights):
                teacher.sam_decoder.load_weights(teacher_start_weights)
            else:
                print(f"Warning: Student phase teacher SAM decoder checkpoint for epoch {start_epoch} not found. Using SAM decoder from dedicated training.")
        else:
            print(f"Warning: Student checkpoint for epoch {start_epoch} not found. Starting Student training fresh for this phase.")
            start_epoch = 0
    
    student = DDP(student, device_ids=[local_rank])

    optimizer_teacher_finetune = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["TEACHER_LR"])
    optimizer_student = torch.optim.Adam(student.parameters(), lr=HYPERPARAMS["STUDENT_LR"])

    print("[Joint Training] Starting joint training")
    for epoch in range(start_epoch, HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]):
        teacher.sam_decoder.train()
        student.train()

        total_teacher_loss = 0.0
        total_student_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_dataloader):
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
                    with torch.no_grad():
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

        # Full Validation Step
        teacher.sam_decoder.eval()
        student.eval()
        total_teacher_val_loss = 0.0
        total_student_val_loss = 0.0
        iterations = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                if batch is not None:
                    for img, mask, txt in zip(images, true_masks, texts):
                        teacher_out = teacher(img, txt)
                        student_out = student(img, txt)

                        teacher_loss = iou_loss(teacher_out, mask)
                        student_loss = student.compute_distill_loss(student_out, teacher_out_for_student, mask)

                        total_teacher_val_loss += teacher_loss.item()
                        total_student_val_loss += student_loss.item()

                    iterations += 1
        avg_teacher_val_loss = 999.9
        avg_student_val_loss = 999.9
        if iterations > 0:
            avg_teacher_val_loss = total_teacher_val_loss / iterations
            avg_student_val_loss = total_student_val_loss / iterations

        if is_main_process():
            # Save checkpoints for this phase
            # Teacher components (text encoder and prior are frozen, but saved if that's the desired package)
            # teacher.text_encoder.store_weights(HYPERPARAMS["CHECKPOINT_DIR"], f"student_phase_teacher_text_encoder_epoch_{epoch+1}")
            # teacher.prior.store_weights(HYPERPARAMS["CHECKPOINT_DIR"], f"student_phase_teacher_prior_epoch_{epoch+1}")
            teacher.sam_decoder.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"student_phase_teacher_sam_decoder_epoch_{epoch+1}_{avg_teacher_val_loss:.4f}"
            )

            student.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"student_phase_student_epoch_{epoch+1}_{avg_student_val_loss:.4f}"
                )

            run.log({"codistillation_teacher_epoch_avg_loss": avg_teacher_val_loss, "codistillation_teacher_epoch": epoch + 1})
            run.log({"codistillation_student_epoch_avg_loss": avg_student_val_loss, "codistillation_student_epoch": epoch + 1})
            print(f"Saved Student Phase checkpoints for epoch {epoch+1} (Teacher Val Loss: {avg_teacher_val_loss:.4f}, Student Val Loss: {avg_student_val_loss:.4f})")

    print("Student training completed.\n")

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_main_process():
    return dist.get_rank() == 0

def get_dataset(dataset_cls, file_list, cache_prefix, split_name, val_size):
    cache_path = f"{cache_prefix}_{split_name}.pth"
        
    if os.path.exists(cache_path):
        print(f"\nLoading cached {split_name} dataset from {cache_path}...")
        start = time.time()
        cache = torch.load(cache_path)
            
        # Create without index build
        dataset = dataset_cls(
            root_dir="./data", 
            file_list=file_list,
            build_index=False,
            verify_files=False,
            split=split_name,
            val_size=val_size
        )
        # Restore state
        dataset.samples = cache['samples']
        dataset.available_files = cache['available_files']
        print(f"Loaded {split_name} dataset in {time.time()-start:.1f}s")
    else:
        print(f"\nBuilding {split_name} dataset index...")
        dataset = dataset_cls(
            root_dir="./data",
            file_list=file_list,
            build_index=True,
            verify_files=True,
            split=split_name,
            val_size=val_size
        )
        if is_main_process():
            print(f"Caching {split_name} dataset...")
            torch.save({
                'samples': dataset.samples,
                'available_files': dataset.available_files
            }, cache_path)
    return dataset

def main(hf_token, wandb_key):
    setup_ddp()

    if is_main_process():
        try:
            wandb.login(key=wandb_key)
        except wandb.errors.UsageError as e:
            print(f"Failed to login to W&B: {e}")
            return 

    if is_main_process():
        os.makedirs(HYPERPARAMS["CHECKPOINT_DIR"], exist_ok=True)
    dist.barrier()

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
    clip_text_start_weights, clip_text_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    clip_img_start_weights, clip_img_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_image")
    clip_wrapper_start_weights, clip_wrapper_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_wrapper")
    prior_start_weights, prior_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    sam_decoder_start_weights, sam_decoder_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")
    teacher_start_weights, teacher_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_teacher")
    student_start_weights, student_start_epoch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_student")
    
    if clip_text_start_epoch < HYPERPARAMS["CLIP_EPOCHS"] or prior_start_epoch < HYPERPARAMS["PRIOR_EPOCHS"]:
        # Streaming LAION dataset
        LAION_train_dataset = get_laion_streaming_dataset(
            HUGGINGFACE = hf_token, 
            text_processor = CLIPTokenize,
            split = "train",
            val_size=HYPERPARAMS["LAION_VAL_SIZE"]
        )
        LAION_train_loader = DataLoader(
            LAION_train_dataset,
            batch_size=HYPERPARAMS["LAION_BATCH_SIZE"],
            collate_fn=adaptive_collate,
            pin_memory=True
        )
    
        LAION_val_dataset = get_laion_streaming_dataset(
            HUGGINGFACE = hf_token, 
            text_processor = CLIPTokenize,
            split = "val",
            val_size=HYPERPARAMS["LAION_VAL_SIZE"]
        )
        LAION_val_loader = DataLoader(
            LAION_val_dataset,
            batch_size=HYPERPARAMS["LAION_BATCH_SIZE"],
            collate_fn=adaptive_collate,
            pin_memory=True
        )

        if clip_text_start_epoch < HYPERPARAMS["CLIP_EPOCHS"]:
            print("Starting CLIP Training Phase")
            train_clip(train_loader=LAION_train_loader,
                       val_loader=LAION_val_loader, 
                       text_start_weights=clip_text_start_weights, 
                       img_start_weights=clip_img_start_weights,
                       wrapper_start_weights=clip_wrapper_start_weights,
                       start_epoch=clip_text_start_epoch + 1, 
                       run=run)
        else:
            print("CLIP training already completed or up to date.")

        if prior_start_epoch < HYPERPARAMS["PRIOR_EPOCHS"]:
            print("Starting Prior Training Phase")
            train_prior(train_loader=LAION_train_loader,
                        val_loader=LAION_val_loader, 
                        start_weights=prior_start_weights, 
                        start_epoch=prior_start_epoch + 1, 
                        run=run)
        else:
            print("Prior training already completed or up to date.")
    else:
        print("CLIP and Prior already trained")


    if sam_decoder_start_epoch < HYPERPARAMS["SAM_DECODER_EPOCHS"] or student_start_epoch < HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]:
        # Mixed Dataset loading
        def load_file_list(file_path):
            with open(file_path) as f:
                return [line.strip().split('\t') for line in f.readlines()[1:]]

        print("Initializing SAM datasets")
        sa1b_files = load_file_list("data/Datasets/SA-1B_dataset_copy.txt")
        sav_files = load_file_list("data/Datasets/SA-V_dataset_copy.txt")

        sa1b_train = get_dataset(SA1BDataset, sa1b_files, "sa1b_cache", "train", HYPERPARAMS["SA_VAL_SIZE"])
        sav_train = get_dataset(SAVDataset, sav_files, "sav_cache", "train", HYPERPARAMS["SAV_VAL_SIZE"])
    
        sa1b_val = get_dataset(SA1BDataset, sa1b_files, "sa1b_cache", "val", HYPERPARAMS["SA_VAL_SIZE"])
        sav_val = get_dataset(SAVDataset, sav_files, "sav_cache", "val", HYPERPARAMS["SAV_VAL_SIZE"])

        # Create Datasets
        train_dataset = torch.utils.data.ConcatDataset([sa1b_train, sav_train])
        val_dataset = torch.utils.data.ConcatDataset([sa1b_val, sav_val])

        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=HYPERPARAMS["SAM_BATCH_SIZE"], 
                                      shuffle=False, 
                                      num_workers=10, 
                                      collate_fn=SAM_adaptive_collate, 
                                      pin_memory=True,
                                      sampler=train_sampler)
    
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=HYPERPARAMS["SAM_BATCH_SIZE"], 
                                    shuffle=False, 
                                    num_workers=10, 
                                    collate_fn=SAM_adaptive_collate, 
                                    pin_memory=True,
                                    sampler=val_sampler)

        if sam_decoder_start_epoch < HYPERPARAMS["SAM_DECODER_EPOCHS"]:
            print("Starting SAM Decoder Training Phase")
            train_SAM_decoder(train_dataloader, 
                              val_dataloader, 
                              start_weights=sam_decoder_start_weights,
                              start_epoch=sam_decoder_start_epoch + 1, 
                              run=run)
        else:
            print("SAM Decoder training already completed or up to date.")

        if student_start_epoch < HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]:
            print("Starting Student Training Phase")
            train_student(train_dataloader, 
                          val_dataloader, 
                          teacher_start_weights=teacher_start_weights,
                          student_start_weights=student_start_weights,
                          start_epoch=student_start_epoch + 1, 
                          run=run)
        else:
            print("Student training already completed or up to date.")
    
    print("All training phases completed")
    run.finish()

# ======== Main ========
if __name__ == "__main__":
    # Configure command line interface
    parser = argparse.ArgumentParser(description="Train pipeline for Zero-Shot Segmentation")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API key")
    args = parser.parse_args()
    
    torch.multiprocessing.freeze_support()
    torch.manual_seed(42)
    
    main(args.token, args.wandb_key)