# test_pipeline.py
import torch
import os
import re
import glob
import wandb
import argparse
from torch.utils.data import DataLoader, ChainDataset, DistributedSampler
from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss
from models.prior_model import create_prior, PriorLoss, TeacherCLIP
from models.SAM_model import iou_loss, create_SAM
from models.distill_model import create_Student
from data.custom400m import get_laion_streaming_dataset, adaptive_collate
from data.segmentation import SAM_adaptive_collate, SA1BDataset, SAVDataset, StaticSA1BDataset, StaticSAVDataset

# ======== Hyperparameters & Setup ========
HYPERPARAMS = {
    "CLIP_EPOCHS": 2,
    "PRIOR_EPOCHS": 2,
    "SAM_DECODER_EPOCHS": 1,
    "TEACHER_STUDENT_EPOCHS": 1,
    "CLIP_LR": 0.0001,
    "PRIOR_LR": 0.0001,
    "DECODER_LR": 0.0001, # For SAM Decoder training
    "TEACHER_LR": 0.00001, # For teacher fine-tuning during student training
    "STUDENT_LR": 0.0001,
    "LAION_VAL_SIZE": 5,
    "LAION_BATCH_SIZE": 5,
    "SA_VAL_TAR_COUNT": 1,
    "SA_VAL_SAMPLE_COUNT": 2,
    "SAV_VAL_TAR_COUNT": 1,
    "SAV_VAL_SAMPLE_COUNT": 2,
    "SAM_BATCH_SIZE": 2,
    "SAVE_FREQ": 1,
    "CHECKPOINT_DIR": "weights_test",
    "WANDB_PROJECT_NAME": "Zero_Shot_Segmentation_Laptop_Test",
    "WANDB_ENTITY_NAME": "adityaasuratkal-rensselaer-polytechnic-institute"
}

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        print("✅ Using Apple MPS (Metal Performance Shaders)")
        return 'mps'
    else:
        return 'cpu'
device = get_device()

def setup_ddp():
    return False

def is_main_process():
    return True

# ======== Helper Function for Checkpoints ========
def get_latest_epoch_checkpoint(directory, prefix):
    if not os.path.isdir(directory):
        return None, 0, 0
    
    files = glob.glob(os.path.join(directory, f"{prefix}_epoch_*"))
    if not files: 
        return None, 0, 0

    latest_parsed_ep = -1
    latest_parsed_ba = -1
    latest_file = None
    is_mid_epoch = False
    
    for f_path in files:
        filename = os.path.basename(f_path)
        
        match_complete = re.search(rf"{prefix}_epoch_(\d+)_complete", filename)
        if match_complete:
            ep = int(match_complete.group(1))
            ba = float('inf')
            if ep > latest_parsed_ep or (ep == latest_parsed_ep and ba > latest_parsed_ba):
                latest_parsed_ep = ep
                latest_parsed_ba = ba
                latest_file = f_path
                is_mid_epoch = False
            continue

        match_new = re.search(rf"{prefix}_epoch_(\d+)_batch_(\d+)_([0-9]+(?:\.[0-9]+)?)", filename)
        if match_new:
            ep, ba = int(match_new.group(1)), int(match_new.group(2))
            if ep > latest_parsed_ep or (ep == latest_parsed_ep and ba > latest_parsed_ba):
                latest_parsed_ep = ep
                latest_parsed_ba = ba
                latest_file = f_path
                is_mid_epoch = True
                    
    if latest_file is None:
        return None, 0, 0
        
    if is_mid_epoch:
        return latest_file, latest_parsed_ep - 1, latest_parsed_ba
    else:
        return latest_file, latest_parsed_ep, 0

def get_best_weights_checkpoint(directory, prefix):
    if not os.path.isdir(directory):
        return None, 0, 0
    
    files = glob.glob(os.path.join(directory, f"{prefix}_epoch_*"))
    best_loss = float('inf')
    best_file = None
    best_epoch = -1
    best_batch = -1
    
    for f_path in files:
        filename = os.path.basename(f_path)
        match_new = re.search(rf"{prefix}_epoch_(\d+)_batch_(\d+)_([0-9]+(?:\.[0-9]+)?)", filename)
        if match_new:
            ep, ba, loss = int(match_new.group(1)), int(match_new.group(2)), float(match_new.group(3))
            if loss < best_loss:
                best_loss, best_file, best_epoch, best_batch = loss, f_path, ep, ba
    
    if best_file is None:
        return get_latest_epoch_checkpoint(directory, prefix)
                
    return best_file, max(0, best_epoch), max(0, best_batch)

# ======== CLIP Training ========
def train_clip(train_loader, val_loader, text_start_weights, img_start_weights, wrapper_start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    print("\n=== Training CLIP ===")
    local_device = device

    text_encoder = create_text_encoder().to(local_device)
    image_encoder = create_image_encoder().to(local_device)

    clip_model:CLIPWrapper = CLIPWrapper(text_encoder, image_encoder).to(local_device)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=HYPERPARAMS["CLIP_LR"])
    
    if start_epoch > 0:
        if os.path.exists(text_start_weights) and os.path.exists(img_start_weights) and os.path.exists(wrapper_start_weights):
            print(f"Resuming CLIP training from epoch {start_epoch} batch {start_batch}")
            clip_model.load_weights(wrapper_start_weights, img_start_weights, text_start_weights)
        else:
            print(f"Warning: Checkpoint for epoch {start_epoch} not found. Starting CLIP from scratch.")
            start_epoch = 0
            start_batch = 0

    for epoch in range(start_epoch, HYPERPARAMS["CLIP_EPOCHS"]):
        clip_model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx > 10: break
            if epoch == start_epoch and (batch_idx <= start_batch and start_batch > 0):
                print(f"Skipping batch {batch_idx}")
                continue
                
            if batch is None: continue
            
            images, texts = batch
            print(images.size())
            print(texts.size())

            images = images.to(local_device)
            texts = texts.to(local_device)
            optimizer.zero_grad()
            
            text_features, image_features, logit_scale = clip_model(texts, images)
            
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()
            loss = clip_contrastive_loss(logits_per_image, logits_per_text)
            
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

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                clip_model.eval()
                val_loss = 0.0
                iters = 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        if v_batch is None: continue
                        v_images, v_texts = v_batch

                        print(v_images.size())
                        print(v_texts.size())

                        v_images = v_images.to(device)
                        v_texts = v_texts.to(device)
            
                        # Forward pass
                        text_features, image_features, v_scale = clip_model(v_texts, v_images)
            
                        # Contrastive loss
                        v_logits_per_image = v_scale * image_features @ text_features.t()
                        v_logits_per_text = v_scale * text_features @ image_features.t()
                        loss = clip_contrastive_loss(v_logits_per_image, v_logits_per_text)
            
                        # Backprop
                        val_loss += loss.item()
                        iters += 1
                        if iters >= 10: break 
                        
                avg_val = val_loss / iters if iters > 0 else 999.9
                clip_model.store_weights(
                    HYPERPARAMS['CHECKPOINT_DIR'], 
                    f"clip_text_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}", 
                    f"clip_image_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}", 
                    f"clip_wrapper_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}"
                )
                print(f"Saved CLIP partial epoch {epoch+1} batch {batch_idx} (Val Loss: {avg_val:.4f})")
                clip_model.train()

        if is_main_process():
            avg_epoch_loss = total_loss / (batch_idx + 1) if batch_idx > -1 else 0
            print(f"CLIP Epoch {epoch+1} Average Train Loss: {avg_epoch_loss:.4f}")
            
            clip_model.store_weights(
                HYPERPARAMS['CHECKPOINT_DIR'], 
                f"clip_text_epoch_{epoch+1}_complete", 
                f"clip_image_epoch_{epoch+1}_complete", 
                f"clip_wrapper_epoch_{epoch+1}_complete"
            )
            print(f"Epoch {epoch+1} fully complete. Saved complete flag checkpoints.")
            
    print("CLIP training completed.")

# ======== Prior Training ========
def train_prior(train_loader, val_loader, start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    print("\n=== Training Prior ===")
    local_device = device

    text_encoder = create_text_encoder().to(local_device)
    prior_teacher = TeacherCLIP().to(local_device)
    
    best_clip_text_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    if not best_clip_text_ckpt: raise FileNotFoundError("Latest CLIP text or image checkpoints not found. Train CLIP first.")
    print(f"Text encoder weights: {best_clip_text_ckpt}")
    text_encoder.load_weights(best_clip_text_ckpt)
    
    for param in text_encoder.parameters(): param.requires_grad_(False)
    text_encoder.eval()
    for param in prior_teacher.parameters(): param.requires_grad_(False)
    prior_teacher.eval()

    prior_model = create_prior().to(local_device)
    optimizer = torch.optim.Adam(prior_model.parameters(), lr=HYPERPARAMS["PRIOR_LR"])

    if start_epoch > 0:
        if os.path.exists(start_weights):
            prior_model.load_weights(start_weights)
        else:
            start_epoch = 0
            start_batch = 0
    
    prior = prior_model

    for epoch in range(start_epoch, HYPERPARAMS["PRIOR_EPOCHS"]):
        prior.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx > 10: break
            if epoch == start_epoch and batch_idx <= start_batch and start_batch > 0:
                print(f"Skipping batch {batch_idx}")
                continue
            if batch is None: continue
            
            images, texts = batch

            print(images.size())
            print(texts.size())

            images = images.to(local_device)
            texts = texts.to(local_device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                text_emb = text_encoder(texts)
                target_grid = prior_teacher(images)
            
            prior_grid = prior(text_emb)
            loss = PriorLoss(prior_grid, target_grid)
            
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

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                prior.eval()
                val_loss = 0.0
                iters = 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        if v_batch is None: continue

                        v_images, v_texts = v_batch

                        print(v_images.size())
                        print(v_texts.size())

                        v_texts = v_texts.to(local_device)
                        v_images = v_images.to(local_device)

                        v_text_emb = text_encoder(v_texts)
                        v_target_grid = prior_teacher(v_images)
                        v_prior_grid = prior(v_text_emb)

                        val_loss += PriorLoss(v_prior_grid, v_target_grid).item()

                        iters += 1

                        if iters >= 10: break
                
                avg_val = val_loss / iters if iters > 0 else 999.9
                prior.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"], 
                    f"prior_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}"
                )
                print(f"Saved Prior partial epoch {epoch+1} batch {batch_idx} (Val Loss: {avg_val:.4f})")
                prior.train()

        if is_main_process():
            avg_epoch_loss = total_loss / (batch_idx + 1) if batch_idx > -1 else 0
            print(f"Prior Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

            prior.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"prior_epoch_{epoch+1}_complete"
            )
            print(f"Epoch {epoch+1} fully complete. Saved complete flag checkpoints.")
            
    print("Prior training completed.\n")

# ======== SAM Teacher Training ========
def train_SAM_decoder(train_dataloader, val_dataloader, start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    print("\n=== Training SAM Decoder (Teacher Component) ===")
    local_device = 'cpu'

    text_encoder = create_text_encoder().to(local_device)
    prior = create_prior().to(local_device)
    sam_decoder = create_SAM().to(local_device)

    best_clip_text_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    if not best_clip_text_ckpt: raise FileNotFoundError("CLIP text checkpoint not found for SAM Decoder training.")
    print(f"Text encoder weights: {best_clip_text_ckpt}")
    text_encoder.load_weights(best_clip_text_ckpt)

    best_prior_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    if not best_prior_ckpt: raise FileNotFoundError("Prior checkpoint not found for SAM Decoder training.")
    print(f"Prior model weights: {best_prior_ckpt}")
    prior.load_weights(best_prior_ckpt)

    if start_epoch > 0:
        if os.path.exists(start_weights):
            sam_decoder.load_weights(start_weights)
        else:
            start_epoch = 0
            start_batch = 0

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

    teacher = TeacherModel().to(local_device)

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
            if batch_idx > 5: break
            if epoch == start_epoch and batch_idx <= start_batch and start_batch > 0:
                print(f"Skipping batch {batch_idx}")
                continue
            if batch is None: continue

            images, true_masks, texts = batch
            current_batch_loss_sum = 0
            num_samples_in_batch = 0
            optimizer_sam_decoder.zero_grad()

            for img, mask, txt in zip(images, true_masks, texts):
                print("Train entry")
                print(img.size())
                print(mask.size())
                print(txt.size())

                mask = mask.to(local_device).float()
                img = img.to(local_device)
                txt = txt.to(local_device)

                pred_mask = teacher.forward(img, txt)
                loss = iou_loss(pred_mask, mask)
                loss.backward()
                
                current_batch_loss_sum += loss.item()
                num_samples_in_batch += 1
        
            optimizer_sam_decoder.step()

            avg_batch_item_loss = current_batch_loss_sum / max(1, num_samples_in_batch)
            total_loss += avg_batch_item_loss
            batch_count += 1

            if batch_idx % 100 == 0 and is_main_process():
                print(f"SAM Decoder Epoch {epoch+1}/{HYPERPARAMS['SAM_DECODER_EPOCHS']} | Batch {batch_idx} Avg Item Loss: {avg_batch_item_loss:.4f}")
                run.log({
                    "sam_decoder_batch_avg_item_loss": avg_batch_item_loss,
                    "sam_decoder_epoch": epoch + 1,
                    "sam_decoder_batch_idx": batch_idx
                })

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                teacher.sam_decoder.eval()
                val_loss = 0.0
                iters = 0
                num_val_samples = 0
                with torch.no_grad():
                    for v_batch in val_dataloader:
                        if v_batch is None: continue
                        v_images, v_true_masks, v_texts = v_batch

                        for v_img, v_mask, v_txt in zip(v_images, v_true_masks, v_texts):
                            print("Val entry")
                            print(v_img.size())
                            print(v_mask.size())
                            print(v_txt.size())

                            v_mask = v_mask.to(local_device).float()
                            v_img = v_img.to(local_device)
                            v_txt = v_txt.to(local_device)

                            v_pred = teacher.forward(v_img, v_txt)
                            
                            val_loss += iou_loss(v_pred, v_mask).item()
                            num_val_samples += 1

                        iters += 1
                        if iters >= 10: break
                        
                avg_val = val_loss / num_val_samples if num_val_samples > 0 else 999.9
                teacher.sam_decoder.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"], 
                    f"sam_decoder_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}"
                )
                print(f"Saved SAM Decoder partial epoch {epoch+1} batch {batch_idx} (Val Loss: {avg_val:.4f})")
                teacher.sam_decoder.train()

            batch_count += 1
            
        if is_main_process():
            avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"SAM Decoder Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

            teacher.sam_decoder.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"sam_decoder_epoch_{epoch+1}_complete"
            )
            print(f"Epoch {epoch+1} fully complete. Saved complete flag checkpoints.")

    print("SAM Decoder training completed.\n")

# ======== SAM Student Training ========
def train_student(train_dataloader, val_dataloader, teacher_start_weights, student_start_weights, run:wandb, start_epoch = 0, start_batch = 0):
    print("\n=== Training Student (with Teacher Fine-tuning) ===")
    local_device = 'cpu'

    text_encoder = create_text_encoder().to(local_device)
    prior = create_prior().to(local_device)
    sam_decoder = create_SAM().to(local_device)

    best_clip_text_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    best_prior_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    best_sam_decoder_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")

    if not best_clip_text_ckpt: raise FileNotFoundError("CLIP text ckpt not found for Student training.")
    if not best_prior_ckpt: raise FileNotFoundError("Prior ckpt not found for Student training.")
    if not best_sam_decoder_ckpt: raise FileNotFoundError("SAM Decoder ckpt not found for Student training.")

    print(f"Text encoder weights: {best_clip_text_ckpt}")
    text_encoder.load_weights(best_clip_text_ckpt)
    print(f"Prior model weights: {best_prior_ckpt}")
    prior.load_weights(best_prior_ckpt)
    print(f"SAM decoder weights: {best_sam_decoder_ckpt}")
    sam_decoder.load_weights(best_sam_decoder_ckpt)

    for param in text_encoder.parameters(): param.requires_grad_(False)
    text_encoder.eval()
    for param in prior.parameters(): param.requires_grad_(False)
    prior.eval()

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

    teacher = TeacherModel().to(local_device)
    student = create_Student().to(local_device)

    if start_epoch > 0:
        if os.path.exists(student_start_weights):
            student.load_weights(student_start_weights)
            if os.path.exists(teacher_start_weights):
                teacher.sam_decoder.load_weights(teacher_start_weights)
        else:
            start_epoch = 0
            start_batch = 0

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
            if batch_idx > 5: break
            if epoch == start_epoch and batch_idx <= start_batch and start_batch > 0:
                print(f"Skipping batch {batch_idx}")
                continue
            if batch is None: continue

            images, true_masks, texts = batch

            current_batch_teacher_loss_sum = 0
            current_batch_student_loss_sum = 0
            num_samples_in_batch = 0

            optimizer_teacher_finetune.zero_grad()
            optimizer_student.zero_grad()

            for img, mask, txt in zip(images, true_masks, texts):
                print("Train entry")
                print(img.size())
                print(mask.size())
                print(txt.size())

                mask = mask.to(local_device).float()
                img = img.to(local_device)
                txt = txt.to(local_device)

                teacher_out = teacher(img, txt)
                student_out = student(img, txt)
                
                with torch.no_grad():
                    teacher_out_for_student = teacher(img, txt).detach()

                teacher_loss = iou_loss(teacher_out, mask)
                student_loss = student.compute_distill_loss(student_out, teacher_out_for_student, mask)

                teacher_loss.backward(retain_graph=True)
                student_loss.backward()

                current_batch_teacher_loss_sum += teacher_loss.item()
                current_batch_student_loss_sum += student_loss.item()
                num_samples_in_batch += 1
                    
            optimizer_teacher_finetune.step()
            optimizer_student.step()

            batch_count += 1
            avg_batch_teacher_loss = current_batch_teacher_loss_sum / max(1, num_samples_in_batch)
            avg_batch_student_loss = current_batch_student_loss_sum / max(1, num_samples_in_batch)
            total_teacher_loss += avg_batch_teacher_loss
            total_student_loss += avg_batch_student_loss
            
            if batch_idx % 100 == 0 and is_main_process():
                print(f"Student Epoch {epoch+1}/{HYPERPARAMS['TEACHER_STUDENT_EPOCHS']} | Batch {batch_idx} | Teacher Loss: {avg_batch_teacher_loss:.4f}, Student Loss: {avg_batch_student_loss:.4f}")
                run.log({
                    "student_phase_batch_teacher_loss": avg_batch_teacher_loss,
                    "student_phase_batch_student_loss": avg_batch_student_loss,
                    "student_phase_epoch": epoch + 1,
                    "student_phase_batch_idx": batch_idx
                })

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                teacher.sam_decoder.eval()
                student.eval()
                val_t_loss, val_s_loss = 0.0, 0.0
                iters = 0
                num_val_samples = 0
                with torch.no_grad():
                    for v_batch in val_dataloader:
                        if v_batch is None: continue
                        v_images, v_true_masks, v_texts = v_batch
                        for v_img, v_mask, v_txt in zip(v_images, v_true_masks, v_texts):
                            print("Val entry")
                            print(v_img.size())
                            print(v_mask.size())
                            print(v_txt.size())

                            v_mask = v_mask.to(local_device).float()
                            v_img = v_img.to(local_device)
                            v_txt = v_txt.to(local_device)

                            t_out = teacher(v_img, v_txt)
                            s_out = student(v_img, v_txt)
                            
                            val_t_loss += iou_loss(t_out, v_mask).item()
                            val_s_loss += student.compute_distill_loss(s_out, t_out, v_mask).item()
                            num_val_samples += 1

                        iters += 1
                        if iters >= 10: break

                avg_t_val = val_t_loss / num_val_samples if num_val_samples > 0 else 999.9
                avg_s_val = val_s_loss / num_val_samples if num_val_samples > 0 else 999.9
                
                teacher.sam_decoder.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"], 
                    f"student_phase_teacher_epoch_{epoch+1}_batch_{batch_idx}_{avg_t_val:.4f}")
                student.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"], 
                    f"student_phase_student_epoch_{epoch+1}_batch_{batch_idx}_{avg_s_val:.4f}")
                print(f"Saved Joint Phase partial epoch {epoch+1} batch {batch_idx}")
                
                teacher.sam_decoder.train()
                student.train()

        if is_main_process():
            avg_epoch_teacher_loss = total_teacher_loss / batch_count if batch_count > 0 else 0
            avg_epoch_student_loss = total_student_loss / batch_count if batch_count > 0 else 0
            print(f"Student Epoch {epoch+1} Avg Losses - Teacher: {avg_epoch_teacher_loss:.4f}, Student: {avg_epoch_student_loss:.4f}")

            teacher.sam_decoder.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"student_phase_teacher_epoch_{epoch+1}_complete")
            student.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"student_phase_student_epoch_{epoch+1}_complete")
            print(f"Epoch {epoch+1} fully complete. Saved complete flag checkpoints.")

    print("Student training completed.\n")

def get_dataset(dataset_cls, file_list, split_name, val_tar_count, val_sample_count=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    dataset = dataset_cls(
        file_list=file_list,
        cache_dir=cache_dir,
        device=device,
        split=split_name,
        val_tar_count=val_tar_count,
        val_sample_count=val_sample_count
    )
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
    clip_text_start_weights, clip_text_start_epoch, clip_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    clip_img_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_image")
    clip_wrapper_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_wrapper")
    prior_start_weights, prior_start_epoch, prior_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    sam_decoder_start_weights, sam_decoder_start_epoch, sam_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")
    teacher_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_teacher")
    student_start_weights, student_start_epoch, student_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_student")

    print(f"CLIP latest weights at epoch {clip_text_start_epoch+1} batch {clip_start_batch}")
    print(f"Prior latest weights at epoch {prior_start_epoch+1} batch {prior_start_batch}")
    print(f"SAM Decoder latest weights at epoch {sam_decoder_start_epoch+1} batch {sam_start_batch}")
    print(f"Teacher-Student latest weights at epoch {student_start_epoch+1} batch {student_start_batch}")

    if clip_text_start_epoch < HYPERPARAMS["CLIP_EPOCHS"] or prior_start_epoch < HYPERPARAMS["PRIOR_EPOCHS"]:
        LAION_train_dataset = get_laion_streaming_dataset(
            HUGGINGFACE = hf_token, 
            text_processor = CLIPTokenize,
            split = "train",
            val_size=HYPERPARAMS["LAION_VAL_SIZE"]+1
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
            val_size=HYPERPARAMS["LAION_VAL_SIZE"]+1
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
                       start_epoch=clip_text_start_epoch, 
                       start_batch=clip_start_batch,
                       run=run)
        else:
            print("CLIP training already completed.")

        if prior_start_epoch < HYPERPARAMS["PRIOR_EPOCHS"]:
            print("Starting Prior Training Phase")
            train_prior(train_loader=LAION_train_loader,
                        val_loader=LAION_val_loader, 
                        start_weights=prior_start_weights, 
                        start_epoch=prior_start_epoch, 
                        start_batch=prior_start_batch,
                        run=run)
        else:
            print("Prior training already completed.")
    else:
        print("CLIP and Prior already trained")
        
    num_workers = 0 if device == 'mps' else 10

    if sam_decoder_start_epoch < HYPERPARAMS["SAM_DECODER_EPOCHS"] or student_start_epoch < HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]:
        def load_file_list(file_path):
            try:
                with open(file_path) as f:
                    return [line.strip().split('\t') for line in f.readlines()[1:]]
            except FileNotFoundError:
                return []

        print("Initializing Iterable SAM datasets...")
        sa1b_files = load_file_list("data/Datasets/SA-1B_dataset.txt")
        sav_files = load_file_list("data/Datasets/SA-V_dataset.txt")

        # Training (Streaming Infinite Iterables)
        sa1b_train = get_dataset(SA1BDataset, sa1b_files, "train", HYPERPARAMS["SA_VAL_TAR_COUNT"])
        sav_train = get_dataset(SAVDataset, sav_files, "train", HYPERPARAMS["SAV_VAL_TAR_COUNT"])
        
        # Validation (Fast Static Loaders)
        sa1b_val = get_dataset(StaticSA1BDataset, sa1b_files, "val", HYPERPARAMS["SA_VAL_TAR_COUNT"], HYPERPARAMS.get("SA_VAL_SAMPLE_COUNT"))
        sav_val = get_dataset(StaticSAVDataset, sav_files, "val", HYPERPARAMS["SAV_VAL_TAR_COUNT"], HYPERPARAMS.get("SAV_VAL_SAMPLE_COUNT"))

        train_dataset = ChainDataset([sa1b_train, sav_train])
        val_dataset = torch.utils.data.ConcatDataset([sa1b_val, sav_val])

        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=HYPERPARAMS["SAM_BATCH_SIZE"], 
                                      shuffle=False, 
                                      num_workers=num_workers, 
                                      collate_fn=SAM_adaptive_collate, 
                                      pin_memory=True, 
                                      sampler=None)
        
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if "LOCAL_RANK" in os.environ else None
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=HYPERPARAMS["SAM_BATCH_SIZE"], 
                                    shuffle=False, 
                                    num_workers=num_workers, 
                                    collate_fn=SAM_adaptive_collate, 
                                    pin_memory=True, 
                                    sampler=val_sampler)

        if sam_decoder_start_epoch < HYPERPARAMS["SAM_DECODER_EPOCHS"]:
            print("Starting SAM Decoder Training Phase")
            train_SAM_decoder(train_dataloader, 
                              val_dataloader, 
                              start_weights=sam_decoder_start_weights,
                              start_epoch=sam_decoder_start_epoch, 
                              start_batch=sam_start_batch,
                              run=run)
        else:
            print("SAM Decoder training already completed.")

        if student_start_epoch < HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]:
            print("Starting Student Training Phase")
            train_student(train_dataloader, 
                          val_dataloader, 
                          teacher_start_weights=teacher_start_weights,
                          student_start_weights=student_start_weights,
                          start_epoch=student_start_epoch,
                          start_batch=student_start_batch,
                          run=run)
        else:
            print("Student training already completed.")
    
    print("All training phases completed")

    best_clip_text_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    best_prior_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    best_sam_decoder_standalone_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")
    best_sam_decoder_teacher_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_teacher")
    best_student_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_student")

    print(f"Best Text Encoder checkpoint: {best_clip_text_ckpt}")
    print(f"Best Prior checkpoint: {best_prior_ckpt}")
    print(f"Best SAM Decoder (Standalone) checkpoint: {best_sam_decoder_standalone_ckpt}")
    print(f"Best SAM Decoder (Teacher) checkpoint: {best_sam_decoder_teacher_ckpt}")
    print(f"Best Student checkpoint: {best_student_ckpt}")

    run.finish()

# ======== Main ========
if __name__ == "__main__":
    # Configure command line interface
    parser = argparse.ArgumentParser(description="Train pipeline for Zero-Shot Segmentation")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API key")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    
    main(args.token, args.wandb_key)