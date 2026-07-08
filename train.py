# train.py
import os
os.environ["HF_HUB_HTTP_TIMEOUT"] = "3600"

import socket
socket.setdefaulttimeout(3600)

import datasets
datasets.config.DOWNLOAD_DEFAULT_TIMEOUT = 3600
datasets.config.MAX_RETRIES = 10

import requests
import time
import random
import datetime

_original_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs['timeout'] = 3600
    for attempt in range(15): 
        response = _original_send(self, request, **kwargs)
        if response.status_code == 429:
            sleep_duration = 310 + random.uniform(0, 120)
            print(f"Worker hit Hugging Face 5-minute quota. Sleeping for {sleep_duration/60:.1f} minutes...")
            time.sleep(sleep_duration)
            continue
        return response
    return _original_send(self, request, **kwargs)

requests.Session.send = _patched_send

_original_request = requests.Session.request
def _patched_request(self, method, url, **kwargs):
    kwargs['timeout'] = 3600
    return _original_request(self, method, url, **kwargs)
requests.Session.request = _patched_request

import torch
import re
import glob
import wandb
import argparse
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.clip_model import create_text_encoder, create_image_encoder, CLIPWrapper, clip_contrastive_loss
from models.prior_model import create_prior, PriorLoss
from models.SAM_model import iou_loss, create_SAM
from models.distill_model import create_Student
from data.custom400m import adaptive_collate, get_laion_dataset
from data.segmentation import SAM_adaptive_collate, get_segmentation_dataset
import math
from torch.optim.lr_scheduler import LambdaLR

# ======== Hyperparameters & Setup ========
HYPERPARAMS = {
    "CLIP_EPOCHS": 3,
    "PRIOR_EPOCHS": 2,
    "SAM_DECODER_EPOCHS": 1,
    "TEACHER_STUDENT_EPOCHS": 2,
    "CLIP_LR": 0.002,
    "PRIOR_LR": 0.0005,
    "DECODER_LR": 0.0001, # For SAM Decoder training
    "TEACHER_LR": 0.00001, # For teacher fine-tuning during student training
    "STUDENT_LR": 0.0001,
    "CLIP_WEIGHT_DECAY": 0.0009239627275656261,
    "PRIOR_WEIGHT_DECAY":   3.0316183452561147e-06,
    "DECODER_WEIGHT_DECAY": 0.002503434387763064,
    "TEACHER_WEIGHT_DECAY": 0.004887852632767088,
    "STUDENT_WEIGHT_DECAY": 0.008092493298107342,
    "WARMUP_STEPS": 1000,
    "MIN_LR_RATIO": 0.01,
    "LAION_VAL_SIZE": 10000,
    "CLIP_BATCH_SIZE": 32,
    "PRIOR_BATCH_SIZE": 32,
    "DECODER_BATCH_SIZE": 1,
    "STUDENT_BATCH_SIZE": 64,
    "LAION_TOTAL_SAMPLES": 200000,
    "SEG_VAL_SIZE": 2000,
    "SAVE_FREQ": 50,
    "CHECKPOINT_DIR": "weights",
    "WANDB_PROJECT_NAME": "Zero Shot Segmentation",
    "WANDB_ENTITY_NAME": "adityaasuratkal-rensselaer-polytechnic-institute"
}

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
device = get_device()

def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(hours=4)
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_main_process():
    return dist.get_rank() == 0

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
        
        match_complete = re.search(rf"{prefix}_epoch_(\d+)_complete_([0-9]+(?:\.[0-9]+)?)", filename)
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
        match_complete = re.search(rf"{prefix}_epoch_(\d+)_complete_([0-9]+(?:\.[0-9]+)?)", filename)
        if match_complete:
            ep, loss = int(match_complete.group(1)), float(match_complete.group(2))
            if loss < best_loss:
                best_loss, best_file, best_epoch, best_batch = loss, f_path, ep, 0

        match_new = re.search(rf"{prefix}_epoch_(\d+)_batch_(\d+)_([0-9]+(?:\.[0-9]+)?)", filename)
        if match_new:
            ep, ba, loss = int(match_new.group(1)), int(match_new.group(2)), float(match_new.group(3))
            if loss < best_loss:
                best_loss, best_file, best_epoch, best_batch = loss, f_path, ep, ba
    
    if best_file is None:
        return get_latest_epoch_checkpoint(directory, prefix)
                
    return best_file, max(0, best_epoch), max(0, best_batch)

def create_lr_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: 0 → 1 over warmup_steps
            return current_step / max(1, warmup_steps)
        else:
            # Cosine decay: 1 → min_lr_ratio over remaining steps
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)

# ======== CLIP Training ========
def train_clip(train_dataset, val_dataset, text_start_weights, img_start_weights, wrapper_start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    if is_main_process():
        print("\n=== Training CLIP ===")
    local_device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}') if "LOCAL_RANK" in os.environ else device

    text_encoder = create_text_encoder().to(local_device)
    image_encoder = create_image_encoder().to(local_device)

    clip_model:CLIPWrapper = CLIPWrapper(text_encoder, image_encoder).to(local_device)
    optimizer = torch.optim.Adam(clip_model.parameters(), lr=HYPERPARAMS["CLIP_LR"], weight_decay=HYPERPARAMS["CLIP_WEIGHT_DECAY"])

    if start_epoch > 0:
        if os.path.exists(text_start_weights) and os.path.exists(img_start_weights) and os.path.exists(wrapper_start_weights):
            print(f"Resuming CLIP training from epoch {start_epoch} batch {start_batch}")
            clip_model.load_weights(wrapper_start_weights, img_start_weights, text_start_weights)
        else:
            print(f"Warning: Checkpoint for epoch {start_epoch} not found. Starting CLIP from scratch.")
            start_epoch = 0
            start_batch = 0

    clip_model = DDP(clip_model, device_ids=[int(os.environ["LOCAL_RANK"])])

    use_dist = dist.is_initialized()
    batch_size = HYPERPARAMS["CLIP_BATCH_SIZE"]
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_dist else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=adaptive_collate,
        pin_memory=True, num_workers=4, prefetch_factor=4,
        persistent_workers=True, drop_last=True,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_dist else None
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, collate_fn=adaptive_collate, pin_memory=True, num_workers=2,
    )

    estimated_total_steps = HYPERPARAMS["CLIP_EPOCHS"] * len(train_loader)
    scheduler = create_lr_warmup_cosine_scheduler(
        optimizer, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )

    if start_epoch > 0 or start_batch > 0:
        skip_steps = start_epoch * len(train_loader) + start_batch
        for _ in range(skip_steps):
            scheduler.step()
        if is_main_process():
            print(f"[Resume] Resuming from epoch {start_epoch}, batch {start_batch}")

    if dist.is_initialized():
        dist.barrier()
        if is_main_process():
            print("[Sync] All ranks ready. Starting CLIP training loop.")

    for epoch in range(start_epoch, HYPERPARAMS["CLIP_EPOCHS"]):
        clip_model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0
        batch_start = start_batch if epoch == start_epoch else 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < batch_start:
                continue
            if batch is None: continue

            images, texts = batch
            images = images.to(local_device)
            texts = texts.to(local_device)
            optimizer.zero_grad()

            text_features, image_features, logit_scale = clip_model(texts, images)

            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()
            loss = clip_contrastive_loss(logits_per_image, logits_per_text)

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0 and is_main_process():
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLIP Epoch {epoch+1}/{HYPERPARAMS['CLIP_EPOCHS']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                run.log({
                    "clip_batch_loss": loss.item(),
                    "clip_epoch": epoch + 1,
                    "clip_batch_idx": batch_idx,
                    "clip_lr": current_lr
                })

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                clip_model.eval()
                val_loss = 0.0
                iters = 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        if v_batch is None: continue
                        v_images, v_texts = v_batch

                        v_images = v_images.to(local_device)
                        v_texts = v_texts.to(local_device)

                        # Forward pass
                        text_features, image_features, v_scale = clip_model(v_texts, v_images)

                        # Contrastive loss
                        v_logits_per_image = v_scale * image_features @ text_features.t()
                        v_logits_per_text = v_scale * text_features @ image_features.t()
                        loss = clip_contrastive_loss(v_logits_per_image, v_logits_per_text)

                        # Backprop
                        val_loss += loss.item()
                        iters += 1

                avg_val = val_loss / iters if iters > 0 else 999.9
                run.log({"clip_val_loss": avg_val, "clip_epoch": epoch + 1, "clip_batch_idx": batch_idx})
                clip_model.module.store_weights(
                    HYPERPARAMS['CHECKPOINT_DIR'],
                    f"clip_text_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}",
                    f"clip_image_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}",
                    f"clip_wrapper_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}"
                )
                print(f"Saved CLIP partial epoch {epoch+1} batch {batch_idx} (Val Loss: {avg_val:.4f})")
                clip_model.train()

        if is_main_process():
            avg_epoch_loss = total_loss / max(1, len(train_loader) - batch_start)
            print(f"CLIP Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
            clip_model.eval()
            val_loss = 0.0
            iters = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    if v_batch is None: continue
                    v_images, v_texts = v_batch

                    v_images = v_images.to(local_device)
                    v_texts = v_texts.to(local_device)

                    # Forward pass
                    text_features, image_features, v_scale = clip_model(v_texts, v_images)

                    # Contrastive loss
                    v_logits_per_image = v_scale * image_features @ text_features.t()
                    v_logits_per_text = v_scale * text_features @ image_features.t()
                    loss = clip_contrastive_loss(v_logits_per_image, v_logits_per_text)

                    # Backprop
                    val_loss += loss.item()
                    iters += 1

            avg_val = val_loss / iters if iters > 0 else 999.9
            clip_model.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"],
                f"clip_text_epoch_{epoch+1}_complete_{avg_val:.4f}",
                f"clip_image_epoch_{epoch+1}_complete_{avg_val:.4f}",
                f"clip_wrapper_epoch_{epoch+1}_complete_{avg_val:.4f}")
            print(f"Saved CLIP epoch {epoch+1}(Val Loss: {avg_val:.4f})")
    if is_main_process():
        print("CLIP training completed.")

# ======== Prior Training ========
def train_prior(train_dataset, val_dataset, start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    if is_main_process():
        print("\n=== Training Prior ===")
    local_device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}') if "LOCAL_RANK" in os.environ else device

    text_encoder = create_text_encoder().to(local_device)
    prior_teacher = create_image_encoder().to(local_device)
    
    best_clip_text_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    best_clip_image_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_image")
    if not best_clip_text_ckpt or not best_clip_image_ckpt: raise FileNotFoundError("Latest CLIP text or image checkpoints not found. Train CLIP first.")
    if is_main_process():
        print(f"Text encoder weights: {best_clip_text_ckpt}")
        print(f"Image encoder weights: {best_clip_image_ckpt}")
    text_encoder.load_weights(best_clip_text_ckpt)
    prior_teacher.load_weights(best_clip_image_ckpt)
    
    for param in text_encoder.parameters(): param.requires_grad_(False)
    text_encoder.half()
    text_encoder.eval()
    for param in prior_teacher.parameters(): param.requires_grad_(False)
    prior_teacher.half()
    prior_teacher.eval()

    prior_model = create_prior().to(local_device)
    optimizer = torch.optim.Adam(prior_model.parameters(), lr=HYPERPARAMS["PRIOR_LR"], weight_decay=HYPERPARAMS["PRIOR_WEIGHT_DECAY"])

    if start_epoch > 0:
        if os.path.exists(start_weights):
            prior_model.load_weights(start_weights)
        else:
            start_epoch = 0
            start_batch = 0
    
    prior = DDP(prior_model, device_ids=[int(os.environ["LOCAL_RANK"])])

    use_dist = dist.is_initialized()
    batch_size = HYPERPARAMS["PRIOR_BATCH_SIZE"]
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_dist else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=adaptive_collate,
        pin_memory=True, num_workers=4, prefetch_factor=4,
        persistent_workers=True, drop_last=True,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_dist else None
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        shuffle=False, collate_fn=adaptive_collate, pin_memory=True, num_workers=2,
    )

    estimated_total_steps = HYPERPARAMS["PRIOR_EPOCHS"] * len(train_loader)
    scheduler = create_lr_warmup_cosine_scheduler(
        optimizer, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )

    if start_epoch > 0 or start_batch > 0:
        skip_steps = start_epoch * len(train_loader) + start_batch
        for _ in range(skip_steps):
            scheduler.step()
        if is_main_process():
            print(f"[Resume] Resuming from epoch {start_epoch}, batch {start_batch}")

    if dist.is_initialized():
        dist.barrier()
        if is_main_process():
            print("[Sync] All ranks ready. Starting Prior training loop.")

    for epoch in range(start_epoch, HYPERPARAMS["PRIOR_EPOCHS"]):
        prior.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0
        batch_start = start_batch if epoch == start_epoch else 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < batch_start:
                continue
            if batch is None: continue

            images, texts = batch
            images = images.to(local_device)
            texts = texts.to(local_device)
            optimizer.zero_grad()

            with torch.no_grad():
                text_emb = text_encoder(texts).float()
                target_grid, _ = prior_teacher(images.half())
                target_grid = target_grid.float()

            prior_grid = prior(text_emb)
            loss = PriorLoss(prior_grid, target_grid)

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0 and is_main_process():
                current_lr = scheduler.get_last_lr()[0]
                print(f"Prior Epoch {epoch+1}/{HYPERPARAMS['PRIOR_EPOCHS']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                run.log({
                    "prior_batch_loss": loss.item(),
                    "prior_epoch": epoch + 1,
                    "prior_batch_idx": batch_idx,
                    "prior_lr": current_lr
                })

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                prior.eval()
                val_loss = 0.0
                iters = 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        if v_batch is None: continue

                        v_images, v_texts = v_batch

                        v_texts = v_texts.to(local_device)
                        v_images = v_images.to(local_device)

                        v_text_emb = text_encoder(v_texts).float()
                        v_target_grid, _ = prior_teacher(v_images.half())
                        v_target_grid = v_target_grid.float()
                        v_prior_grid = prior(v_text_emb)

                        val_loss += PriorLoss(v_prior_grid, v_target_grid).item()

                        iters += 1

                avg_val = val_loss / iters if iters > 0 else 999.9
                run.log({"prior_val_loss": avg_val, "prior_epoch": epoch + 1, "prior_batch_idx": batch_idx})
                prior.module.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"],
                    f"prior_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}"
                )
                print(f"Saved Prior partial epoch {epoch+1} batch {batch_idx} (Val Loss: {avg_val:.4f})")
                prior.train()

        if is_main_process():
            avg_epoch_loss = total_loss / max(1, len(train_loader) - batch_start)
            print(f"Prior Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
            prior.eval()
            val_loss = 0.0
            iters = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    if v_batch is None: continue

                    v_images, v_texts = v_batch

                    v_texts = v_texts.to(local_device)
                    v_images = v_images.to(local_device)

                    v_text_emb = text_encoder(v_texts).float()
                    v_target_grid, _ = prior_teacher(v_images.half())
                    v_target_grid = v_target_grid.float()
                    v_prior_grid = prior(v_text_emb)

                    val_loss += PriorLoss(v_prior_grid, v_target_grid).item()

                    iters += 1

            avg_val = val_loss / iters if iters > 0 else 999.9
            prior.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"],
                f"prior_epoch_{epoch+1}_complete_{avg_val:.4f}")
            print(f"Saved Prior epoch {epoch+1} (Val Loss: {avg_val:.4f})")
    if is_main_process():
        print("Prior training completed.\n")

# ======== SAM Teacher Training ========
def train_SAM_decoder(train_dataloader, val_dataloader, start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    if is_main_process():
        print("\n=== Training SAM Decoder (Teacher Component) ===")
    local_device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}') if "LOCAL_RANK" in os.environ else device

    text_encoder = create_text_encoder().to(local_device)
    prior = create_prior().to(local_device)
    sam_decoder = create_SAM().to(local_device)

    best_clip_text_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    if not best_clip_text_ckpt: raise FileNotFoundError("CLIP text checkpoint not found for SAM Decoder training.")
    if is_main_process():
        print(f"Text encoder weights: {best_clip_text_ckpt}")
    text_encoder.load_weights(best_clip_text_ckpt)

    best_prior_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    if not best_prior_ckpt: raise FileNotFoundError("Prior checkpoint not found for SAM Decoder training.")
    if is_main_process():
        print(f"Prior model weights: {best_prior_ckpt}")
    prior.load_weights(best_prior_ckpt)

    if start_epoch > 0:
        if os.path.exists(start_weights):
            sam_decoder.load_weights(start_weights)
        else:
            start_epoch = 0
            start_batch = 0

    sam_decoder = DDP(sam_decoder, device_ids=[int(os.environ["LOCAL_RANK"])])

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder
            self.prior = prior
            self.sam_decoder = sam_decoder

        def forward_frame(self, frame, text_tokens, memory, t=0):
            with torch.no_grad():
                text_emb  = self.text_encoder(text_tokens)
                prior_emb = self.prior(text_emb).float()
            mask, new_memory = self.sam_decoder.module.forward(frame, prior_emb, memory, t)
            return mask, new_memory
 
        def init_memory(self, B, device):
            return self.sam_decoder.module.init_memory(B, device)

    teacher = TeacherModel().to(local_device)

    for param in teacher.text_encoder.parameters(): param.requires_grad_(False)
    for param in teacher.prior.parameters(): param.requires_grad_(False)
    teacher.text_encoder.half()
    teacher.text_encoder.eval()
    teacher.prior.half()
    teacher.prior.eval()
            
    optimizer_sam_decoder = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["DECODER_LR"], weight_decay=HYPERPARAMS["DECODER_WEIGHT_DECAY"])

    estimated_total_steps = HYPERPARAMS["SAM_DECODER_EPOCHS"] * 2000
    scheduler = create_lr_warmup_cosine_scheduler(
        optimizer_sam_decoder, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )
    if start_epoch > 0 or start_batch > 0:
        skip_steps = start_epoch * 2000 + start_batch
        for _ in range(skip_steps):
            scheduler.step()

    if dist.is_initialized():
        dist.barrier()
        if is_main_process():
            print("[Sync] All ranks ready. Starting SAM decoder training loop.")

    for epoch in range(start_epoch, HYPERPARAMS["SAM_DECODER_EPOCHS"]):
        teacher.sam_decoder.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            if epoch == start_epoch and batch_idx <= start_batch and start_batch > 0:
                print(f"Skipping batch {batch_idx}")
                continue
            if batch is None: continue

            images, true_masks, texts = batch
            current_batch_loss_sum = 0
            num_samples_in_batch = 0
            optimizer_sam_decoder.zero_grad()

            for img, mask, txt in zip(images, true_masks, texts):
                mask = mask.to(local_device).float()
                img = img.to(local_device)
                txt = txt.to(local_device)
                T = img.shape[1]
                memory = teacher.init_memory(img.shape[0], local_device)

                for t in range(T):
                    pred_mask, new_memory = teacher.forward_frame(img[:, t], txt, memory, t)
                    loss = iou_loss(pred_mask, mask[:, t])
                    loss.backward()
                    memory = new_memory.detach()
                
                    current_batch_loss_sum += loss.item()
                num_samples_in_batch += 1
        
            optimizer_sam_decoder.step()
            scheduler.step()

            avg_batch_item_loss = current_batch_loss_sum / max(1, num_samples_in_batch)
            total_loss += avg_batch_item_loss
            batch_count += 1

            if batch_idx % 100 == 0 and is_main_process():
                current_lr = scheduler.get_last_lr()[0]
                print(f"SAM Decoder Epoch {epoch+1}/{HYPERPARAMS['SAM_DECODER_EPOCHS']} | Batch {batch_idx} Avg Item Loss: {avg_batch_item_loss:.4f}")
                run.log({
                    "sam_decoder_batch_avg_item_loss": avg_batch_item_loss,
                    "sam_decoder_epoch": epoch + 1,
                    "sam_decoder_batch_idx": batch_idx,
                    "sam_decoder_lr": current_lr
                })

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                teacher.sam_decoder.eval()
                val_loss = 0.0
                num_val_samples = 0
                with torch.no_grad():
                    for v_batch in val_dataloader:
                        if v_batch is None: continue
                        v_images, v_true_masks, v_texts = v_batch

                        for v_img, v_mask, v_txt in zip(v_images, v_true_masks, v_texts):
                            v_mask = v_mask.to(local_device).float()
                            v_img = v_img.to(local_device)
                            v_txt = v_txt.to(local_device)

                            T = v_img.shape[1]
                            memory = teacher.init_memory(v_img.shape[0], local_device)

                            for t in range(T):
                                v_pred, memory = teacher.forward_frame(v_img[:, t], v_txt, memory, t)
                                val_loss += iou_loss(v_pred, v_mask[:,t]).item()
                            num_val_samples += 1
                        
                avg_val = val_loss / num_val_samples if num_val_samples > 0 else 999.9
                teacher.sam_decoder.module.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"], 
                    f"sam_decoder_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}"
                )
                print(f"Saved SAM Decoder partial epoch {epoch+1} batch {batch_idx} (Val Loss: {avg_val:.4f})")
                teacher.sam_decoder.train()
            
        if is_main_process():
            avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"SAM Decoder Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
            teacher.sam_decoder.eval()
            val_loss = 0.0
            num_val_samples = 0
            with torch.no_grad():
                for v_batch in val_dataloader:
                    if v_batch is None: continue
                    v_images, v_true_masks, v_texts = v_batch

                    for v_img, v_mask, v_txt in zip(v_images, v_true_masks, v_texts):
                        v_mask = v_mask.to(local_device).float()
                        v_img = v_img.to(local_device)
                        v_txt = v_txt.to(local_device)

                        T = v_img.shape[1]
                        memory = teacher.init_memory(v_img.shape[0], local_device)

                        for t in range(T):
                            v_pred, memory = teacher.forward_frame(v_img[:, t], v_txt, memory, t)
                            val_loss += iou_loss(v_pred, v_mask[:,t]).item()

                        num_val_samples += 1
                        
            avg_val = val_loss / num_val_samples if num_val_samples > 0 else 999.9
            teacher.sam_decoder.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"sam_decoder_epoch_{epoch+1}_complete_{avg_val:.4f}"
            )
            print(f"Saved SAM Decoder epoch {epoch+1} (Val Loss: {avg_val:.4f})")
    if is_main_process():
        print("SAM Decoder training completed.\n")

# ======== SAM Student Training ========
def train_student(train_dataloader, val_dataloader, teacher_start_weights, student_start_weights,
                   ablation_student_start_weights, run:wandb, start_epoch = 0, start_batch = 0):
    if is_main_process():
        print("\n=== Training Student (with Teacher Fine-tuning) ===")
    local_device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}') if "LOCAL_RANK" in os.environ else device

    text_encoder = create_text_encoder().to(local_device)
    prior = create_prior().to(local_device)
    sam_decoder = create_SAM().to(local_device)

    best_clip_text_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    best_prior_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    best_sam_decoder_ckpt, _, _ = get_best_weights_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")

    if not best_clip_text_ckpt: raise FileNotFoundError("CLIP text ckpt not found for Student training.")
    if not best_prior_ckpt: raise FileNotFoundError("Prior ckpt not found for Student training.")
    if not best_sam_decoder_ckpt: raise FileNotFoundError("SAM Decoder ckpt not found for Student training.")

    if is_main_process():
        print(f"Text encoder weights: {best_clip_text_ckpt}")
    text_encoder.load_weights(best_clip_text_ckpt)
    if is_main_process():
        print(f"Prior model weights: {best_prior_ckpt}")
    prior.load_weights(best_prior_ckpt)
    if is_main_process():
        print(f"SAM decoder weights: {best_sam_decoder_ckpt}")
    sam_decoder.load_weights(best_sam_decoder_ckpt)

    for param in text_encoder.parameters(): param.requires_grad_(False)
    text_encoder.half()
    text_encoder.eval()
    for param in prior.parameters(): param.requires_grad_(False)
    prior.half()
    prior.eval()

    sam_decoder = DDP(sam_decoder, device_ids=[int(os.environ["LOCAL_RANK"])])

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder
            self.prior = prior
            self.sam_decoder = sam_decoder

        def forward_frame(self, frame, text_tokens, memory, t=0):
            with torch.no_grad():
                text_emb  = self.text_encoder(text_tokens)
                prior_emb = self.prior(text_emb).float()
            mask, new_memory = self.sam_decoder.module.forward(frame, prior_emb, memory, t)
            return mask, new_memory

        def init_memory(self, B, device):
            return self.sam_decoder.module.init_memory(B, device)

    teacher = TeacherModel().to(local_device)
    student = create_Student().to(local_device)
    # Ablation baseline: a completely separate student instance trained on the exact
    # same batches/frames as `student`, but with no teacher influence at all (no
    # distillation term, ground-truth supervision only) — for comparing against the
    # distilled student's performance.
    ablation_student = create_Student().to(local_device)

    if start_epoch > 0:
        if os.path.exists(student_start_weights):
            student.load_weights(student_start_weights)
            if os.path.exists(teacher_start_weights):
                teacher.sam_decoder.load_weights(teacher_start_weights)
            if os.path.exists(ablation_student_start_weights):
                ablation_student.load_weights(ablation_student_start_weights)
        else:
            start_epoch = 0
            start_batch = 0

    student = DDP(student, device_ids=[int(os.environ["LOCAL_RANK"])])
    ablation_student = DDP(ablation_student, device_ids=[int(os.environ["LOCAL_RANK"])])

    optimizer_teacher_finetune = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["TEACHER_LR"], weight_decay=HYPERPARAMS["TEACHER_WEIGHT_DECAY"])
    optimizer_student = torch.optim.Adam(student.parameters(), lr=HYPERPARAMS["STUDENT_LR"], weight_decay=HYPERPARAMS["STUDENT_WEIGHT_DECAY"])
    # Same LR/weight decay/optimizer type as the distilled student — the only
    # intended difference between the two is the loss they're trained on.
    optimizer_ablation_student = torch.optim.Adam(ablation_student.parameters(), lr=HYPERPARAMS["STUDENT_LR"], weight_decay=HYPERPARAMS["STUDENT_WEIGHT_DECAY"])

    estimated_total_steps = HYPERPARAMS["TEACHER_STUDENT_EPOCHS"] * 2000
    scheduler_teacher = create_lr_warmup_cosine_scheduler(
        optimizer_teacher_finetune, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )
    scheduler_student = create_lr_warmup_cosine_scheduler(
        optimizer_student, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )
    scheduler_ablation_student = create_lr_warmup_cosine_scheduler(
        optimizer_ablation_student, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )
    if start_epoch > 0 or start_batch > 0:
        skip_steps = start_epoch * 2000 + start_batch
        for _ in range(skip_steps):
            scheduler_teacher.step()
            scheduler_student.step()
            scheduler_ablation_student.step()

    if dist.is_initialized():
        dist.barrier()
        if is_main_process():
            print("[Sync] All ranks ready. Starting joint training loop.")

    for epoch in range(start_epoch, HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]):
        teacher.sam_decoder.train()
        student.train()
        ablation_student.train()

        total_teacher_loss = 0.0
        total_student_loss = 0.0
        total_ablation_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_dataloader):
            if epoch == start_epoch and batch_idx <= start_batch and start_batch > 0:
                print(f"Skipping batch {batch_idx}")
                continue
            if batch is None: continue

            images, true_masks, texts = batch

            current_batch_teacher_loss_sum = 0
            current_batch_student_loss_sum = 0
            current_batch_ablation_loss_sum = 0
            num_samples_in_batch = 0

            optimizer_teacher_finetune.zero_grad()
            optimizer_student.zero_grad()
            optimizer_ablation_student.zero_grad()

            for img, mask, txt in zip(images, true_masks, texts):
                mask = mask.to(local_device).float()
                img = img.to(local_device)
                txt = txt.to(local_device)

                T = img.shape[1]
                mem_t = teacher.init_memory(img.shape[0], local_device)
                mem_s = student.module.init_memory(img.shape[0], local_device)
                mem_a = ablation_student.module.init_memory(img.shape[0], local_device)

                for t in range(T):
                    teacher_out, mem_t_new = teacher.forward_frame(img[:, t], txt, mem_t, t)
                    student_out, mem_s_new = student.forward(img[:, t], txt, mem_s, t)
                    ablation_out, mem_a_new = ablation_student.forward(img[:, t], txt, mem_a, t)

                    with torch.no_grad():
                        teacher_out_for_student = teacher_out.detach()

                    teacher_loss = iou_loss(teacher_out, mask[:, t])
                    student_loss = student.module.compute_distill_loss(student_out, teacher_out_for_student, mask[:, t])
                    # No teacher influence whatsoever: ground-truth supervision only.
                    ablation_loss = iou_loss(ablation_out, mask[:, t])

                    teacher_loss.backward(retain_graph=True)
                    student_loss.backward()
                    ablation_loss.backward()

                    current_batch_teacher_loss_sum += teacher_loss.item()
                    current_batch_student_loss_sum += student_loss.item()
                    current_batch_ablation_loss_sum += ablation_loss.item()

                    mem_t = mem_t_new.detach()
                    mem_s = mem_s_new.detach()
                    mem_a = mem_a_new.detach()
                num_samples_in_batch += 1

            optimizer_teacher_finetune.step()
            optimizer_student.step()
            optimizer_ablation_student.step()
            scheduler_teacher.step()
            scheduler_student.step()
            scheduler_ablation_student.step()

            batch_count += 1
            avg_batch_teacher_loss = current_batch_teacher_loss_sum / max(1, num_samples_in_batch)
            avg_batch_student_loss = current_batch_student_loss_sum / max(1, num_samples_in_batch)
            avg_batch_ablation_loss = current_batch_ablation_loss_sum / max(1, num_samples_in_batch)
            total_teacher_loss += avg_batch_teacher_loss
            total_student_loss += avg_batch_student_loss
            total_ablation_loss += avg_batch_ablation_loss

            if batch_idx % 100 == 0 and is_main_process():
                t_lr = scheduler_teacher.get_last_lr()[0]
                s_lr = scheduler_student.get_last_lr()[0]
                a_lr = scheduler_ablation_student.get_last_lr()[0]
                print(f"Student Epoch {epoch+1}/{HYPERPARAMS['TEACHER_STUDENT_EPOCHS']} | Batch {batch_idx} | Teacher Loss: {avg_batch_teacher_loss:.4f}, Student Loss: {avg_batch_student_loss:.4f}, Ablation Loss: {avg_batch_ablation_loss:.4f}")
                run.log({
                    "student_phase_batch_teacher_loss": avg_batch_teacher_loss,
                    "student_phase_batch_student_loss": avg_batch_student_loss,
                    "student_phase_batch_ablation_loss": avg_batch_ablation_loss,
                    "student_phase_epoch": epoch + 1,
                    "student_phase_batch_idx": batch_idx,
                    "student_phase_teacher_lr": t_lr,
                    "student_phase_student_lr": s_lr,
                    "student_phase_ablation_lr": a_lr
                })

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                teacher.sam_decoder.eval()
                student.eval()
                ablation_student.eval()
                val_t_loss, val_s_loss, val_a_loss = 0.0, 0.0, 0.0
                num_val_samples = 0
                with torch.no_grad():
                    for v_batch in val_dataloader:
                        if v_batch is None: continue
                        v_images, v_true_masks, v_texts = v_batch
                        for v_img, v_mask, v_txt in zip(v_images, v_true_masks, v_texts):
                            v_mask = v_mask.to(local_device).float()
                            v_img = v_img.to(local_device)
                            v_txt = v_txt.to(local_device)

                            T = v_img.shape[1]
                            mem_t = teacher.init_memory(v_img.shape[0], local_device)
                            mem_s = student.module.init_memory(v_img.shape[0], local_device)
                            mem_a = ablation_student.module.init_memory(v_img.shape[0], local_device)

                            for t in range(T):
                                teacher_out, mem_t = teacher.forward_frame(v_img[:, t], v_txt, mem_t, t)
                                student_out, mem_s = student.forward(v_img[:, t], v_txt, mem_s, t)
                                ablation_out, mem_a = ablation_student.forward(v_img[:, t], v_txt, mem_a, t)

                                val_t_loss += iou_loss(teacher_out, v_mask[:, t]).item()
                                val_s_loss += student.module.compute_distill_loss(student_out, teacher_out, v_mask[:, t]).item()
                                val_a_loss += iou_loss(ablation_out, v_mask[:, t]).item()

                            num_val_samples += 1

                avg_t_val = val_t_loss / num_val_samples if num_val_samples > 0 else 999.9
                avg_s_val = val_s_loss / num_val_samples if num_val_samples > 0 else 999.9
                avg_a_val = val_a_loss / num_val_samples if num_val_samples > 0 else 999.9

                teacher.sam_decoder.module.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"],
                    f"student_phase_teacher_epoch_{epoch+1}_batch_{batch_idx}_{avg_t_val:.4f}")
                student.module.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"],
                    f"student_phase_student_epoch_{epoch+1}_batch_{batch_idx}_{avg_s_val:.4f}")
                ablation_student.module.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"],
                    f"student_phase_student_ablation_epoch_{epoch+1}_batch_{batch_idx}_{avg_a_val:.4f}")
                print(f"Saved Joint Phase partial epoch {epoch+1} batch {batch_idx}(Teacher Val Loss: {avg_t_val:.4f}, Student Val Loss: {avg_s_val:.4f}, Ablation Val Loss: {avg_a_val:.4f})")

                teacher.sam_decoder.train()
                student.train()
                ablation_student.train()

        if is_main_process():
            avg_epoch_teacher_loss = total_teacher_loss / batch_count if batch_count > 0 else 0
            avg_epoch_student_loss = total_student_loss / batch_count if batch_count > 0 else 0
            avg_epoch_ablation_loss = total_ablation_loss / batch_count if batch_count > 0 else 0
            print(f"Student Epoch {epoch+1} Avg Losses - Teacher: {avg_epoch_teacher_loss:.4f}, Student: {avg_epoch_student_loss:.4f}, Ablation: {avg_epoch_ablation_loss:.4f}")
            teacher.sam_decoder.eval()
            student.eval()
            ablation_student.eval()
            val_t_loss, val_s_loss, val_a_loss = 0.0, 0.0, 0.0
            num_val_samples = 0
            with torch.no_grad():
                for v_batch in val_dataloader:
                    if v_batch is None: continue
                    v_images, v_true_masks, v_texts = v_batch
                    for v_img, v_mask, v_txt in zip(v_images, v_true_masks, v_texts):
                        v_mask = v_mask.to(local_device).float()
                        v_img = v_img.to(local_device)
                        v_txt = v_txt.to(local_device)

                        T = v_img.shape[1]
                        mem_t = teacher.init_memory(v_img.shape[0], local_device)
                        mem_s = student.module.init_memory(v_img.shape[0], local_device)
                        mem_a = ablation_student.module.init_memory(v_img.shape[0], local_device)

                        for t in range(T):
                            teacher_out, mem_t = teacher.forward_frame(v_img[:, t], v_txt, mem_t, t)
                            student_out, mem_s = student.forward(v_img[:, t], v_txt, mem_s, t)
                            ablation_out, mem_a = ablation_student.forward(v_img[:, t], v_txt, mem_a, t)

                            val_t_loss += iou_loss(teacher_out, v_mask[:, t]).item()
                            val_s_loss += student.module.compute_distill_loss(student_out, teacher_out, v_mask[:, t]).item()
                            val_a_loss += iou_loss(ablation_out, v_mask[:, t]).item()

                        num_val_samples += 1

            avg_t_val = val_t_loss / num_val_samples if num_val_samples > 0 else 999.9
            avg_s_val = val_s_loss / num_val_samples if num_val_samples > 0 else 999.9
            avg_a_val = val_a_loss / num_val_samples if num_val_samples > 0 else 999.9
            teacher.sam_decoder.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"],
                f"student_phase_teacher_epoch_{epoch+1}_complete_{avg_t_val:.4f}")
            student.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"],
                f"student_phase_student_epoch_{epoch+1}_complete_{avg_s_val:.4f}")
            ablation_student.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"],
                f"student_phase_student_ablation_epoch_{epoch+1}_complete_{avg_a_val:.4f}")
            print(f"Saved Joint Phase partial epoch {epoch+1} batch {batch_idx}(Teacher Val Loss: {avg_t_val:.4f}, Student Val Loss: {avg_s_val:.4f}, Ablation Val Loss: {avg_a_val:.4f})")
    if is_main_process():
        print("Student training completed.\n")

def get_ddp_laion_datasets(hf_token):
    # One process per NODE (not per GPU) downloads — every node ends up with its own
    # full copy, matching the old sharded implementation's per-node concurrency scale
    # (LAIONDataset caps concurrent fetches itself), instead of every GPU process on
    # a node redundantly repeating the same download.
    is_local_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0
    kwargs = dict(
        hf_token=hf_token,
        total_samples=HYPERPARAMS["LAION_TOTAL_SAMPLES"],
        val_size=HYPERPARAMS["LAION_VAL_SIZE"],
    )
    if is_local_rank0:
        train_dataset = get_laion_dataset(split="train", **kwargs)
        val_dataset = get_laion_dataset(split="val", **kwargs)
    if dist.is_initialized():
        dist.barrier()
    if not is_local_rank0:
        train_dataset = get_laion_dataset(split="train", **kwargs)
        val_dataset = get_laion_dataset(split="val", **kwargs)
    return train_dataset, val_dataset

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
    clip_text_start_weights, clip_text_start_epoch, clip_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    clip_img_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_image")
    clip_wrapper_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_wrapper")
    prior_start_weights, prior_start_epoch, prior_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    sam_decoder_start_weights, sam_decoder_start_epoch, sam_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")
    teacher_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_teacher")
    student_start_weights, student_start_epoch, student_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_student")
    ablation_student_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_student_ablation")

    if clip_text_start_epoch < HYPERPARAMS["CLIP_EPOCHS"] or prior_start_epoch < HYPERPARAMS["PRIOR_EPOCHS"]:
        laion_train_dataset, laion_val_dataset = get_ddp_laion_datasets(hf_token)
        if is_main_process():
            print(f"[LAION] Dataset ready: {len(laion_train_dataset)} train / "
                  f"{len(laion_val_dataset)} val samples")

        if clip_text_start_epoch < HYPERPARAMS["CLIP_EPOCHS"]:
            print("Starting CLIP Training Phase")
            train_clip(train_dataset=laion_train_dataset,
                       val_dataset=laion_val_dataset,
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
            train_prior(train_dataset=laion_train_dataset,
                        val_dataset=laion_val_dataset,
                        start_weights=prior_start_weights,
                        start_epoch=prior_start_epoch,
                        start_batch=prior_start_batch,
                        run=run)
        else:
            print("Prior training already completed.")
    else:
        print("CLIP and Prior already trained")

    num_workers = 0 if device == 'mps' else 4

    if sam_decoder_start_epoch < HYPERPARAMS["SAM_DECODER_EPOCHS"] or student_start_epoch < HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]:
        print("Initializing SA-1B/SA-V segmentation dataset...")
        seg_train_dataset = get_segmentation_dataset(split="train", val_size=HYPERPARAMS["SEG_VAL_SIZE"])
        seg_val_dataset = get_segmentation_dataset(split="val", val_size=HYPERPARAMS["SEG_VAL_SIZE"])
        if is_main_process():
            print(f"[Segmentation] {len(seg_train_dataset)} train / {len(seg_val_dataset)} val samples")

        use_dist = dist.is_initialized()

        if sam_decoder_start_epoch < HYPERPARAMS["SAM_DECODER_EPOCHS"]:
            print("Starting SAM Decoder Training Phase")
            train_sampler = DistributedSampler(seg_train_dataset, shuffle=True) if use_dist else None
            train_dataloader = DataLoader(seg_train_dataset,
                                      batch_size=HYPERPARAMS["DECODER_BATCH_SIZE"],
                                      shuffle=(train_sampler is None),
                                      sampler=train_sampler,
                                      num_workers=num_workers,
                                      collate_fn=SAM_adaptive_collate,
                                      pin_memory=True)

            val_sampler = DistributedSampler(seg_val_dataset, shuffle=False) if use_dist else None
            val_dataloader = DataLoader(seg_val_dataset,
                                    batch_size=HYPERPARAMS["DECODER_BATCH_SIZE"],
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=SAM_adaptive_collate,
                                    pin_memory=True,
                                    sampler=val_sampler)

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
            train_sampler = DistributedSampler(seg_train_dataset, shuffle=True) if use_dist else None
            train_dataloader = DataLoader(seg_train_dataset,
                                      batch_size=HYPERPARAMS["STUDENT_BATCH_SIZE"],
                                      shuffle=(train_sampler is None),
                                      sampler=train_sampler,
                                      num_workers=num_workers,
                                      collate_fn=SAM_adaptive_collate,
                                      pin_memory=True)

            val_sampler = DistributedSampler(seg_val_dataset, shuffle=False) if use_dist else None
            val_dataloader = DataLoader(seg_val_dataset,
                                    batch_size=HYPERPARAMS["STUDENT_BATCH_SIZE"],
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=SAM_adaptive_collate,
                                    pin_memory=True,
                                    sampler=val_sampler)
            train_student(train_dataloader,
                          val_dataloader,
                          teacher_start_weights=teacher_start_weights,
                          student_start_weights=student_start_weights,
                          ablation_student_start_weights=ablation_student_start_weights,
                          start_epoch=student_start_epoch,
                          start_batch=student_start_batch,
                          run=run)
        else:
            print("Student training already completed.")
    
    print("All training phases completed")
    run.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

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