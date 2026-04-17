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
from torch.utils.data import DataLoader, ChainDataset, DistributedSampler, IterableDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss
from models.prior_model import create_prior, PriorLoss, TeacherCLIP
from models.SAM_model import iou_loss, create_SAM
from models.distill_model import create_Student
from data.custom400m import get_laion_streaming_dataset, get_laion_test_dataset, adaptive_collate, ChunkedLAIONManager
from data.segmentation import SAM_adaptive_collate, SA1BDataset, SAVDataset, StaticSA1BDataset, StaticSAVDataset
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
    "WARMUP_STEPS": 1000,
    "MIN_LR_RATIO": 0.01,
    "LAION_VAL_SIZE": 10000,
    "LAION_BATCH_SIZE": 128,
    "LAION_CACHE_SAMPLES": 200000,
    "LAION_CHUNK_SIZE": 10000,
    "SA_VAL_TAR_COUNT": 1,  
    "SA_VAL_SAMPLE_COUNT": 5000,
    "SAV_VAL_TAR_COUNT": 1,
    "SAV_VAL_SAMPLE_COUNT": 5000,
    "SAM_BATCH_SIZE": 512,
    "EST_SAMPLES_PER_TAR": 10000,
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
def train_clip(chunk_manager:ChunkedLAIONManager, text_start_weights, img_start_weights, wrapper_start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    print("\n=== Training CLIP ===")
    local_device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}') if "LOCAL_RANK" in os.environ else device

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

    clip_model = DDP(clip_model, device_ids=[int(os.environ["LOCAL_RANK"])])

    num_chunks = chunk_manager.num_chunks if not isinstance(chunk_manager.num_chunks, float) else 20
    batches_per_chunk = int(chunk_manager.chunk_size * (1 - chunk_manager.val_ratio)) // HYPERPARAMS["LAION_BATCH_SIZE"]
    estimated_total_steps = HYPERPARAMS["CLIP_EPOCHS"] * num_chunks * batches_per_chunk
    scheduler = create_lr_warmup_cosine_scheduler(
        optimizer, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )
 
    # Compute which chunk to resume from based on saved global_batch_idx
    start_chunk = 0
    start_batch_in_chunk = 0
    if start_epoch > 0 or start_batch > 0:
        start_chunk = start_batch // batches_per_chunk if batches_per_chunk > 0 else 0
        start_batch_in_chunk = start_batch % batches_per_chunk if batches_per_chunk > 0 else 0
        skip_steps = start_epoch * num_chunks * batches_per_chunk + start_batch
        for _ in range(skip_steps):
            scheduler.step()
        if is_main_process():
            print(f"[Resume] Resuming from epoch {start_epoch}, chunk {start_chunk}, batch {start_batch_in_chunk}")
    
    # Skip the stream forward to the right chunk position
    chunk_manager.skip_chunks(start_chunk)
 
    # Download static val set once (persists across all chunks and epochs)
    val_loader = chunk_manager.prepare_val_loader(
        HYPERPARAMS["LAION_BATCH_SIZE"], 
        num_val_samples=HYPERPARAMS["LAION_VAL_SIZE"]
    )

    if dist.is_initialized():
        dist.barrier()
        if is_main_process():
            print("[Sync] All ranks ready. Starting CLIP training loop.")

    for epoch in range(start_epoch, HYPERPARAMS["CLIP_EPOCHS"]):
        clip_model.train()
        total_loss = 0.0
        global_batch_idx = start_chunk * batches_per_chunk if epoch == start_epoch else 0
        epoch_start_chunk = start_chunk if epoch == start_epoch else 0
        
        for chunk_idx in range(epoch_start_chunk, num_chunks):
            chunk_manager.prepare_chunk(chunk_idx)
            train_loader = chunk_manager.get_loaders(chunk_idx, HYPERPARAMS["LAION_BATCH_SIZE"])
            
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            if is_main_process():
                print(f"CLIP Epoch {epoch+1} | Chunk {chunk_idx+1}/{num_chunks}")
        
            for batch_idx, batch in enumerate(train_loader):
                if epoch == start_epoch and chunk_idx == epoch_start_chunk and batch_idx < start_batch_in_chunk and start_batch > 0:
                    global_batch_idx += 1
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
            
                if global_batch_idx % 100 == 0 and is_main_process():
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"CLIP Epoch {epoch+1}/{HYPERPARAMS['CLIP_EPOCHS']} | Chunk {chunk_idx+1}/{num_chunks} | Batch {global_batch_idx} | Loss: {loss.item():.4f}")
                    run.log({
                        "clip_batch_loss": loss.item(), 
                        "clip_epoch": epoch + 1,
                        "clip_batch_idx": global_batch_idx,
                        "clip_lr": current_lr
                    })

                if global_batch_idx > 0 and global_batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                    clip_model.eval()
                    val_loss = 0.0
                    iters = 0
                    with torch.no_grad():
                        for v_batch in val_loader:
                            if v_batch is None: continue
                            v_images, v_texts = v_batch

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
                        
                    avg_val = val_loss / iters if iters > 0 else 999.9
                    run.log({"clip_val_loss": avg_val, "clip_epoch": epoch + 1, "clip_batch_idx": global_batch_idx})
                    clip_model.module.store_weights(
                        HYPERPARAMS['CHECKPOINT_DIR'], 
                        f"clip_text_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}", 
                        f"clip_image_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}", 
                        f"clip_wrapper_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}"
                    )
                    print(f"Saved CLIP partial epoch {epoch+1} batch {batch_idx} (Val Loss: {avg_val:.4f})")
                    clip_model.train()
                global_batch_idx += 1
            
            # Done with this chunk — delete it, next chunk already prefetching
            chunk_manager.cleanup(chunk_idx)

        if is_main_process():
            avg_epoch_loss = total_loss / max(1, global_batch_idx)
            print(f"CLIP Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}")
            clip_model.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"],
                f"clip_text_epoch_{epoch+1}_complete",
                f"clip_image_epoch_{epoch+1}_complete",
                f"clip_wrapper_epoch_{epoch+1}_complete")

    chunk_manager.shutdown()
    print("CLIP training completed.")

# ======== Prior Training ========
def train_prior(chunk_manager:ChunkedLAIONManager, start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    print("\n=== Training Prior ===")
    local_device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}') if "LOCAL_RANK" in os.environ else device

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
    
    prior = DDP(prior_model, device_ids=[int(os.environ["LOCAL_RANK"])])

    num_chunks = chunk_manager.num_chunks if not isinstance(chunk_manager.num_chunks, float) else 20
    batches_per_chunk = int(chunk_manager.chunk_size * (1 - chunk_manager.val_ratio)) // HYPERPARAMS["LAION_BATCH_SIZE"]
    estimated_total_steps = HYPERPARAMS["PRIOR_EPOCHS"] * num_chunks * batches_per_chunk
    scheduler = create_lr_warmup_cosine_scheduler(
        optimizer, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )
 
    start_chunk = 0
    start_batch_in_chunk = 0
    if start_epoch > 0 or start_batch > 0:
        start_chunk = start_batch // batches_per_chunk if batches_per_chunk > 0 else 0
        start_batch_in_chunk = start_batch % batches_per_chunk if batches_per_chunk > 0 else 0
        skip_steps = start_epoch * num_chunks * batches_per_chunk + start_batch
        for _ in range(skip_steps):
            scheduler.step()
        if is_main_process():
            print(f"[Resume] Resuming from epoch {start_epoch}, chunk {start_chunk}, batch {start_batch_in_chunk}")
    
    chunk_manager.skip_chunks(start_chunk)
 
    # Download static val set once (persists across all chunks and epochs)
    val_loader = chunk_manager.prepare_val_loader(
        HYPERPARAMS["LAION_BATCH_SIZE"],
        num_val_samples=HYPERPARAMS["LAION_VAL_SIZE"]
    )
 
    if dist.is_initialized():
        dist.barrier()
        if is_main_process():
            print("[Sync] All ranks ready. Starting Prior training loop.")

    for epoch in range(start_epoch, HYPERPARAMS["PRIOR_EPOCHS"]):
        prior.train()
        total_loss = 0.0
        global_batch_idx = start_chunk * batches_per_chunk if epoch == start_epoch else 0
        epoch_start_chunk = start_chunk if epoch == start_epoch else 0
        
        for chunk_idx in range(epoch_start_chunk, num_chunks):
            chunk_manager.prepare_chunk(chunk_idx)
            train_loader = chunk_manager.get_loaders(chunk_idx, HYPERPARAMS["LAION_BATCH_SIZE"])
            
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            if is_main_process():
                print(f"Prior Epoch {epoch+1} | Chunk {chunk_idx+1}/{num_chunks}")
            
            for batch_idx, batch in enumerate(train_loader):
                if epoch == start_epoch and chunk_idx == epoch_start_chunk and batch_idx < start_batch_in_chunk and start_batch > 0:
                    global_batch_idx += 1
                    continue
                if batch is None: continue
            
                images, texts = batch
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
                scheduler.step()
                total_loss += loss.item()
            
                if global_batch_idx % 100 == 0 and is_main_process():
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"CLIP Epoch {epoch+1}/{HYPERPARAMS['CLIP_EPOCHS']} | Chunk {chunk_idx+1}/{num_chunks} | Batch {global_batch_idx} | Loss: {loss.item():.4f}")
                    run.log({
                        "prior_batch_loss": loss.item(), 
                        "prior_epoch": epoch + 1,
                        "prior_batch_idx": global_batch_idx,
                        "prior_lr": current_lr
                    })

                if global_batch_idx > 0 and global_batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                    prior.eval()
                    val_loss = 0.0
                    iters = 0
                    with torch.no_grad():
                        for v_batch in val_loader:
                            if v_batch is None: continue

                            v_images, v_texts = v_batch

                            v_texts = v_texts.to(local_device)
                            v_images = v_images.to(local_device)

                            v_text_emb = text_encoder(v_texts)
                            v_target_grid = prior_teacher(v_images)
                            v_prior_grid = prior(v_text_emb)

                            val_loss += PriorLoss(v_prior_grid, v_target_grid).item()

                            iters += 1
                
                    avg_val = val_loss / iters if iters > 0 else 999.9
                    run.log({"prior_val_loss": avg_val, "prior_epoch": epoch + 1, "prior_batch_idx": global_batch_idx})
                    prior.module.store_weights(
                        HYPERPARAMS["CHECKPOINT_DIR"], 
                        f"prior_epoch_{epoch+1}_batch_{batch_idx}_{avg_val:.4f}"
                    )
                    print(f"Saved Prior partial epoch {epoch+1} batch {batch_idx} (Val Loss: {avg_val:.4f})")
                    prior.train()
                global_batch_idx += 1
            chunk_manager.cleanup(chunk_idx)

        if is_main_process():
            avg_epoch_loss = total_loss / max(1, global_batch_idx)
            print(f"Prior Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.4f}")
            prior.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"prior_epoch_{epoch+1}_complete")
            
    chunk_manager.shutdown()   
    print("Prior training completed.\n")

# ======== SAM Teacher Training ========
def train_SAM_decoder(train_dataloader, val_dataloader, start_weights, run: wandb, start_epoch = 0, start_batch = 0):
    print("\n=== Training SAM Decoder (Teacher Component) ===")
    local_device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}') if "LOCAL_RANK" in os.environ else device

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

    sam_decoder = DDP(sam_decoder, device_ids=[int(os.environ["LOCAL_RANK"])])

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

                pred_mask = teacher.forward(img, txt)
                loss = iou_loss(pred_mask, mask)
                loss.backward()
                
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

                            v_pred = teacher.forward(v_img, v_txt)

                            val_loss += iou_loss(v_pred, v_mask).item()
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

            teacher.sam_decoder.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"sam_decoder_epoch_{epoch+1}_complete"
            )
            print(f"Epoch {epoch+1} fully complete. Saved complete flag checkpoints.")

    print("SAM Decoder training completed.\n")

# ======== SAM Student Training ========
def train_student(train_dataloader, val_dataloader, teacher_start_weights, student_start_weights, run:wandb, start_epoch = 0, start_batch = 0):
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

    sam_decoder = DDP(sam_decoder, device_ids=[int(os.environ["LOCAL_RANK"])])

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
    
    student = DDP(student, device_ids=[int(os.environ["LOCAL_RANK"])])

    optimizer_teacher_finetune = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=HYPERPARAMS["TEACHER_LR"])
    optimizer_student = torch.optim.Adam(student.parameters(), lr=HYPERPARAMS["STUDENT_LR"])

    estimated_total_steps = HYPERPARAMS["TEACHER_STUDENT_EPOCHS"] * 2000
    scheduler_teacher = create_lr_warmup_cosine_scheduler(
        optimizer_teacher_finetune, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )
    scheduler_student = create_lr_warmup_cosine_scheduler(
        optimizer_student, HYPERPARAMS["WARMUP_STEPS"], estimated_total_steps, HYPERPARAMS["MIN_LR_RATIO"]
    )
    if start_epoch > 0 or start_batch > 0:
        skip_steps = start_epoch * 2000 + start_batch
        for _ in range(skip_steps):
            scheduler_teacher.step()
            scheduler_student.step()

    if dist.is_initialized():
        dist.barrier()
        if is_main_process():
            print("[Sync] All ranks ready. Starting joint training loop.")

    print("[Joint Training] Starting joint training")
    for epoch in range(start_epoch, HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]):
        teacher.sam_decoder.train()
        student.train()

        total_teacher_loss = 0.0
        total_student_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_dataloader):
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
            scheduler_teacher.step()
            scheduler_student.step()
            
            batch_count += 1
            avg_batch_teacher_loss = current_batch_teacher_loss_sum / max(1, num_samples_in_batch)
            avg_batch_student_loss = current_batch_student_loss_sum / max(1, num_samples_in_batch)
            total_teacher_loss += avg_batch_teacher_loss
            total_student_loss += avg_batch_student_loss
            
            if batch_idx % 100 == 0 and is_main_process():
                t_lr = scheduler_teacher.get_last_lr()[0]
                s_lr = scheduler_student.get_last_lr()[0]
                print(f"Student Epoch {epoch+1}/{HYPERPARAMS['TEACHER_STUDENT_EPOCHS']} | Batch {batch_idx} | Teacher Loss: {avg_batch_teacher_loss:.4f}, Student Loss: {avg_batch_student_loss:.4f}")
                run.log({
                    "student_phase_batch_teacher_loss": avg_batch_teacher_loss,
                    "student_phase_batch_student_loss": avg_batch_student_loss,
                    "student_phase_epoch": epoch + 1,
                    "student_phase_batch_idx": batch_idx,
                    "student_phase_teacher_lr": t_lr,
                    "student_phase_student_lr": s_lr
                })

            if batch_idx > 0 and batch_idx % HYPERPARAMS["SAVE_FREQ"] == 0 and is_main_process():
                teacher.sam_decoder.eval()
                student.eval()
                val_t_loss, val_s_loss = 0.0, 0.0
                num_val_samples = 0
                with torch.no_grad():
                    for v_batch in val_dataloader:
                        if v_batch is None: continue
                        v_images, v_true_masks, v_texts = v_batch
                        for v_img, v_mask, v_txt in zip(v_images, v_true_masks, v_texts):
                            v_mask = v_mask.to(local_device).float()
                            v_img = v_img.to(local_device)
                            v_txt = v_txt.to(local_device)

                            t_out = teacher(v_img, v_txt)
                            s_out = student(v_img, v_txt)
                            
                            val_t_loss += iou_loss(t_out, v_mask).item()
                            val_s_loss += student.module.compute_distill_loss(s_out, t_out, v_mask).item()

                            num_val_samples += 1
                        
                avg_t_val = val_t_loss / num_val_samples if num_val_samples > 0 else 999.9
                avg_s_val = val_s_loss / num_val_samples if num_val_samples > 0 else 999.9
                
                teacher.sam_decoder.module.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"], 
                    f"student_phase_teacher_epoch_{epoch+1}_batch_{batch_idx}_{avg_t_val:.4f}")
                student.module.store_weights(
                    HYPERPARAMS["CHECKPOINT_DIR"], 
                    f"student_phase_student_epoch_{epoch+1}_batch_{batch_idx}_{avg_s_val:.4f}")
                print(f"Saved Joint Phase partial epoch {epoch+1} batch {batch_idx}")
                
                teacher.sam_decoder.train()
                student.train()

        if is_main_process():
            avg_epoch_teacher_loss = total_teacher_loss / batch_count if batch_count > 0 else 0
            avg_epoch_student_loss = total_student_loss / batch_count if batch_count > 0 else 0
            print(f"Student Epoch {epoch+1} Avg Losses - Teacher: {avg_epoch_teacher_loss:.4f}, Student: {avg_epoch_student_loss:.4f}")

            teacher.sam_decoder.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"student_phase_teacher_epoch_{epoch+1}_complete")
            student.module.store_weights(
                HYPERPARAMS["CHECKPOINT_DIR"], 
                f"student_phase_student_epoch_{epoch+1}_complete")
            print(f"Epoch {epoch+1} fully complete. Saved complete flag checkpoints.")
    print("Student training completed.\n")

def get_dataset(dataset_cls, file_list, split_name, val_tar_count, val_sample_count=None, skip_tars=0):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    kwargs = dict(
        file_list=file_list,
        cache_dir=cache_dir,
        device=device,
        split=split_name,
        val_tar_count=val_tar_count,
        val_sample_count=val_sample_count
    )
    if skip_tars > 0 and issubclass(dataset_cls, IterableDataset):
        kwargs['skip_tars'] = skip_tars
    
    dataset = dataset_cls(**kwargs)
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
    clip_text_start_weights, clip_text_start_epoch, clip_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_text")
    clip_img_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_image")
    clip_wrapper_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "clip_wrapper")
    prior_start_weights, prior_start_epoch, prior_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "prior")
    sam_decoder_start_weights, sam_decoder_start_epoch, sam_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "sam_decoder")
    teacher_start_weights, _, _ = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_teacher")
    student_start_weights, student_start_epoch, student_start_batch = get_latest_epoch_checkpoint(HYPERPARAMS['CHECKPOINT_DIR'], "student_phase_student")

    if clip_text_start_epoch < HYPERPARAMS["CLIP_EPOCHS"] or prior_start_epoch < HYPERPARAMS["PRIOR_EPOCHS"]:
        chunk_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache", "chunks")
        
        laion_chunk_manager = ChunkedLAIONManager(
            hf_token=hf_token,
            chunk_size=HYPERPARAMS.get("LAION_CHUNK_SIZE", 10000),
            total_samples=HYPERPARAMS.get("LAION_CACHE_SAMPLES", 200000),
            val_ratio=0.05,
            cache_dir=chunk_cache_dir,
            text_processor=CLIPTokenize,
            num_workers=4,
            prefetch_factor=4,
            collate_fn=adaptive_collate,
        )
 
        if clip_text_start_epoch < HYPERPARAMS["CLIP_EPOCHS"]:
            print("Starting CLIP Training Phase")
            train_clip(chunk_manager=laion_chunk_manager,
                       text_start_weights=clip_text_start_weights, 
                       img_start_weights=clip_img_start_weights,
                       wrapper_start_weights=clip_wrapper_start_weights,
                       start_epoch=clip_text_start_epoch, 
                       start_batch=clip_start_batch,
                       run=run)
        else:
            print("CLIP training already completed.")
 
        if prior_start_epoch < HYPERPARAMS["PRIOR_EPOCHS"]:
            laion_chunk_manager.reset_stream()
            
            print("Starting Prior Training Phase")
            train_prior(chunk_manager=laion_chunk_manager,
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
        def load_file_list(file_path):
            try:
                with open(file_path) as f:
                    return [line.strip().split('\t') for line in f.readlines()[1:]]
            except FileNotFoundError:
                return []
 
        print("Initializing Iterable SAM datasets...")
        sa1b_files = load_file_list("data/Datasets/SA-1B_dataset.txt")
        sav_files = load_file_list("data/Datasets/SA-V_dataset.txt")
 
        samples_per_tar = HYPERPARAMS.get("EST_SAMPLES_PER_TAR", 3000)
        batches_per_tar = max(1, samples_per_tar // HYPERPARAMS["SAM_BATCH_SIZE"])
        
        # Use the active phase's start_batch to determine skip
        if sam_decoder_start_epoch < HYPERPARAMS["SAM_DECODER_EPOCHS"]:
            active_skip_tars = sam_start_batch // batches_per_tar if sam_start_batch > 0 else 0
        elif student_start_epoch < HYPERPARAMS["TEACHER_STUDENT_EPOCHS"]:
            active_skip_tars = student_start_batch // batches_per_tar if student_start_batch > 0 else 0
        else:
            active_skip_tars = 0
        
        if active_skip_tars > 0 and is_main_process():
            print(f"[SAM Resume] Skipping ~{active_skip_tars} tars ({active_skip_tars * samples_per_tar} est. samples)")
 
        # Training (Streaming with prefetch + skip)
        sa1b_train = get_dataset(SA1BDataset, sa1b_files, "train", HYPERPARAMS["SA_VAL_TAR_COUNT"], skip_tars=active_skip_tars)
        sav_train = get_dataset(SAVDataset, sav_files, "train", HYPERPARAMS["SAV_VAL_TAR_COUNT"], skip_tars=active_skip_tars)
        
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