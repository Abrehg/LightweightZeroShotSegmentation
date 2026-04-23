# tune.py
#
# Sequential hyperparameter tuning for a 4-phase pipeline:
#   Phase 1 (clip)    — tunes from scratch
#   Phase 2 (prior)   — loads trained CLIP text encoder
#   Phase 3 (decoder) — loads trained CLIP text encoder + Prior
#   Phase 4 (student) — loads full trained teacher (CLIP + Prior + SAM w/ 3 epochs),
#                        warms up teacher, then tunes student hyperparameters
#
# Usage examples:
#   python tune.py --token HF_TOKEN --phase clip --trials 30
#   python tune.py --token HF_TOKEN --phase prior --trials 30 \
#       --weights_dir models/trained --txt_enc_layers 12
#   python tune.py --token HF_TOKEN --phase decoder --trials 30 \
#       --weights_dir models/trained --txt_enc_layers 12 --prior_layers 10
#   python tune.py --token HF_TOKEN --phase student --trials 30 \
#       --weights_dir models/trained --txt_enc_layers 12 --prior_layers 10 \
#       --sam_layers 8 --sam_memory 10 --teacher_warmup_epochs 3

import torch
import gc
import os
import copy
import optuna
import argparse
import warnings
warnings.filterwarnings('ignore')
import json
from torch.utils.data import DataLoader, ConcatDataset
from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss
from models.prior_model import create_prior, PriorLoss, TeacherCLIP
from models.SAM_model import iou_loss, create_SAM
from models.distill_model import create_Student
from data.custom400m import get_laion_test_dataset, adaptive_collate
from data.segmentation import SAM_adaptive_collate, StaticSA1BDataset, StaticSAVDataset

# Hardware Setup (Single Node)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Micro-Sweep Configurations
TUNE_EPOCHS = 1
TUNE_TOTAL_SAMPLES = 2500
TUNE_VAL_RATIO = 0.2

# Standard weight filenames (saved by store_weights in each model)
CLIP_TXT_FILENAME = "txtEncWeights"
CLIP_IMG_FILENAME = "imgEncWeights"
CLIP_WRAPPER_FILENAME = "CLIPWrapperWeights"
PRIOR_FILENAME = "PriorWeights"
SAM_FILENAME = "SAMWeights"

# ======================== Dataset Caching ========================
# Loaded once on first call, reused across all trials.

_laion_cache = {}
_sam_cache = {}

def _get_laion_datasets(hf_token):
    if 'train' not in _laion_cache:
        print(f"[Cache] Downloading {TUNE_TOTAL_SAMPLES} LAION samples (one-time)...")
        full_dataset = get_laion_test_dataset(
            hf_token, val_boundary=0, num_samples=TUNE_TOTAL_SAMPLES,
            text_processor=CLIPTokenize, split="train"
        )
        
        total = len(full_dataset)
        val_size = max(1, int(total * TUNE_VAL_RATIO))
        train_size = total - val_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        _laion_cache['train'] = train_subset
        _laion_cache['val'] = val_subset
        print(f"[Cache] Split into {train_size} train / {val_size} val samples")
    
    return _laion_cache['train'], _laion_cache['val']

def get_laion_loaders(hf_token, batch_size):
    train_dataset, val_dataset = _get_laion_datasets(hf_token)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=adaptive_collate, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=adaptive_collate, shuffle=False, num_workers=4)
    return train_loader, val_loader

def _get_sam_datasets():
    if 'train' not in _sam_cache:
        def load_file_list(file_path):
            try:
                with open(file_path) as f:
                    return [line.strip().split('\t') for line in f.readlines()[1:]]
            except FileNotFoundError:
                print(f"[SAM] WARNING: File list not found: {file_path}")
                return []
 
        sa1b_files = load_file_list("data/Datasets/SA-1B_dataset.txt")
        sav_files = load_file_list("data/Datasets/SA-V_dataset.txt")
        
        print(f"[SAM] Found {len(sa1b_files)} SA-1B entries, {len(sav_files)} SA-V entries")
 
        # Use barn for tar cache to avoid home quota limits
        cache_dir = os.environ.get(
            "SAM_CACHE_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")
        )
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[SAM] Cache directory: {cache_dir}")
 
        print("[Cache] Loading SAM train dataset (one-time)...")
        _sam_cache['train'] = ConcatDataset([
            StaticSA1BDataset(sa1b_files, cache_dir, device, "train", val_tar_count=1, val_sample_count=1000),
            StaticSAVDataset(sav_files, cache_dir, device, "train", val_tar_count=1, val_sample_count=1000)
        ])
        print(f"[SAM] Train dataset: {len(_sam_cache['train'])} samples")
        
        print("[Cache] Loading SAM val dataset (one-time)...")
        _sam_cache['val'] = ConcatDataset([
            StaticSA1BDataset(sa1b_files, cache_dir, device, "val", val_tar_count=1, val_sample_count=250),
            StaticSAVDataset(sav_files, cache_dir, device, "val", val_tar_count=1, val_sample_count=250)
        ])
        print(f"[SAM] Val dataset: {len(_sam_cache['val'])} samples")
        
        if len(_sam_cache['train']) == 0:
            print("[SAM] ERROR: Train dataset is empty! Check that:")
            print("  1. data/Datasets/SA-1B_dataset.txt and SA-V_dataset.txt have valid URLs")
            print(f"  2. Cache dir {cache_dir} has enough disk space")
            print("  3. URLs haven't expired (SA-1B/SA-V links rotate periodically)")
    
    return _sam_cache['train'], _sam_cache['val']
 
def get_sam_loaders(batch_size):
    train_dataset, val_dataset = _get_sam_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=SAM_adaptive_collate, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=SAM_adaptive_collate, shuffle=False, num_workers=4)
    return train_loader, val_loader

# ======================== Helpers ========================

def gpu_cleanup():
    """Free GPU memory between trials to prevent fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# Base directory for saving trial weights
TUNE_CHECKPOINT_DIR = "weights/tune"

def save_trial_weights(phase, trial, val_loss, models_dict):
    trial_dir = os.path.join(
        TUNE_CHECKPOINT_DIR, phase,
        f"trial_{trial.number:02d}_loss_{val_loss:.4f}"
    )
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save each model's weights
    for filename, model in models_dict.items():
        if filename == "clip_wrapper":
            # CLIPWrapper has its own multi-file save
            model.store_weights(
                trial_dir, 
                CLIP_TXT_FILENAME, CLIP_IMG_FILENAME, CLIP_WRAPPER_FILENAME
            )
        elif hasattr(model, 'store_weights'):
            model.store_weights(trial_dir, filename)
        else:
            # Fallback: raw state_dict save
            torch.save(model.state_dict(), os.path.join(trial_dir, filename))
    
    # Save hyperparameters alongside weights
    params = dict(trial.params)
    params["val_loss"] = val_loss
    params["trial_number"] = trial.number
    with open(os.path.join(trial_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"  [Save] Trial {trial.number} weights saved to {trial_dir} (val_loss={val_loss:.4f})")

def load_trained_text_encoder(args):
    txt_path = os.path.join(args.weights_dir, CLIP_TXT_FILENAME)
    text_encoder = create_text_encoder(num_layers=args.txt_enc_layers).to(device)
    if os.path.exists(txt_path):
        print(f"[Weights] Loading trained text encoder ({args.txt_enc_layers} layers) from {txt_path}")
        text_encoder.load_weights(txt_path)
    else:
        print(f"[Weights] WARNING: Trained text encoder not found at {txt_path}")
        print(f"  Using random initialization. Tuning results will approximate — retune after CLIP training for best accuracy.")
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    return text_encoder

def load_trained_prior(args):
    prior_path = os.path.join(args.weights_dir, PRIOR_FILENAME)
    prior = create_prior(num_layers=args.prior_layers).to(device)
    if os.path.exists(prior_path):
        print(f"[Weights] Loading trained prior ({args.prior_layers} layers) from {prior_path}")
        prior.load_weights(prior_path)
    else:
        print(f"[Weights] WARNING: Trained prior not found at {prior_path}")
        print(f"  Using random initialization. Tuning results will approximate — retune after Prior training for best accuracy.")
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False
    return prior

def load_trained_sam(args):
    sam_path = os.path.join(args.weights_dir, SAM_FILENAME)
    sam_decoder = create_SAM(max_memory_length=args.sam_memory, num_layers=args.sam_layers).to(device)
    if os.path.exists(sam_path):
        print(f"[Weights] Loading trained SAM decoder ({args.sam_layers} layers, memory={args.sam_memory}) from {sam_path}")
        sam_decoder.load_weights(sam_path)
    else:
        print(f"[Weights] WARNING: Trained SAM decoder not found at {sam_path}")
        print(f"  Using random initialization. Tuning results will approximate — retune after SAM training for best accuracy.")
    return sam_decoder

# ======================== PHASE 1: CLIP ========================
def objective_clip(trial, hf_token):
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 24)

    estimated_mem_factor = batch_size * num_layers
    if estimated_mem_factor > 2048:
        raise optuna.exceptions.TrialPruned()
    
    gpu_cleanup()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_laion_loaders(hf_token, batch_size)
    
    text_encoder = create_text_encoder(num_layers=num_layers).to(device)
    image_encoder = create_image_encoder().to(device)
    model = CLIPWrapper(text_encoder, image_encoder).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(TUNE_EPOCHS):
        model.train()
        for batch in train_loader:
            if batch is None: continue
            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                text_features, image_features, logit_scale = model(texts, images)
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text  = logit_scale * text_features  @ image_features.t()
                loss = clip_contrastive_loss(logits_per_image, logits_per_text)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                images, texts = batch
                images = images.to(device)
                texts = texts.to(device)

                with torch.cuda.amp.autocast():
                    text_features, image_features, logit_scale = model(texts, images)
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text  = logit_scale * text_features  @ image_features.t()
                    val_loss += clip_contrastive_loss(logits_per_image, logits_per_text).item()
                
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    save_trial_weights("clip", trial, val_loss, {"clip_wrapper": model})
        
    return val_loss

# ======================== PHASE 2: Prior ========================
# Depends on: trained CLIP text encoder (frozen).
def objective_prior(trial, hf_token, args):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 24)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    estimated_mem_factor = batch_size * num_layers
    if estimated_mem_factor > 1024:
        raise optuna.exceptions.TrialPruned()

    gpu_cleanup()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_laion_loaders(hf_token, batch_size=batch_size)
    
    # Load trained text encoder from CLIP phase (frozen)
    text_encoder = load_trained_text_encoder(args)
    teacher = TeacherCLIP().to(device).eval()
    
    prior = create_prior(num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr)

    for epoch in range(TUNE_EPOCHS):
        prior.train()
        for batch in train_loader:
            if batch is None: continue
            images, texts = batch

            images = images.to(device)
            texts = texts.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    text_emb    = text_encoder(texts)
                    target_grid = teacher(images)
                prior_grid = prior(text_emb)
                loss = PriorLoss(prior_grid, target_grid)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        prior.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                images, texts = batch

                images = images.to(device)
                texts = texts.to(device)
            
                with torch.cuda.amp.autocast():
                    text_emb    = text_encoder(texts)
                    target_grid = teacher(images)
                    prior_grid  = prior(text_emb)
                    val_loss   += PriorLoss(prior_grid, target_grid).item()
                
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        
    save_trial_weights("prior", trial, val_loss, {PRIOR_FILENAME: prior})

    return val_loss

# ======================== PHASE 3: SAM Decoder ========================
# Depends on: trained CLIP text encoder + Prior (both frozen).
def objective_decoder(trial, args):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1])
    max_memory_length = trial.suggest_int("Max Memory Length", 1, 3)
    enc_num_layers = trial.suggest_int("encoder num layers", 1, 4)
    dec_num_layers = trial.suggest_int("decoder num layers", 1, 4)
    
    if max_memory_length > 3:
        raise optuna.exceptions.TrialPruned()
    if batch_size > 4:
        raise optuna.exceptions.TrialPruned()

    gpu_cleanup()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_sam_loaders(batch_size)
    
    # Load trained components from previous phases (frozen)
    text_encoder = load_trained_text_encoder(args)
    prior = load_trained_prior(args)
    
    # SAM decoder is what we're tuning — fresh init
    sam_decoder = create_SAM(max_memory_length=max_memory_length, 
                             enc_num_layers=enc_num_layers, 
                             dec_num_layers=dec_num_layers).to(device)

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder
            self.prior = prior
            self.sam_decoder = sam_decoder

        def forward_frame(self, frame, text_tokens, memory, t=0):
            with torch.no_grad():
                text_emb  = self.text_encoder(text_tokens)
                prior_emb = self.prior(text_emb)
            mask, new_memory = self.sam_decoder(frame, prior_emb, memory, t)
            return mask, new_memory
 
        def init_memory(self, B, device):
            return self.sam_decoder.init_memory(B, device)

    teacher = TeacherModel().to(device)

    optimizer = torch.optim.Adam(sam_decoder.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(TUNE_EPOCHS):
        sam_decoder.train()
        for batch in train_loader:
            if batch is None: continue
            images, masks, texts = batch
            optimizer.zero_grad()
            
            for img, mask, txt in zip(images, masks, texts):
                mask = mask.to(device).float()
                img = img.to(device)
                txt = txt.to(device)

                T = img.shape[1]
                memory = teacher.init_memory(img.shape[0], device)

                for t in range(T):
                    with torch.cuda.amp.autocast():
                        pred_mask, new_memory = teacher.forward_frame(img, txt, memory, t)
                        loss = iou_loss(pred_mask, mask[:, t])
                    scaler.scale(loss).backward()
                    memory = new_memory.detach()
            optimizer.step()
            
        sam_decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                images, masks, texts = batch
                for img, mask, txt in zip(images, masks, texts):
                    mask = mask.to(device).float()
                    img = img.to(device)
                    txt = txt.to(device)

                    T = img.shape[1]
                    memory = teacher.init_memory(img.shape[0], device)

                    for t in range(T):
                        with torch.cuda.amp.autocast():
                            pred_mask, memory = teacher.forward_frame(img, txt, memory, t)
                            val_loss += iou_loss(pred_mask, mask[:, t])
                    
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    save_trial_weights("decoder", trial, val_loss, {SAM_FILENAME: sam_decoder})
        
    return val_loss

# ======================== PHASE 4: Student (Co-Distillation) ========================
# Depends on: full trained teacher pipeline (CLIP + Prior + SAM with 3 epochs).
# The teacher is loaded from checkpoints, warmed up, then co-trained with the
# student — the SAM decoder continues improving while the student distills from it.
# Text encoder and prior remain frozen throughout.

def warmup_teacher(teacher, train_loader, warmup_epochs, lr=1e-4):
    print(f"[Teacher Warmup] Running {warmup_epochs} warm-up epoch(s) on SAM decoder...")
    optimizer = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=lr)
    
    for epoch in range(warmup_epochs):
        teacher.sam_decoder.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if batch is None: continue
            images, masks, texts = batch
            optimizer.zero_grad()
            
            for img, mask, txt in zip(images, masks, texts):
                mask = mask.to(device).float()
                img = img.to(device)
                txt = txt.to(device)

                T = img.shape[1]
                memory = teacher.init_memory(img.shape[0], device)

                for t in range(T):
                    pred_mask, new_memory = teacher.forward_frame(img, txt, memory, t)
                    loss = iou_loss(pred_mask, mask[:, t])
                    loss.backward()
                    memory = new_memory.detach()
            
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"  Warmup epoch {epoch+1}/{warmup_epochs} -- avg loss: {avg_loss:.4f}")
    
    teacher.sam_decoder.eval()
    print("[Teacher Warmup] Complete.")

# Cache the warmed SAM decoder state_dict so warm-up runs once.
# Each trial gets a fresh deep-copy (since co-distillation modifies the teacher).
_warmed_sam_state = None
_frozen_text_encoder = None
_frozen_prior = None

def get_warmed_teacher_components(args):
    """Build the teacher, warm up SAM decoder once, cache the resulting
    state_dict and frozen upstream components for reuse across trials."""
    global _warmed_sam_state, _frozen_text_encoder, _frozen_prior
    
    if _warmed_sam_state is not None:
        return _frozen_text_encoder, _frozen_prior, _warmed_sam_state
    
    _frozen_text_encoder = load_trained_text_encoder(args)
    _frozen_prior = load_trained_prior(args)
    sam_decoder = load_trained_sam(args)
    
    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = _frozen_text_encoder
            self.prior = _frozen_prior
            self.sam_decoder = sam_decoder

        def forward_frame(self, frame, text_tokens, memory, t=0):
            with torch.no_grad():
                text_emb  = self.text_encoder(text_tokens)
                prior_emb = self.prior(text_emb)
            mask, new_memory = self.sam_decoder(frame, prior_emb, memory, t)
            return mask, new_memory
 
        def init_memory(self, B, device):
            return self.sam_decoder.init_memory(B, device)

    teacher = TeacherModel().to(device)
    
    if args.teacher_warmup_epochs > 0:
        warmup_loader = get_sam_loaders(batch_size=64)[0]
        warmup_teacher(teacher, warmup_loader, args.teacher_warmup_epochs)
    
    # Cache the warmed SAM weights (deep copy so later trials don't conflict)
    _warmed_sam_state = copy.deepcopy(sam_decoder.state_dict())
    
    print("[Cache] Warmed SAM decoder state_dict cached for all student trials.")
    return _frozen_text_encoder, _frozen_prior, _warmed_sam_state

def objective_student(trial, args):
    teacher_lr = trial.suggest_float("teacher_lr", 1e-5, 1e-2, log=True)
    student_lr = trial.suggest_float("student_lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1])
    input_layers = trial.suggest_int("input layers", 1, 10)
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 8)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 8)
    max_memory_length = trial.suggest_int("Max Memory Length", 1, 20)

    if batch_size * (input_layers + num_decoder_layers) > 256:
        raise optuna.exceptions.TrialPruned()

    gpu_cleanup()
    scaler_teacher = torch.cuda.amp.GradScaler()
    scaler_student = torch.cuda.amp.GradScaler()
    train_loader, val_loader = get_sam_loaders(batch_size)

    # Get cached components (warm-up runs only on first trial)
    text_encoder, prior, warmed_sam_state = get_warmed_teacher_components(args)

    # Build a fresh teacher with warmed SAM weights for this trial
    sam_decoder = create_SAM(
        max_memory_length=args.sam_memory, 
        enc_num_layers=args.sam_enc_layers, 
        dec_num_layers=args.sam_dec_layers
    ).to(device)
    sam_decoder.load_state_dict(warmed_sam_state)

    class TeacherModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = text_encoder
            self.prior = prior
            self.sam_decoder = sam_decoder

        def forward_frame(self, frame, text_tokens, memory, t=0):
            with torch.no_grad():
                text_emb  = self.text_encoder(text_tokens)
                prior_emb = self.prior(text_emb)
            mask, new_memory = self.sam_decoder(frame, prior_emb, memory, t)
            return mask, new_memory
 
        def init_memory(self, B, device):
            return self.sam_decoder.init_memory(B, device)

    teacher = TeacherModel().to(device)

    student = create_Student(
        text_transformer_layers=input_layers, 
        max_memory_length=max_memory_length,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers
    ).to(device)

    # Co-distillation: both the teacher SAM decoder and student train jointly
    optimizer_teacher = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=teacher_lr)
    optimizer_student = torch.optim.Adam(student.parameters(), lr=student_lr)

    for epoch in range(TUNE_EPOCHS):
        teacher.sam_decoder.train()
        student.train()

        for _, batch in enumerate(train_loader):
            if batch is None: continue
            images, true_masks, texts = batch

            optimizer_teacher.zero_grad()
            optimizer_student.zero_grad()

            for img, mask, txt in zip(images, true_masks, texts):
                mask = mask.to(device).float()
                img = img.to(device)
                txt = txt.to(device)

                T = img.shape[1]
                mem_t = teacher.init_memory(img.shape[0], device)
                mem_s = student.init_memory(img.shape[0], device)

                for t in range(T):
                    with torch.cuda.amp.autocast():
                        teacher_out, mem_t_new = teacher.forward_frame(img, txt, mem_t, t)
                        student_out, mem_s_new = student.forward(img, txt, mem_s, t)
                
                        with torch.no_grad():
                            teacher_out_for_student = teacher_out.detach()

                        teacher_loss = iou_loss(teacher_out, mask[:, t])
                        student_loss = student.compute_distill_loss(student_out, teacher_out_for_student, mask[:, t])
               
                    scaler_teacher.scale(teacher_loss).backward(retain_graph=True)
                    scaler_student.scale(student_loss).backward()
                    mem_t = mem_t_new.detach()
                    mem_s = mem_s_new.detach()
 
            scaler_teacher.step(optimizer_teacher)
            scaler_teacher.update()
            scaler_student.step(optimizer_student)
            scaler_student.update()

        teacher.sam_decoder.eval()
        student.eval()
        val_t_loss, val_s_loss = 0.0, 0.0
        with torch.no_grad():
            for v_batch in val_loader:
                if v_batch is None: continue
                v_images, v_true_masks, v_texts = v_batch
                for v_img, v_mask, v_txt in zip(v_images, v_true_masks, v_texts):
                    v_mask = v_mask.to(device).float()
                    v_img = v_img.to(device)
                    v_txt = v_txt.to(device)

                    T = img.shape[1]
                    mem_t = teacher.init_memory(img.shape[0], device)
                    mem_s = student.init_memory(img.shape[0], device)

                    for t in range(T):
                        with torch.cuda.amp.autocast():
                            teacher_out, mem_t = teacher.forward_frame(img, txt, mem_t, t)
                            student_out, mem_s = student.forward(img, txt, mem_s, t)

                            val_t_loss += iou_loss(teacher_out, mask[:, t])
                            val_s_loss += student.compute_distill_loss(student_out, teacher_out, mask[:, t])

        val_loss = val_t_loss + val_s_loss
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
    
    save_trial_weights("student", trial, val_loss, {
        "StudentWeights": student,
        SAM_FILENAME: sam_decoder,  # co-distilled teacher SAM
    })

    return val_loss

# ======================== Main ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for sequential model pipeline")
    
    # Required
    parser.add_argument("--token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--phase", type=str, choices=["clip", "prior", "decoder", "student"], required=True)
    parser.add_argument("--trials", type=int, default=30)
    
    # Weight loading (required for phases after clip)
    parser.add_argument("--weights_dir", type=str, default="models/trained",
                        help="Directory containing trained weights from previous phases")
    
    # Structural params from previous tuning results (must match saved weights)
    parser.add_argument("--txt_enc_layers", type=int, default=12,
                        help="num_layers for the trained text encoder (from CLIP tuning best trial)")
    parser.add_argument("--prior_layers", type=int, default=12,
                        help="num_layers for the trained prior (from Prior tuning best trial)")
    parser.add_argument("--sam_layers", type=int, default=2,
                        help="num_layers for the trained SAM decoder (from Decoder tuning best trial)")
    parser.add_argument("--sam_memory", type=int, default=10,
                        help="max_memory_length for the trained SAM decoder (from Decoder tuning best trial)")
    
    # Student-specific
    parser.add_argument("--teacher_warmup_epochs", type=int, default=3,
                        help="Epochs to warm up teacher SAM decoder before student tuning. "
                             "Set to 0 if SAM weights already have sufficient training.")
    
    args = parser.parse_args()
    
    # Validate that weights exist for phases that need them
    if args.phase in ("prior", "decoder", "student"):
        os.makedirs(args.weights_dir, exist_ok=True)
        
        # Check which weights exist and warn about missing ones upfront
        required_weights = {}
        if args.phase in ("prior", "decoder", "student"):
            required_weights["Text encoder"] = os.path.join(args.weights_dir, CLIP_TXT_FILENAME)
        if args.phase in ("decoder", "student"):
            required_weights["Prior"] = os.path.join(args.weights_dir, PRIOR_FILENAME)
        if args.phase == "student":
            required_weights["SAM decoder"] = os.path.join(args.weights_dir, SAM_FILENAME)
        
        missing = [name for name, path in required_weights.items() if not os.path.exists(path)]
        if missing:
            print(f"\n{'='*60}")
            print(f"  WARNING: Missing trained weights for phase '{args.phase}':")
            for name in missing:
                print(f"    - {name}: {required_weights[name]}")
            print(f"\n  Tuning will proceed with RANDOM initialization for these")
            print(f"  components. Results are approximate — consider retuning")
            print(f"  after full training for best accuracy.")
            print(f"{'='*60}\n")
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    catch_errors = (torch.cuda.OutOfMemoryError, RuntimeError)
    
    print(f"=== Starting Optuna Micro-Sweep for Phase: {args.phase.upper()} ===")
    if args.phase != "clip":
        print(f"  Loading pretrained weights from: {args.weights_dir}")
    
    if args.phase == "clip":
        study.optimize(lambda t: objective_clip(t, args.token), n_trials=args.trials, catch=catch_errors)
    elif args.phase == "prior":
        study.optimize(lambda t: objective_prior(t, args.token, args), n_trials=args.trials, catch=catch_errors)
    elif args.phase == "decoder":
        study.optimize(lambda t: objective_decoder(t, args), n_trials=args.trials, catch=catch_errors)
    elif args.phase == "student":
        study.optimize(lambda t: objective_student(t, args), n_trials=args.trials, catch=catch_errors)
        
    print(f"\nBest {args.phase} trial:")
    print(f"  Value (Validation Loss): {study.best_trial.value}")
    print("  Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    best_dir = os.path.join(
        TUNE_CHECKPOINT_DIR, args.phase,
        f"trial_{study.best_trial.number:02d}_loss_{study.best_trial.value:.4f}"
    )
    if os.path.isdir(best_dir):
        print(f"  Best trial weights: {best_dir}")
    
    # Summary of all saved trials
    phase_dir = os.path.join(TUNE_CHECKPOINT_DIR, args.phase)
    if os.path.isdir(phase_dir):
        saved_trials = sorted(os.listdir(phase_dir))
        print(f"\n  All saved trial weights ({len(saved_trials)} trials):")
        for d in saved_trials:
            print(f"    {d}")