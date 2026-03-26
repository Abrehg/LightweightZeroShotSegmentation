# tuneModel.py
#
# Fine-tune the trained teacher and student on referring expression datasets
# (gRefCOCO + Ref-YouTube-VOS) to improve generalization to human-written prompts.
#
# Run after the main train.py pipeline has completed all 4 phases.
# Loads best checkpoints from the training run, then fine-tunes on natural
# language referring expressions with multi-target, no-target, and temporal grounding.
#
# Usage:
#   # Single GPU
#   python finetune.py --checkpoint_dir weights --wandb_key KEY
#
#   # Multi-GPU via torchrun
#   torchrun --nproc_per_node=4 finetune.py --checkpoint_dir weights --wandb_key KEY

import os
import re
import glob
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import wandb
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.clip_model import create_text_encoder, CLIPTokenize
from models.prior_model import create_prior
from models.SAM_model import iou_loss, create_SAM
from models.distill_model import create_Student
from data.referring import (
    gRefCOCODataset, RefYouTubeVOSDataset, referring_collate,
    get_grefcoco_loaders, get_refytvos_loaders
)

# ======== Hyperparameters ========
FINETUNE_PARAMS = {
    "EPOCHS": 5,
    "TEACHER_LR": 1e-5,      # Low LR to avoid catastrophic forgetting
    "STUDENT_LR": 5e-5,
    "BATCH_SIZE": 32,
    "SAVE_FREQ": 100,
    "GREFCOCO_DIR": "data/grefcoco",
    "COCO_IMAGE_DIR": "data/images/train2014",
    "REFYTVOS_DIR": "data/ref-youtube-vos",
    "MAX_VIDEO_FRAMES": 8,
    "WANDB_PROJECT": "Zero Shot Segmentation",
    "WANDB_ENTITY": "adityaasuratkal-rensselaer-polytechnic-institute",
}

# ======== Utility Functions ========

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_distributed():
    return dist.is_initialized()

def is_main_process():
    return not is_distributed() or dist.get_rank() == 0

def get_local_device():
    if "LOCAL_RANK" in os.environ:
        return torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}')
    return torch.device(get_device())

def get_best_weights_checkpoint(directory, prefix):
    """Find the best checkpoint by validation loss, matching train.py's naming convention."""
    if not os.path.isdir(directory):
        return None
    
    files = glob.glob(os.path.join(directory, f"{prefix}_epoch_*"))
    best_loss = float('inf')
    best_file = None
    
    for f_path in files:
        filename = os.path.basename(f_path)
        match = re.search(rf"{prefix}_epoch_(\d+)_batch_(\d+)_([0-9]+(?:\.[0-9]+)?)", filename)
        if match:
            loss = float(match.group(3))
            if loss < best_loss:
                best_loss, best_file = loss, f_path
    
    # Fallback to latest complete epoch
    if best_file is None:
        complete_files = [f for f in files if "complete" in f]
        if complete_files:
            best_file = sorted(complete_files)[-1]
    
    return best_file

# ======== Load Trained Pipeline ========

def load_trained_pipeline(checkpoint_dir, local_device):
    """Load the full trained teacher pipeline and student from train.py checkpoints."""
    
    # Text encoder
    clip_text_ckpt = get_best_weights_checkpoint(checkpoint_dir, "clip_text")
    if not clip_text_ckpt:
        raise FileNotFoundError("No CLIP text encoder checkpoint found.")
    text_encoder = create_text_encoder().to(local_device)
    text_encoder.load_weights(clip_text_ckpt)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    print(f"[Load] Text encoder: {clip_text_ckpt}")
    
    # Prior
    prior_ckpt = get_best_weights_checkpoint(checkpoint_dir, "prior")
    if not prior_ckpt:
        raise FileNotFoundError("No Prior checkpoint found.")
    prior = create_prior().to(local_device)
    prior.load_weights(prior_ckpt)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False
    print(f"[Load] Prior: {prior_ckpt}")
    
    # SAM decoder (teacher) — use student-phase teacher if available, else standalone SAM
    teacher_sam_ckpt = get_best_weights_checkpoint(checkpoint_dir, "student_phase_teacher")
    if not teacher_sam_ckpt:
        teacher_sam_ckpt = get_best_weights_checkpoint(checkpoint_dir, "sam_decoder")
    if not teacher_sam_ckpt:
        raise FileNotFoundError("No SAM decoder checkpoint found.")
    sam_decoder = create_SAM().to(local_device)
    sam_decoder.load_weights(teacher_sam_ckpt)
    print(f"[Load] SAM decoder: {teacher_sam_ckpt}")
    
    # Student
    student_ckpt = get_best_weights_checkpoint(checkpoint_dir, "student_phase_student")
    if not student_ckpt:
        raise FileNotFoundError("No Student checkpoint found.")
    student = create_Student().to(local_device)
    student.load_weights(student_ckpt)
    print(f"[Load] Student: {student_ckpt}")
    
    # Build teacher wrapper
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
    
    return teacher, student

# ======== Fine-Tuning Loop ========

def finetune(checkpoint_dir, wandb_key):
    setup_ddp()
    local_device = get_local_device()
    
    # W&B
    if is_main_process():
        wandb.login(key=wandb_key)
        run = wandb.init(
            project=FINETUNE_PARAMS["WANDB_PROJECT"],
            entity=FINETUNE_PARAMS["WANDB_ENTITY"],
            config=FINETUNE_PARAMS,
            name="finetune-referring"
        )
    else:
        run = wandb.init(mode="disabled")
    
    # Load trained models
    teacher, student = load_trained_pipeline(checkpoint_dir, local_device)
    
    # Wrap in DDP if distributed
    if is_distributed():
        teacher.sam_decoder = DDP(teacher.sam_decoder, device_ids=[int(os.environ["LOCAL_RANK"])])
        student = DDP(student, device_ids=[int(os.environ["LOCAL_RANK"])])
    
    # Build combined dataset: gRefCOCO (static) + Ref-YouTube-VOS (video)
    print("\n[Data] Loading gRefCOCO...")
    grefcoco_train = gRefCOCODataset(
        grefcoco_dir=FINETUNE_PARAMS["GREFCOCO_DIR"],
        image_dir=FINETUNE_PARAMS["COCO_IMAGE_DIR"],
        split="train"
    )
    grefcoco_val = gRefCOCODataset(
        grefcoco_dir=FINETUNE_PARAMS["GREFCOCO_DIR"],
        image_dir=FINETUNE_PARAMS["COCO_IMAGE_DIR"],
        split="val"
    )
    
    print("\n[Data] Loading Ref-YouTube-VOS...")
    refytvos_full = RefYouTubeVOSDataset(
        root_dir=FINETUNE_PARAMS["REFYTVOS_DIR"],
        split="train",
        max_frames=FINETUNE_PARAMS["MAX_VIDEO_FRAMES"]
    )
    # Split off a val portion since competition val annotations aren't public
    total = len(refytvos_full)
    val_size = min(500, total // 10)
    refytvos_train, refytvos_val = torch.utils.data.random_split(
        refytvos_full, [total - val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Combine static + video for training
    train_dataset = ConcatDataset([grefcoco_train, refytvos_train])
    val_dataset = ConcatDataset([grefcoco_val, refytvos_val])
    
    bs = FINETUNE_PARAMS["BATCH_SIZE"]
    train_sampler = DistributedSampler(train_dataset) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=referring_collate,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        sampler=val_sampler, collate_fn=referring_collate,
        num_workers=4, pin_memory=True
    )
    
    print(f"\n[Data] Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Optimizers (low LR to avoid forgetting pretraining)
    optimizer_teacher = torch.optim.Adam(
        teacher.sam_decoder.parameters(), lr=FINETUNE_PARAMS["TEACHER_LR"]
    )
    optimizer_student = torch.optim.Adam(
        (student.module if is_distributed() else student).parameters(),
        lr=FINETUNE_PARAMS["STUDENT_LR"]
    )
    
    # Co-distillation fine-tuning loop
    print(f"\n=== Starting Referring Expression Fine-Tuning ({FINETUNE_PARAMS['EPOCHS']} epochs) ===\n")
    
    for epoch in range(FINETUNE_PARAMS["EPOCHS"]):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        teacher.sam_decoder.train()
        if is_distributed():
            student.module.train()
        else:
            student.train()
        
        total_teacher_loss = 0.0
        total_student_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            images, true_masks, texts = batch
            
            optimizer_teacher.zero_grad()
            optimizer_student.zero_grad()
            
            batch_t_loss = 0.0
            batch_s_loss = 0.0
            n_samples = 0
            
            for img, mask, txt in zip(images, true_masks, texts):
                mask = mask.to(local_device).float()
                img = img.to(local_device)
                txt = txt.to(local_device)
                
                # Teacher forward + loss
                teacher_out = teacher(img, txt)
                t_loss = iou_loss(teacher_out, mask)
                
                # Student forward + distillation loss
                student_out = student(img, txt)
                with torch.no_grad():
                    teacher_detached = teacher_out.detach()
                s_loss = (student.module if is_distributed() else student).compute_distill_loss(
                    student_out, teacher_detached, mask
                )
                
                t_loss.backward(retain_graph=True)
                s_loss.backward()
                
                batch_t_loss += t_loss.item()
                batch_s_loss += s_loss.item()
                n_samples += 1
            
            optimizer_teacher.step()
            optimizer_student.step()
            
            avg_t = batch_t_loss / max(1, n_samples)
            avg_s = batch_s_loss / max(1, n_samples)
            total_teacher_loss += avg_t
            total_student_loss += avg_s
            batch_count += 1
            
            if batch_idx % 50 == 0 and is_main_process():
                print(f"  Epoch {epoch+1}/{FINETUNE_PARAMS['EPOCHS']} | "
                      f"Batch {batch_idx} | Teacher: {avg_t:.4f} | Student: {avg_s:.4f}")
                run.log({
                    "ft_teacher_loss": avg_t,
                    "ft_student_loss": avg_s,
                    "ft_epoch": epoch + 1,
                    "ft_batch": batch_idx,
                })
            
            # Periodic validation + save
            if batch_idx > 0 and batch_idx % FINETUNE_PARAMS["SAVE_FREQ"] == 0 and is_main_process():
                val_loss = _validate(teacher, student, val_loader, local_device)
                print(f"  [Val] Epoch {epoch+1} Batch {batch_idx} — Val Loss: {val_loss:.4f}")
                run.log({"ft_val_loss": val_loss, "ft_epoch": epoch + 1})
                
                _save_checkpoint(teacher, student, checkpoint_dir, epoch + 1, batch_idx, val_loss)
                teacher.sam_decoder.train()
                if is_distributed():
                    student.module.train()
                else:
                    student.train()
        
        # End-of-epoch validation + save
        if is_main_process():
            avg_epoch_t = total_teacher_loss / max(1, batch_count)
            avg_epoch_s = total_student_loss / max(1, batch_count)
            print(f"\n  Epoch {epoch+1} Complete — "
                  f"Avg Teacher: {avg_epoch_t:.4f}, Avg Student: {avg_epoch_s:.4f}")
            
            val_loss = _validate(teacher, student, val_loader, local_device)
            print(f"  [Val] Epoch {epoch+1} — Val Loss: {val_loss:.4f}\n")
            run.log({"ft_epoch_val_loss": val_loss, "ft_epoch": epoch + 1})
            
            _save_checkpoint(teacher, student, checkpoint_dir, epoch + 1, "complete", val_loss)
    
    print("Fine-tuning complete.")
    run.finish()


def _validate(teacher, student, val_loader, device):
    """Run validation and return combined loss."""
    teacher.sam_decoder.eval()
    if isinstance(student, DDP):
        student.module.eval()
    else:
        student.eval()
    
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            images, masks, texts = batch
            for img, mask, txt in zip(images, masks, texts):
                mask = mask.to(device).float()
                img = img.to(device)
                txt = txt.to(device)
                
                t_out = teacher(img, txt)
                s_out = student(img, txt)
                
                t_loss = iou_loss(t_out, mask).item()
                s_loss = (student.module if isinstance(student, DDP) else student).compute_distill_loss(
                    s_out, t_out, mask
                ).item()
                total_loss += t_loss + s_loss
                n_samples += 1
    
    return total_loss / max(1, n_samples)


def _save_checkpoint(teacher, student, checkpoint_dir, epoch, batch_info, val_loss):
    """Save fine-tuned checkpoints with ft_ prefix to distinguish from main training."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    sam_module = teacher.sam_decoder.module if isinstance(teacher.sam_decoder, DDP) else teacher.sam_decoder
    stu_module = student.module if isinstance(student, DDP) else student
    
    sam_module.store_weights(
        checkpoint_dir, f"ft_teacher_epoch_{epoch}_batch_{batch_info}_{val_loss:.4f}"
    )
    stu_module.store_weights(
        checkpoint_dir, f"ft_student_epoch_{epoch}_batch_{batch_info}_{val_loss:.4f}"
    )
    print(f"  [Save] Fine-tune checkpoints saved (epoch {epoch}, val_loss={val_loss:.4f})")


# ======== Main ========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune on referring expression datasets")
    parser.add_argument("--checkpoint_dir", type=str, default="weights",
                        help="Directory with trained model checkpoints from train.py")
    parser.add_argument("--wandb_key", type=str, required=True)
    args = parser.parse_args()
    
    torch.manual_seed(42)
    finetune(args.checkpoint_dir, args.wandb_key)