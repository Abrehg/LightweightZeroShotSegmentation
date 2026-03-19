# tune.py
import torch
import optuna
import argparse
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader, ConcatDataset
from models.clip_model import create_text_encoder, create_image_encoder, CLIPTokenize, CLIPWrapper, clip_contrastive_loss
from models.prior_model import create_prior, PriorLoss, TeacherCLIP
from models.SAM_model import iou_loss, create_SAM
from models.distill_model import create_Student
from data.custom400m import get_laion_test_dataset, adaptive_collate
from data.segmentation import SAM_adaptive_collate, StaticSA1BDataset, StaticSAVDataset

# Hardware Setup (Single Node)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Micro-Sweep Configurations (Fast runs!)
TUNE_EPOCHS = 1
TUNE_TRAIN_SAMPLES = 2000 
TUNE_VAL_SAMPLES = 500

def get_laion_loaders(hf_token, batch_size):
    train_dataset = get_laion_test_dataset(hf_token, val_boundary=10000, num_samples=TUNE_TRAIN_SAMPLES, text_processor=CLIPTokenize, split="train")
    val_dataset = get_laion_test_dataset(hf_token, val_boundary=10000, num_samples=TUNE_VAL_SAMPLES, text_processor=CLIPTokenize, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=adaptive_collate, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=adaptive_collate, shuffle=False, num_workers=4)
    return train_loader, val_loader

def get_sam_loaders(batch_size):
    def load_file_list(file_path):
        try:
            with open(file_path) as f:
                return [line.strip().split('\t') for line in f.readlines()[1:]]
        except FileNotFoundError:
            return []

    sa1b_files = load_file_list("data/Datasets/SA-1B_dataset.txt")
    sav_files = load_file_list("data/Datasets/SA-V_dataset.txt")

    # Use Static versions for fast disk I/O during tuning
    train_dataset = ConcatDataset([
        StaticSA1BDataset(sa1b_files, "data/cache", device, "train", val_tar_count=1, val_sample_count=TUNE_TRAIN_SAMPLES//2),
        StaticSAVDataset(sav_files, "data/cache", device, "train", val_tar_count=1, val_sample_count=TUNE_TRAIN_SAMPLES//2)
    ])
    val_dataset = ConcatDataset([
        StaticSA1BDataset(sa1b_files, "data/cache", device, "val", val_tar_count=1, val_sample_count=TUNE_VAL_SAMPLES//2),
        StaticSAVDataset(sav_files, "data/cache", device, "val", val_tar_count=1, val_sample_count=TUNE_VAL_SAMPLES//2)
    ])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=SAM_adaptive_collate, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=SAM_adaptive_collate, shuffle=False, num_workers=4)
    return train_loader, val_loader

# ======== PHASE 1: CLIP Objective ========
def objective_clip(trial, hf_token):
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    num_layers = trial.suggest_int("num_layers", 1, 30)
    
    train_loader, val_loader = get_laion_loaders(hf_token, batch_size)
    
    # Passing structural tuning params to model
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
            
            text_features, image_features, logit_scale = model(texts, images)
            
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()
            loss = clip_contrastive_loss(logits_per_image, logits_per_text)

            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                images, texts = batch
                images = images.to(device)
                texts = texts.to(device)
            
                text_features, image_features, logit_scale = model(texts, images)
            
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logit_scale * text_features @ image_features.t()
                val_loss += clip_contrastive_loss(logits_per_image, logits_per_text).item()
                
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        
    return val_loss

# ======== PHASE 2: Prior Objective ========
def objective_prior(trial, hf_token):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 30)
    
    train_loader, val_loader = get_laion_loaders(hf_token, batch_size=batch_size)
    
    text_encoder = create_text_encoder(num_layers=10).to(device).eval()
    teacher = TeacherCLIP().to(device).eval()
    
    # Passing structural tuning params to Prior
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
            
            with torch.no_grad():
                text_emb = text_encoder(texts)
                target_grid = teacher(images)
            
            prior_grid = prior(text_emb)
            loss = PriorLoss(prior_grid, target_grid)
            
            loss.backward()
            optimizer.step()
            
        prior.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                images, texts = batch

                images = images.to(device)
                texts = texts.to(device)
            
                text_emb = text_encoder(texts)
                target_grid = teacher(images)
            
                prior_grid = prior(text_emb)
                val_loss += PriorLoss(prior_grid, target_grid).item()
                
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        
    return val_loss

# ======== PHASE 3: SAM Decoder Objective ========
def objective_decoder(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    max_memory_length = trial.suggest_int("Max Memory Length", 1, 30)
    num_layers = trial.suggest_int("num layers", 1, 30)
    
    train_loader, val_loader = get_sam_loaders(batch_size)
    
    text_encoder = create_text_encoder(num_layers=10).to(device).eval()
    prior = create_prior(num_layers=10).to(device).eval()
    sam_decoder = create_SAM(max_memory_length=max_memory_length, num_layers=num_layers).to(device)

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

                pred_mask = teacher.forward(img, txt)
                loss = iou_loss(pred_mask, mask)
                loss.backward()
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

                    pred_mask = teacher.forward(img, txt)
                    val_loss += iou_loss(pred_mask, mask).item()
                    
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        
    return val_loss

def objective_student(trial):
    teacher_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    student_lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    input_layers = trial.suggest_int("input layers", 1, 20)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 8)
    max_memory_length = trial.suggest_int("Max Memory Length", 1, 30)

    train_loader, val_loader = get_sam_loaders(batch_size)

    text_encoder = create_text_encoder(num_layers=10).to(device).eval()
    prior = create_prior(num_layers=10).to(device).eval()
    sam_decoder = create_SAM(max_memory_length=10, num_layers=10).to(device)

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

    teacher = TeacherModel().to(device)

    student = create_Student(text_transformer_layers=input_layers, 
                             max_memory_length=max_memory_length,
                             num_decoder_layers=num_decoder_layers)

    optimizer_teacher_finetune = torch.optim.Adam(teacher.sam_decoder.parameters(), lr=teacher_lr)
    optimizer_student = torch.optim.Adam(student.parameters(), lr=student_lr)

    for epoch in range(TUNE_EPOCHS):
        teacher.sam_decoder.train()
        student.train()

        total_teacher_loss = 0.0
        total_student_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None: continue
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

                    t_out = teacher(v_img, v_txt)
                    s_out = student(v_img, v_txt)
                            
                    val_t_loss += iou_loss(t_out, v_mask).item()
                    val_s_loss += student.compute_distill_loss(s_out, t_out, v_mask).item()
        val_loss = val_t_loss + val_s_loss
        trial.report(val_loss, epoch)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
    
    return val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--phase", type=str, choices=["clip", "prior", "decoder", "student"], required=True)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    print(f"=== Starting Optuna Micro-Sweep for Phase: {args.phase.upper()} ===")
    if args.phase == "clip":
        study.optimize(lambda t: objective_clip(t, args.token), n_trials=args.trials)
    elif args.phase == "prior":
        study.optimize(lambda t: objective_prior(t, args.token), n_trials=args.trials)
    elif args.phase == "decoder":
        study.optimize(objective_decoder, n_trials=args.trials)
    elif args.phase == "student":
        study.optimize(objective_student, n_trials=args.trials)
        
    print(f"\\nBest {args.phase} trial:")
    print(f"  Value (Validation Loss): {study.best_trial.value}")
    print("  Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")