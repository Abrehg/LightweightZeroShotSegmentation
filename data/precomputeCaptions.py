# precompute_captions.py
#
# Pre-compute InstructBLIP captions for SA-1B and SA-V datasets.
# Run this ONCE on a single GPU before distributed training.
# Output is a small JSON file (~10-50MB) that all training ranks can read
# without loading the 4GB InstructBLIP model.
#
# Usage:
#   python precompute_captions.py
#
# Output:
#   data/cache/caption_cache.json
#
# This replaces on-the-fly caption generation during training, saving
# ~4GB GPU memory per rank and eliminating per-sample inference latency.

import os
import json
import tarfile
import argparse
from PIL import Image
import torch
from torchvision import transforms
from pycocotools import mask as coco_mask
from .segmentation import CaptionGenerator, download_tar_file
from Helpers.sav_utils import SAVDatasetHelper

NUM_PERSONAS = 4  # 4 caption styles per sample

def precompute_sa1b_captions(file_list, cache_dir, caption_generator, caption_cache, max_samples=None):
    """Generate captions for all SA-1B samples across all tar files."""
    transform = transforms.Compose([transforms.ToTensor()])
    total = 0
    
    for filename, url in file_list:
        dest_path = os.path.join(cache_dir, filename)
        if not os.path.exists(dest_path):
            print(f"  Downloading {filename}...")
            if not download_tar_file(url, dest_path):
                print(f"  Failed to download {filename}, skipping.")
                continue
        
        tar_basename = os.path.basename(dest_path)
        
        try:
            with tarfile.open(dest_path, "r") as tar:
                members = tar.getmembers()
                json_files = [m for m in members if m.name.endswith('.json')]
                
                for json_member in json_files:
                    try:
                        data = json.load(tar.extractfile(json_member))
                        image_name = data['image']['file_name']
                        image_member = next(
                            (m for m in members if m.isfile() and os.path.basename(m.name) == image_name), 
                            None
                        )
                        if not image_member:
                            continue
                        
                        image = Image.open(tar.extractfile(image_member)).convert("RGB")
                        image_tensor = transform(image)
                        
                        for ann_idx, ann in enumerate(data['annotations']):
                            cache_key = f"{tar_basename}::{json_member.name}::{ann_idx}"
                            
                            if cache_key in caption_cache:
                                continue  # Already cached
                            
                            try:
                                rle = {
                                    'counts': ann['segmentation']['counts'],
                                    'size': ann['segmentation']['size']
                                }
                                mask_tensor = torch.from_numpy(coco_mask.decode(rle)).long()
                                
                                captions = []
                                for persona_idx in range(NUM_PERSONAS):
                                    caption = caption_generator.generate_all_captions(
                                        image_tensor, mask_tensor, persona_idx
                                    )
                                    captions.append(caption)
                                
                                caption_cache[cache_key] = captions
                                total += 1
                                
                                if total % 100 == 0:
                                    print(f"  SA-1B: {total} samples captioned...")
                                
                                if max_samples and total >= max_samples:
                                    return total
                                    
                            except Exception:
                                continue
                    except Exception:
                        continue
        except Exception:
            continue
    
    return total

def precompute_sav_captions(file_list, cache_dir, caption_generator, caption_cache, max_samples=None):
    """Generate captions for all SA-V video samples."""
    transform = transforms.Compose([transforms.ToTensor()])
    sav_helper = SAVDatasetHelper(os.path.dirname(cache_dir))
    total = 0
    
    for filename, url in file_list:
        dest_path = os.path.join(cache_dir, filename)
        if not os.path.exists(dest_path):
            print(f"  Downloading {filename}...")
            if not download_tar_file(url, dest_path):
                print(f"  Failed to download {filename}, skipping.")
                continue
        
        tar_basename = os.path.basename(dest_path)
        
        try:
            with tarfile.open(dest_path, "r") as tar:
                members = tar.getmembers()
                mp4_files = [m for m in members if m.name.endswith('.mp4') and m.isfile()]
                json_files = [m for m in members if m.name.endswith('.json') and m.isfile()]
                
                for video_member in mp4_files:
                    video_id = os.path.splitext(os.path.basename(video_member.name))[0]
                    cache_key = f"{tar_basename}::{video_id}"
                    
                    if cache_key in caption_cache:
                        continue
                    
                    annot_member = next(
                        (m for m in json_files 
                         if f"{video_id}_manual.json" in m.name or f"{video_id}_auto.json" in m.name),
                        None
                    )
                    if not annot_member:
                        continue
                    
                    try:
                        # Extract video to temp file for frame reading
                        temp_path = os.path.join(cache_dir, f"precompute_tmp_{video_id}.mp4")
                        with open(temp_path, 'wb') as f:
                            f.write(tar.extractfile(video_member).read())
                        
                        frames = sav_helper.read_frames(temp_path)
                        os.remove(temp_path)
                        
                        if not frames:
                            continue
                        
                        data = json.load(tar.extractfile(annot_member))
                        
                        # Find first valid frame with a mask
                        first_valid_frame, first_valid_mask = None, None
                        for frame_idx, frame in enumerate(frames):
                            if data and data.get('masklet') and len(data['masklet']) > frame_idx:
                                frame_masks = []
                                for ann in data['masklet'][frame_idx]:
                                    rle = {'counts': ann['counts'], 'size': ann['size']}
                                    frame_masks.append(torch.from_numpy(coco_mask.decode(rle)).long())
                                if frame_masks:
                                    first_valid_frame = transform(Image.fromarray(frame))
                                    first_valid_mask = torch.stack(frame_masks).any(dim=0).float()
                                    break
                        
                        if first_valid_frame is None:
                            caption_cache[cache_key] = ["object"] * NUM_PERSONAS
                        else:
                            captions = []
                            for persona_idx in range(NUM_PERSONAS):
                                caption = caption_generator.generate_all_captions(
                                    first_valid_frame, first_valid_mask, persona_idx
                                )
                                captions.append(caption)
                            caption_cache[cache_key] = captions
                        
                        total += 1
                        if total % 50 == 0:
                            print(f"  SA-V: {total} videos captioned...")
                        
                        if max_samples and total >= max_samples:
                            return total
                    
                    except Exception:
                        continue
        except Exception:
            continue
    
    return total

def main():
    parser = argparse.ArgumentParser(description="Pre-compute captions for SA-1B and SA-V datasets")
    parser.add_argument("--output", type=str, default="data/cache/caption_cache.json",
                        help="Output path for the caption cache JSON")
    parser.add_argument("--max_sa1b", type=int, default=None,
                        help="Max SA-1B samples to caption (None = all)")
    parser.add_argument("--max_sav", type=int, default=None,
                        help="Max SA-V videos to caption (None = all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    cache_dir = os.path.dirname(args.output)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load existing cache if present (supports incremental precomputation)
    caption_cache = {}
    if os.path.exists(args.output):
        print(f"Loading existing cache from {args.output}...")
        with open(args.output, 'r') as f:
            caption_cache = json.load(f)
        print(f"  {len(caption_cache)} entries already cached.")
    
    # Load file lists
    def load_file_list(file_path):
        try:
            with open(file_path) as f:
                return [line.strip().split('\t') for line in f.readlines()[1:]]
        except FileNotFoundError:
            return []
    
    sa1b_files = load_file_list("Datasets/SA-1B_dataset.txt")
    sav_files = load_file_list("Datasets/SA-V_dataset.txt")
    
    print(f"\nFound {len(sa1b_files)} SA-1B tar files, {len(sav_files)} SA-V tar files")
    
    # Initialize caption generator (loaded once, used for all samples)
    print(f"\nLoading InstructBLIP captioning model on {args.device}...")
    caption_generator = CaptionGenerator(device=args.device)
    print("Model loaded.\n")
    
    # Process SA-1B
    if sa1b_files:
        print("=== Processing SA-1B samples ===")
        sa1b_count = precompute_sa1b_captions(
            sa1b_files, cache_dir, caption_generator, caption_cache, args.max_sa1b
        )
        print(f"  SA-1B: {sa1b_count} new samples captioned.\n")
    
    # Save intermediate (in case SA-V fails)
    with open(args.output, 'w') as f:
        json.dump(caption_cache, f)
    print(f"Intermediate save: {len(caption_cache)} total entries.")
    
    # Process SA-V
    if sav_files:
        print("\n=== Processing SA-V videos ===")
        sav_count = precompute_sav_captions(
            sav_files, cache_dir, caption_generator, caption_cache, args.max_sav
        )
        print(f"  SA-V: {sav_count} new videos captioned.\n")
    
    # Final save
    with open(args.output, 'w') as f:
        json.dump(caption_cache, f)
    
    cache_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nCaption cache saved: {args.output}")
    print(f"  Total entries: {len(caption_cache)}")
    print(f"  File size: {cache_size_mb:.1f} MB")
    print(f"\nDone. You can now run training without the InstructBLIP model.")

if __name__ == "__main__":
    main()