# data/referring.py
import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO
from models.clip_model import CLIPTokenize


# ======================== gRefCOCO ========================

class gRefCOCODataset(Dataset):
    """
    Generalized Referring Expression Segmentation dataset.
    
    Supports:
      - Single-target: expression refers to one object
      - Multi-target:  expression refers to multiple objects (masks unioned)
      - No-target:     expression refers to nothing present (all-zeros mask)
    
    Each sample is one (image, expression, mask) triplet.
    """

    def __init__(self, 
                 grefcoco_dir="data/grefcoco",
                 image_dir="data/images/train2014",
                 split="train",
                 image_size=224):
        super().__init__()
        self.image_dir = image_dir
        self.split = split
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

        # Load COCO instance annotations (for segmentation masks)
        instances_path = os.path.join(grefcoco_dir, "instances.json")
        print(f"[gRefCOCO] Loading COCO instances from {instances_path}...")
        self.coco = COCO(instances_path)

        # Load gRefCOCO referring expressions
        grefs_path = os.path.join(grefcoco_dir, "grefs(unc).json")
        print(f"[gRefCOCO] Loading gRefCOCO annotations from {grefs_path}...")
        with open(grefs_path, 'r') as f:
            all_refs = json.load(f)

        # Filter by split and build sample list
        # Each ref has: image_id, ann_id (list), sentences (list of {sent, ...}), split
        self.samples = []
        for ref in all_refs:
            if ref.get('split') != split:
                continue
            image_id = ref['image_id']
            ann_ids = ref.get('ann_id', [])
            sentences = ref.get('sentences', [])
            
            for sent_info in sentences:
                sentence = sent_info.get('sent', sent_info) if isinstance(sent_info, dict) else str(sent_info)
                self.samples.append({
                    'image_id': image_id,
                    'ann_ids': ann_ids if isinstance(ann_ids, list) else [ann_ids],
                    'sentence': sentence,
                })

        print(f"[gRefCOCO] Split '{split}': {len(self.samples)} samples "
              f"({sum(1 for s in self.samples if len(s['ann_ids']) == 0)} no-target, "
              f"{sum(1 for s in self.samples if len(s['ann_ids']) > 1)} multi-target)")

    def __len__(self):
        return len(self.samples)

    def _get_mask(self, ann_ids, image_info):
        """Build a binary mask from annotation IDs. Empty ann_ids → all-zeros mask."""
        h, w = image_info['height'], image_info['width']
        
        if not ann_ids:
            # No-target expression: return empty mask
            return torch.zeros(h, w, dtype=torch.long)
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for ann_id in ann_ids:
            try:
                ann = self.coco.anns[ann_id]
                if isinstance(ann['segmentation'], list):
                    # Polygon format → convert to RLE → decode
                    rle = coco_mask_util.frPyObjects(ann['segmentation'], h, w)
                    rle = coco_mask_util.merge(rle)
                elif isinstance(ann['segmentation'], dict):
                    # RLE format already
                    rle = ann['segmentation']
                else:
                    continue
                mask = coco_mask_util.decode(rle)
                combined_mask = np.maximum(combined_mask, mask)
            except (KeyError, Exception):
                continue
        
        return torch.from_numpy(combined_mask).long()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        
        # Load image
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        # Build mask (union of all target annotations, or all-zeros for no-target)
        mask = self._get_mask(sample['ann_ids'], image_info)
        
        # Apply spatial transforms to image
        image_tensor = self.transform(image).unsqueeze(0)  # [1, C, H, W]
        
        # Resize mask to match image_tensor spatial dims
        _, _, th, tw = image_tensor.shape
        mask_resized = transforms.functional.resize(
            mask.unsqueeze(0).unsqueeze(0).float(),  # [1, 1, H, W]
            [th, tw],
            interpolation=transforms.InterpolationMode.NEAREST
        ).squeeze(0).long()  # [1, H, W]
        
        # Tokenize the referring expression directly (no captioning model needed)
        caption_tokens = CLIPTokenize(sample['sentence'])  # [1, 77]
        
        return image_tensor, mask_resized, caption_tokens


# ======================== Ref-YouTube-VOS ========================

class RefYouTubeVOSDataset(Dataset):
    """
    Referring Video Object Segmentation dataset based on YouTube-VOS.
    
    Each sample is a (video_clip, per-frame_masks, expression) triplet.
    Masks are palette-encoded PNGs where pixel value = object_id.
    Frames without the referred object naturally have all-zeros masks.
    """

    def __init__(self,
                 root_dir="data/ref-youtube-vos",
                 split="train",
                 max_frames=8,
                 frame_sample_rate=1):
        super().__init__()
        self.split = split
        self.max_frames = max_frames
        self.frame_sample_rate = frame_sample_rate
        
        self.jpeg_dir = os.path.join(root_dir, split, "JPEGImages")
        self.annot_dir = os.path.join(root_dir, split, "Annotations")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load meta_expressions
        meta_path = os.path.join(root_dir, "meta_expressions", split, "meta_expressions.json")
        print(f"[RefYTVOS] Loading meta expressions from {meta_path}...")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Build sample list: one entry per (video, expression) pair
        self.samples = []
        for video_id, video_info in meta['videos'].items():
            frames = sorted(video_info.get('frames', []))
            expressions = video_info.get('expressions', {})
            
            for exp_id, exp_info in expressions.items():
                expression = exp_info.get('exp', '')
                # obj_id can be a single int or list
                obj_ids = exp_info.get('obj_id', [])
                if isinstance(obj_ids, (int, str)):
                    obj_ids = [int(obj_ids)]
                else:
                    obj_ids = [int(oid) for oid in obj_ids]
                
                self.samples.append({
                    'video_id': video_id,
                    'frames': frames,
                    'expression': expression,
                    'obj_ids': obj_ids,
                })

        print(f"[RefYTVOS] Split '{split}': {len(self.samples)} expression-video pairs "
              f"across {len(meta['videos'])} videos")

    def __len__(self):
        return len(self.samples)

    def _sample_frames(self, frame_list):
        """Subsample frames if the video is longer than max_frames."""
        if self.frame_sample_rate > 1:
            frame_list = frame_list[::self.frame_sample_rate]
        
        if len(frame_list) <= self.max_frames:
            return frame_list
        
        # Uniformly sample max_frames from the available frames
        indices = np.linspace(0, len(frame_list) - 1, self.max_frames, dtype=int)
        return [frame_list[i] for i in indices]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['video_id']
        obj_ids = sample['obj_ids']
        
        # Select frame subset
        frame_names = self._sample_frames(sample['frames'])
        
        frames = []
        masks = []
        
        for frame_name in frame_names:
            # Load frame
            frame_path = os.path.join(self.jpeg_dir, video_id, f"{frame_name}.jpg")
            frame = Image.open(frame_path).convert("RGB")
            frame_tensor = self.transform(frame)  # [C, H, W]
            frames.append(frame_tensor)
            
            # Load annotation mask (palette-encoded PNG)
            mask_path = os.path.join(self.annot_dir, video_id, f"{frame_name}.png")
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path)
                mask_array = np.array(mask_img)  # pixel values = object IDs
                
                # Build binary mask: 1 where any referred object is present
                binary_mask = np.zeros_like(mask_array, dtype=np.uint8)
                for oid in obj_ids:
                    binary_mask = np.maximum(binary_mask, (mask_array == oid).astype(np.uint8))
                
                masks.append(torch.from_numpy(binary_mask).long())
            else:
                # Frame without annotation → zeros (object may not be visible)
                h, w = frame_tensor.shape[1], frame_tensor.shape[2]
                masks.append(torch.zeros(h, w, dtype=torch.long))
        
        # Stack into video tensors
        frames_tensor = torch.stack(frames).unsqueeze(0)  # [1, T, C, H, W]
        masks_tensor = torch.stack(masks).unsqueeze(0)    # [1, T, H, W]
        
        # Tokenize the referring expression
        caption_tokens = CLIPTokenize(sample['expression'])  # [1, 77]
        
        return frames_tensor, masks_tensor, caption_tokens


# ======================== Collate & Loader Helpers ========================

def referring_collate(batch):
    """Same interface as SAM_adaptive_collate: returns lists of variable-size tensors."""
    images, masks, texts = zip(*batch)
    return list(images), list(masks), list(texts)


def get_grefcoco_loaders(batch_size, grefcoco_dir="data/grefcoco", image_dir="data/images/train2014",
                         num_workers=4, image_size=224):
    """Create train and val DataLoaders for gRefCOCO."""
    train_dataset = gRefCOCODataset(
        grefcoco_dir=grefcoco_dir, image_dir=image_dir, split="train", image_size=image_size
    )
    val_dataset = gRefCOCODataset(
        grefcoco_dir=grefcoco_dir, image_dir=image_dir, split="val", image_size=image_size
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=referring_collate, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=referring_collate, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def get_refytvos_loaders(batch_size, root_dir="data/ref-youtube-vos",
                         num_workers=4, max_frames=8):
    """Create train DataLoader for Ref-YouTube-VOS (val annotations not publicly available)."""
    train_dataset = RefYouTubeVOSDataset(
        root_dir=root_dir, split="train", max_frames=max_frames
    )
    
    # Split train into train/val since competition val annotations aren't public
    total = len(train_dataset)
    val_size = min(500, total // 10)
    train_size = total - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        collate_fn=referring_collate, num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        collate_fn=referring_collate, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader