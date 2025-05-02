import os
import tarfile
import requests
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
import torchvision.transforms.functional as F
import json
from pycocotools import mask as coco_mask
from models.clip_model import CLIPTokenize
from .sav_utils import SAVDatasetHelper, decode_video, show_anns

def SAM_adaptive_collate(batch):
    """Handle variable-sized images by stacking as lists"""
    images, masks, texts = zip(*batch)
    return list(images), list(masks), list(texts)

class BaseTarDataset(Dataset):
    """Base class for tar-based segmentation datasets"""
    def __init__(self, root_dir, file_list, max_retries=3, verify_files=True):
        """
        Args:
            root_dir (str): Root directory for storage
            file_list (list): List of (filename, url) tuples
            transform (callable): Optional transform
            max_retries (int): Number of download retries
        """
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.max_retries = max_retries
        self.cache_dir = os.path.join(root_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if verify_files:  # New conditional
            self.available_files = self._verify_files()
        else:
            self.available_files = []

    def _download_file(self, url, dest_path):
        """Download with retries and progress tracking"""
        print(f"Attempting to download: {url}")
        for _ in range(self.max_retries):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(dest_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Successfully downloaded: {dest_path}")
                return True
            except Exception as e:
                print(f"Download failed: {str(e)}")
        return False

    def _verify_files(self):
        """Check local cache and download missing files"""
        print(f"\n=== Verifying {len(self.file_list)} files ===")
        valid_files = []
        for filename, url in self.file_list:
            dest_path = os.path.join(self.cache_dir, filename)
            
            if not os.path.exists(dest_path):
                print(f"Downloading {filename}...")
                if self._download_file(url, dest_path):
                    valid_files.append(dest_path)
            else:
                valid_files.append(dest_path)
        print(f"Verified {len(valid_files)} files, {len(self.file_list)-len(valid_files)} missing")
        return valid_files

    def _load_from_tar(self, tar_path):
        """Load images and masks from tar file"""
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            
            # SA-1B specific structure: {image_id}.jpg and {image_id}.mask.png
            # SA-V specific structure: video_id/frame_{n}.jpg and video_id/mask_{n}.png
            # Implement this in child classes
            
        #return images, masks
    
    @property
    def available_files(self):
        return self._available_files

    @available_files.setter 
    def available_files(self, value):
        self._available_files = value

class SA1BDataset(BaseTarDataset):
    """Segment Anything 1B Dataset"""
    def __init__(self, root_dir, file_list, 
                 device='cuda' if torch.cuda.is_available() else 'cpu', build_index = True, verify_files=True):
        super().__init__(root_dir, file_list, max_retries=3, verify_files=verify_files)
        self.device = device

        if build_index:
            self.samples = self._build_index()
        else:
            self.samples = []

        
        # Initialize BLIP model for caption generation
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device).eval()

    def _generate_object_caption(self, image, mask):
        """Generate caption for masked object using BLIP model"""
        # Convert to PIL Images
        image_pil = F.to_pil_image(image)
        mask_pil = F.to_pil_image(mask.float())
        
        # Crop to masked region
        bbox = mask_pil.getbbox()
        if not bbox:  # Empty mask
            return "object"
            
        cropped_image = image_pil.crop(bbox)
        
        # Generate caption
        inputs = self.processor(cropped_image, return_tensors="pt").to(self.device)
        outputs = self.caption_model.generate(**inputs, max_new_tokens=20)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption

    def _build_index(self):
        """Create index with enhanced validation using pycocotools"""
        print("\n=== Building SA-1B dataset index ===")
        samples = []
        tar_idx = 1
        for tar_path in self.available_files:
            print(f"\nProcessing tar file {tar_idx}/{len(self.available_files)}: {os.path.basename(tar_path)}")
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                json_files = [m for m in members if m.name.endswith('.json')]
                print(f"Found {len(json_files)} JSON files in archive")
            
                json_idx = 1
                for json_member in json_files:
                    print(f"\nProcessing JSON {json_idx}/{len(json_files)}: {json_member.name}")
                    try:
                        json_file = tar.extractfile(json_member)
                        data = json.load(json_file)
                        image_name = data['image']['file_name']
                    
                        image_member = next(
                            (m for m in members 
                            if m.isfile() and 
                            os.path.basename(m.name) == image_name),
                            None
                        )
                        if not image_member:
                            continue
                        
                        valid_ann_indices = []
                        print(f"Found {len(data['annotations'])} annotations")
                        for ann_idx in range(len(data['annotations'])):
                            try:
                                ann = data['annotations'][ann_idx]
                                # Validate with pycocotools
                                rle = {
                                    'counts': ann['segmentation']['counts'],
                                    'size': ann['segmentation']['size']
                                }
                                coco_mask.decode(rle)  # This performs the validation
                                valid_ann_indices.append(ann_idx)
                            except Exception as e:
                                print(f"Skipping invalid annotation {ann_idx}: {str(e)}")
                            
                        print(f"Valid annotations: {len(valid_ann_indices)}/{len(data['annotations'])}")
                        for ann_idx in valid_ann_indices:
                            samples.append((tar_path, json_member.name, ann_idx, image_member.name))
                    except Exception as e:
                        print(f"Error processing {json_member.name}: {str(e)}")
                    json_idx = json_idx + 1
            tar_idx = tar_idx + 1

        print(f"\n=== Index built. Total samples: {len(samples)} ===")
        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tar_path, json_name, ann_idx, image_path = self.samples[idx]
        
        with tarfile.open(tar_path, "r") as tar:
            # Load JSON data
            json_file = tar.extractfile(json_name)
            data = json.load(json_file)
            
            # Load image
            image_member = tar.getmember(image_path)
            image_file = tar.extractfile(image_member)
            image = Image.open(image_file).convert("RGB")
            image = self.transform(image) if self.transform else transforms.ToTensor()(image)
            
            # Process annotation using pycocotools
            ann = data['annotations'][ann_idx]
            
            # Convert segmentation to COCO-compatible format
            rle = {
                'counts': ann['segmentation']['counts'],
                'size': ann['segmentation']['size']
            }
            
            # Decode mask
            mask = coco_mask.decode(rle)
            mask = torch.from_numpy(mask).long()
            mask = mask.unsqueeze(0)
            
            # Generate caption
            with torch.no_grad():
                caption = self._generate_object_caption(image, mask)
            
            caption = CLIPTokenize(caption)
            caption = caption.squeeze(1)

            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
                
        return image, mask, caption
    
    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = value

class SAVDataset(BaseTarDataset):
    """SA-V dataset with flexible annotation handling"""
    def __init__(self, root_dir, file_list, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 caption_strategy='video', 
                 build_index=True):
        super().__init__(root_dir, file_list, max_retries=3, verify_files=True)
        self.device = device
        self.caption_strategy = caption_strategy
        self.extract_dir = os.path.join(root_dir, "sav_extracted")
        os.makedirs(self.extract_dir, exist_ok=True)
        
        # Initialize components
        self.sav_helper = SAVDatasetHelper(self.extract_dir)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device).eval()

        if build_index:
            self.samples = self._build_index()

    def _load_from_tar(self, tar_path):
        """SA-V tar processing with flexible annotation support"""
        video_id = os.path.splitext(os.path.basename(tar_path))[0]
        extract_path = os.path.join(self.extract_dir, video_id)
        
        if os.path.exists(extract_path):
            return extract_path
            
        os.makedirs(extract_path, exist_ok=True)
        
        with tarfile.open(tar_path, "r") as tar:
            # Extract all files first
            tar.extractall(path=extract_path)
            
            # Post-extraction validation
            extracted_files = set(os.listdir(extract_path))
            required = {f"{video_id}.mp4"}
            auto_file = f"{video_id}_auto.json"
            manual_file = f"{video_id}_manual.json"
            
            # Check for essential files
            if not required.issubset(extracted_files):
                raise ValueError(f"Missing video file in {tar_path}")
                
            if auto_file not in extracted_files and manual_file not in extracted_files:
                print(f"No annotations found for {video_id}, skipping")
                return None
                
        return extract_path

    def _build_index(self):
        """Build index handling mixed annotation availability"""
        samples = []
        for tar_path in self.available_files:
            video_id = os.path.splitext(os.path.basename(tar_path))[0]
            extract_path = self._load_from_tar(tar_path)
            
            if extract_path is None:  # Skip invalid extractions
                continue
                
            # Path setup with existence checks
            mp4_path = os.path.join(extract_path, f"{video_id}.mp4")
            auto_path = os.path.join(extract_path, f"{video_id}_auto.json")
            manual_path = os.path.join(extract_path, f"{video_id}_manual.json")
            
            # Determine available annotations
            annot_sources = []
            if os.path.exists(manual_path):
                annot_sources.append(('manual', manual_path))
            if os.path.exists(auto_path):
                annot_sources.append(('auto', auto_path))

            for source_type, annot_path in annot_sources:
                try:
                    with open(annot_path) as f:
                        annotations = json.load(f)
                        
                    # Validate annotation structure
                    if not self._validate_annotations(annotations):
                        continue
                        
                    # Process frames
                    frames = decode_video(mp4_path)
                    
                    for frame_idx, frame_anns in enumerate(annotations['masklet']):
                        for masklet_idx in range(len(frame_anns)):
                            samples.append({
                                'extract_path': extract_path,
                                'video_id': video_id,
                                'frame_idx': frame_idx,
                                'masklet_idx': masklet_idx,
                                'annot_source': source_type
                            })
                except Exception as e:
                    print(f"Skipping {source_type} annotations for {video_id}: {str(e)}")
        
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        extract_path = sample['extract_path']
        video_id = sample['video_id']
        
        # Load frames
        mp4_path = os.path.join(extract_path, f"{video_id}.mp4")
        frames = decode_video(mp4_path)
        
        # Load annotations
        json_path = os.path.join(extract_path, 
            f"{video_id}_{sample['annot_source']}.json")
        with open(json_path) as f:
            annotations = json.load(f)
            
        # Process masklet
        rle = annotations['masklet'][sample['frame_idx']][sample['masklet_idx']]
        mask = coco_mask.decode(rle)
        
        # Generate caption with annotation priority
        if self.caption_strategy == 'video':
            caption = self._video_level_caption(extract_path, video_id)
        else:
            caption = self._generate_caption(frames[sample['frame_idx']], mask)

        # Convert to tensors
        image_tensor = F.to_tensor(frames[sample['frame_idx']])
        mask_tensor = torch.from_numpy(mask).long()
        caption_tensor = CLIPTokenize(caption).squeeze(0)

        return image_tensor, mask_tensor, caption_tensor
    
    def _validate_annotations(self, annot_data):
        """Validate annotation file structure"""
        return all(k in annot_data for k in ['masklet', 'video_id', 'masklet_num'])

    def _video_level_caption(self, video_id):
        """New method: Generate caption from aggregated video masks"""
        mp4_path = os.path.join(self.root_dir, f"{video_id}.mp4")
        frames = decode_video(mp4_path)
        
        # Create composite mask across all frames
        composite_mask = np.zeros_like(frames[0][..., 0], dtype=bool)
        with open(os.path.join(self.root_dir, f"{video_id}_auto.json")) as f:
            annotations = json.load(f)
            
        for frame_anns in annotations['masklet']:
            for rle in frame_anns:
                composite_mask |= coco_mask.decode(rle)

        # Generate caption from composite mask
        return self._generate_caption(frames[0], composite_mask)

    def _generate_caption(self, frame, mask):
        """Unified caption generation method"""
        image_pil = Image.fromarray(frame)
        mask_pil = Image.fromarray(mask.astype(np.uint8))
        
        bbox = mask_pil.getbbox()
        if not bbox:
            return "objects in video"
            
        cropped = image_pil.crop(bbox)
        inputs = self.processor(cropped, return_tensors="pt").to(self.device)
        outputs = self.caption_model.generate(**inputs, max_new_tokens=20)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def visualize_sample(self, idx):
        """Integrated visualization using sav_utils"""
        sample = self.samples[idx]
        self.sav_helper.visualize_annotation(
            sample['video_id'],
            sample['frame_idx'],
            show_manual=(sample['annot_source'] == 'manual')
        )
    
    def _video_level_caption(self, video_dir, video_id):
        """Generate caption from all frames in video"""
        composite_mask = np.zeros((1080, 1920), dtype=bool)  # Default size
        json_path = os.path.join(video_dir, f"{video_id}_auto.json")
        
        try:
            with open(json_path) as f:
                annotations = json.load(f)
            
            # Create composite mask
            for frame_anns in annotations['masklet']:
                for rle in frame_anns:
                    composite_mask |= coco_mask.decode(rle)
        except:
            pass
            
        # Get first frame for context
        mp4_path = os.path.join(video_dir, f"{video_id}.mp4")
        frames = decode_video(mp4_path)
        
        return self._generate_caption(frames[0], composite_mask)

    
# # Usage Example
# if __name__ == "__main__":
#     # Load dataset from text files
#     def load_file_list(file_path):
#         with open(file_path) as f:
#             return [line.strip().split('\t') for line in f.readlines()[1:]]  # Skip header

#     sa1b_files = load_file_list("SA-1B_dataset copy.txt")
#     sav_files = load_file_list("SA-V_dataset copy.txt")

#     # Create datasets
#     sa1b_dataset = SA1BDataset(
#         root_dir="./data",
#         #text_processor=text_processor,
#         file_list=sa1b_files
#     )

#     sav_dataset = SAVDataset(
#         root_dir="./data",
#         #text_processor=text_processor,
#         file_list=sav_files
#     )

#     # Create combined dataset
#     combined_dataset = torch.utils.data.ConcatDataset([sa1b_dataset, sav_dataset])

#     # Create dataloader
#     dataloader = DataLoader(
#         combined_dataset,
#         batch_size=8,
#         shuffle=True,
#         num_workers=4,
#         collate_fn=SAM_adaptive_collate,
#         pin_memory=True
#     )