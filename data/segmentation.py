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

def SAM_adaptive_collate(batch):
    """Handle variable-sized images by stacking as lists"""
    images, masks, texts = zip(*batch)
    return list(images), list(masks), list(texts)

def SAV_adaptive_collate(batch):
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
    """Segment Anything Video Dataset"""
    def __init__(self, root_dir, file_list, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(root_dir, file_list, max_retries=3)
        self.device = device
        self.caption_cache = {}
        self.samples = self._build_index()
        
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
        """Index video sequences using first frame's mask for caption generation"""
        print("\n=== Building SA-V dataset index ===")
        samples = []
        tar_idx = 1
        for tar_path in self.available_files:
            print(f"\nProcessing tar file {tar_idx}/{len(self.available_files)}: {os.path.basename(tar_path)}")
            with tarfile.open(tar_path, "r") as tar:
                all_members = tar.getmembers()
                member_names = {m.name: m for m in all_members}
            
                # Process video directories
                video_dirs = [m for m in all_members if m.isdir()]

                print(f"Found {len(video_dirs)} video directories")
                vid_idx = 1
                for video_dir in video_dirs:
                    print(f"\nProcessing video {vid_idx}/{len(video_dirs)}: {video_dir.name}")
                    video_id = video_dir.name
                    frames = sorted([m for m in tar.getmembers() 
                                   if m.name.startswith(video_id) 
                                   and "frame_" in m.name])
                    masks = sorted([m for m in tar.getmembers() 
                                  if m.name.startswith(video_id)
                                  and "mask_" in m.name])

                    print(f"Found {len(frames)} frames and {len(masks)} masks")
                    if not masks:
                        continue  # Skip videos without masks

                    # Get first frame with mask
                    first_frame = frames[0]
                    first_mask = masks[0]
                    
                    # Check for existing caption
                    caption_path = os.path.join(video_id, "caption.txt")
                    try:
                        caption_file = tar.extractfile(caption_path)
                        caption = caption_file.read().decode('utf-8').strip()
                    except KeyError:
                        # Generate caption from first frame
                        frame_file = tar.extractfile(first_frame)
                        mask_file = tar.extractfile(first_mask)
                        
                        image = Image.open(frame_file).convert("RGB")
                        mask = Image.open(mask_file)
                        caption = self._generate_object_caption(image, mask)
                        
                        # Cache generated caption for all frames
                        self.caption_cache[video_id] = caption

                    # Create samples for all frames with cached caption
                    for frame, mask in zip(frames, masks):
                        samples.append((tar_path, frame.name, mask.name, video_id))
                    vid_idx = vid_idx + 1
            tar_idx = tar_idx + 1
        print(f"\n=== Index built. Total video frames: {len(samples)} ===")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        print(f"\nLoading video frame {idx+1}/{len(self)}")
        tar_path, frame_path, mask_path, video_id = self.samples[idx]
        
        with tarfile.open(tar_path, "r") as tar:
            image_file = tar.extractfile(frame_path)
            mask_file = tar.extractfile(mask_path)
            
            image = Image.open(image_file).convert("RGB")
            mask = Image.open(mask_file)
            
            # Retrieve cached caption or use generated
            caption = self.caption_cache.get(video_id, "object in video")


        caption = CLIPTokenize(caption)
        caption = caption.squeeze(0)
            
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(np.array(mask)).long()

        print(f"Using caption: {caption[:50]}...")
            
        return image, mask, caption

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