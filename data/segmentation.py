import os
import tarfile
import requests
from PIL import Image
import torch
from torch.utils.data import IterableDataset, Dataset
from torchvision import transforms
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torchvision.transforms.functional as F
import json
from pycocotools import mask as coco_mask
from models.clip_model import CLIPTokenize
from .sav_utils import SAVDatasetHelper
import re
import time

def SAM_adaptive_collate(batch):
    images, masks, texts = zip(*batch)
    return list(images), list(masks), list(texts)

# class CaptionGenerator:
#     def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         print("⚠️ WARNING: Using DUMMY CaptionGenerator to save memory for testing! ⚠️")
    
#     def _post_process_caption(self, caption):
#         return caption

#     def generate_all_captions(self, image, mask, index):
#         # Instantly return dummy text to bypass the expensive LLM inference
#         dummy_captions = [
#             "a descriptive search query for the object",
#             "a simple description of the red object doing something",
#             "a brief one-sentence caption for this image.",
#             "factual object label"
#         ]
        
#         # Fallback just in case an invalid index is passed
#         if index < 0 or index > 3:
#             return "invalid_persona_index"
            
#         return dummy_captions[index]

class CaptionGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Use flash_attention_2 for performance on CUDA, otherwise eager
        attn_implementation = "flash_attention_2" if self.device == 'cuda' else "eager"
        self.compute_dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32

        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            dtype=self.compute_dtype,
            attn_implementation=attn_implementation
        ).to(self.device).eval()
    
    def _post_process_caption(self, caption):
        prefixes = [
            "a photo of", "a close-up of", "it's a photo of", "the image shows",
            "the image depicts", "the image features", "the picture shows", "in this image,",
            "this image shows", "this is a photo of", "this is a picture of",
            "the subject of the image is", "the main subject of this image is",
        ]
        normalized_caption = caption.lower()
        for prefix in prefixes:
            if normalized_caption.startswith(prefix):
                caption = caption[len(prefix):].lstrip()
                break

        background_pattern = r'\s*(on|in|with|against|in front of)\s+(a|the)\s+black\s+(background|surface|area)\b'
        caption = re.sub(background_pattern, '', caption, flags=re.IGNORECASE)

        sentences = caption.split('.')
        cleaned_sentences = []
        for sentence in sentences:
            if "background is black" not in sentence.lower() and sentence.strip():
                cleaned_sentences.append(sentence)
        caption = '. '.join(cleaned_sentences)
        
        return caption.strip(" .,")

    def generate_all_captions(self, image, mask, index):
        # Convert to PIL Image and crop to the masked region's bounding box
        image_pil = F.to_pil_image(image)
        mask_pil = F.to_pil_image(mask.float())
        bbox = mask_pil.getbbox()
        if not bbox:
            return ["object"] * 4
        image_rgba = image_pil.convert("RGBA")
        cropped_image_rgba = image_rgba.crop(bbox)
        cropped_mask = mask_pil.crop(bbox)
        cropped_image_rgba.putalpha(cropped_mask)

        background = Image.new("RGB", cropped_image_rgba.size, (0, 0, 0))
        background.paste(cropped_image_rgba, mask=cropped_image_rgba.split()[3])
        final_masked_image_rgb = background


        with torch.no_grad():
            if index == 0:
                # Persona 1: The Search Query Generator
                prompt = "Describe the subject of the image using a short, descriptive search query."

            elif index == 1:
                # Persona 2: The Object-Focused Labeler
                prompt = "A simple description of the subject, including its main color and what it is doing."

            elif index == 2:
                # Persona 3: The Natural Language Captioner
                prompt = "Write a brief, one-sentence caption for this image."

            elif index == 3:
                # Persona 4: The Literal Labeler
                prompt = "A factual label for the subject in the image."

            else:
                # Handle invalid index case
                return "invalid_persona_index"

            if prompt:
                inputs = self.processor(
                    images=final_masked_image_rgb, text=prompt, return_tensors="pt"
                ).to(self.device, self.compute_dtype)
                outputs = self.model.generate(
                    **inputs, 
                    max_length=75, 
                    min_length=1, 
                    num_beams=3,
                    repetition_penalty=1.5
                )
                caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                return self._post_process_caption(caption)

def download_tar_file(url, dest_path, max_retries=3):
    tmp_dest_path = dest_path + ".tmp"
    for _ in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=20)
            response.raise_for_status()
            expected_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(tmp_dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
            
            if expected_size > 0 and downloaded_size != expected_size:
                raise Exception(f"Incomplete download")
            
            os.rename(tmp_dest_path, dest_path)
            return True
        except Exception:
            if os.path.exists(tmp_dest_path):
                try: os.remove(tmp_dest_path)
                except OSError: pass
            time.sleep(1)
    return False

class StreamingBaseTarDataset(IterableDataset):
    def __init__(self, file_list, cache_dir, split='train', val_tar_count=1, val_sample_count=None, max_retries=3):
        super().__init__()
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.split = split
        
        if split == 'val':
            self.file_list = file_list[:val_tar_count]
        else:
            self.file_list = file_list[val_tar_count:]

    def _get_worker_shard(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        dist_world_size = int(os.environ.get("WORLD_SIZE", 1)) if "LOCAL_RANK" in os.environ else 1
        dist_rank = int(os.environ.get("RANK", 0)) if "LOCAL_RANK" in os.environ else 0

        total_shards = num_workers * dist_world_size
        global_worker_id = (dist_rank * num_workers) + worker_id

        shard = [f for i, f in enumerate(self.file_list) if i % total_shards == global_worker_id]
        return shard, global_worker_id

class SA1BDataset(StreamingBaseTarDataset):
    def __init__(self, file_list, cache_dir, device='cpu', split='train', val_tar_count=1, val_sample_count=None):
        super().__init__(file_list, cache_dir, split, val_tar_count=val_tar_count)
        self.device = device
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.caption_generator = None 

    def _process_tar(self, tar_path):
        if self.caption_generator is None:
            self.caption_generator = CaptionGenerator(device=self.device)

        try:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                json_files = [m for m in members if m.name.endswith('.json')]
                
                for json_member in json_files:
                    try:
                        json_file = tar.extractfile(json_member)
                        data = json.load(json_file)
                        image_name = data['image']['file_name']
                        
                        image_member = next((m for m in members if m.isfile() and os.path.basename(m.name) == image_name), None)
                        if not image_member: continue
                            
                        image = Image.open(tar.extractfile(image_member)).convert("RGB")
                        image_tensor = self.transform(image).unsqueeze(0) # [1, C, H, W]
                        
                        for ann_idx, ann in enumerate(data['annotations']):
                            try:
                                rle = {'counts': ann['segmentation']['counts'], 'size': ann['segmentation']['size']}
                                mask = coco_mask.decode(rle)
                                mask_tensor = torch.from_numpy(mask).long().unsqueeze(0) # [1, H, W]
                                
                                for caption_idx in range(4):
                                    caption = self.caption_generator.generate_all_captions(
                                        image_tensor.squeeze(0), mask_tensor.squeeze(0), caption_idx
                                    )
                                    caption_tokens = CLIPTokenize(caption) # Pre-unsqueezed sequence [1, 77]
                                    yield image_tensor, mask_tensor, caption_tokens
                            except Exception: continue
                    except Exception: continue
        except Exception: return

    def __iter__(self):
        shard, global_worker_id = self._get_worker_shard()
        
        for url_info in shard:
            filename, url = url_info[0], url_info[1]
            temp_tar_path = os.path.join(self.cache_dir, f"sa1b_w{global_worker_id}_{filename}")
            
            if not os.path.exists(temp_tar_path):
                success = download_tar_file(url, temp_tar_path)
            else: success = True

            if success:
                yield from self._process_tar(temp_tar_path)
                try: os.remove(temp_tar_path) 
                except OSError: pass

class SAVDataset(StreamingBaseTarDataset):
    def __init__(self, file_list, cache_dir, device='cpu', split='train', val_tar_count=1, val_sample_count=None):
        super().__init__(file_list, cache_dir, split, val_tar_count=val_tar_count)
        self.device = device
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.caption_generator = None
        self.sav_helper = SAVDatasetHelper(os.path.dirname(cache_dir)) 

    def _process_tar(self, tar_path, global_worker_id):
        if self.caption_generator is None:
            self.caption_generator = CaptionGenerator(device=self.device)

        try:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                mp4_files = [m for m in members if m.name.endswith('.mp4') and m.isfile()]
                json_files = [m for m in members if m.name.endswith('.json') and m.isfile()]
                
                for video_member in mp4_files:
                    try:
                        video_id = os.path.splitext(os.path.basename(video_member.name))[0]
                        annot_member = next((m for m in json_files if f"{video_id}_manual.json" in m.name or f"{video_id}_auto.json" in m.name), None)
                        if not annot_member: continue
                        
                        temp_video_path = os.path.join(self.cache_dir, f"tmp_vid_{global_worker_id}_{video_id}.mp4")
                        with open(temp_video_path, 'wb') as f:
                            f.write(tar.extractfile(video_member).read())
                            
                        frames = self.sav_helper.read_frames(temp_video_path)
                        os.remove(temp_video_path) 
                        
                        if not frames: continue
                        frames_tensor = torch.stack([self.transform(Image.fromarray(f)) for f in frames]).unsqueeze(0) # [1, T, C, H, W]

                        data = json.load(tar.extractfile(annot_member))
                        
                        first_valid_frame, first_valid_mask = None, None
                        masks = []
                        for frame_idx in range(len(frames_tensor.squeeze(0))):
                            frame_masks = []
                            if data and data.get('masklet') and len(data['masklet']) > frame_idx:
                                for ann in data['masklet'][frame_idx]:
                                    rle = {'counts': ann['counts'], 'size': ann['size']}
                                    frame_masks.append(torch.from_numpy(coco_mask.decode(rle)).long())
                            if frame_masks:
                                frame_mask = torch.stack(frame_masks).any(dim=0).float()
                                masks.append(frame_mask)
                                if first_valid_frame is None:
                                    first_valid_frame = frames_tensor.squeeze(0)[frame_idx]
                                    first_valid_mask = frame_mask
                            else:
                                masks.append(torch.zeros_like(frames_tensor.squeeze(0)[0].mean(dim=0)))
                        masks_tensor = torch.stack(masks).unsqueeze(0) # [1, T, H, W]

                        if first_valid_frame is not None:
                            for caption_idx in range(4):
                                caption = self.caption_generator.generate_all_captions(first_valid_frame, first_valid_mask, caption_idx)
                                caption_tokens = CLIPTokenize(caption) # [1, seq_len]
                                yield frames_tensor, masks_tensor, caption_tokens
                        else:
                            yield frames_tensor, masks_tensor, CLIPTokenize("object")

                    except Exception: continue
        except Exception: return

    def __iter__(self):
        shard, global_worker_id = self._get_worker_shard()
        for url_info in shard:
            filename, url = url_info[0], url_info[1]
            temp_tar_path = os.path.join(self.cache_dir, f"sav_w{global_worker_id}_{filename}")
            
            if not os.path.exists(temp_tar_path): success = download_tar_file(url, temp_tar_path)
            else: success = True

            if success:
                yield from self._process_tar(temp_tar_path, global_worker_id)
                try: os.remove(temp_tar_path) 
                except OSError: pass

class StaticSA1BDataset(Dataset):
    def __init__(self, file_list, cache_dir, device='cpu', split='val', val_tar_count=1, val_sample_count=None):
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.file_list = file_list[:val_tar_count]
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        self.available_files = []
        for filename, url in self.file_list:
            dest_path = os.path.join(self.cache_dir, filename)
            if not os.path.exists(dest_path):
                download_tar_file(url, dest_path)
            if os.path.exists(dest_path):
                self.available_files.append(dest_path)
                
        self.samples = self._build_index()
        
        if val_sample_count is not None and val_sample_count > 0:
            self.samples = self.samples[:val_sample_count]
            
        self.caption_generator = CaptionGenerator(device=self.device)
        
    def _build_index(self):
        samples = []
        for tar_path in self.available_files:
            try:
                with tarfile.open(tar_path, "r") as tar:
                    members = tar.getmembers()
                    json_files = [m for m in members if m.name.endswith('.json')]
                    for json_member in json_files:
                        try:
                            json_file = tar.extractfile(json_member)
                            data = json.load(json_file)
                            image_name = data['image']['file_name']
                            image_member = next((m for m in members if m.isfile() and os.path.basename(m.name) == image_name), None)
                            if not image_member: continue
                            for ann_idx in range(len(data['annotations'])):
                                samples.append((tar_path, json_member.name, ann_idx, image_member.name))
                        except Exception: pass
            except Exception: pass
        return samples

    def __len__(self):
        return len(self.samples) * 4

    def __getitem__(self, idx):
        original_sample_idx = idx // 4
        caption_idx = idx % 4
        tar_path, json_name, ann_idx, image_path = self.samples[original_sample_idx]
        
        with tarfile.open(tar_path, "r") as tar:
            data = json.load(tar.extractfile(json_name))
            image = Image.open(tar.extractfile(image_path)).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0) # [1, C, H, W]
            
            ann = data['annotations'][ann_idx]
            rle = {'counts': ann['segmentation']['counts'], 'size': ann['segmentation']['size']}
            mask = torch.from_numpy(coco_mask.decode(rle)).long().unsqueeze(0) # [1, H, W]
            
            caption = self.caption_generator.generate_all_captions(image_tensor.squeeze(0), mask.squeeze(0), caption_idx)
            caption_tokens = CLIPTokenize(caption) # [1, seq_len]
                
        return image_tensor, mask, caption_tokens

class StaticSAVDataset(Dataset):
    def __init__(self, file_list, cache_dir, device='cpu', split='val', val_tar_count=1, val_sample_count=None):
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.file_list = file_list[:val_tar_count]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.sav_helper = SAVDatasetHelper(os.path.dirname(cache_dir))
        
        self.available_files = []
        for filename, url in self.file_list:
            dest_path = os.path.join(self.cache_dir, filename)
            if not os.path.exists(dest_path):
                download_tar_file(url, dest_path)
            if os.path.exists(dest_path):
                self.available_files.append(dest_path)
                
        self.samples = self._build_index()
        
        if val_sample_count is not None and val_sample_count > 0:
            self.samples = self.samples[:val_sample_count]
            
        self.caption_generator = CaptionGenerator(device=self.device)

    def _build_index(self):
        samples = []
        for tar_path in self.available_files:
            try:
                with tarfile.open(tar_path, "r") as tar:
                    members = tar.getmembers()
                    mp4_files = [m for m in members if m.name.endswith('.mp4') and m.isfile()]
                    json_files = [m for m in members if m.name.endswith('.json') and m.isfile()]
                    for video_member in mp4_files:
                        video_id = os.path.splitext(os.path.basename(video_member.name))[0]
                        annot_member = next((m for m in json_files if f"{video_id}_manual.json" in m.name or f"{video_id}_auto.json" in m.name), None)
                        if annot_member:
                            samples.append((tar_path, video_id, annot_member.name))
            except Exception: pass
        return samples

    def __len__(self):
        return len(self.samples) * 4
        
    def __getitem__(self, idx):
        original_sample_idx = idx // 4
        caption_idx = idx % 4
        
        tar_path, video_id, annot_name = self.samples[original_sample_idx]
        with tarfile.open(tar_path, "r") as tar:
            video_path = f"{video_id}.mp4"
            video_member = next((m for m in tar.getmembers() if m.isfile() and m.name.endswith(video_path)), None)
            
            temp_video_path = os.path.join(self.cache_dir, f"static_tmp_{video_id}_{idx}.mp4")
            with open(temp_video_path, 'wb') as f:
                f.write(tar.extractfile(video_member).read())
                
            frames = self.sav_helper.read_frames(temp_video_path)
            os.remove(temp_video_path)
            
            frames_tensor = torch.stack([self.transform(Image.fromarray(f)) for f in frames]).unsqueeze(0) # [1, T, C, H, W]
            
            data = json.load(tar.extractfile(annot_name))
            
            masks = []
            first_valid_frame, first_valid_mask = None, None
            for frame_idx in range(len(frames_tensor.squeeze(0))):
                frame_masks = []
                if data and data.get('masklet') and len(data['masklet']) > frame_idx:
                    for ann in data['masklet'][frame_idx]:
                        rle = {'counts': ann['counts'], 'size': ann['size']}
                        frame_masks.append(torch.from_numpy(coco_mask.decode(rle)).long())
                if frame_masks:
                    frame_mask = torch.stack(frame_masks).any(dim=0).float()
                    masks.append(frame_mask)
                    if first_valid_frame is None:
                        first_valid_frame = frames_tensor.squeeze(0)[frame_idx]
                        first_valid_mask = frame_mask
                else:
                    masks.append(torch.zeros_like(frames_tensor.squeeze(0)[0].mean(dim=0)))
            
            masks_tensor = torch.stack(masks).unsqueeze(0) # [1, T, H, W]
            
            if first_valid_frame is not None:
                caption = self.caption_generator.generate_all_captions(first_valid_frame, first_valid_mask, caption_idx)
                caption_tokens = CLIPTokenize(caption)
            else:
                caption_tokens = CLIPTokenize("object")
                
        return frames_tensor, masks_tensor, caption_tokens

# def test_captioning_models(image_path: str, mask_path: str):
#     print(f"🧪 Testing with:\n  Image: {image_path}\n  Mask:  {mask_path}\n")

#     ## 1. Load and Preprocess Image and Mask
#     try:
#         # Load the image and mask from the specified paths
#         input_image = Image.open(image_path).convert("RGB")
#         mask_image = Image.open(mask_path).convert("L")  # Convert mask to grayscale
#     except FileNotFoundError as e:
#         print(f"❌ Error: Could not find a file. Please check your paths.\n{e}")
#         return

#     # Define a transform to convert PIL images to PyTorch tensors
#     to_tensor = transforms.ToTensor()
#     image_tensor = to_tensor(input_image)
#     mask_tensor = to_tensor(mask_image)

#     # Validate that image and mask dimensions match
#     if image_tensor.shape[1:] != mask_tensor.shape[1:]:
#         print("❌ Error: Image and mask dimensions must be the same.")
#         print(f"Image shape: {image_tensor.shape}, Mask shape: {mask_tensor.shape}")
#         return

#     ## 2. Initialize the Caption Generator
#     # This step loads all five large language models and may require significant
#     # GPU memory and time.
#     print("🧠 Initializing captioning models... (This may take a moment)")
#     try:
#         caption_generator = CaptionGenerator()
#         print("✅ Models initialized successfully.")
#     except Exception as e:
#         print(f"❌ Error during model initialization: {e}")
#         return

#     ## 4. Display Results
#     model_names = ["The Search Query Generator", "The Object-Focused Labeler", "The Natural Language Captioner", "The Literal Labeler"]
    
#     print("\n--- 📸 Generated Captions ---")
#     for i in range(len(model_names)):
#         try:
#             caption = caption_generator.generate_all_captions(image_tensor, mask_tensor, i)
#             print(f"🔹 **{model_names[i]}**: {caption}")
#         except Exception as e:
#             print(f"\n❌ An error occurred during caption generation: {e}")
        
#     print("----------------------------\n")

# test_captioning_models(image_path=image_file, mask_path=mask_file)