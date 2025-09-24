import os
import tarfile
import requests
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq,
    Florence2ForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM,
    LlavaForConditionalGeneration
)
import torchvision.transforms.functional as F
import json
from pycocotools import mask as coco_mask
from models.clip_model import CLIPTokenize
from .sav_utils import SAVDatasetHelper
import random

# pip install pytorchvideo

def SAM_adaptive_collate(batch):
    """Handle variable-sized images by stacking as lists"""
    images, masks, texts = zip(*batch)
    return list(images), list(masks), list(texts)

class CaptionGenerator:
    """A helper class to centralize all captioning models and logic."""
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("=== Initializing Captioning Models ===")

        # 1. BLIP Model
        self.processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model_blip = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            attn_implementation="eager"
        ).to(self.device).eval()

        # 2. PaliGemma Model
        self.processor_paligemma = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
        self.model_paligemma = AutoModelForVision2Seq.from_pretrained(
            "google/paligemma-3b-pt-224",
            torch_dtype=torch.float32,
            attn_implementation="eager"
        ).to(self.device).eval()

        # 3. Florence-2 Pipeline
        self.processor_florence2 = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        self.model_florence2 = Florence2ForConditionalGeneration.from_pretrained(
            "microsoft/Florence-2-base",
            # ✅ VERIFY THIS LINE IS CORRECT
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device).eval()

        # 4. Llava-1.5 Model
        self.model_llava_name = "llava-hf/llava-1.5-7b-hf"
        self.processor_llava = AutoProcessor.from_pretrained(self.model_llava_name)
        self.model_llava = LlavaForConditionalGeneration.from_pretrained(
            self.model_llava_name,
            torch_dtype=torch.float32, 
            attn_implementation="eager"
        ).to(self.device).eval()

    def _parse_output(self, output):
        """Helper to safely extract generated text from pipeline output."""
        if output and isinstance(output, list) and 'generated_text' in output[0]:
            return output[0]['generated_text']
        return "caption generation failed"
    
    def generate_all_captions(self, image, mask):
        """
        Generates a single caption for a masked object from a random choice of models.
        """
        # Convert to PIL Image and crop to the masked region's bounding box
        image_pil = F.to_pil_image(image)
        mask_pil = F.to_pil_image(mask.float())
        bbox = mask_pil.getbbox()
        if not bbox:
            return ["object"] * 4
        cropped_image = image_pil.crop(bbox)

        captions = []
        with torch.no_grad():
            # Generate captions using each pipeline
            # 1. BLIP Caption
            print("Starting BLIP Caption")
            inputs_blip = self.processor_blip(cropped_image, return_tensors="pt").to(self.device)
            outputs_blip = self.model_blip.generate(**inputs_blip, max_new_tokens=20)
            captions.append(self.processor_blip.decode(outputs_blip[0], skip_special_tokens=True))
            print("Finished BLIP caption")

            # 2. PaliGemma Caption
            print("Starting PaliGemma Caption")
            prompt_paligemma = "<image>\nDescribe this image"
            inputs_paligemma = self.processor_paligemma(
                text=prompt_paligemma, 
                images=cropped_image, 
                return_tensors="pt"
            ).to(self.device)

            outputs_paligemma = self.model_paligemma.generate(**inputs_paligemma, max_new_tokens=50)
            paligemma_caption = self.processor_paligemma.decode(outputs_paligemma[0], skip_special_tokens=True)
            # The output might still contain the prompt, so clean it up.
            paligemma_caption = paligemma_caption.replace(prompt_paligemma, '').strip()
            captions.append(paligemma_caption)
            print("Finished PaliGemma caption")
            
            # 3. Florence-2 Caption
            print("Starting Florence-2 Caption")
            Florence_prompt = "<CAPTION>"
            
            # ✅ VERIFY THIS BLOCK IS CORRECT
            inputs_florence2 = self.processor_florence2(
                text=Florence_prompt, images=cropped_image, return_tensors="pt"
            )

            inputs_florence2 = {key: val.to(self.device) for key, val in inputs_florence2.items()}
            inputs_florence2["pixel_values"] = inputs_florence2["pixel_values"].to(self.model_florence2.dtype)

            outputs = self.model_florence2.generate(
                **inputs_florence2,
                max_new_tokens=50,
                num_beams=3
            )
            generated_text = self.processor_florence2.batch_decode(outputs, skip_special_tokens=True)[0]
            parsed_florence = generated_text.replace(Florence_prompt, "").strip()
            captions.append(parsed_florence)
            print("Finished Florence 2 caption")

            # 4. Llava-1.5 Caption
            print("Starting Llava-1.5 Caption")
            prompt_llava = "USER: <image>\nDescribe this image. ASSISTANT:"
            inputs_llava = self.processor_llava(text=prompt_llava, images=cropped_image, return_tensors="pt").to(self.device)
            outputs_llava = self.model_llava.generate(**inputs_llava, max_new_tokens=128, do_sample=True, temperature=0.2)
            llava_caption = self.processor_llava.batch_decode(outputs_llava, skip_special_tokens=True)[0]
            # It's better to split by ASSISTANT: to get only the model's response
            if "ASSISTANT:" in llava_caption:
                 llava_caption = llava_caption.split("ASSISTANT:")[1].strip()
            else:
                 llava_caption = llava_caption.replace(prompt_llava, "").strip() # Fallback
            captions.append(llava_caption)
            print("Finished Llava-1.5 caption")
            
        return captions

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
        
        if verify_files:
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
            random.shuffle(self.samples)
        else:
            self.samples = []
        
        # Initialize models for caption generation
        self.caption_generator = CaptionGenerator(device=self.device)

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
        return len(self.samples)*4

    def __getitem__(self, idx):
        # Map the flat index to an original sample index and a caption index
        original_sample_idx = idx // 4
        caption_idx = idx % 4
        
        tar_path, json_name, ann_idx, image_path = self.samples[original_sample_idx]
        
        with tarfile.open(tar_path, "r") as tar:
            json_file = tar.extractfile(json_name)
            data = json.load(json_file)
            
            image_member = tar.getmember(image_path)
            image_file = tar.extractfile(image_member)
            image = Image.open(image_file).convert("RGB")
            image = self.transform(image) if self.transform else transforms.ToTensor()(image)
            
            ann = data['annotations'][ann_idx]
            rle = {'counts': ann['segmentation']['counts'], 'size': ann['segmentation']['size']}
            mask = coco_mask.decode(rle)
            mask = torch.from_numpy(mask).long().unsqueeze(0)
            
            # Generate all 5 captions and select the one for this index
            all_captions = self.caption_generator.generate_all_captions(image, mask)
            caption = all_captions[caption_idx]
            
            caption = CLIPTokenize(caption).squeeze(1)
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

    def __init__(self, root_dir, file_list,
                 device='cuda' if torch.cuda.is_available() else 'cpu', build_index=True,
                 verify_files=True):
        super().__init__(root_dir, file_list, max_retries=3, verify_files=verify_files)
        self.device = device
        self.sav_helper = SAVDatasetHelper(root_dir)
        self.transform = transforms.Compose([transforms.ToTensor()])

        if build_index:
            self.samples = self._build_index()
            random.shuffle(self.samples)
        else:
            self.samples = []

        # Initialize models for caption generation
        self.caption_generator = CaptionGenerator(device=self.device)

    def _build_index(self):
        """Create index for SA-V dataset by iterating through video files"""

        print("\n=== Building SA-V dataset index ===")
        samples = []
        tar_idx = 1
        for tar_path in self.available_files:
            print(f"\nProcessing tar file {tar_idx}/{len(self.available_files)}: {os.path.basename(tar_path)}")
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                #print(f"Tar file members: {[m.name for m in members]}")
                mp4_files = [m for m in members if m.name.endswith('.mp4') and m.isfile()]
                #print(f"Found MP4 files: {[f.name for f in mp4_files]}")
                json_files = [m for m in members if m.name.endswith('.json') and m.isfile()]
                #print(f"Found JSON files: {[f.name for f in json_files]}")

                for video_member in mp4_files:
                    video_id = os.path.splitext(os.path.basename(video_member.name))[0]
                    print(f"\nProcessing video ID: {video_id}")
                    try:
                        # Check for annotation files (manual preferred)
                        manual_annot_name = f"{video_id}_manual.json"
                        auto_annot_name = f"{video_id}_auto.json"

                        annot_member = None
                        for json_member in json_files:
                            parts = json_member.name.split('/')
                            if len(parts) > 2:  # Ensure there are at least 3 parts
                                path_after_second_slash = '/'.join(parts[2:])
                                if manual_annot_name in path_after_second_slash:
                                    annot_member = json_member
                                    print(f"Found manual annotation file: {annot_member.name}")
                                    break  # Prioritize manual
                                elif auto_annot_name in path_after_second_slash:
                                    annot_member = json_member
                                    print(f"Found automatic annotation file: {annot_member.name}")

                        if annot_member:
                            samples.append((tar_path, video_id, annot_member.name))
                        else:
                            print(f"No annotation found for video ID: {video_id}")
                    except Exception as e:
                        print(f"Error processing video ID {video_id}: {str(e)}")
            tar_idx += 1

        print(f"\n=== Index built. Total samples: {len(samples)} ===")
        return samples

    def __len__(self):
        return len(self.samples) * 4

    def __getitem__(self, idx):
        # Map the flat index to an original sample index and a caption index
        original_sample_idx = idx // 4
        caption_idx = idx % 4

        tar_path, video_id, annot_name = self.samples[original_sample_idx]
        with tarfile.open(tar_path, "r") as tar:
            video_path = f"{video_id}.mp4"
            video_member = next((m for m in tar.getmembers() if m.isfile() and m.name.endswith(video_path)), None)
            if not video_member:
                raise ValueError(f"Video file {video_path} not found in tar archive")
            
            video_file = tar.extractfile(video_member)
            temp_video_path = os.path.join(self.cache_dir, video_path)
            with open(temp_video_path, 'wb') as f:
                f.write(video_file.read())
            frames = self.sav_helper.read_frames(temp_video_path)
            os.remove(temp_video_path)

            if frames is None or len(frames) == 0:
                raise ValueError(f"Failed to load frames from {video_path}")

            frames = torch.stack([self.transform(Image.fromarray(frame)) for frame in frames])

            annot_file = tar.extractfile(annot_name)
            data = json.load(annot_file)

            first_valid_frame = None
            first_valid_mask = None
            masks = []
            for frame_idx in range(len(frames)):
                frame_masks = []
                if data and data.get('masklet') and len(data['masklet']) > frame_idx:
                    for ann in data['masklet'][frame_idx]:
                        rle = {'counts': ann['counts'], 'size': ann['size']}
                        mask = torch.from_numpy(coco_mask.decode(rle)).long()
                        frame_masks.append(mask)
                
                if frame_masks:
                    frame_mask = torch.stack(frame_masks).any(dim=0).unsqueeze(0).float()
                    masks.append(frame_mask)
                    if first_valid_frame is None:
                        first_valid_frame = frames[frame_idx]
                        first_valid_mask = frame_mask
                else:
                    masks.append(torch.zeros_like(frames[0].mean(dim=0, keepdim=True)))
            masks = torch.stack(masks)

            if first_valid_frame is not None:
                # Generate all 5 captions and select the one for this index
                all_captions = self.caption_generator.generate_all_captions(first_valid_frame, first_valid_mask)
                caption = all_captions[caption_idx]
            else:
                caption = "object"

            caption = CLIPTokenize(caption).squeeze(1)

        return frames, masks, caption



def test_captioning_models(image_path: str, mask_path: str):
    print(f"🧪 Testing with:\n  Image: {image_path}\n  Mask:  {mask_path}\n")

    ## 1. Load and Preprocess Image and Mask
    try:
        # Load the image and mask from the specified paths
        input_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")  # Convert mask to grayscale
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find a file. Please check your paths.\n{e}")
        return

    # Define a transform to convert PIL images to PyTorch tensors
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(input_image)
    mask_tensor = to_tensor(mask_image)

    # Validate that image and mask dimensions match
    if image_tensor.shape[1:] != mask_tensor.shape[1:]:
        print("❌ Error: Image and mask dimensions must be the same.")
        print(f"Image shape: {image_tensor.shape}, Mask shape: {mask_tensor.shape}")
        return

    ## 2. Initialize the Caption Generator
    # This step loads all five large language models and may require significant
    # GPU memory and time.
    print("🧠 Initializing captioning models... (This may take a moment)")
    try:
        caption_generator = CaptionGenerator()
        print("✅ Models initialized successfully.")
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        return
        
    ## 3. Generate Captions
    print("\n✍️ Generating captions for the masked object...")
    try:
        all_captions = caption_generator.generate_all_captions(image_tensor, mask_tensor)
    except Exception as e:
        print(f"\n❌ An error occurred during caption generation: {e}")
        return

    ## 4. Display Results
    model_names = ["BLIP", "PaliGemma", "Florence-2", "Llava-1.5"]
    
    print("\n--- 📸 Generated Captions ---")
    if len(all_captions) == len(model_names):
        for name, caption in zip(model_names, all_captions):
            print(f"🔹 **{name}**: {caption}")
    else:
        print("An unexpected number of captions were returned.")
        print(all_captions)
    print("----------------------------\n")


image_file = "/Users/adityaasuratkal/Downloads/Gemini_Generated_Image_wpocdvwpocdvwpoc.png"
mask_file = "/Users/adityaasuratkal/Downloads/Gemini_Generated_Image_velvxjvelvxjvelv.png"

test_captioning_models(image_path=image_file, mask_path=mask_file)
    
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