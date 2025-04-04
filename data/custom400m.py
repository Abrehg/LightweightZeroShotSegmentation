import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import requests
import io
from torchvision import transforms
from typing import Optional, Callable

def _conditional_crop_pad(img: Image.Image, target_size: int) -> Image.Image:
    """Custom transform to crop/pad images to target_size without resizing"""
    width, height = img.size
    
    # Phase 1: Center cropping for oversized dimensions
    if width > target_size:
        left = (width - target_size) // 2
        img = img.crop((left, 0, left + target_size, height))
        width = target_size
        
    if height > target_size:
        top = (height - target_size) // 2
        img = img.crop((0, top, width, top + target_size))
        height = target_size
    
    # Phase 2: Symmetric padding for undersized dimensions
    pad_w = max(target_size - width, 0)
    pad_h = max(target_size - height, 0)
    
    if pad_w > 0 or pad_h > 0:
        img = transforms.functional.pad(
            img, 
            padding=(pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2),
            fill=0,
            padding_mode='constant'
        )
    
    return img

class ConditionalCropPad:
    """Serializable version of the crop/pad transform"""
    def __init__(self, target_size: int):
        self.target_size = target_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return _conditional_crop_pad(img, self.target_size)

class StreamingLAIONDataset(IterableDataset):
    """Full streaming dataset for training"""
    def __init__(
        self,
        HUGGINGFACE_TOKEN,
        text_processor: Optional[Callable] = None,
        max_retries: int = 3,
        min_size: int = 32,
        max_aspect_ratio: float = 4.0,
        sample_timeout: int = 10,
        image_size: int = 512
    ):
        self.text_processor = text_processor
        self.max_retries = max_retries
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio
        self.sample_timeout = sample_timeout
        self.image_size = image_size
        self.transform = transforms.Compose([
            ConditionalCropPad(self.image_size),
            transforms.ToTensor(),
        ])
        
        self.hf_dataset = load_dataset(
            "laion/laion400m",
            split="train",
            streaming=True,
            token=HUGGINGFACE_TOKEN
        )

    def _is_valid(self, img: Image.Image) -> bool:
        """Validate image dimensions"""
        w, h = img.size
        return (
            min(w, h) >= self.min_size and
            max(w/h, h/w) <= self.max_aspect_ratio
        )

    def _process_sample(self, sample: dict) -> Optional[tuple]:
        """Process single sample with retries"""
        for _ in range(self.max_retries):
            try:
                response = requests.get(
                    sample['url'],
                    timeout=self.sample_timeout
                )
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                
                if not self._is_valid(img):
                    return None
                
                text = sample['caption']
                if self.text_processor:
                    text = self.text_processor(text)
                
                return self.transform(img), text
            except Exception as e:
                continue
        return None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shard = self.hf_dataset
        if worker_info is not None:
            shard = shard.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id
            )
        
        for sample in shard:
            processed = self._process_sample(sample)
            if processed is not None:
                yield processed

class CachedLAIONDataset(Dataset):
    """Fixed-size cached dataset for testing/validation"""
    def __init__(
        self,
        HUGGINGFACE_TOKEN,
        num_samples: int = 5,
        text_processor: Optional[Callable] = None,
        min_size: int = 32,
        max_aspect_ratio: float = 4.0,
        seed: int = 42,
        image_size: int = 512
    ):
        self.num_samples = num_samples
        self.text_processor = text_processor
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio
        self.seed = seed
        self.image_size = image_size
        self.transform = transforms.Compose([
            ConditionalCropPad(self.image_size),  # Replaces lambda
            transforms.ToTensor(),
        ])

        self.samples = self._precache_samples(HUGGINGFACE_TOKEN)

    def _is_valid(self, img: Image.Image) -> bool:
        w, h = img.size
        return (
            min(w, h) >= self.min_size and
            max(w/h, h/w) <= self.max_aspect_ratio
        )

    def _precache_samples(self, HUGGINGFACE_TOKEN) -> list:
        """Pre-download and cache validated samples"""
        print(f"\n[Dataset] Precaching {self.num_samples} samples...")
        dataset = load_dataset(
            "laion/laion400m",
            split="train",
            streaming=True,
            token=HUGGINGFACE_TOKEN
        ).shuffle(seed=self.seed)

        samples = []
        for idx, sample in enumerate(dataset):
            try:
                print(f"[Cached Dataset] Attempting sample {idx+1}/{self.num_samples}")
                response = requests.get(sample['url'], timeout=10)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
            
                if not self._is_valid(img):
                    print(f"[Cached Dataset] Sample {idx+1} failed validation")
                    continue
            
                text = sample['caption']
                if self.text_processor:
                    text = self.text_processor(text)
            
                samples.append((self.transform(img), text))
                print(f"[Cached Dataset] Sample {len(samples)}/{self.num_samples} cached")
            
                if len(samples) >= self.num_samples:
                    print("[Cached Dataset] Precaching complete")
                    break
            except Exception as e:
                print(f"[Cached Dataset] Error processing sample {idx+1}: {str(e)}")
                continue
    
        if len(samples) < self.num_samples:
            print(f"\n⚠️ Warning: Only collected {len(samples)}/{self.num_samples} samples!")
    
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def adaptive_collate(batch):
    """Collate function that stacks images and texts"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    
    images, texts = zip(*batch)
    images = torch.stack(images)
    
    # Stack text tensors if they're not already batched
    if isinstance(texts[0], torch.Tensor):
        texts = torch.stack(texts).squeeze(1)
    
    return images, texts

def get_laion_test_dataset(HUGGINGFACE, num_samples=5, text_processor=None):
    return CachedLAIONDataset(HUGGINGFACE_TOKEN=HUGGINGFACE, num_samples=num_samples, text_processor=text_processor)

def get_laion_streaming_dataset(HUGGINGFACE, text_processor=None):
    return StreamingLAIONDataset(HUGGINGFACE_TOKEN=HUGGINGFACE, text_processor=text_processor)