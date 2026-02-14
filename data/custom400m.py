import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import requests
import io
from torchvision import transforms
from typing import Optional, Callable

# Standard CLIP stats
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

#Streaming version of LAION Dataset for training phase
class StreamingLAIONDataset(IterableDataset):
    def __init__(
        self,
        HUGGINGFACE_TOKEN,
        text_processor: Optional[Callable] = None,
        max_retries: int = 3,
        min_size: int = 32,
        max_aspect_ratio: float = 4.0,
        sample_timeout: int = 10,
        image_size: int = 224,
        split_mode: str = "train",
        val_size: int = 10000
    ):
        self.text_processor = text_processor
        self.max_retries = max_retries
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio
        self.sample_timeout = sample_timeout
        self.image_size = image_size
        self.split_mode = split_mode
        self.val_size = val_size

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        ])
        
        self.hf_dataset = load_dataset(
            "laion/laion400m",
            split="train",
            streaming=True,
            token=HUGGINGFACE_TOKEN
        )

        if self.split_mode == "train":
            self.hf_dataset = self.hf_dataset.skip(self.val_size)
            self.hf_dataset = self.hf_dataset.shuffle(seed=42, buffer_size=10000)
        elif self.split_mode == "val":
            self.hf_dataset = self.hf_dataset.take(self.val_size)
        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode}")

    def _is_valid(self, img: Image.Image) -> bool:
        w, h = img.size
        return (
            min(w, h) >= self.min_size and
            max(w/h, h/w) <= self.max_aspect_ratio
        )

    def _process_sample(self, sample: dict) -> Optional[tuple]:
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

#Cached version of LAION Dataset for testing/validation
class CachedLAIONDataset(Dataset):
    def __init__(
        self,
        HUGGINGFACE_TOKEN,
        num_samples: int = 5,
        text_processor: Optional[Callable] = None,
        min_size: int = 32,
        max_aspect_ratio: float = 4.0,
        seed: int = 42,
        image_size: int = 224,
        split_mode: str = "train",
        val_split_boundary: int = 10000
    ):
        self.num_samples = num_samples
        self.text_processor = text_processor
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio
        self.seed = seed
        self.image_size = image_size
        self.split_mode = split_mode
        self.val_split_boundary = val_split_boundary

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        ])

        self.samples = self._precache_samples(HUGGINGFACE_TOKEN)

    def _is_valid(self, img: Image.Image) -> bool:
        w, h = img.size
        return (
            min(w, h) >= self.min_size and
            max(w/h, h/w) <= self.max_aspect_ratio
        )

    def _precache_samples(self, HUGGINGFACE_TOKEN) -> list:
        print(f"\n[CachedDataset] Mode: {self.split_mode}. Target Samples: {self.num_samples}")
        
        try:
            dataset = load_dataset(
                "laion/laion400m",
                split="train",
                streaming=True,
                token=HUGGINGFACE_TOKEN
            )
            
            if self.split_mode == "val":
                dataset = dataset.take(self.val_split_boundary)
            elif self.split_mode == "train":
                dataset = dataset.skip(self.val_split_boundary)
            
            dataset = dataset.shuffle(seed=self.seed)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

        samples = []
        iterator = iter(dataset)
        
        while len(samples) < self.num_samples:
            try:
                sample = next(iterator)
                response = requests.get(sample['url'], timeout=10)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
            
                if not self._is_valid(img):
                    continue
            
                text = sample['caption']
                if self.text_processor:
                    text = self.text_processor(text)
            
                samples.append((self.transform(img), text))
                
                if len(samples) % 100 == 0:
                     print(f"Cached {len(samples)}/{self.num_samples}...")

            except StopIteration:
                print("Stream exhausted before filling cache.")
                break
            except Exception as e:
                continue
    
        if len(samples) < self.num_samples:
            print(f"\n Warning: Only collected {len(samples)}/{self.num_samples} samples!")
        else:
            print("[Cached Dataset] Precaching complete")
    
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def adaptive_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    images, texts = zip(*batch)
    images = torch.stack(images)
    
    if len(texts) > 0 and isinstance(texts[0], torch.Tensor):
        texts = torch.stack(texts).squeeze(1)
    
    return images, texts

# Call twice for train/val splits. once with split="train" and once with split="val"
def get_laion_streaming_dataset(HUGGINGFACE, val_size, text_processor=None, split="train"):
    return StreamingLAIONDataset(
        HUGGINGFACE_TOKEN=HUGGINGFACE, 
        text_processor=text_processor,
        split_mode=split,
        val_size=val_size
    )

# Call twice for train/val splits. once with split="train" and once with split="val"
def get_laion_test_dataset(HUGGINGFACE, val_boundary, num_samples=1000, text_processor=None, split = "train"):
    return CachedLAIONDataset(
        HUGGINGFACE_TOKEN=HUGGINGFACE, 
        num_samples=num_samples, 
        text_processor=text_processor,
        split_mode=split,
        val_split_boundary=val_boundary
    )