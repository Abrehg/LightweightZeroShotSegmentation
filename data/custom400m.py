import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader, DistributedSampler
import os
import time
import threading
import torch.distributed as dist
from datasets import load_dataset
from PIL import Image
import requests
import io
from torchvision import transforms
from typing import Optional, Callable
import itertools

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
        val_size: int = 10000,
        skip_items_per_worker: int = 0
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
            "laion/relaion400m",
            split="train",
            streaming=True,
            token=HUGGINGFACE_TOKEN
        )

        self.skip_items_per_worker = skip_items_per_worker

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
        
        # 1. Get Global DDP details
        if dist.is_initialized():
            global_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            global_rank = 0
            world_size = 1
            
        # 2. Get local DataLoader worker details
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        # 3. Calculate absolute worker rank across the entire supercomputer
        total_shards = world_size * num_workers
        global_worker_rank = (global_rank * num_workers) + worker_id
        
        def safe_hf_generator():
            original_get_worker_info = torch.utils.data.get_worker_info
            
            torch.utils.data.get_worker_info = lambda: None
            try:
                hf_iterator = iter(self.hf_dataset)
            finally:
                torch.utils.data.get_worker_info = original_get_worker_info
                
            while True:
                torch.utils.data.get_worker_info = lambda: None
                try:
                    sample = next(hf_iterator)
                except StopIteration:
                    torch.utils.data.get_worker_info = original_get_worker_info
                    break
                except Exception as e:
                    torch.utils.data.get_worker_info = original_get_worker_info
                    raise e
                    
                torch.utils.data.get_worker_info = original_get_worker_info
                yield sample
                
        shard_iterator = itertools.islice(safe_hf_generator(), global_worker_rank, None, total_shards)
        
        if self.skip_items_per_worker > 0:
            shard_iterator = itertools.islice(shard_iterator, self.skip_items_per_worker, None)
        
        for sample in shard_iterator:
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
                "laion/relaion400m",
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

class _ChunkedTensorDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
 
class ChunkedLAIONManager:
    def __init__(self, hf_token, chunk_size=10000, total_samples=None,
                 val_ratio=0.05, cache_dir="data/cache/chunks",
                 text_processor=None, collate_fn=None,
                 num_workers=4, prefetch_factor=4):
        self.hf_token = hf_token
        self.chunk_size = chunk_size
        self.total_samples = total_samples
        self.val_ratio = val_ratio
        self.cache_dir = cache_dir
        self.text_processor = text_processor
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.num_chunks = max(1, total_samples // chunk_size) if total_samples else float('inf')
        
        self._prefetch_thread = None
        self._stream_iter = None
        self._skip_items = 0  # Stream offset for resume
        self._is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        
        if self._is_rank0:
            self._init_stream()
    
    def _init_stream(self):
        stream = StreamingLAIONDataset(
            HUGGINGFACE_TOKEN=self.hf_token,
            text_processor=self.text_processor,
            split_mode="train",
            val_size=0,
            skip_items_per_worker=self._skip_items
        )
        self._stream_iter = iter(DataLoader(
            stream, batch_size=1, num_workers=1,
            collate_fn=lambda x: x[0]
        ))
    
    def skip_chunks(self, n):
        if n <= 0:
            return
        self._skip_items = n * self.chunk_size
        if self._is_rank0:
            print(f"[ChunkedLAION] Skipping {n} chunks ({self._skip_items} stream items) for resume")
            self._init_stream()  # Re-init with new skip offset
    
    def reset_stream(self):
        self._skip_items = 0
        if self._is_rank0:
            self._init_stream()
    
    def _chunk_path(self, idx):
        return os.path.join(self.cache_dir, f"chunk_{idx:04d}.pt")
    
    def _download_chunk(self, chunk_idx):
        path = self._chunk_path(chunk_idx)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[Chunk {chunk_idx}] Already cached, skipping.")
            return True
        
        print(f"[Chunk {chunk_idx}] Downloading {self.chunk_size} samples...")
        samples, failures = [], 0
        t0 = time.time()
        
        while len(samples) < self.chunk_size:
            try:
                sample = next(self._stream_iter)
                if sample is not None:
                    samples.append(sample)
                else:
                    failures += 1
            except StopIteration:
                print(f"[Chunk {chunk_idx}] Stream exhausted at {len(samples)} samples.")
                break
            except Exception:
                failures += 1
                if failures > self.chunk_size:
                    break
        
        if not samples:
            print(f"[Chunk {chunk_idx}] No samples downloaded!")
            return False
        
        torch.save(samples, path)
        elapsed = time.time() - t0
        mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[Chunk {chunk_idx}] Saved {len(samples)} samples ({mb:.0f} MB) in {elapsed:.0f}s")
        return True
    
    def _prefetch_bg(self, chunk_idx):
        if not self._is_rank0:
            return
        def _worker():
            try:
                self._download_chunk(chunk_idx)
            except Exception as e:
                print(f"[Prefetch] Chunk {chunk_idx} failed: {e}")
        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()
    
    def _wait_prefetch(self):
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()
            self._prefetch_thread = None
    
    def prepare_chunk(self, chunk_idx):
        if self._is_rank0:
            self._wait_prefetch()
            if not os.path.exists(self._chunk_path(chunk_idx)):
                self._download_chunk(chunk_idx)
        if dist.is_initialized():
            dist.barrier()
        # Start prefetching next chunk in background
        next_idx = chunk_idx + 1
        if isinstance(self.num_chunks, float) or next_idx < self.num_chunks:
            self._prefetch_bg(next_idx)
    
    def get_loaders(self, chunk_idx, batch_size):
        path = self._chunk_path(chunk_idx)
        samples = torch.load(path, map_location='cpu')
        full = _ChunkedTensorDataset(samples)
        
        use_dist = dist.is_initialized()
        train_sampler = DistributedSampler(full, shuffle=True) if use_dist else None
        
        train_loader = DataLoader(
            full, batch_size=batch_size, sampler=train_sampler,
            shuffle=(train_sampler is None), collate_fn=self.collate_fn,
            pin_memory=True, num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor, persistent_workers=True, drop_last=True
        )
        return train_loader
    
    def prepare_val_loader(self, batch_size, num_val_samples=10000):
        if self._is_rank0:
            print(f"[ChunkedLAION] Downloading {num_val_samples} static val samples (one-time)...")
        
        val_dataset = CachedLAIONDataset(
            HUGGINGFACE_TOKEN=self.hf_token,
            num_samples=num_val_samples,
            text_processor=self.text_processor,
            split_mode="val",
            val_split_boundary=num_val_samples
        )
        
        if self._is_rank0:
            print(f"[ChunkedLAION] Val dataset ready: {len(val_dataset)} samples")
        
        # Sync so all ranks wait for rank 0's download
        if dist.is_initialized():
            dist.barrier()
        
        use_dist = dist.is_initialized()
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_dist else None
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            shuffle=False, collate_fn=self.collate_fn,
            pin_memory=True, num_workers=min(2, self.num_workers)
        )
        return val_loader

    def cleanup(self, chunk_idx):
        path = self._chunk_path(chunk_idx)
        if os.path.exists(path):
            try:
                os.remove(path)
                if self._is_rank0:
                    print(f"[Chunk {chunk_idx}] Deleted.")
            except OSError:
                pass
    
    def shutdown(self):
        self._wait_prefetch()

def adaptive_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    images, texts = zip(*batch)
    images = torch.stack(images)
    
    if len(texts) > 0 and isinstance(texts[0], torch.Tensor):
        texts = torch.stack(texts).squeeze(1)
    
    return images, texts

def get_laion_streaming_dataset(HUGGINGFACE, val_size, text_processor=None, split="train", skip_items=0):
    return StreamingLAIONDataset(
        HUGGINGFACE_TOKEN=HUGGINGFACE, 
        text_processor=text_processor,
        split_mode=split,
        val_size=val_size,
        skip_items_per_worker=skip_items
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