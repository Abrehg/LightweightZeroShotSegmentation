import torch
from torch.utils.data import Dataset
import os
import time
import itertools
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from PIL import Image
import requests
import io
import json
import gzip
import base64
from torchvision import transforms
from typing import Optional, Callable

from models.clip_model import CLIPTokenize

# Standard CLIP stats
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

_CLIP_NORMALIZE = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

# ============================================================================
# LAION (laion/relaion400m) dataset
#
# On-disk spec: images are JPEG-encoded (quality LAION_JPEG_QUALITY), resized so the
# short side matches LAION_IMAGE_SIZE and center-cropped to
# (LAION_IMAGE_SIZE, LAION_IMAGE_SIZE); captions are pre-tokenized so training never
# re-runs the tokenizer. Each split (train/val) is stored as a single gzip-compressed
# JSON array of {"image": base64 JPEG, "tokens": [...]} records under
# LAION_LOCAL_DEFAULT_DIR. Constructing LAIONDataset (or calling get_laion_dataset)
# is enough to get a usable dataset: if the local file for the requested split is
# already there, it's just loaded; if not, it's downloaded in full right then and
# there using the given hf_token. Callers (e.g. train.py) don't need a separate
# download step.
# ============================================================================
LAION_IMAGE_SIZE = 256
LAION_JPEG_QUALITY = 85
LAION_LOCAL_DEFAULT_DIR = os.environ.get(
    "LAION_LOCAL_DIR", "/gpfs/u/home/ZSIS/ZSISsrtk/scratch/laion"
)


def prepare_image_for_storage(pil_img: Image.Image, size: int = LAION_IMAGE_SIZE) -> Image.Image:
    """Resize (short side = size) then center-crop to (size, size)."""
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    scale = size / min(w, h)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return pil_img.crop((left, top, left + size, top + size))


def encode_sample_to_record(pil_img: Image.Image, token_ids) -> dict:
    """pre-cropped PIL image + pre-tokenized ids -> a JSON-serializable record."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=LAION_JPEG_QUALITY)
    return {
        "image": base64.b64encode(buf.getvalue()).decode("ascii"),
        "tokens": [int(t) for t in token_ids],
    }


def decode_record(record: dict):
    """record -> (uint8 CHW image tensor, long token-id tensor). No network access."""
    img_bytes = base64.b64decode(record["image"])
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transforms.PILToTensor()(img)
    tokens = torch.tensor(record["tokens"], dtype=torch.long)
    return img_tensor, tokens


def write_json_gz(path: str, records: list):
    """Write records as a gzip JSON array, atomically (write to tmp, then rename)."""
    tmp_path = path + f".tmp{os.getpid()}"
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        json.dump(records, f)
    os.replace(tmp_path, path)


def read_json_gz(path: str) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


class LAIONDataset(Dataset):
    """
    Unified LAION dataset handler. Just constructing it is enough:
      - If data_dir already has a local dump for this split, it's loaded (no network).
      - Otherwise, hf_token is required and the full split is downloaded right away —
        streamed from HuggingFace sample by sample (fetch by URL, validate, retry on
        failure, same as plain streaming iteration), postprocessed to the on-disk spec
        below, and dumped to local/shared storage — before __init__ returns.

    val_size controls how many entries are withheld from training: the "train" split
    skips exactly the first val_size stream entries before sampling total_samples from
    the remainder, and the "val" split takes exactly those first val_size entries — so
    the same val_size passed to both splits keeps them disjoint and the model never
    trains on validation data.

    On-disk spec: images -> JPEG (quality LAION_JPEG_QUALITY), resized (short side)
    and center-cropped to (LAION_IMAGE_SIZE, LAION_IMAGE_SIZE); captions -> pre-
    tokenized token-id lists via text_processor (CLIPTokenize by default) so training
    never re-runs the tokenizer.
    """
    def __init__(
        self,
        split: str = "train",
        data_dir: str = LAION_LOCAL_DEFAULT_DIR,
        hf_token: Optional[str] = None,
        text_processor: Optional[Callable] = CLIPTokenize,
        total_samples: int = 200000,
        val_size: int = 10000,
        min_size: int = 32,
        max_aspect_ratio: float = 4.0,
        sample_timeout: int = 10,
        max_retries: int = 3,
        seed: int = 42,
        shuffle_buffer: int = 10000,
        download_workers: int = 4,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.data_dir = data_dir
        self.hf_token = hf_token
        self.text_processor = text_processor
        self.total_samples = total_samples
        self.val_size = val_size
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio
        self.sample_timeout = sample_timeout
        self.max_retries = max_retries
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        # Caps concurrent fetches to roughly the same per-node scale the old
        # sharded streaming implementation used (dl_workers = min(4, num_nodes)), so
        # every node downloading the full dataset doesn't multiply cluster-wide
        # request volume against LAION's image hosts by num_nodes.
        self.download_workers = download_workers

        if os.path.exists(self.local_path) and os.path.getsize(self.local_path) > 0:
            self.records = read_json_gz(self.local_path)
        else:
            if not hf_token:
                raise ValueError(
                    f"No local LAION '{split}' data at {self.local_path} and no "
                    f"hf_token was given to download it."
                )
            self.records = self._download_all()

    @property
    def local_path(self) -> str:
        filename = "train.json.gz" if self.split == "train" else "val.json.gz"
        return os.path.join(self.data_dir, filename)

    def _is_valid(self, img: Image.Image) -> bool:
        w, h = img.size
        return (
            min(w, h) >= self.min_size and
            max(w / h, h / w) <= self.max_aspect_ratio
        )

    def _hf_stream(self):
        hf_dataset = load_dataset(
            "laion/relaion400m", split="train", streaming=True, token=self.hf_token
        )
        if self.split == "train":
            hf_dataset = hf_dataset.skip(self.val_size)
            hf_dataset = hf_dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        else:
            hf_dataset = hf_dataset.take(self.val_size)
        return hf_dataset

    def _process_sample(self, sample: dict) -> Optional[dict]:
        for _ in range(self.max_retries):
            try:
                response = requests.get(sample['url'], timeout=self.sample_timeout)
                img = Image.open(io.BytesIO(response.content)).convert("RGB")

                if not self._is_valid(img):
                    return None

                img = prepare_image_for_storage(img, size=LAION_IMAGE_SIZE)

                tokens = sample['caption']
                if self.text_processor:
                    tokens = self.text_processor(tokens)
                    if isinstance(tokens, torch.Tensor):
                        tokens = tokens.squeeze(0).tolist()

                return encode_sample_to_record(img, tokens)
            except Exception:
                continue
        return None

    def _download_all(self) -> list:
        """Streams and processes the entire configured split, then dumps it to
        self.local_path. Assumes it must fetch the whole split (total_samples /
        val_size), not a partial one. Fetches are done download_workers-at-a-time
        (not one giant burst) so a node downloading the full dataset stays around the
        same concurrent-request scale a single sharded worker used to be."""
        os.makedirs(self.data_dir, exist_ok=True)
        target = self.val_size if self.split == "val" else self.total_samples
        print(f"[LAION:{self.split}] No local copy at {self.local_path} — downloading "
              f"{target} samples now ({self.download_workers} concurrent fetches).", flush=True)

        records, failures = [], 0
        last_logged = 0
        t0 = time.time()
        sample_iter = iter(self._hf_stream())

        with ThreadPoolExecutor(max_workers=self.download_workers) as executor:
            while len(records) < target:
                batch = list(itertools.islice(sample_iter, self.download_workers))
                if not batch:
                    break
                for record in executor.map(self._process_sample, batch):
                    if record is not None:
                        records.append(record)
                    else:
                        failures += 1
                if len(records) - last_logged >= 1000:
                    last_logged = len(records)
                    elapsed = time.time() - t0
                    print(f"[LAION:{self.split}] {len(records)}/{target} "
                          f"({failures} failed) in {elapsed/60:.1f}m", flush=True)

        records = records[:target]
        write_json_gz(self.local_path, records)
        print(f"[LAION:{self.split}] Done: {len(records)} samples -> {self.local_path}")
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img, tokens = decode_record(self.records[idx])
        img = _CLIP_NORMALIZE(img.float().div(255.0))
        return img, tokens


def get_laion_dataset(split="train", data_dir=LAION_LOCAL_DEFAULT_DIR, hf_token=None, **kwargs) -> LAIONDataset:
    """
    Single factory for the LAION dataset: loads the local split at data_dir if it's
    already there, otherwise downloads it in full using hf_token before returning.
    """
    return LAIONDataset(split=split, data_dir=data_dir, hf_token=hf_token, **kwargs)


def adaptive_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images, texts = zip(*batch)
    if images[0].dtype == torch.uint8:
        images = [_CLIP_NORMALIZE(img.float().div(255.0)) for img in images]
    images = torch.stack(images)

    if len(texts) > 0 and isinstance(texts[0], torch.Tensor):
        texts = torch.stack(texts).squeeze(1)

    return images, texts
