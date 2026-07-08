import os
import io
import re
import json
import random
import base64
import tarfile
import tempfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from pycocotools import mask as coco_mask
from models.clip_model import CLIPTokenize
from .custom400m import write_json_gz, read_json_gz

# ============================================================================
# SA-1B + SA-V segmentation dataset
#
# Raw input: the user drops SA-1B tar/tar.gz files into SA1B_RAW_DIR (each tar has
# flat sa_N.jpg/sa_N.json pairs, COCO-RLE masks) and SA-V tar/tar.gz files into
# SAV_RAW_DIR (each tar has JPEGImages_24fps/<video_id>/<frame_num>.jpg — every
# frame, 24fps — and Annotations_6fps/<video_id>/<object_id>/<frame_num>.png — one
# subfolder per tracked object, each holding that object's own binary mask at the
# frame numbers it was annotated on). No network access here — everything is
# already local.
#
# On-disk spec: images/frames -> downscaled 4x (bilinear) -> JPEG q=SEG_JPEG_QUALITY;
# masks -> downscaled 4x (nearest) -> binary PNG, ranked by area descending and kept
# only if area > SEG_MASK_AREA_THRESHOLD of the image (variable count per sample/
# frame, not a fixed top-K). SA-1B image samples and SA-V video samples are pooled
# together and shuffled (interleaved) before a single combined-pool val split is
# taken — see SegmentationDataset.
#
# Captions are NOT generated here — SegmentationDataset writes captions as None;
# precomputeCaptions.py fills them in (raw text, then tokenized) in a separate pass.
# ============================================================================
SEG_DOWNSCALE_FACTOR = 4
SEG_JPEG_QUALITY = 85
SEG_MASK_AREA_THRESHOLD = 0.05

SA1B_RAW_DIR = os.environ.get("SA1B_RAW_DIR", "/gpfs/u/home/ZSIS/ZSISsrtk/scratch/sa1b_raw")
SAV_RAW_DIR = os.environ.get("SAV_RAW_DIR", "/gpfs/u/home/ZSIS/ZSISsrtk/scratch/sav_raw")
SEG_LOCAL_DEFAULT_DIR = os.environ.get(
    "SEG_LOCAL_DIR", "/gpfs/u/home/ZSIS/ZSISsrtk/scratch/segmentation"
)


def SAM_adaptive_collate(batch):
    images, masks, texts = zip(*batch)
    images = [
        img.float().div(255.0) if img.dtype == torch.uint8 else img
        for img in images
    ]
    return list(images), list(masks), list(texts)


# ---------------------------------------------------------------------------
# Caption tri-state: None (not generated) -> [text, text, text, text] (pass 1,
# for the user to inspect) -> [[ids...], [ids...], [ids...], [ids...]] (pass 2,
# tokenized; precomputeCaptions.py deletes the text once this exists).
# ---------------------------------------------------------------------------
def is_caption_tokenized(entry) -> bool:
    return isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], list)


def is_caption_raw(entry) -> bool:
    return isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], str)


# ---------------------------------------------------------------------------
# Encode / decode helpers
# ---------------------------------------------------------------------------
def _downscale_image(pil_img: Image.Image, factor: int) -> Image.Image:
    w, h = pil_img.size
    new_w, new_h = max(1, round(w / factor)), max(1, round(h / factor))
    return pil_img.resize((new_w, new_h), Image.BILINEAR)


def _downscale_mask(mask_bool: np.ndarray, factor: int) -> Image.Image:
    h, w = mask_bool.shape
    new_w, new_h = max(1, round(w / factor)), max(1, round(h / factor))
    pil = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
    return pil.resize((new_w, new_h), Image.NEAREST)


def _encode_image_b64(pil_img: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_mask_b64(pil_mask_L: Image.Image) -> str:
    arr = np.array(pil_mask_L)
    bin_img = Image.fromarray(((arr > 127).astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    bin_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_seg_image(b64str: str) -> torch.Tensor:
    """base64 JPEG -> uint8 CHW tensor. No network access."""
    img = Image.open(io.BytesIO(base64.b64decode(b64str))).convert("RGB")
    return transforms.PILToTensor()(img)


def decode_seg_mask(b64str: str) -> torch.Tensor:
    """base64 PNG -> uint8 HxW binary (0/1) tensor. No network access."""
    m = Image.open(io.BytesIO(base64.b64decode(b64str))).convert("L")
    arr = (np.array(m) > 127).astype(np.uint8)
    return torch.from_numpy(arr)


def _rank_and_filter_masks(items, min_area_fraction: float):
    """items: [(key, bool HxW ndarray), ...] -> same, filtered to
    area/total > min_area_fraction and sorted by area descending (variable count)."""
    kept = []
    for key, mask in items:
        total = mask.shape[0] * mask.shape[1]
        area = float(mask.sum())
        if total > 0 and area / total > min_area_fraction:
            kept.append((key, mask, area))
    kept.sort(key=lambda t: t[2], reverse=True)
    return [(key, mask) for key, mask, _ in kept]


def _find_raw_tars(raw_dir: str):
    if not os.path.isdir(raw_dir):
        return []
    return sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.endswith(".tar") or f.endswith(".tar.gz") or f.endswith(".tgz")
    )


def _open_tar(path: str) -> tarfile.TarFile:
    mode = "r:gz" if path.endswith((".tar.gz", ".tgz")) else "r:"
    return tarfile.open(path, mode)


# ---------------------------------------------------------------------------
# SA-1B: flat sa_N.jpg / sa_N.json pairs, COCO-RLE masks, one sample per image.
# ---------------------------------------------------------------------------
def _process_sa1b_tar(tar_path, downscale_factor, jpeg_quality, min_area_fraction):
    try:
        tar = _open_tar(tar_path)
    except Exception:
        return
    with tar:
        members = tar.getmembers()
        json_members = [m for m in members if m.name.endswith('.json') and m.isfile()]

        for jm in json_members:
            try:
                data = json.load(tar.extractfile(jm))
                image_name = data['image']['file_name']
                image_member = next(
                    (m for m in members if m.isfile() and os.path.basename(m.name) == image_name),
                    None,
                )
                if image_member is None:
                    continue

                mask_items = []
                for ann_idx, ann in enumerate(data.get('annotations', [])):
                    try:
                        rle = {'counts': ann['segmentation']['counts'], 'size': ann['segmentation']['size']}
                        mask_items.append((ann_idx, coco_mask.decode(rle).astype(bool)))
                    except Exception:
                        continue
                kept = _rank_and_filter_masks(mask_items, min_area_fraction)
                if not kept:
                    continue

                pil_img = Image.open(tar.extractfile(image_member)).convert("RGB")
                small_img = _downscale_image(pil_img, downscale_factor)

                yield {
                    "type": "image",
                    "image": _encode_image_b64(small_img, jpeg_quality),
                    "masks": [_encode_mask_b64(_downscale_mask(m, downscale_factor)) for _, m in kept],
                    "captions": [None] * len(kept),
                }
            except Exception:
                continue


def _iter_sa1b_raw_samples(raw_dir, downscale_factor, jpeg_quality, min_area_fraction):
    tars = _find_raw_tars(raw_dir)
    print(f"[SegBuild:SA-1B] Found {len(tars)} tar(s) in {raw_dir}")
    for tar_path in tars:
        count = 0
        for record in _process_sa1b_tar(tar_path, downscale_factor, jpeg_quality, min_area_fraction):
            count += 1
            yield record
        print(f"[SegBuild:SA-1B] {os.path.basename(tar_path)}: {count} samples")


# ---------------------------------------------------------------------------
# SA-V: JPEGImages_24fps/<video_id>/<frame_num>.jpg (all frames, 24fps) +
# Annotations_6fps/<video_id>/<object_id>/<frame_num>.png (one subfolder per
# tracked object, each holding that object's own binary mask at the frames it was
# annotated on — filenames are the true 24fps frame number, e.g. 00000, 00004,
# 00008..., so they line up with JPEGImages_24fps directly, no fixed stride math
# needed). One sample per video, containing every frame that has at least one
# object mask passing the area filter, plus a list of the distinct tracked objects
# (for per-object captioning).
# ---------------------------------------------------------------------------
def _find_subdir(root: str, name: str):
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return os.path.join(dirpath, name)
    return None


def _find_frame_jpg(jpeg_dir: str, frame_num: int):
    for width in (5, 6, 4, 7):
        for ext in (".jpg", ".jpeg"):
            candidate = os.path.join(jpeg_dir, f"{frame_num:0{width}d}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def _process_sav_video(jpeg_dir, annot_dir, downscale_factor, jpeg_quality, min_area_fraction):
    object_ids = sorted(
        d for d in os.listdir(annot_dir) if os.path.isdir(os.path.join(annot_dir, d))
    )
    if not object_ids:
        return None

    # Gather frame_num -> {object_id: mask_path} across all per-object folders.
    frame_to_masks = {}
    for object_id in object_ids:
        obj_dir = os.path.join(annot_dir, object_id)
        for fname in os.listdir(obj_dir):
            if not fname.lower().endswith(".png"):
                continue
            try:
                frame_num = int(os.path.splitext(fname)[0])
            except ValueError:
                continue
            frame_to_masks.setdefault(frame_num, {})[object_id] = os.path.join(obj_dir, fname)

    if not frame_to_masks:
        return None

    frames_out = []
    object_first_seen = {}

    for frame_num in sorted(frame_to_masks.keys()):
        jpg_path = _find_frame_jpg(jpeg_dir, frame_num)
        if jpg_path is None:
            continue

        mask_items = []
        for object_id, mask_path in frame_to_masks[frame_num].items():
            mask_bool = np.array(Image.open(mask_path).convert("L")) > 127
            if mask_bool.any():
                mask_items.append((object_id, mask_bool))
        kept = _rank_and_filter_masks(mask_items, min_area_fraction)
        if not kept:
            continue

        pil_frame = Image.open(jpg_path).convert("RGB")
        small_frame = _downscale_image(pil_frame, downscale_factor)

        frame_out_idx = len(frames_out)
        frames_out.append({
            "image": _encode_image_b64(small_frame, jpeg_quality),
            "masks": [
                {"object_id": oid, "mask": _encode_mask_b64(_downscale_mask(m, downscale_factor))}
                for oid, m in kept
            ],
        })
        for oid, _ in kept:
            if oid not in object_first_seen:
                object_first_seen[oid] = frame_out_idx

    if not frames_out or not object_first_seen:
        return None

    objects = [
        {"object_id": oid, "rep_frame_idx": rep_idx, "captions": None}
        for oid, rep_idx in object_first_seen.items()
    ]
    return {"type": "video", "frames": frames_out, "objects": objects}


def _process_sav_tar(tar_path, downscale_factor, jpeg_quality, min_area_fraction):
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            tar = _open_tar(tar_path)
        except Exception:
            return
        with tar:
            tar.extractall(tmpdir)

        jpeg_root = _find_subdir(tmpdir, "JPEGImages_24fps")
        annot_root = _find_subdir(tmpdir, "Annotations_6fps")
        if jpeg_root is None or annot_root is None:
            return

        for video_id in sorted(os.listdir(jpeg_root)):
            video_jpeg_dir = os.path.join(jpeg_root, video_id)
            video_annot_dir = os.path.join(annot_root, video_id)
            if not os.path.isdir(video_jpeg_dir) or not os.path.isdir(video_annot_dir):
                continue
            try:
                record = _process_sav_video(
                    video_jpeg_dir, video_annot_dir,
                    downscale_factor, jpeg_quality, min_area_fraction,
                )
                if record is not None:
                    yield record
            except Exception:
                continue


def _iter_sav_raw_samples(raw_dir, downscale_factor, jpeg_quality, min_area_fraction):
    tars = _find_raw_tars(raw_dir)
    print(f"[SegBuild:SA-V] Found {len(tars)} tar(s) in {raw_dir}")
    for tar_path in tars:
        count = 0
        for record in _process_sav_tar(tar_path, downscale_factor, jpeg_quality, min_area_fraction):
            count += 1
            yield record
        print(f"[SegBuild:SA-V] {os.path.basename(tar_path)}: {count} video samples")


# ---------------------------------------------------------------------------
# Unified dataset
# ---------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    """
    Unified SA-1B + SA-V dataset. Just constructing it is enough:
      - If data_dir already has a local dump for this split, it's loaded directly.
      - Otherwise every tar/tar.gz file in sa1b_raw_dir and sav_raw_dir is processed
        right away (see module docstring for the on-disk spec), SA-1B image samples
        and SA-V video samples are pooled together and shuffled (interleaved), and a
        single val_size slice is withheld from that combined pool. Both
        train.json.gz and val.json.gz are written in this one pass — there's one
        finite local pool to split, unlike LAION's live stream — and this instance
        then holds just the requested split.

    __getitem__ returns (image_tensor[1,T,C,H,W], mask_tensor[1,T,H,W], caption
    tokens[1,seq]) — T=1 for SA-1B images, T=num frames for SA-V videos — so
    SAM_adaptive_collate / the SAM training loop treat both sources identically.
    Each access randomly picks one of the sample's available (mask/object, caption)
    pairs, preferring ones that already have tokenized captions.
    """
    def __init__(
        self,
        split: str = "train",
        data_dir: str = SEG_LOCAL_DEFAULT_DIR,
        sa1b_raw_dir: str = SA1B_RAW_DIR,
        sav_raw_dir: str = SAV_RAW_DIR,
        val_size: int = 2000,
        downscale_factor: int = SEG_DOWNSCALE_FACTOR,
        jpeg_quality: int = SEG_JPEG_QUALITY,
        mask_area_threshold: float = SEG_MASK_AREA_THRESHOLD,
        seed: int = 42,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.data_dir = data_dir
        self.sa1b_raw_dir = sa1b_raw_dir
        self.sav_raw_dir = sav_raw_dir
        self.val_size = val_size
        self.downscale_factor = downscale_factor
        self.jpeg_quality = jpeg_quality
        self.mask_area_threshold = mask_area_threshold
        self.seed = seed

        if os.path.exists(self.local_path) and os.path.getsize(self.local_path) > 0:
            self.records = read_json_gz(self.local_path)
        else:
            self.records = self._build_all()

    @property
    def local_path(self) -> str:
        filename = "train.json.gz" if self.split == "train" else "val.json.gz"
        return os.path.join(self.data_dir, filename)

    def _build_all(self) -> list:
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"[SegBuild] No local dump at {self.data_dir} — processing raw tars now.")

        sa1b_samples = list(_iter_sa1b_raw_samples(
            self.sa1b_raw_dir, self.downscale_factor, self.jpeg_quality, self.mask_area_threshold
        ))
        sav_samples = list(_iter_sav_raw_samples(
            self.sav_raw_dir, self.downscale_factor, self.jpeg_quality, self.mask_area_threshold
        ))
        combined = sa1b_samples + sav_samples
        if not combined:
            raise RuntimeError(
                f"No SA-1B or SA-V samples found under {self.sa1b_raw_dir} / "
                f"{self.sav_raw_dir}. Transfer tar/tar.gz files there first."
            )

        rng = random.Random(self.seed)
        rng.shuffle(combined)  # pool SA-1B + SA-V together, interleaved

        val_count = min(self.val_size, len(combined) - 1) if len(combined) > 1 else 0
        val_records = combined[:val_count]
        train_records = combined[val_count:]

        write_json_gz(os.path.join(self.data_dir, "train.json.gz"), train_records)
        write_json_gz(os.path.join(self.data_dir, "val.json.gz"), val_records)
        print(f"[SegBuild] Done: {len(train_records)} train / {len(val_records)} val "
              f"({len(sa1b_samples)} SA-1B, {len(sav_samples)} SA-V before split)")

        return train_records if self.split == "train" else val_records

    def _pick_caption_tokens(self, captions_entry):
        if captions_entry is not None and is_caption_tokenized(captions_entry):
            persona = random.randint(0, 3)
            return torch.tensor(captions_entry[persona], dtype=torch.long).unsqueeze(0)
        return CLIPTokenize("object")

    def _get_image_item(self, rec):
        masks, captions = rec["masks"], rec["captions"]
        tokenized = [i for i, c in enumerate(captions) if is_caption_tokenized(c)]
        i = random.choice(tokenized if tokenized else list(range(len(masks))))

        image = decode_seg_image(rec["image"])
        mask = decode_seg_mask(masks[i])
        img_tensor = image.unsqueeze(0).unsqueeze(0)   # [1, T=1, C, H, W]
        mask_tensor = mask.unsqueeze(0).unsqueeze(0)    # [1, T=1, H, W]
        return img_tensor, mask_tensor, self._pick_caption_tokens(captions[i])

    def _get_video_item(self, rec):
        objects = rec["objects"]
        tokenized = [o for o in objects if is_caption_tokenized(o["captions"])]
        obj = random.choice(tokenized if tokenized else objects)
        object_id = obj["object_id"]

        frame_imgs, frame_masks = [], []
        for frame in rec["frames"]:
            img = decode_seg_image(frame["image"])
            frame_imgs.append(img)
            mask_b64 = next((m["mask"] for m in frame["masks"] if m["object_id"] == object_id), None)
            frame_masks.append(
                decode_seg_mask(mask_b64).float() if mask_b64 is not None
                else torch.zeros(img.shape[-2:], dtype=torch.float)
            )

        img_tensor = torch.stack(frame_imgs).unsqueeze(0)     # [1, T, C, H, W]
        mask_tensor = torch.stack(frame_masks).unsqueeze(0)   # [1, T, H, W]
        return img_tensor, mask_tensor, self._pick_caption_tokens(obj["captions"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return self._get_image_item(rec) if rec["type"] == "image" else self._get_video_item(rec)


def get_segmentation_dataset(split: str = "train", **kwargs) -> SegmentationDataset:
    return SegmentationDataset(split=split, **kwargs)


# ---------------------------------------------------------------------------
# Captioning (used by precomputeCaptions.py — not run during training/dataset build)
# ---------------------------------------------------------------------------
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

        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

        self.processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            use_fast=False
            )
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
        image_pil = TF.to_pil_image(image)
        mask_pil = TF.to_pil_image(mask.float())
        bbox = mask_pil.getbbox()
        if not bbox:
            return "object"
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
                return "invalid_persona_index"

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
