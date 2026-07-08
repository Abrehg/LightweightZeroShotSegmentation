# precomputeCaptions.py
#
# Two-pass caption pipeline for the SA-1B/SA-V segmentation dataset built by
# data/segmentation.py (train.json.gz / val.json.gz under SEG_LOCAL_DEFAULT_DIR).
#
# Each run does, in order:
#   1) Tokenize: any caption slot holding raw text (written by a *previous* run) is
#      tokenized and the text is deleted. This is the automatic "second pass".
#   2) Generate: any caption slot still empty (None) gets 4 new raw-text captions
#      from the InstructBLIP-based CaptionGenerator, one per persona.
#
# So: run this once -> captions appear as text in the JSON for you to inspect. Run
# it again with no changes -> that text is tokenized and removed, replaced by token
# ids; anything newly added in the meantime gets its own text captions in that same
# run, ready to be tokenized on the run after that.
#
# Usage (from the repo root, so the `data` package's relative imports resolve):
#   python -m data.precomputeCaptions [--data-dir PATH] [--device cuda]

import os
import argparse
import torch

from .segmentation import SEG_LOCAL_DEFAULT_DIR, CaptionGenerator, decode_seg_image, decode_seg_mask, is_caption_raw
from .custom400m import read_json_gz, write_json_gz
from models.clip_model import CLIPTokenize

NUM_PERSONAS = 4


def _tokenize_entry(text_entry):
    return [CLIPTokenize(text).squeeze(0).tolist() for text in text_entry]


def _iter_caption_slots(records):
    """Yields (get_media, current_captions, set_captions) for every mask (image
    records) / tracked object (video records) in the dataset."""
    for rec in records:
        if rec["type"] == "image":
            for i in range(len(rec["masks"])):
                def get_media(rec=rec, i=i):
                    return decode_seg_image(rec["image"]), decode_seg_mask(rec["masks"][i])

                def set_captions(value, rec=rec, i=i):
                    rec["captions"][i] = value

                yield get_media, rec["captions"][i], set_captions
        else:
            for obj in rec["objects"]:
                def get_media(rec=rec, obj=obj):
                    frame = rec["frames"][obj["rep_frame_idx"]]
                    image = decode_seg_image(frame["image"])
                    mask_b64 = next(m["mask"] for m in frame["masks"] if m["object_id"] == obj["object_id"])
                    return image, decode_seg_mask(mask_b64)

                def set_captions(value, obj=obj):
                    obj["captions"] = value

                yield get_media, obj["captions"], set_captions


def _process_split(path, caption_generator_holder, device):
    if not os.path.exists(path):
        print(f"  {path} not found, skipping.")
        return

    records = read_json_gz(path)
    tokenized_count, generated_count = 0, 0

    # Pass 2 (of the *previous* run's output): tokenize any raw text, delete it.
    for _, captions, set_captions in _iter_caption_slots(records):
        if is_caption_raw(captions):
            set_captions(_tokenize_entry(captions))
            tokenized_count += 1

    # Pass 1 (of *this* run): generate captions for anything still empty.
    for get_media, captions, set_captions in _iter_caption_slots(records):
        if captions is not None:
            continue
        if caption_generator_holder[0] is None:
            print(f"  Loading InstructBLIP captioning model on {device}...")
            caption_generator_holder[0] = CaptionGenerator(device=device)
        caption_generator = caption_generator_holder[0]

        image, mask = get_media()
        captions_out = [caption_generator.generate_all_captions(image, mask, p) for p in range(NUM_PERSONAS)]
        set_captions(captions_out)
        generated_count += 1
        if generated_count % 100 == 0:
            print(f"  {os.path.basename(path)}: {generated_count} new caption sets generated...")

    if tokenized_count or generated_count:
        write_json_gz(path, records)
    print(f"  {os.path.basename(path)}: tokenized {tokenized_count} previous entries, "
          f"generated {generated_count} new caption sets.")


def main():
    parser = argparse.ArgumentParser(description="Two-pass captioning for the SA-1B/SA-V segmentation dataset")
    parser.add_argument("--data-dir", type=str, default=SEG_LOCAL_DEFAULT_DIR,
                         help="Directory holding train.json.gz / val.json.gz (see data/segmentation.py)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    caption_generator_holder = [None]  # lazy-loaded only if generation is actually needed

    for split_file in ("train.json.gz", "val.json.gz"):
        path = os.path.join(args.data_dir, split_file)
        print(f"Processing {path}...")
        _process_split(path, caption_generator_holder, args.device)

    print("Done.")


if __name__ == "__main__":
    main()
