import os
import json
import torch
from torchvision import transforms
import torchvision.transforms.functional as F_tv # Alias to avoid conflict
from PIL import Image, UnidentifiedImageError
from pycocotools import mask as coco_mask # For RLE encoding
import numpy as np
import time
from collections import defaultdict

# --- Model Imports (adjust paths as per your project structure) ---
from models.SAM_model import VideoSAM 
from models.clip_model import create_text_encoder, CLIPTokenize 
from models.prior_model import create_prior
from models.distill_model import DistilledMemoryStudent

# --- OVIS API Helper Class (COCO-inspired, from previous discussion) ---
class OVISApi:
    def __init__(self, annotation_file=None):
        self.dataset = {}
        self.anns = {}
        self.vidToAnns = defaultdict(list)
        self.catToVids = defaultdict(list)
        self.vids = {}
        self.cats = {}
        if annotation_file:
            print(f"Loading OVIS annotations into memory from: {annotation_file}...")
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert isinstance(dataset, dict), f"Annotation file {annotation_file} does not load as a dict."
            print(f"Done (t={time.time() - tic:.2f}s)")
            self.dataset = dataset
            self._create_index()

    def _create_index(self):
        print("Creating OVISApi index...")
        anns, cats, vids = {}, {}, {}
        vidToAnns = defaultdict(list)
        catToVids = defaultdict(list)

        if 'annotations' in self.dataset and self.dataset['annotations'] is not None:
            for ann in self.dataset['annotations']:
                vidToAnns[ann['video_id']].append(ann['id'])
                anns[ann['id']] = ann
        
        if 'videos' in self.dataset and self.dataset['videos'] is not None:
            for vid in self.dataset['videos']:
                vids[vid['id']] = vid

        if 'categories' in self.dataset and self.dataset['categories'] is not None:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and self.dataset['annotations'] is not None and \
           'categories' in self.dataset and self.dataset['categories'] is not None:
            for ann in self.dataset['annotations']:
                if 'category_id' in ann:
                    catToVids[ann['category_id']].append(ann['video_id'])
        
        self.anns, self.vidToAnns, self.catToVids, self.vids, self.cats = anns, vidToAnns, catToVids, vids, cats
        print("OVISApi index created.")

    def getAnnIds(self, vidIds=[], catIds=[]):
        vidIds = vidIds if isinstance(vidIds, list) else [vidIds]
        catIds = catIds if isinstance(catIds, list) else [catIds]
        
        current_annotations = self.dataset.get('annotations', [])
        if current_annotations is None: current_annotations = []

        if not vidIds and not catIds: 
            anns_data = current_annotations
        else:
            if vidIds:
                anns_data_intermediate = []
                for vidId in vidIds:
                    for ann_id in self.vidToAnns.get(vidId, []):
                        if ann_id in self.anns:
                            anns_data_intermediate.append(self.anns[ann_id])
                anns_data = anns_data_intermediate
            else: 
                anns_data = current_annotations
            if catIds:
                anns_data = [ann for ann in anns_data if ann.get('category_id') in catIds]
        return [ann['id'] for ann in anns_data]

    def getVidIds(self, vidIds=[], catIds=[]):
        vidIds = vidIds if isinstance(vidIds, list) else [vidIds]
        catIds = catIds if isinstance(catIds, list) else [catIds]
        ids = set(vidIds) if vidIds else set(self.vids.keys())
        if catIds:
            vids_with_cats = set()
            first = True
            for catId in catIds:
                v_for_c = set(self.catToVids.get(catId, []))
                if first: vids_with_cats = v_for_c; first = False
                else: vids_with_cats &= v_for_c
            if vidIds: ids &= vids_with_cats
            else: ids = vids_with_cats
        return list(ids)

    def loadAnns(self, ids=[]):
        if isinstance(ids, list): return [self.anns[id] for id in ids if id in self.anns]
        elif isinstance(ids, int): return [self.anns[ids]] if ids in self.anns else []
        return []

    def loadVids(self, ids=[]):
        if isinstance(ids, list): return [self.vids[id] for id in ids if id in self.vids]
        elif isinstance(ids, int): return [self.vids[ids]] if ids in self.vids else []
        return []
    
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """ get cat ids for given filter values (categories, supercategories, ids)
        :param catNms (str array): category names
        :param supNms (str array): supercategory names
        :param catIds (int array): category ids
        :return: ids (int array)  : integer array of category ids
        """
        catNms = catNms if isinstance(catNms, list) else [catNms]
        supNms = supNms if isinstance(supNms, list) else [supNms]
        catIds = catIds if isinstance(catIds, list) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats_data = self.dataset.get('categories', [])
        else:
            cats_data = self.dataset.get('categories', [])
            if not isinstance(cats_data, list): cats_data = []

            if len(catNms) > 0:
                cats_data = [cat for cat in cats_data if cat['name'] in catNms]
            if len(supNms) > 0:
                cats_data = [cat for cat in cats_data if cat['supercategory'] in supNms]
            if len(catIds) > 0:
                cats_data = [cat for cat in cats_data if cat['id'] in catIds]
        ids = [cat['id'] for cat in cats_data]
        return ids

    def loadCats(self, ids=[]):
        """ load cats with the specified ids
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if isinstance(ids, list):
            return [self.cats[id] for id in ids if id in self.cats]
        elif isinstance(ids, int):
            return [self.cats[ids]] if ids in self.cats else []
        return []

# --- Helper Functions ---
def load_model(model_class, checkpoint_path, device, **kwargs):
    # Filter out None kwargs, as some models might not expect them
    valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    model = model_class(**valid_kwargs).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint for {model_class.__name__} from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path} for {model_class.__name__}. Model is randomly initialized.")
    model.eval()
    return model

def masks_to_rle(masks_tensor_th): # Input (T, H, W) or (T, 1, H, W) binary [0,1] torch tensor
    """Converts a batch of binary masks to a list of RLEs."""
    if masks_tensor_th.ndim == 4:
        masks_tensor_th = masks_tensor_th.squeeze(1) # (T, H, W)
    
    rle_list = []
    for i in range(masks_tensor_th.shape[0]):
        mask_frame_np = masks_tensor_th[i].cpu().numpy().astype(np.uint8) # Ensure uint8
        if np.any(mask_frame_np): 
            rle = coco_mask.encode(np.asfortranarray(mask_frame_np))
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('utf-8')
            rle_list.append(rle)
        else:
            rle_list.append(None) 
    return rle_list

def process_video(video_frames_paths, transform_fn):
    """Loads and transforms a list of video frames."""
    frames = []
    original_dims = None
    for frame_path in video_frames_paths:
        try:
            img = Image.open(frame_path).convert("RGB")
            if original_dims is None:
                original_dims = (img.height, img.width)
            img_tensor = transform_fn(img)
            frames.append(img_tensor)
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Error loading frame {frame_path}: {e}. Skipping frame.")
            return None, None 
    if not frames:
        return None, None
    # Stacking here, device placement will be done in the main loop
    return torch.stack(frames), original_dims

def run_inference():
    # --- Configuration ---
    config = {
        "ovis_root_dir": '/Users/adityaasuratkal/Downloads/Research_Datasets/OVIS', # IMPORTANT: Set this
        "output_dir": "./ovis_inference_results",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        
        # Teacher (VideoSAM) Checkpoints and Parameters
        "sam_checkpoint": "/path/to/your/sam_decoder_epoch_X.pt", # IMPORTANT: Set this
        "text_encoder_checkpoint": "/path/to/your/clip_text_epoch_X.pt", # IMPORTANT: Set this
        "prior_checkpoint": "/path/to/your/prior_epoch_X.pt", # IMPORTANT: Set this

        # Student (DistilledMemoryStudent) Checkpoints and Parameters
        "student_checkpoint": "/path/to/your/student_phase_student_epoch_X.pt",

        # Common Inference Parameters
        "text_prompt": "an object", # Generic prompt
        "segmentation_threshold": 0.5,
    }

    device = torch.device(config["device"])
    os.makedirs(config["output_dir"], exist_ok=True)

    # --- 1. Load Models ---
    print("Loading models...")
    
    # Teacher Model (VideoSAM)
    teacher_sam_model = load_model(VideoSAM, config["sam_checkpoint"], device)
    
    # Components for generating text prior (shared by teacher and potentially student)
    text_encoder = load_model(create_text_encoder, config["text_encoder_checkpoint"], device)
    # Assuming create_prior() doesn't take arguments related to input/output dims directly,
    # but is trained to map from text_encoder's output to text_embed_dim_for_sam_prior
    prior_model = load_model(create_prior, config["prior_checkpoint"], device)

    # Student Model (DistilledMemoryStudent)
    student_model = load_model(DistilledMemoryStudent, config["student_checkpoint"], device)
    
    print("Models loaded.")

    # --- 2. Image Transform ---
    img_transform = transforms.Compose([transforms.ToTensor()])

    # --- 3. Process Splits ---
    for split in ['valid', 'test']:
        print(f"\n--- Processing split: {split} ---")
        ovis_api = OVISApi(os.path.join(config["ovis_root_dir"], f'annotations_{split}.json'))
        if not ovis_api.dataset or not ovis_api.cats: # Check if categories were loaded
            print(f"No categories found or annotation data loaded for split '{split}'. Skipping.")
            continue
        
        categories = ovis_api.dataset.get('categories', [])
        if not categories:
            print(f"No 'categories' list in annotation JSON for split '{split}'. Skipping.")
            continue
        print(f"Found {len(categories)} categories for split '{split}'.")

        all_results_teacher_split = []
        all_results_student_split = []

        for cat_idx, category_info in enumerate(categories):
            category_id = category_info['id']
            category_name = category_info['name']
            print(f"\n  Processing Category {cat_idx+1}/{len(categories)}: ID={category_id}, Name='{category_name}'")

            # --- Prepare Text Prior Embedding for current category ---
            current_text_prompt = category_name 
            tokenized_text = CLIPTokenize(current_text_prompt) 
            if tokenized_text.ndim == 1: tokenized_text = tokenized_text.unsqueeze(0)
            tokenized_text = tokenized_text.to(device)

            with torch.no_grad():
                text_embedding = text_encoder(tokenized_text)
                text_prior_embedding_for_sam = prior_model(text_embedding) 
                student_token_ids_input = tokenized_text # For DistilledMemoryStudent

            video_ids = ovis_api.getVidIds() # Process all videos for this category prompt
            print(f"    Found {len(video_ids)} videos to process for category '{category_name}'.")
            
            for i, video_id in enumerate(video_ids):
                # print(f"      Processing video {i+1}/{len(video_ids)} (ID: {video_id}) for category '{category_name}'")
                video_info_list = ovis_api.loadVids(ids=[video_id])
                if not video_info_list:
                    # print(f"        Could not load video info for video_id: {video_id}. Skipping.")
                    continue
                video_info = video_info_list[0]

                frame_relative_paths = video_info.get('file_names')
                if not frame_relative_paths:
                    # print(f"        'file_names' not found for video_id: {video_id}. Skipping.")
                    continue

                full_frame_paths = [os.path.join(config["ovis_root_dir"], split, rel_path) for rel_path in frame_relative_paths]
                video_tensor_cpu, original_dims = process_video(full_frame_paths, img_transform)
                if video_tensor_cpu is None:
                    # print(f"        Failed to process frames for video {video_id}. Skipping.")
                    continue
                video_tensor_input_device = video_tensor_cpu.to(device) 
                original_H, original_W = original_dims

                # --- Run Teacher (VideoSAM) Model Inference ---
                with torch.no_grad():
                    teacher_pred_logits = teacher_sam_model(video_tensor_input_device.unsqueeze(0), text_prior_embedding_for_sam) 
                    teacher_pred_logits = teacher_pred_logits.squeeze(0) 
                teacher_pred_probs = torch.sigmoid(teacher_pred_logits)
                teacher_binary_masks_model_res = (teacher_pred_probs > config["segmentation_threshold"]).byte()
                teacher_binary_masks_orig_res = F_tv.interpolate(teacher_binary_masks_model_res.float(), size=(original_H, original_W), mode='nearest').byte()
                teacher_rle_segmentations = masks_to_rle(teacher_binary_masks_orig_res)
                try: video_id_int = int(video_id)
                except ValueError: video_id_int = str(video_id) 
                teacher_result_entry = {"video_id": video_id_int, "category_id": category_id, "segmentations": teacher_rle_segmentations, "score": teacher_pred_probs.mean().item()}
                all_results_teacher_split.append(teacher_result_entry)

                # --- Run Student (DistilledMemoryStudent) Model Inference ---
                with torch.no_grad():
                    student_pred_logits = student_model(video_tensor_input_device, student_token_ids_input.to(device)) 
                student_pred_probs = torch.sigmoid(student_pred_logits)
                student_binary_masks_model_res = (student_pred_probs > config["segmentation_threshold"]).byte()
                student_binary_masks_orig_res = F_tv.interpolate(student_binary_masks_model_res.float(), size=(original_H, original_W), mode='nearest').byte()
                student_rle_segmentations = masks_to_rle(student_binary_masks_orig_res)
                student_result_entry = {"video_id": video_id_int, "category_id": category_id, "segmentations": student_rle_segmentations, "score": student_pred_probs.mean().item()}
                all_results_student_split.append(student_result_entry)

        # --- Save results for the split (now contains results for all categories) ---
        output_teacher_json_path = os.path.join(config["output_dir"], f"ovis_results_teacher_{split}_per_category.json")
        print(f"Saving {len(all_results_teacher_split)} teacher results for split '{split}' to {output_teacher_json_path}")
        with open(output_teacher_json_path, 'w') as f: json.dump(all_results_teacher_split, f)

        output_student_json_path = os.path.join(config["output_dir"], f"ovis_results_student_{split}_per_category.json")
        print(f"Saving {len(all_results_student_split)} student results for split '{split}' to {output_student_json_path}")
        with open(output_student_json_path, 'w') as f: json.dump(all_results_student_split, f)
            
    print("\nInference completed.")

if __name__ == "__main__":
    run_inference()
