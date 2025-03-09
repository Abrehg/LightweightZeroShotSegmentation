# dataset.py
import os
import json
import pyarrow.parquet as pq
import io
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import clip

class CLIPTokenizer:
    def __init__(self):
        self.tokenizer = clip.tokenize
        self.context_length = 77  # CLIP's standard context length
        
    def __call__(self, text):
        # Returns tensor of shape (context_length,)
        return self.tokenizer(text, truncate=True).squeeze(0)

class COCODataset(Dataset):
    def __init__(self, root_dir, ann_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = CLIPTokenizer()
        
        with open(ann_path) as f:
            self.annotations = json.load(f)['annotations']
        
        # Build image ID to file path mapping
        self.image_paths = {
            img['id']: os.path.join(root_dir, img['file_name'])
            for img in json.load(open(os.path.join(root_dir, 'images_info.json')))
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_path = self.image_paths[ann['image_id']]
        image = Image.open(image_path).convert('RGB')
        text = ann['caption']
        
        if self.transform:
            image = self.transform(image)
            
        if self.tokenizer:
            text = self.tokenizer(text)
            
        return image, text

class CC3MDataset(Dataset):
    def __init__(self, tsv_path, image_root, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.tokenizer = CLIPTokenizer()
        
        with open(tsv_path) as f:
            self.samples = [line.strip().split('\t') for line in f]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, caption = self.samples[idx]
        image_path = os.path.join(self.image_root, rel_path)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.tokenizer:
            text = self.tokenizer(caption)
            
        return image, text

class Custom400MDataset(Dataset):
    def __init__(self, parquet_root, transform=None):
        self.parquet_files = [
            os.path.join(parquet_root, f) 
            for f in os.listdir(parquet_root) 
            if f.endswith('.parquet')
        ]
        self.transform = transform
        self.tokenizer = CLIPTokenizer()
        
    def __len__(self):
        return len(self.parquet_files) * 100000  # Approximate per-file size

    def __getitem__(self, idx):
        # Implement memory-mapped parquet loading
        file_idx = idx // 100000
        sample_idx = idx % 100000
        
        # Using pyarrow for efficient loading
        table = pq.read_table(self.parquet_files[file_idx])
        sample = table.slice(sample_idx, 1).to_pydict()
        
        image = Image.open(io.BytesIO(sample['image'][0])).convert('RGB')
        text = sample['text'][0]
        
        if self.transform:
            image = self.transform(image)
            
        if self.tokenizer:
            text = self.tokenizer(text)
            
        return image, text

def build_datasets(config, transform):
    datasets = []
    
    # Load COCO
    if config.get('coco_path'):
        coco_train = COCODataset(
            root_dir=os.path.join(config['coco_path'], 'train2017'),
            ann_path=os.path.join(config['coco_path'], 'annotations/captions_train2017.json'),
            transform=transform
        )
        datasets.append(coco_train)
    
    # Load CC3M
    if config.get('cc3m_path'):
        cc3m = CC3MDataset(
            tsv_path=os.path.join(config['cc3m_path'], 'Train_GCC-training.tsv'),
            image_root=os.path.join(config['cc3m_path'], 'images'),
            transform=transform
        )
        datasets.append(cc3m)
    
    # Load Custom 400M
    if config.get('custom400m_path'):
        custom_ds = Custom400MDataset(
            parquet_root=config['custom400m_path'],
            transform=transform
        )
        datasets.append(custom_ds)
    
    return ConcatDataset(datasets)