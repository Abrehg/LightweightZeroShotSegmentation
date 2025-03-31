import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms

# Constants
IMAGE_SIZE = 256
TIMEOUT = 10

def get_laion_train_dataset():
    """Returns streaming IterableDataset for training (all data, including NSFW)"""
    class LAIONIterableDataset(IterableDataset):
        def __init__(self, transform=None):
            self.hf_dataset = load_dataset(
                "laion/laion400m-multi",
                split="train",
                streaming=True
            )

        def _process_sample(self, sample):
            try:
                # Download and process image
                response = requests.get(sample["URL"], timeout=TIMEOUT)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                return image, sample["TEXT"]
            except:
                # Return placeholder with empty text
                return (
                    torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
                    ""
                )

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            dataset = self.hf_dataset
            
            # Handle multi-worker sharding
            if worker_info is not None:
                dataset = dataset.shard(
                    num_shards=worker_info.num_workers,
                    index=worker_info.id
                )
                
            for sample in dataset:
                yield self._process_sample(sample)

    return LAIONIterableDataset(transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ]))

def get_laion_test_dataset(num_samples=5):
    """Returns small test Dataset with first N samples"""
    class LAIONMapDataset(Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            try:
                response = requests.get(sample["URL"], timeout=TIMEOUT)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                return image, sample["TEXT"]
            except:
                return (
                    torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
                    ""
                )

    # Load and cache first N samples
    hf_dataset = load_dataset("laion/laion400m-multi", split="train", streaming=True)
    test_samples = list(hf_dataset.take(num_samples))
    
    return LAIONMapDataset(test_samples, transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ]))

# Usage example
if __name__ == "__main__":
    # For training
    train_dataset = get_laion_train_dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=4  # Enables parallel fetching
    )

    # For testing
    test_dataset = get_laion_test_dataset(5)
    test_loader = DataLoader(test_dataset, batch_size=5)

    # Test the datasets
    print("Test samples:")
    for images, texts in test_loader:
        print(f"Images shape: {images.shape}")
        print(f"Texts: {texts}")

    print("\nTraining stream started:")
    for i, (batch_images, batch_texts) in enumerate(train_loader):
        print(f"Batch {i} - Images: {batch_images.shape}, Texts: {len(batch_texts)}")
        # Add your training loop here
        if i >= 2:  # Just show 3 batches for demonstration
            break