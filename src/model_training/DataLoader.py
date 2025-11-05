import torchvision
import torch
import os

from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.utils.configParser import ConfigParser
from safetensors import safe_open
from safetensors.torch import save_file


def save_images_as_safetensors_separate_chunks(dataset_folder, output_folder='tensor_cache', chunk_size=10000):
    """Save as separate chunk files to avoid OOM"""
    os.makedirs(output_folder, exist_ok=True)

    splits = {
        'train': os.path.join(dataset_folder, 'train'),
        'test': os.path.join(dataset_folder, 'test'),
        'val': os.path.join(dataset_folder, 'val')
    }

    transform = transforms.Compose([transforms.ToTensor()])

    for split_name, split_folder in splits.items():
        if not os.path.exists(split_folder):
            print(f"Skipping {split_name} (folder not found)")
            continue

        print(f"\nProcessing {split_name} set...")
        dataset = torchvision.datasets.ImageFolder(root=split_folder, transform=transform)
        print(f"Found {len(dataset)} images in {split_name}")

        # Process and save chunks immediately
        num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
        chunk_files = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(dataset))

            print(f"Loading chunk {chunk_idx + 1}/{num_chunks} (images {start_idx} to {end_idx})...")

            chunk_images = []
            chunk_labels = []

            for i in tqdm(range(start_idx, end_idx), desc=f"Chunk {chunk_idx + 1}"):
                img, label = dataset[i]
                chunk_images.append(img)
                chunk_labels.append(label)

            # Stack and save immediately
            chunk_images_tensor = torch.stack(chunk_images)
            chunk_labels_tensor = torch.tensor(chunk_labels)

            chunk_file = os.path.join(output_folder, f'{split_name}_chunk_{chunk_idx}.safetensors')
            save_file({
                'images': chunk_images_tensor,
                'labels': chunk_labels_tensor
            }, chunk_file)

            chunk_files.append(f'{split_name}_chunk_{chunk_idx}.safetensors')
            print(f"Saved chunk {chunk_idx + 1}: {chunk_images_tensor.shape}")

            # Free memory immediately
            del chunk_images, chunk_labels, chunk_images_tensor, chunk_labels_tensor

        # Save metadata
        metadata = {
            'classes': dataset.classes,
            'total_samples': len(dataset),
            'chunk_files': chunk_files,
            'chunk_size': chunk_size
        }
        torch.save(metadata, os.path.join(output_folder, f'{split_name}_metadata.pt'))

        print(f"✓ Saved {split_name} in {len(chunk_files)} chunks")
        del dataset


class MultiChunkSafeTensorDataset(Dataset):
    """Dataset that loads from multiple safetensor chunk files"""

    def __init__(self, tensor_folder, split_name, device='cpu'):
        metadata_path = os.path.join(tensor_folder, f'{split_name}_metadata.pt')
        self.metadata = torch.load(metadata_path)
        self.tensor_folder = tensor_folder
        self.device = device
        self.total_samples = self.metadata['total_samples']
        self.chunk_size = self.metadata['chunk_size']
        self.chunk_files = self.metadata['chunk_files']

        # Cache for currently loaded chunk
        self._current_chunk_idx = None
        self._current_file_handle = None
        self._current_images_slice = None
        self._current_labels_slice = None

    def _load_chunk(self, chunk_idx):
        """Load a specific chunk file"""
        if chunk_idx != self._current_chunk_idx:
            # Close previous handle
            if self._current_file_handle is not None:
                del self._current_file_handle

            # Open new chunk
            chunk_file = os.path.join(self.tensor_folder, self.chunk_files[chunk_idx])
            self._current_file_handle = safe_open(chunk_file, framework="pt", device=self.device)
            self._current_images_slice = self._current_file_handle.get_slice('images')
            self._current_labels_slice = self._current_file_handle.get_slice('labels')
            self._current_chunk_idx = chunk_idx

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which chunk this index belongs to
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        # Load chunk if needed
        self._load_chunk(chunk_idx)

        # Get image and label from current chunk
        image = self._current_images_slice[local_idx]
        label = self._current_labels_slice[local_idx]

        return image, label.item()


def load_dataset_from_safetensors_multichunk(tensor_folder='tensor_cache', batch_size=16, device='cpu', num_workers=4):
    """Load dataset from multiple chunk files"""
    print("Setting up multi-chunk lazy loading with safetensors...")

    trainset = MultiChunkSafeTensorDataset(tensor_folder, 'train', device=device)
    testset = MultiChunkSafeTensorDataset(tensor_folder, 'test', device=device)
    validateset = MultiChunkSafeTensorDataset(tensor_folder, 'val', device=device)

    print(f"Train: {len(trainset)}, Test: {len(testset)}, Val: {len(validateset)}")

    # Use num_workers=0 to avoid multiprocessing issues with file handles
    # You can increase this but may need to adjust the dataset class
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    validateloader = DataLoader(validateset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

    print(f"✓ Ready to train!")

    return trainloader, validateloader, testloader

def load_validation_from_safetensors_multichunk(tensor_folder='tensor_cache', batch_size=16, device='cpu', num_workers=4):
    """Load validation dataset from multiple chunk files"""
    print("Setting up multi-chunk lazy loading for validation with safetensors...")

    validateset = MultiChunkSafeTensorDataset(tensor_folder, 'val', device=device)

    print(f"Val: {len(validateset)}")

    # Use num_workers=0 to avoid multiprocessing issues with file handles
    # You can increase this but may need to adjust the dataset class
    validateloader = DataLoader(validateset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)

    print(f"✓ Ready for validation!")

    return validateloader

if __name__ == '__main__':
    config_parser = ConfigParser("config.ini")
    # dataset_folder = config_parser.get("Data Loader", "Dataset Folder")
    # output_tensor_file = config_parser.get("Data Loader", "Output Tensor File")
    for s in ["small","medium"]:
        save_images_as_safetensors_separate_chunks(f"./data/dataset/images/{s}", f"./data/dataset/images/{s}", chunk_size=12500)

