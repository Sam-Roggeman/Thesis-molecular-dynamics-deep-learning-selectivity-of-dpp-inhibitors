import torch
import concurrent.futures
from pathlib import Path
import tarfile
import numpy as np
from pathlib import Path
import tempfile
import os

import torchvision
from tqdm import tqdm
import torch.nn.functional as F

from src.model_training.DataLoader import DataLoader, save_as_safetensor
from src.utils.configParser import ConfigParserWrapper
from src.model_training.train_test_split import train_val_test_split, create_splits_randomsplit, \
    create_ligand_splits_1_train_1_valtest

from torch.utils.data import TensorDataset

from src.utils.utils import get_binding_classes, parse_pdb_from_string, parse_filename


def load_trajectory_fast(tar_path, progress_id=None):
    """Modified to show progress for individual trajectories"""

    all_coords = []
    with tarfile.open(tar_path, 'r') as tar:
        # Process members in the order they appear in tar (fastest)
        # loop over every 10th frame
        members = tar.getmembers()

        # only process 10% of the frames
        for i in range(0, 10001, 10):
            member = members[i]
            if not (member.name.startswith('frame_') and member.name.endswith('.pdb')):
                continue
            # Parse directly from tar
            extracted_file = tar.extractfile(member)
            if extracted_file:
                pdb_content = extracted_file.read().decode('utf-8')
                coords = parse_pdb_from_string(pdb_content)
                all_coords.append(coords)
            if i % 1000 == 0 and progress_id is not None:
                print(f"Trajectory {progress_id}: Processed frame {i}/{len(members)}")
    tensor = torch.stack(all_coords)
    filename = Path(tar_path).name

    return tensor, filename


def process_single_trajectory(tar_path, progress_id):
    """Modified to accept progress ID"""
    return load_trajectory_fast(tar_path, progress_id)


def process_parallel(raw_data_path, max_workers=2):
    """Process individual tensors with progress bar"""

    tar_files = sorted(Path(raw_data_path).glob("*"))
    trajectory_tensors = []
    filenames = []
    print(f"Processing {len(tar_files)} individual trajectories...")

    with tqdm(total=len(tar_files), desc="Trajectories") as pbar:
        if max_workers == 1:
            # Process sequentially
            for idx, tf in enumerate(tar_files):
                tensor, filename = process_single_trajectory(tf, idx)
                trajectory_tensors.append(tensor)
                filenames.append(filename)

                pbar.update(1)
                pbar.set_postfix({
                    "Latest": tf.name,
                    "Shape": str(tensor.shape)
                })
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_file = {executor.submit(process_single_trajectory, tf, idx): tf for idx, tf in enumerate(tar_files)}

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    tensor, filename = future.result()
                    trajectory_tensors.append(tensor)
                    filenames.append(filename)


                    pbar.update(1)
                    pbar.set_postfix({
                        "Latest": future_to_file[future].name,
                        "Shape": str(tensor.shape)
                    })

    return trajectory_tensors, filenames


def pad_and_size(tensor, target_size, pad_value=0):
    """Pad tensor
    :param tensor: Tensor of shape (frames, nr_atoms, 3)
    :param target_size: int, target number
    :param pad_value: value to use for padding
    :return: padded tensor and the nr of real atoms before padding (size)
    """
    start_size = tensor.shape[1]
    # if padding is needed
    if start_size < target_size:
        pad_size = target_size - start_size
        tensor = F.pad(tensor, (0, 0, 0, pad_size), value=pad_value)
    return tensor, start_size
def apply_padding(tensors, target_size=168*168):
    # target_size = size^2 of the image we want
    pad_value = 0
    # Pad all tensors and create size list
    padded_data = []
    sizes = []
    print(f"Padding trajectories to size {target_size}")
    for i, tensor in enumerate(tensors):
        _temp_tensor, _size = pad_and_size(tensor=tensor, target_size=target_size, pad_value=pad_value)
        padded_data.append(_temp_tensor)
        sizes.append(_size)
        print(f"Trajectory {i}: padded from {tensor.shape[1]} atoms to {target_size}")
    print(f"Finished padding tensors")
    # Stack results
    data_tensor = torch.stack(padded_data)  # Shape: (trajectories, frames, target_size, 3)
    size = data_tensor.shape

    return data_tensor, size


def merge_tensors(tensor_dict):
    """Merge a list of tensors into a single tensor by concatenation along the first dimension.
    :param tensor_dict: dict of {class_name: [tensors]}
    :return: a tensor containing the whole dataset, labelled by class_name
    """
    merged_tensors = {}
    for class_name, tensor_list in tensor_dict.items():
        if tensor_list:
            merged_tensor = torch.cat(tensor_list, dim=0)
            merged_tensors[class_name] = merged_tensor
            print(f"Merged class {class_name}: {merged_tensor.shape}")
        else:
            print(f"No tensors to merge for class {class_name}")
    return merged_tensors

def classify_tensors(trajectory_tensors, filenames):
    """
    Classify tensors based on filename substrings.
    :param trajectory_tensors:
    """
    print(f"Classifying {len(trajectory_tensors)} tensors")
    classes = get_binding_classes()
    # empty dict to hold classified tensors
    classified_tensors = {cls: [] for cls in classes}
    # add the tensors to their classes
    for i, tensor in enumerate(trajectory_tensors):
        print(f"Trajectory {i}: {tensor.shape}")
        filename = filenames[i].replace('.tar.gz', '')

        dpp_class, ligand_name, binding_type = parse_filename(filename)
        class_name = binding_type
        tensor.ligand_name = ligand_name  # attach ligand name to tensor for later use
        classified_tensors[class_name].append(tensor)
        print(f"\tClassified as {class_name}")

    return classified_tensors

def create_torch_dataset_from_tensors(classified_tensors):
    """Create PyTorch datasets from classified tensors.
    :param classified_tensors: dict of {class_name: merged_tensor}
    :return: tensor dataset
    """
    data = []
    labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(classified_tensors.keys())}
    print(f"Creating PyTorch dataset from tensors")
    for class_name, tensor in classified_tensors.items():
        print(f"Class {class_name}: {tensor.shape}")
        data.append(tensor)
        labels.append(torch.full((tensor.shape[0],), class_to_idx[class_name]))
    # Concatenate all data and labels
    data_tensor = torch.cat(data, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    print(f"Final dataset shape: {data_tensor.shape}, labels shape: {labels_tensor.shape}")
    dataset = TensorDataset(data_tensor, labels_tensor)
    return dataset
def save_dataset(dataset, dataset_name,dataset_folder='./data/dataset/tensors',  chunk_size=10000):
    """Save as separate chunk files to avoid OOM"""
    os.makedirs(dataset_folder, exist_ok=True)
    print(f"\nProcessing {dataset_name} set...")
    save_as_safetensor(dataset, dataset_folder, dataset_name, chunk_size)
    print(f"✓ Saved {dataset_name} dataset to {dataset_folder}")

def save_datasets(train_dataset, val_dataset, test_dataset, size_ratio):
    """Save datasets to folders"""
    # reverse the sizes dict to have size_name -> ratio
    print("Saving datasets to folders...")

    for split in [(train_dataset, 'train'), (val_dataset, 'val'), (test_dataset, 'test')]:
        dataset, split_name = split
        # for each size
        for size_name, ratio in size_ratio.items():
            subset_size = int(len(dataset) * ratio)
            # create a subset of the dataset
            subset, _ = torch.utils.data.random_split(dataset, [subset_size, len(dataset) - subset_size])
            save_dataset(subset, f"{split_name}_{size_name}", dataset_folder=f'./data/dataset/tensors/{size_name}/{split_name}', chunk_size=5000)
            print( f"✓ Saved {split_name} dataset of size {size_name} ({subset_size} samples, ratio {ratio*100:.2f}%).")
    print( "✓ All datasets saved successfully.")

def save_classified_tensors(classified_tensors, dataset_folder='./data/dataset/tensors/raw_classified', chunk_size=10000):
    """
    Save classified tensors separately. Uses chunking to avoid OOM.
    The folder structure will be: /dataset_folder/enzyme/binding_type/chunk_0.safetensors
    :param classified_tensors: dict of {enzyme:{binding_type: [tensors]}}
    :param dataset_folder: folder to save datasets
    :param chunk_size: chunk size for saving
    """
    os.makedirs(dataset_folder, exist_ok=True)
    print(f"\nSaving classified tensors to {dataset_folder}...")
    for dpp_name, classified_entry in classified_tensors.items():
        path = os.path.join(dataset_folder, dpp_name)
        os.makedirs(path, exist_ok=True)
        for class_name, tensor_list in classified_entry.items():
            path = os.path.join(dataset_folder, dpp_name, class_name)
            os.makedirs(path, exist_ok=True)
            # save the tensor      


if __name__ == "__main__":
    configparser = ConfigParserWrapper()
    raw_data_path = configparser.get_raw_data_folder()

    logs_parent_dir, logging_enabled, console_enabled = configparser.get_logging()
    log_subdir = "data_preprocessing"
    log_dir = os.path.join(logs_parent_dir, log_subdir)
    print("Starting data preprocessing...")
    # parallel processing
    trajectory_tensors,filenames = process_parallel(raw_data_path, max_workers=6) # [Shape: (10.001, nr_atoms, 3)]
    # Pad the tensors
    trajectory_tensors, trajectory_sizes = apply_padding(trajectory_tensors, 168*168) # Shape: (18, 10.001, 168*168, 3)

    # Classify the tensors
    classified_tensors = classify_tensors(trajectory_tensors, filenames) # {class_name: [tensors]}
    del trajectory_tensors, filenames, trajectory_sizes

    # # split the tensor into train, validation and test
    # train_dataset, val_dataset, test_dataset = create_traject_splits_6_1_1(classified_tensors)
    # # Save datasets to folders
    # save_datasets(train_dataset, val_dataset, test_dataset, size_ratio={"traject_splits_6_1_1_10%": 1.0})

    # split the tensor into train, validation and test
    classified_train_tensors, classified_val_tensors, classified_test_tensors = create_ligand_splits_1_train_1_valtest(classified_tensors)
    train_dataset = create_torch_dataset_from_tensors(merge_tensors(classified_train_tensors))
    val_dataset = create_torch_dataset_from_tensors(merge_tensors(classified_val_tensors))
    test_dataset = create_torch_dataset_from_tensors(merge_tensors(classified_test_tensors))
    # Save datasets to folders
    save_datasets(train_dataset, val_dataset, test_dataset, size_ratio={"ligand_splits_1_train_1_valtest_10%": 1.0})


    # Merge tensors in each class
    merged_classified_tensors = merge_tensors(classified_tensors)  # {class_name: merged_tensor}
    del classified_tensors, classified_train_tensors, classified_val_tensors, classified_test_tensors

    # PyTorch dataset from tensors
    dataset = create_torch_dataset_from_tensors(merged_classified_tensors)

    # split the tensor into train, validation and test
    train_dataset, val_dataset, test_dataset = create_splits_randomsplit(dataset)
    # Save datasets to folders
    save_datasets(train_dataset, val_dataset, test_dataset, size_ratio={"random_split_10%": 1.0})

    print("✓ Data preprocessing completed successfully.")








