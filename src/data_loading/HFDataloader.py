import shutil

import datasets
import os

import torch

from src.utils.training_config import TrainingConfig

import resource

def _download_dataset(config: TrainingConfig, splits=None) -> datasets.DatasetDict:
    print("Downloading dataset...")
    dataset_size = config.dataset_size
    assert dataset_size > 0 and dataset_size <= 1, "Dataset size must be between 0 and 1"
    print(f"\tDownloading {dataset_size} of {config.dataset_location}")
    # download only a subset of the dataset if dataset_size < 1
    split_size = int(dataset_size * 100)
    cache_dir = os.environ.get("HF_DOWNLOADED_DATASET_DIR")
    

    if splits is not None:
        print(f"\t\tOnly downloading splits: {splits}")
        _split_arg = {split: f"{split}[:{split_size}%]" for split in splits}
    else:
        for split in ['train', 'validation', 'test']:
            print(f"\t\tDownloading {split_size}% of {split} split")
        _split_arg = {
            "train": f"train[:{split_size}%]",
            "validation": f"validation[:{split_size}%]",
            "test": f"test[:{split_size}%]",
        }    
    # cache dir
    print(f"\tUsing cache directory: {cache_dir}")
    dataset_dict: datasets.DatasetDict = datasets.load_dataset(
        config.dataset_location,
        split=_split_arg,
        cache_dir=os.environ.get("HF_DOWNLOADED_DATASET_DIR"),
        token=os.environ.get("HF_TOKEN"),
        num_proc= config.num_cpus
    )
    print("\t...downloading_dataset complete")
    # wrap the dataset in a DatasetDict if it's not already one
    if not isinstance(dataset_dict, datasets.DatasetDict):
        dataset_dict = datasets.DatasetDict({splits[0]: dataset_dict})
    return dataset_dict

def _download_streaming_dataset(config: TrainingConfig, splits: list = ["train", "validation", "test"]) -> datasets.IterableDatasetDict:
    dataset_size = config.dataset_size
    assert dataset_size > 0 and dataset_size <= 1, "Dataset size must be between 0 and 1"
    print(f"\tDownloading {dataset_size} of {config.dataset_location}")
    dataset: datasets.IterableDatasetDict = datasets.load_dataset(
        config.dataset_location,
        token=os.environ.get("HF_TOKEN"),
        streaming=True
    )
    # shuffle the trainssplit of the dataset
    dataset["train"] = dataset["train"].shuffle(seed=config.shuffle_seed, buffer_size=config.shuffle_buffer_size)
    if dataset_size < 1:
        splitinfo: datasets.DatasetInfo = dataset["train"].info
        for split in dataset.keys():
            try:
                total_samples = splitinfo.splits[split].num_examples
                n_samples = int(total_samples * config.dataset_size)
                n_samples = max(config.batch_size, n_samples)
                n_samples = ((n_samples + config.batch_size - 1) // config.batch_size) * config.batch_size
                print(f"\tUsing streaming subset for {split}: {n_samples}/{total_samples} samples")
                dataset[split] = dataset[split].take(n_samples)
            except Exception:
                print(
                    f"Warning: Could not determine split size for streaming subset on {split}. "
                    "Falling back to full streamed split."
                )
    print("\t...downloading_streaming_dataset complete")
    # take only the specified splits
    return datasets.IterableDatasetDict({split: dataset[split] for split in splits if split in dataset})   

# Define a type for the dataloader dict
DataLoaderDict = dict[str, torch.utils.data.DataLoader]

def initialize_dataloaders(config: TrainingConfig, splits=None, keep_all_columns: bool = False) -> DataLoaderDict:
    """
    Initialize the dataloader for training.
    """
    if splits is None:
        splits = ['train', 'validation', 'test']
    # Download the dataset
    print("Initializing dataloader...")
    print("\tDownloading dataset...")
    dataset_dict: datasets.DatasetDict = _download_dataset(config, splits=splits)
    print("\t...initializing_dataloader complete")
    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    # apply the training transform to the training split and the validation transform to the validation and test splits
    
    print("\tApplying transforms...")
    map_args = {
        "batched": True, 
        "batch_size": config.transform_batch_size, 
        "num_proc": config.num_cpus,
        "input_columns": ['data', 'labels', "num_atoms"], 
        }

    if not keep_all_columns:
        map_args["remove_columns"] = ['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms', 'replica_id']
    # if cache_exists:
    #     print(f"\tFound existing mapped dataset cache at {mapped_cache_dir}, using it to speed up dataloader initialization...")
    #     # copy the cache dir to the faster local storage if running in GPULAB
    #     if "GPULAB_CPUS_RESERVED" in os.environ:
    #         # copy the cache dir to the faster local storage if it doesn't already exist there
    #         for file in os.listdir(mapped_cache_dir):
    #             print(f"Copying mapped dataset cache from {mapped_cache_dir} to {fast_cache_dir} for faster access...")
    #             if file.endswith(".arrow"):
    #                 fp = os.path.join(fast_cache_dir, file)
    #                 if not os.path.exists(fp):
    #                     shutil.copy(os.path.join(mapped_cache_dir, file), fp)
    #             print(f"\t...copying complete")    
                
    # print(f"\tUsing mapped dataset cache directory: {mapped_cache_dir}")
    for split in splits:
        print(f"\t\tApplying transforms to {split}')")
        if split == 'train':
            transform_fn = config.training_transform
        else:
            transform_fn = config.validation_transform
        dataset_dict[split] = dataset_dict[split].map(
            transform_fn,
            **map_args,
            load_from_cache_file=True,
        )
    # if not cache_exists:
    #     print(f"\tFinished applying transforms and caching mapped dataset at {fast_cache_dir}")
    #     print(f"\tCopying mapped dataset from {fast_cache_dir} cache to {mapped_cache_dir} for future runs...")
    #     for file in os.listdir(fast_cache_dir):
    #         if file.endswith(".arrow"):
    #             fp = os.path.join(mapped_cache_dir, file)
    #             if not os.path.exists(fp):
    #                 shutil.copy(os.path.join(fast_cache_dir, file), fp)
    #     print(f"\t...copying complete")
        
    print("\t...applying_transforms complete")
    dataloader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_cpus,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True,
        "prefetch_factor": 4, 
    }
    resource.setrlimit(resource.RLIMIT_NOFILE, (10810, 10810))
    
    dataset_dict = dataset_dict.with_format(
        type="torch",
        columns=["data", "labels"],
        output_all_columns=keep_all_columns,
    )
    dataloader_dict = {}
    for split in splits:
        print(f"\t\tCreating dataloader for {split} split with batch size {config.batch_size} and num_workers {config.num_cpus}...")
        dataloader = torch.utils.data.DataLoader(dataset_dict[split], **dataloader_args)
        dataloader_dict[split] = dataloader
    return dataloader_dict

def initialize_streaming_dataloader(config: TrainingConfig, keep_all_columns: bool = False, splits: list = ["train", "validation", "test"]) -> DataLoaderDict:
    """
    Initialize the streaming dataloader for training.
    """
    # Initialize the streaming dataloader
    print("Initializing streaming dataloader...")
    dataset_dict: datasets.IterableDatasetDict = _download_streaming_dataset(config, splits=splits)
    if "train" in splits:
        print("\tShuffling training split...")
        # shuffle the training split of the dataset
        dataset_dict["train"] = dataset_dict["train"].shuffle(seed=config.shuffle_seed, buffer_size=config.shuffle_buffer_size)
    dataset_dict = dataset_dict.rename_columns({'coordinates': 'data', 'binding_type': 'labels'})
    # set the format of the dataset to torch tensors and create dataloaders for each split
    dataset_dict = dataset_dict.with_format(type="torch")
    # apply the training transform to the training split and the validation transform to the validation and test splits
    map_args = {"batched": True, "batch_size": config.transform_batch_size, 
                "input_columns": ['data', 'labels', "num_atoms"], 
                }
    if not keep_all_columns:
        map_args["remove_columns"] = ['pdb_id', 'dpp_class', 'ligand_name', 'num_atoms']
    dataloader_dict = {}
    for split in splits:
        print(f"\t\tApplying transforms to {split} split...")
        if split == 'train':
            transform_fn = config.training_transform
        else:
            transform_fn = config.validation_transform
        dataset_dict[split] = dataset_dict[split].map(
            transform_fn,
            **map_args,
        )
        dataloader_args = {"batch_size": config.batch_size, "num_workers": config.num_cpus, "pin_memory": True}    
        dataloader = torch.utils.data.DataLoader(dataset_dict[split], **dataloader_args)
        dataloader_dict[split] = dataloader
    print("\t...applying_transforms complete")

    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    
    print("\t...initializing_streaming_dataloader complete")
    return dataloader_dict

if __name__ == "__main__":
    import argparse
    # load and cache the dataset and dataloader
    # read command line arguments for dataset size and batch size
    parser = argparse.ArgumentParser(description="Prepare cache for Hugging Face dataset and dataloader")
    parser.add_argument("--dataset-size", type=int, default=100, help="Size of the dataset to use for training.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--cpus", type=int, default=1, help="Number of CPUs to use for training.")
    parser.add_argument("--use-all-columns", action="store_true", help="Whether to keep all columns in the dataset or remove extra columns.")
    
    args = parser.parse_args()
    config = TrainingConfig(    
        dataset_location="Sam-Roggeman/SamRoggeman_Thesis_Dataset",
        dataset_size=args.dataset_size / 100,
        batch_size=args.batch_size,
        transform_batch_size=128,
        num_cpus=args.cpus,
        
    )
    
        
    dataloader_dict = initialize_dataloaders(config, keep_all_columns=args.use_all_columns)
    print("Dataloaders initialized and cached successfully.")
    
    