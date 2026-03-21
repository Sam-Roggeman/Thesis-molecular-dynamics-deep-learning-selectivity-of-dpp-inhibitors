
import os

from src.utils.training_config import TrainingConfig
from datasets import IterableDatasetDict, config, load_dataset


class HFStreamingDataloader:
    config: TrainingConfig
    dataset_dict: IterableDatasetDict
    def __init__(self, config:TrainingConfig):
        print("HFStreamingDataloader.__init__")
        self.config = config
        self.cache_folder = os.environ.get("HF_CACHE_DIR")
        self.hf_token = os.environ.get("HF_TOKEN")

        # download the dataset
        self.download_dataset()




    def download_dataset(self) -> IterableDatasetDict:
        print("\tdownload_dataset")
        dataset_size = self.config.dataset_size

        assert dataset_size > 0 and dataset_size <= 1, "Dataset size must be between 0 and 1"
        print(f"\t\tDownloading {dataset_size} of {self.config.dataset_name}")
        
        dataset: IterableDatasetDict = load_dataset(
            self.config.dataset_name,
            cache_dir=self.cache_folder,
            token=self.hf_token,
            streaming=True
        )
        # shuffle the trainssplit of the dataset
        dataset["train"] = dataset["train"].shuffle(
                buffer_size=self.config.shuffle_buffer_size, 
                seed=self.config.seed
            )

        if dataset_size < 1:
            for split in dataset.keys():
                try:
                    total_samples = dataset.info.splits[split].num_examples
                    n_samples = int(total_samples * config.dataset_size)
                    n_samples = max(config.batch_size, n_samples)
                    n_samples = ((n_samples + config.batch_size - 1) // config.batch_size) * config.batch_size
                    print(f"Using streaming subset for {split}: {n_samples}/{total_samples} samples")
                    dataset = dataset.take(n_samples)
                except Exception:
                    print(
                        f"Warning: Could not determine split size for streaming subset on {split}. "
                        "Falling back to full streamed split."
                    )
        print(f"\t...downloading_dataset complete")
        self.dataset_dict = dataset
