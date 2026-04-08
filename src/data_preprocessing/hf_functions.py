import os
import datasets
from dotenv import load_dotenv

from huggingface_hub import HfApi
def initialize_hf_api():
    load_dotenv()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return api

def load_dataset_from_hf(api, repo_id, cpu_cores):
    # Load the whole dataset from HuggingFace Hub
    dataset = datasets.load_dataset(repo_id, token=api.token, num_proc=cpu_cores)
    return dataset
def convert_col_to_int(dataset: datasets.Dataset, column_name:str, cpu_cores):
    # Convert the specified column to integers
    def convert_to_int(example):
        example[column_name] = int(example[column_name])
        return example

    converted_dataset = dataset.map(convert_to_int, num_proc=cpu_cores)
    return converted_dataset
def remove_12i_entries(dataset, cpu_cores):


    # remove the entries in the dataset where the column "ligand_name" has the value "12i"
    filtered_dataset = dataset.filter(lambda x: x["ligand_name"] != "12i", num_proc=cpu_cores)
    print(f"Cleaned dataset size: {len(filtered_dataset)}")
    print(any(x == "12i" for x in filtered_dataset["train"]["ligand_name"]))  # should be False
    print(any(x == "12i" for x in filtered_dataset["test"]["ligand_name"]))  # should be False
    print(any(x == "12i" for x in filtered_dataset["validation"]["ligand_name"]))  # should be False
    return filtered_dataset
def add_replica_id_column(dataset, cpu_cores):
    # add a column "replica_id" to the dataset, which is 1 for all entries
    def add_replica_id(example):
        example["replica_id"] = 1
        return example

    dataset = dataset.map(add_replica_id, num_proc=cpu_cores)
    return dataset



def append_to_hf_dataset(dataset, new_datapath, cpu_cores):
    # Load the existing dataset from HuggingFace Hub
    train_datapath = os.path.join(new_datapath, "train")
    test_datapath = os.path.join(new_datapath, "test")
    validation_datapath = os.path.join(new_datapath, "val")
    print("Reading new dataset from disk...")
    # read new dataset from save_to_disk
    new_dataset_train: datasets.Dataset = datasets.load_from_disk(train_datapath)
    new_dataset_test: datasets.Dataset = datasets.load_from_disk(test_datapath)
    new_dataset_validation: datasets.Dataset = datasets.load_from_disk(validation_datapath)
    print("New dataset loaded from disk.")
    print("Converting 'replica_id' column to integers...")

    new_features = new_dataset_train.features.copy()
    new_features["replica_id"] = datasets.Value("int8")
    new_features["num_atoms"] = datasets.Value("int16")

    # convert the column "replica_id" to integers
    new_dataset_train = new_dataset_train.cast(new_features, num_proc=cpu_cores)
    new_dataset_test = new_dataset_test.cast(new_features, num_proc=cpu_cores)
    new_dataset_validation = new_dataset_validation.cast(new_features, num_proc=cpu_cores)
    dataset = dataset.cast(new_features)

    print("Column 'replica_id' converted to integers.")
    print("Appending new dataset to existing dataset...")
    # Append the new data to the existing dataset
    combined_train: datasets.Dataset = datasets.concatenate_datasets([dataset["train"], new_dataset_train])
    combined_test: datasets.Dataset = datasets.concatenate_datasets([dataset["test"], new_dataset_test])
    combined_validation: datasets.Dataset = datasets.concatenate_datasets([dataset["validation"], new_dataset_validation])
    print("Shuffling the combined dataset...")
    # shuffle the combined dataset
    combined_dataset = datasets.DatasetDict({
        "train": combined_train.shuffle(seed=42),
        "test": combined_test.shuffle(seed=42),
        "validation": combined_validation.shuffle(seed=42)
    })
    print(f"Combined dataset size: {len(combined_dataset['train'])} train, {len(combined_dataset['test'])} test, {len(combined_dataset['validation'])} validation")
    print("Pushing the combined dataset back to HuggingFace Hub...")
    # Push the combined dataset back to HuggingFace Hub
    return combined_dataset

def append_custom_split_to_hf_dataset(new_datapath, new_split_name, repo_id, cpu_cores):
    """ Append new data from disk to an existing HuggingFace dataset as a new split, then push to the Hub.
    Args:
        new_datapath (_type_): Path to the new split saved on disk that belongs to the new split (e.g., "unique_test_runs") 
        new_split_name (_type_): Name of the new split (e.g., "unique_test_runs")
        repo_id (_type_): HuggingFace Hub repository ID (e.g., "username/dataset_name")
        cpu_cores (_type_): Number of CPU cores to use for processing
    """
    # Load the existing dataset from HuggingFace Hub
    api = initialize_hf_api()
    dataset = load_dataset_from_hf(api, repo_id, cpu_cores)

    # Load the new split from disk
    print(f"Loading new split '{new_split_name}' from disk at: {new_datapath}")
    new_split_dataset: datasets.Dataset = datasets.load_from_disk(new_datapath)
    print(f"New split '{new_split_name}' loaded with size: {len(new_split_dataset)}")

    # Add the new split to the existing dataset
    dataset[new_split_name] = new_split_dataset

    print(f"New split '{new_split_name}' added to the existing dataset. Total splits now: {list(dataset.keys())}")

    # Push the updated dataset back to HuggingFace Hub
    print("Pushing the updated dataset with the new split back to HuggingFace Hub...")
    dataset.push_to_hub(repo_id, token=api.token, num_proc=cpu_cores)
    print("Dataset successfully updated on HuggingFace Hub with the new split.")
    
    
    
def main_append():  
    starting_repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset"
    repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"
    cpu_cores = 8
    new_datapath = "/project_antwerp/dataset/full_dataset/full_dataset"

    api = initialize_hf_api()
    dataset = load_dataset_from_hf(api, starting_repo_id, cpu_cores)
    dataset = remove_12i_entries(dataset, cpu_cores)
    dataset = add_replica_id_column(dataset, cpu_cores)
    dataset = append_to_hf_dataset(dataset, new_datapath, cpu_cores)
    dataset.push_to_hub(repo_id, token=api.token, num_proc=cpu_cores)
    
    print("Combined dataset successfully pushed to HuggingFace Hub.")

    
    


if __name__ == "__main__":
    # take args from command line to specify the new datapath, new split name, repo id and cpu cores
    import argparse
    load_dotenv()  # Load environment variables from .env file
    parser = argparse.ArgumentParser(description="Append a new split to an existing HuggingFace dataset and push to the Hub")
    parser.add_argument("--new-datapath", required=True, help="Path to the new split saved on disk (e.g., 'unique_test_runs')")
    parser.add_argument("--new-split-name", required=True, help="Name of the new split (e.g., 'unique_test_runs')")
    parser.add_argument("--repo-id", required=True, help="HuggingFace Hub repository ID (e.g., 'username/dataset_name')", default="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full")
    parser.add_argument("--cpu-cores", type=int, default=1, help="Number of CPU cores to use for processing", default=8)
    args = parser.parse_args()
    append_custom_split_to_hf_dataset(
        new_datapath=args.new_datapath,
        new_split_name=args.new_split_name,
        repo_id=args.repo_id,
        cpu_cores=args.cpu_cores,
    )


    




