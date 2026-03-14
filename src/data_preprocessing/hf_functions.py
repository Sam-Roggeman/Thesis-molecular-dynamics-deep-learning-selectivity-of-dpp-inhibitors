import os
import datasets
from dotenv import load_dotenv

from huggingface_hub import HfApi
def initialize_hf_api():
    load_dotenv()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return api

def load_dataset_from_hf(api, repo_id, cpu_cores):
    api = initialize_hf_api()
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
def remove_12i_entries(dataset, api, repo_id, cpu_cores):
    repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"
    api= initialize_hf_api()

    dataset = load_dataset_from_hf(api, repo_id, cpu_cores)

    # remove the entries in the dataset where the column "ligand_name" has the value "12i"
    filtered_dataset = dataset.filter(lambda x: x["ligand_name"] != "12i", num_proc=cpu_cores)
    print(f"Cleaned dataset size: {len(filtered_dataset)}")
    print(any(x == "12i" for x in filtered_dataset["train"]["ligand_name"]))  # should be False
    print(any(x == "12i" for x in filtered_dataset["test"]["ligand_name"]))  # should be False
    print(any(x == "12i" for x in filtered_dataset["validation"]["ligand_name"]))  # should be False
    filtered_dataset.push_to_hub(repo_id, token=api.token, num_proc=cpu_cores)




def append_to_hf_dataset(api, repo_id, new_datapath, cpu_cores):
    api = initialize_hf_api()
    # Load the existing dataset from HuggingFace Hub
    dataset:  datasets.DatasetDict = datasets.load_dataset(repo_id, token=api.token, num_proc=cpu_cores)
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
    new_dataset_train = new_dataset_train.cast(new_features)
    new_dataset_test = new_dataset_test.cast(new_features)
    new_dataset_validation = new_dataset_validation.cast(new_features)
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
    combined_dataset.push_to_hub(repo_id, token=api.token, num_proc=cpu_cores)
    print("Combined dataset successfully pushed to HuggingFace Hub.")

def main_append():  
    repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"
    api = initialize_hf_api()
    cpu_cores = 8
    new_datapath = "/project_antwerp/dataset/full_dataset/full_dataset"
    append_to_hf_dataset(api, repo_id, new_datapath, cpu_cores)


if __name__ == "__main__":
    main_append()


    




