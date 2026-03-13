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
    dataset:  datasets.Dataset = datasets.load_dataset(repo_id, token=api.token, num_proc=cpu_cores)


    # read new dataset from save_to_disk
    new_dataset: datasets.Dataset = datasets.load_from_disk(new_datapath)
    # Append the new data to the existing dataset
    combined_dataset: datasets.Dataset = datasets.concatenate_datasets([dataset, new_dataset])
    # shuffle the combined dataset
    combined_dataset = combined_dataset.shuffle(seed=42, num_proc=cpu_cores)
    # Push the combined dataset back to HuggingFace Hub
    combined_dataset.push_to_hub(repo_id, token=api.token, num_proc=cpu_cores)

def main_append():  
    repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"
    api = initialize_hf_api()
    cpu_cores = 8
    new_datapath = "/project_antwerp/dataset/full_dataset/full_dataset"
    append_to_hf_dataset(api, repo_id, new_datapath, cpu_cores)


if __name__ == "__main__":
    main_append()


    




