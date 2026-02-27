import os
from datasets import load_dataset
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"


# Load the whole dataset from HuggingFace Hub
dataset = load_dataset(repo_id, token=api.token)
print(f"Original dataset size: {len(dataset)}")

# remove the entries in the dataset where the column "ligand_name" has the value "12i"
filtered_dataset = dataset.filter(lambda x: x["ligand_name"] != "12i", num_proc=16)
print(f"Cleaned dataset size: {len(filtered_dataset)}")
print(any(x == "12i" for x in filtered_dataset["train"]["lingand_name"]))  # should be False
print(any(x == "12i" for x in filtered_dataset["test"]["lingand_name"]))  # should be False
print(any(x == "12i" for x in filtered_dataset["val"]["lingand_name"]))  # should be False
filtered_dataset.push_to_hub(repo_id, token=api.token)






