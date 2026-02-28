import os
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"
cpu_cores = os.cpu_count() or 1

# Load the whole dataset from HuggingFace Hub
dataset = load_dataset(repo_id, token=api.token, num_proc=cpu_cores)
print(f"Original dataset size: {len(dataset)}")

# remove the entries in the dataset where the column "ligand_name" has the value "12i"
filtered_dataset = dataset.filter(lambda x: x["ligand_name"] != "12i", num_proc=cpu_cores)
print(f"Cleaned dataset size: {len(filtered_dataset)}")
print(any(x == "12i" for x in filtered_dataset["train"]["lingand_name"]))  # should be False
print(any(x == "12i" for x in filtered_dataset["test"]["lingand_name"]))  # should be False
print(any(x == "12i" for x in filtered_dataset["val"]["lingand_name"]))  # should be False
filtered_dataset.push_to_hub(repo_id, token=api.token)






