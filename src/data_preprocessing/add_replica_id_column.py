import os
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "Sam-Roggeman/SamRoggeman_Thesis_Dataset_full"
cpu_cores = 8

# Load the whole dataset from HuggingFace Hub
dataset = load_dataset(repo_id, token=api.token, num_proc=cpu_cores)

# add a column "replica_id" to the dataset, which is 1 for all entries
def add_replica_id(example):
    example["replica_id"] = 1
    return example

dataset = dataset.map(add_replica_id, num_proc=cpu_cores)

dataset.push_to_hub(repo_id, token=api.token, num_proc=cpu_cores)






