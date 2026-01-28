"""
Upload Dataset to Huggingface Hub
"""
from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo, upload_large_folder
from src.api_keys import huggingface_token

def upload_dataset_to_huggingface(dataset_path: str, repo_name: str, hf_token: str, organization: str = None):
    """
    Uploads a dataset stored on disk to the Huggingface Hub.

    Args:
        dataset_path (str): Path to the dataset directory on disk.
        repo_name (str): Name of the repository to create on Huggingface Hub.
        hf_token (str): Huggingface authentication token.
        organization (str, optional): Organization name if uploading to an organization. Defaults to None.
    """
    print(f"Uploading dataset from {dataset_path} to Huggingface Hub repository '{repo_name}'...")

    api = HfApi(token=huggingface_token)

    # Create the repository on Huggingface Hub if it doesnt exist
    api.upload_large_folder(
        repo_id="SamRoggeman_Thesis_Dataset",
        private=False,
        repo_type="dataset",
        folder_path="/mnt/d/School/trajects/dataset/full_dataset",
        num_workers=8
    )





    print(f"Dataset successfully uploaded to '{repo_name}'.")

if __name__ == "__main__":
    dataset_path = "/mnt/d/School/trajects/dataset/full_dataset"
    repo_name = "thesis-dataset"

    upload_dataset_to_huggingface(dataset_path=dataset_path, repo_name="SamRoggeman_Thesis_Dataset", organization="UA", hf_token="<PASSWORD>")
