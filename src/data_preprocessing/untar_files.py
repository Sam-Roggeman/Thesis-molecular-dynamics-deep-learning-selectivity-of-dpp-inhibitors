


"""
Loop through all the .tar.gz files in the specified directory, extract their contents, and save them to a new directory with the same name
"""

import os
import tarfile
from pathlib import Path

def extract_tar_gz_files(input_dir: str, output_dir: str):
    """Extract all .tar.gz files from input_dir to output_dir"""
    for filename in os.listdir(input_dir):
        # Check if the file is a .tar.gz file
        if filename.endswith('.tar.gz'):
            # Construct the full path to the tar.gz file and the output directory
            tar_path = os.path.join(input_dir, filename)
            extract_path = os.path.join(output_dir, Path(filename).stem)
            # Create the output directory if it doesn't exist
            os.makedirs(extract_path, exist_ok=True)
            print(f"Extracting {tar_path} to {extract_path}...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            print(f"Finished extracting {filename}.")

if __name__ == "__main__":
    input_directory = "/project_antwerp/"
    output_directory = "/mnt/x/School/trajects/raw_data_uncompressed/"
    extract_tar_gz_files(input_directory, output_directory)