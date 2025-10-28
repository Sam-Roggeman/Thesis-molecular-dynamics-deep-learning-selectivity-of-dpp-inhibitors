import os
from src.utils.utils import convert_pdb_to_npy

if __name__ == '__main__':
    source_path = "/mnt/x/School/trajects/pre_processed_data/"
    destination_path = "/mnt/x/School/trajects/pre_processed_data_npy/"
    # create destination folder if it does not exist
    os.makedirs(destination_path, exist_ok=True)

    convert_pdb_to_npy(source_path, destination_path, batch_size=50, num_processes=16)