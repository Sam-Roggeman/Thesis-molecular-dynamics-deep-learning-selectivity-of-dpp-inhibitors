import os
from tqdm import tqdm

from src.utils.configParser import ConfigParser


def filename_to_classname(filename):
    """Map filename to class name based on substrings."""
    filename = filename.lower()
    if "apo" in filename:
        return "apo"
    elif "nonbinder" in filename:
        return "nonbinder"
    elif "dpp9selective" in filename:
        return "dpp9selective"
    elif "dpp8selective" in filename:
        return "dpp8selective"
    elif "aselective" in filename:
        return "aselective"
    else:
        raise ValueError(f"Filename {filename} does not match any known class.")

def filename_to_dpp_classname(filename):
    """Map filename to DPP class name based on substrings."""
    filename = filename.lower()
    if "_dpp8_" in filename:
        return "dpp8"
    elif "_dpp9_" in filename:
        return "dpp9"
    else:
        raise ValueError(f"Filename {filename} does not match any known DPP class.")

def prepare_datasets():
    """Prepare datasets for training, testing, and validation."""
    config_parser = ConfigParser("config.ini")
    input_folder = config_parser.get("Data Embedding", "Output Folder")
    output_folder_small = config_parser.get("Data Splitting", "Small Dataset Folder")
    output_folder_medium = config_parser.get("Data Splitting", "Medium Dataset Folder")
    output_folder_large = config_parser.get("Data Splitting", "Large Dataset Folder")
    output_folders = [output_folder_small, output_folder_medium, output_folder_large]
    classes = ["apo", "nonbinder", "dpp9selective", "dpp8selective", "aselective"]
    # create a folder for each class in output_folder
    for output_folder in output_folders:
        os.makedirs(output_folder, exist_ok=True)
        for cls in classes:
            path = os.path.join(output_folder, cls)
            os.makedirs(path, exist_ok=True)
    traject_folders = os.listdir(input_folder)
    # loop over the input folders and add them to their respective class folder
    for traject_folder in tqdm(traject_folders, desc="Processing folders"):
        source_path = os.path.join(input_folder, traject_folder)
        target_class = filename_to_classname(traject_folder)
        target_path_large = os.path.join(output_folder_large, target_class)
        dpp_class = filename_to_dpp_classname(traject_folder)
        # copy all images in source_path to target_path
        for file in os.listdir(source_path):
            if file.endswith(".png"):
                frame_nr =filename_to_framenumber(file)
                new_file = f"{dpp_class}_{file}"
                source_file = os.path.join(source_path, file)
                target_file_large = os.path.join(target_path_large, new_file)
                if not os.path.isfile(target_file_large):
                    os.system(f"cp {source_file} {target_file_large}")
                if frame_nr % 4 == 0:
                    target_path_medium = os.path.join(output_folder_medium, target_class)
                    target_file_medium = os.path.join(target_path_medium, new_file)
                    os.system(f"cp {source_file} {target_file_medium}")
                if frame_nr % 100 == 0:
                    target_path_small = os.path.join(output_folder_small, target_class)
                    target_file_small = os.path.join(target_path_small, new_file)
                    os.system(f"cp {source_file} {target_file_small}")

def filename_to_framenumber(filename):
    """Extract frame number from filename."""
    part = filename.split('_')[1]
    if part.endswith('.png'):
        part = part[:-4]
        return int(part)
    raise ValueError(f"Filename {filename} does not contain a frame number.")



if __name__ == '__main__':
    prepare_datasets()
