import functools

from datasets import Dataset, load_dataset, NamedSplit, concatenate_datasets, load_from_disk
from datasets.data_files import DownloadConfig
import tarfile
import os
import tempfile
from pathlib import Path
import numpy as np
from Bio import PDB
import multiprocessing as mp

from src.model_training.utils import train_val_test_split
from src.utils.utils import parse_filename
from src.utils.configParser import ConfigParserWrapper
from src.utils.logger import setup_logger, replace_output


def parse_pdb_streaming(frames, dpp_class, ligand_name, binding_type):
    """Stream PDB files directly from TAR archives without extraction.

    :yield: dict with pdb_id, coordinates, chain_info
    """
    for frame in frames:
        if frame.name.endswith('.pdb'):
            f = tar.extractfile(frame)
            pdb_id = Path(frame.name).stem

            try:
                coords = extract_coordinates(f, pdb_id)
                yield {
                    'pdb_id': pdb_id,
                    'dpp_class': dpp_class,
                    'ligand_name': ligand_name,
                    'binding_type': binding_type,
                    'coordinates': coords,
                    'num_atoms': len(coords)
                }
            except Exception as e:
                print(f"Error parsing {pdb_id}: {e}")
                continue


def extract_coordinates(pdb_file, pdb_id):
    """Extract 3D coordinates from PDB file object."""
    from io import StringIO

    # Convert bytes to string
    content = pdb_file.read().decode('utf-8')

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, StringIO(content))

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.coord)

    return np.array(coords, dtype=np.float32)


def preprocess_batch(batch):
    """Process a batch of PDB data in parallel."""
    processed = {
        'pdb_id': [],
        'coordinates': [],
        'center_of_mass': [],
        'num_atoms': []
    }

    for pdb_id, coords in zip(batch['pdb_id'], batch['coordinates']):
        # Normalize coordinates
        coords_normalized = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        center = coords.mean(axis=0)

        processed['pdb_id'].append(pdb_id)
        processed['coordinates'].append(coords_normalized.tolist())
        processed['center_of_mass'].append(center.tolist())
        processed['num_atoms'].append(len(coords))

    return processed

if __name__ == "__main__":
    regenerate = False  # Set to True to regenerate the dataset

    configparser = ConfigParserWrapper()
    raw_data_path = configparser.get_raw_data_folder()

    logs_parent_dir, logging_enabled, console_enabled = configparser.get_logging()
    log_subdir = "data_preprocessing"
    log_dir = os.path.join(logs_parent_dir, log_subdir)
    logger = setup_logger(log_file="", log_dir=log_dir, logging_enabled=logging_enabled,
                          console_enabled=False)
    replace_output(logger)
    print(f"✓ Logger set up. Logs will be saved to {log_dir}")
    print("Starting data preprocessing...")

    streaming_pdb_dataset_path = f"./data/temp/streaming_pdb_dataset/"
    if not os.path.exists(streaming_pdb_dataset_path) or regenerate:
        print("Loading streaming dataset from TAR files...")
        # Load streaming dataset
        tar_files = sorted(Path('/mnt/x/School/trajects/raw_data_uncompressed/').glob('*.tar'))
        for tar_file in tar_files:
            filename = tar_file.stem
            with tarfile.open(tar_file, 'r') as tar:
                frames = tar.getmembers()
                dpp_class, ligand_name, binding_type = parse_filename(filename)
                split_name = f"{dpp_class}_{binding_type}_{ligand_name}"
                func = functools.partial(
                    parse_pdb_streaming,
                    dpp_class=dpp_class,
                    ligand_name=ligand_name,
                    binding_type=binding_type
                )

                dataset = Dataset.from_generator(
                    func,
                    gen_kwargs={
                        'frames': frames,
                    },
                    num_proc=1,
                    split = NamedSplit(split_name)
                )
                res_dir = os.path.join(streaming_pdb_dataset_path, split_name)
                os.makedirs(res_dir, exist_ok=True)
                # store the dataset to disk as safetensors
                dataset.save_to_disk(res_dir, max_shard_size="4GB")
    train_set = {
        "12i": 1.0, # Nonbinder
        "42": 1.0, # dpp9 selective
        "000808": 1.0, # DPP8selective
        "0003822": 1.0,# Aselective
        "apo": 0.5,
    }
    test_val_set = {
        "0000157": 1.0, # Nonbinder
        "0005356": 1.0, # dpp9 selective
        "0005862": 1.0, # DPP8selective
        "0005362": 1.0,# Aselective
    }
    dss_train = []
    dss_val_test = []
    for dpp in ["dpp8", "dpp9"]:
        for binding in ["aselective", "dpp8selective", "dpp9selective", "nonbinder", "apo"]:
            sub_name = f"{dpp}_{binding}"
            # find the ligands available for this dpp and binding type
            available_ligands = [ f.name.split('_')[-1] for f in Path(streaming_pdb_dataset_path).glob(f"{sub_name}_*") if f.is_dir()]
            for ligand in available_ligands:
                split_name = f"{dpp}_{binding}_{ligand}"
                path = os.path.join(streaming_pdb_dataset_path, split_name)
                if binding == "apo":
                    frac = train_set["apo"]
                    ds = load_from_disk(path)
                    ds_dict = ds.train_test_split(test_size=1-frac, seed=42, shuffle=True)
                    dss_val_test.append(ds_dict['test'])
                    dss_train.append(ds_dict['train'])
                elif ligand in train_set.keys():
                    ds = load_from_disk(path )
                    dss_train.append(ds)
                elif ligand in test_val_set.keys():
                    ds = load_from_disk(path)
                    dss_val_test.append(ds)
                else:
                    raise ValueError(f"Unknown ligand {ligand}.")
    full_training_set = concatenate_datasets(dss_train)
    full_val_test_set = concatenate_datasets(dss_val_test)
    full_set = concatenate_datasets([full_training_set, full_val_test_set])
    full_val_test_set = full_val_test_set.train_test_split(test_size=0.5, seed=42, shuffle=True)
    full_val_set = full_val_test_set['train']
    full_test_set = full_val_test_set['test']

    # print shapes
    print(f"\tFinal training set size: {len(full_training_set)}")
    print(f"\tFinal validation set size: {len(full_val_set)}")
    print(f"\tFinal test set size: {len(full_test_set)}")
    print(f"\tFull dataset size: {len(full_set)}")
    dataset_dir = os.path.dirname('./data/dataset/')
    print("Saving final datasets to disk...")
    ligand_path = os.path.join(dataset_dir, "ligand_set")
    final_train_path = os.path.join(ligand_path, "train")
    final_val_path = os.path.join(ligand_path, "val")
    final_test_path = os.path.join(ligand_path, "test")
    final_full_set_path = os.path.join(dataset_dir, "full_dataset")
    os.makedirs(final_train_path, exist_ok=True)
    os.makedirs(final_val_path, exist_ok=True)
    os.makedirs(final_test_path, exist_ok=True)
    full_training_set.save_to_disk(final_train_path, max_shard_size="4GB", num_proc=8)
    full_val_set.save_to_disk(final_val_path, max_shard_size="4GB", num_proc=8)
    full_test_set.save_to_disk(final_test_path, max_shard_size="4GB", num_proc=8)
    for split_name, dset in train_val_test_split(full_set, train_fraction=0.7, val_fraction=0.15, seed=42).items():
        set_path = os.path.join(final_full_set_path, split_name)
        os.makedirs(set_path, exist_ok=True)
        dset.save_to_disk(set_path, max_shard_size="4GB", num_proc=8)


    print("Datasets saved.")


    # else:
    #     print("Loading dataset from disk...")
    #     dataset = Dataset.load_from_disk(streaming_pdb_dataset_path)
    # print("Dataset loaded. Applying batch preprocessing...")
    #
    # # Apply batch preprocessing
    # dataset = dataset.map(
    #     preprocess_batch,
    #     batched=True,
    #     batch_size=32,
    #     num_proc=8  # Use all CPU cores
    # )
