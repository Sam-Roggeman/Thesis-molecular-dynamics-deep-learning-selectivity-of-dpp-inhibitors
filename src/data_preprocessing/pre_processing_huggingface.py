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


def parse_pdb_streaming(tar, frames, dpp_class, ligand_name, binding_type, replica_id):
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
                    'num_atoms': len(coords),
                    'replica_id': replica_id
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




def generate_dataset_from_tars(streaming_pdb_dataset_path, tar_folder, regenerate=False, num_proc=32, skip_existing=True):
    if not os.path.exists(streaming_pdb_dataset_path) or regenerate:
        print("Loading streaming dataset from TAR files...")
        # Load streaming dataset
        tar_files = sorted(Path(tar_folder).glob('*.tar'))
        # 4 GB in bytes
        min_expected_size = 3.5 * 10**9
        counter = 0


        for tar_file in tar_files:
            filename = tar_file.stem

            print(f"Processing file {counter}:\t\t{filename}...")
            counter += 1
            with tarfile.open(tar_file, 'r') as tar:
                frames = tar.getmembers()
                dpp_class, ligand_name, binding_type, replica_id = parse_filename(filename)
                split_name = f"{dpp_class}_{binding_type}_{ligand_name}_{replica_id}"
                res_dir = os.path.join(streaming_pdb_dataset_path, split_name)
                # if the dataset existence should be skipped, doesnt exists, or is not of expected size, skip processing
                if os.path.exists(res_dir):
                    existing_size = 0
                    for element in os.scandir(res_dir):
                        if element.is_file():
                            existing_size += os.path.getsize(element)
                    if skip_existing and existing_size >= min_expected_size:
                        print(f"✓ Skipping {filename} as it already exists and is of expected size.")
                        continue
                func = functools.partial(
                    parse_pdb_streaming,
                    tar = tar,
                    dpp_class=dpp_class,
                    ligand_name=ligand_name,
                    binding_type=binding_type,
                    replica_id=replica_id
                )

                dataset = Dataset.from_generator(
                    func,
                    gen_kwargs={
                        'frames': frames,
                    },
                    num_proc=num_proc,
                    split = NamedSplit(split_name)
                )
                os.makedirs(res_dir, exist_ok=True)
                # store the dataset to disk as safetensors
                dataset.save_to_disk(res_dir, max_shard_size="4GB")
                print(f"✓ Processed {filename} and saved to {res_dir}.")
    print("✓ Streaming dataset loaded and saved to disk.")
if __name__ == "__main__":
    regenerate = True  # Set to True to regenerate the dataset
    tar_folder = "/project_antwerp/dataset/decompressed/"
    print("Starting data preprocessing...")
    streaming_pdb_dataset_path = f"/project_antwerp/dataset/temp/streaming_pdb_dataset/"
    # create the directory if it doesn't exist
    os.makedirs(streaming_pdb_dataset_path, exist_ok=True)
    num_proc = 16

    generate_dataset_from_tars(streaming_pdb_dataset_path, tar_folder, regenerate=regenerate, num_proc=num_proc, skip_existing=False)

    full_ds = []
    for dpp in ["dpp8", "dpp9"]:
        for binding in ["aselective", "dpp8selective", "dpp9selective", "nonbinder", "apo"]:
            sub_name = f"{dpp}_{binding}"
            # find the ligands available for this dpp and binding type
            available_ligands = [ f.name.split('_')[-1] for f in Path(streaming_pdb_dataset_path).glob(f"{sub_name}_*") if f.is_dir()]
            for ligand in available_ligands:
                split_name = f"{dpp}_{binding}_{ligand}"
                print(f"Loading dataset for split: {split_name}...")
                path = os.path.join(streaming_pdb_dataset_path, split_name)
                partial_ds = load_from_disk(path)
                full_ds.append(partial_ds)

    full_ds = concatenate_datasets(full_ds)
    print(f"✓ Full dataset loaded with {len(full_ds)} samples.")
    # split into train, val, test
    full_set = full_ds.shuffle(seed=42)
    train_val_test = train_val_test_split(full_set, train_fraction=0.7, val_fraction=0.15, seed=42)
    full_training_set = train_val_test['train']
    full_val_set = train_val_test['val']  # Use validation split for
    full_test_set = train_val_test['test']

    # print shapes
    print(f"\tFinal training set size: {len(full_training_set)}")
    print(f"\tFinal validation set size: {len(full_val_set)}")
    print(f"\tFinal test set size: {len(full_test_set)}")

    dataset_dir = os.path.dirname('/project_antwerp/dataset/full_dataset/')
    os.makedirs(dataset_dir, exist_ok=True)
    print("Saving final datasets to disk...")
    final_full_set_path = os.path.join(dataset_dir, "full_dataset")
    for split_name, dset in train_val_test_split(full_set, train_fraction=0.7, val_fraction=0.15, seed=42).items():
        set_path = os.path.join(final_full_set_path, split_name)
        os.makedirs(set_path, exist_ok=True)
        dset.save_to_disk(set_path, max_shard_size="4GB", num_proc=num_proc)


    print("Datasets saved.")


