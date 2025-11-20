import functools

from datasets import Dataset, load_dataset, NamedSplit
from datasets.data_files import DownloadConfig
import tarfile
import os
import tempfile
from pathlib import Path
import numpy as np
from Bio import PDB
import multiprocessing as mp
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
    regenerate = True  # Set to True to regenerate the dataset


    configparser = ConfigParserWrapper()
    raw_data_path = configparser.get_raw_data_folder()

    logs_parent_dir, logging_enabled, console_enabled = configparser.get_logging()
    log_subdir = "data_preprocessing"
    log_dir = os.path.join(logs_parent_dir, log_subdir)
    logger = setup_logger(log_file="", log_dir=log_dir, logging_enabled=logging_enabled,
                          console_enabled=console_enabled)
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
                # only take every 1000th frame to reduce dataset size for testing
                frames = [ frame for frame in frames if frame.name.endswith('000.pdb')]
                dpp_class, ligand_name, binding_type = parse_filename(filename)
                func = functools.partial(
                    parse_pdb_streaming,
                    dpp_class=dpp_class,
                    ligand_name=ligand_name,
                    binding_type=binding_type
                )

                # split into 6 parts for multiprocessing
                dataset = Dataset.from_generator(
                    func,
                    gen_kwargs={
                        'frames': frames,
                    },
                    num_proc=1,
                    split = NamedSplit(f"{dpp_class}_{binding_type}_{ligand_name}")
                )
                res_dir = os.path.join(streaming_pdb_dataset_path, filename)
                os.makedirs(res_dir, exist_ok=True)
                # store the dataset to disk as safetensors
                dataset.save_to_disk(res_dir, max_shard_size="4GB")
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
