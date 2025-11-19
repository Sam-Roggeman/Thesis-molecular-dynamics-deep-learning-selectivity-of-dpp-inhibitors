import os
import mdtraj as md
import numpy as np
import torch


# inline function to get classes
def get_binding_classes():
    return ["apo", "nonbinder", "dpp9selective", "dpp8selective", "aselective"]



def parse_pdb_from_string(pdb_content):
    """Parse PDB from string content instead of file"""
    coords = []
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except (ValueError, IndexError):
                continue
    return torch.tensor(coords)
def parse_filename(filename):
    """
    Parse filename to get DPP class, binding type and ligand name.
    :param filename: str in format sep_prot_frames_{DPP_class}_{Ligand_name}_{Binding_type}.{extension}
    """
    filename = remove_extension(filename)
    filename = filename.lower()
    parts = filename.split('_')
    if len(parts) == 5:
        parts.insert(4, 'none')  # Insert 'none' for ligand name if missing
    if len(parts) != 6:
        raise ValueError(f"Filename {filename} is not in the expected format.")
    dpp_class = parts[3]
    ligand_name = parts[4]
    binding_type = parts[5]
    # validate    dpp_class and binding_type
    if dpp_class not in ['dpp8', 'dpp9']:
        raise ValueError(f"Filename {filename} has unknown DPP class {dpp_class}.")
    if binding_type not in get_binding_classes() and binding_type != 'none':
        raise ValueError(f"Filename {filename} has unknown binding type {binding_type}.")
    return dpp_class, ligand_name, binding_type

def prime_factorization(n):
    """
    Perform prime factorization of a given integer n.
    :param n: int
    :return:
    """
    ans = []
    # Loop from 2 to n
    for i in range(2, n + 1):

        # n % i == 0 means n is divisible by i
        while n % i == 0 and n > 0:
            ans.append(i)

            # divide n by i to remove this factor
            n = n // i
    return ans

def calculate_image_size(n_pixels):
    factors = prime_factorization(n_pixels)
    width = height = 1
    while factors:
        factor = factors.pop()
        if width <= height:
            width *= factor
        else:
            height *= factor
    return width, height

def remove_extension(filename):
    """
    Remove the extension from a filename.
    :param filename: str
    :return:
    """
    return '.'.join(filename.split('.')[:-1])


import functools
import os
from tqdm import tqdm
from multiprocessing import Pool

def process_single_pdb(args):
    """Process a single PDB file"""
    pdb_file, source_folder, target_folder = args
    if pdb_file.endswith(".pdb"):
        try:
            filepath_input_pdb = os.path.join(source_folder, pdb_file)
            traj = md.load_pdb(filepath_input_pdb)
            coords = traj.xyz[0]
            output_filepath = os.path.join(target_folder, f"{pdb_file[:-4]}.npy")
            np.save(output_filepath, coords)
            return f"Processed {pdb_file}"
        except Exception as e:
            return f"Error processing {pdb_file}: {str(e)}"
    return None

def convert_pdb_to_npy(source_folder, target_folder, batch_size=64, num_processes=16):
    """
    Convert PDB files in batches with progress tracking
    """
    pdb_files = []
    os.makedirs(target_folder, exist_ok=True)
    traj_dirs = os.listdir(source_folder)
    for trajectory_dir in traj_dirs:
        _source_folder = os.path.join(source_folder, trajectory_dir)
        _target_folder = os.path.join(target_folder, trajectory_dir)
        pdb_files += [[f, _source_folder, _target_folder ] for f in os.listdir(_source_folder) if f.endswith(".pdb") and not os.path.isfile(os.path.join(_target_folder, f"{f[:-4]}.npy"))]
        os.makedirs(_target_folder, exist_ok=True)

    # Process in batches to manage memory
    for i in tqdm(range(0, len(pdb_files), batch_size), desc="Processing batches"):
        batch = pdb_files[i:i + batch_size]

        if num_processes and num_processes > 1:
            # Parallel processing within batch
            with Pool(processes=min(num_processes, len(batch))) as pool:
                pool.map(process_single_pdb, batch)
        else:
            # Sequential processing
            for pdb_file in batch:
                process_single_pdb(pdb_file)
