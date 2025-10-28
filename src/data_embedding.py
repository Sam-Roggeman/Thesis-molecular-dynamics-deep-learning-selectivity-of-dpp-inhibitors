import functools
import time
import datetime
from multiprocessing.pool import Pool
import numpy as np
from matplotlib import pyplot as plt
from mdtraj.formats import XTCTrajectoryFile, PDBTrajectoryFile
import os
from tqdm import tqdm

"""
    Reference: A Machine Learning Approach for the Discovery of Ligand-Specific Functional Mechanisms of GPCRs
    Raw Input data file: .xtc
    -> extract frames and coordinates of atoms from trajectory file
    -> scramble the external degrees of freedom (translation and rotation)
"""
from numpy import ndarray
from src.utils.randomGenerator import RandomGenerator
from src.data_embedding.DataEmbedder import DataEmbedder
from src.utils.configParser import ConfigParser
import src.utils.utils as utils
from concurrent.futures import ThreadPoolExecutor
import threading
def progress(current, total, start_time, bar_length=40, ):
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '=' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current == total else '\r'
    elapsed = time.time() - start_time
    # elapsed time in hh:mm:ss
    elapsed = str(datetime.timedelta(seconds=int(elapsed)))
    print(f'Progress: {current}/{total} [{arrow}{padding}] {int(fraction*100)}% Elapsed Time: {elapsed}s', end=ending)

def read_data_function(extension:str):
    """
    return the function to read data from files with the given extension
    """
    if extension == "xtc":
        return XTCTrajectoryFile
    if extension == "pdb":
        return PDBTrajectoryFile

    else:
        raise ValueError(f"Unsupported file extension: {extension}")


def embed_single_npy_frame(args, data_embedder):
    """
    Embed a single npy frame into an image and save it
    :param args: list of npy_file (name), source folder (where input is stored), target folder (where output is stored), diameter
    :param data_embedder: data embedder
    :return:
    """
    npy_file, source_folder, target_folder = args

    if npy_file.endswith(".npy"):
        try:
            filepath_input_npy = os.path.join(source_folder, npy_file)
            output_filepath = os.path.join(target_folder, f"{npy_file[:-4]}.png")

            coords = np.load(filepath_input_npy)
            image = data_embedder.frame_embedding(coords)
            plt.imsave(output_filepath, image)
            return f"Embedded {npy_file}"
        except Exception as e:
            print(e)
            return f"Error embedding {npy_file}: {str(e)}"
    return None
def data_embedding():
    config_parser  = ConfigParser("config.ini")

    # Folder containing the clean data
    data_folder = config_parser.get("Data Embedding", "input folder")
    file_extension_input = config_parser.get("Data Embedding", "Input Extension")
    output_folder = config_parser.get("Data Embedding","output folder")
    npy_files = []
    traj_dirs = os.listdir(data_folder)

    batch_size = int(config_parser.get("Data Embedding", "batch size"))
    nr_threads = int(config_parser.get("Data Embedding", "nr of workers"))
    height = int(config_parser.get("Data Embedding", "image height"))
    width = int(config_parser.get("Data Embedding", "image width"))
    data_embedder = DataEmbedder(width=width, height=height)

    os.makedirs(output_folder, exist_ok=True)
    for traj_dir in traj_dirs:
        source_path = os.path.join(data_folder, traj_dir)
        target_path = os.path.join(output_folder, traj_dir)
        # create dir in output folder
        os.makedirs(target_path, exist_ok=True)
        npy_files += [[f,source_path,target_path] for f in os.listdir(source_path) if f.endswith(".npy") and not os.path.isfile(os.path.join(target_path, f"{f[:-4]}.png"))]
    nr_threads = 1
    # Process in batches to manage memory
    for i in tqdm(range(0, len(npy_files), batch_size), desc="Processing batches"):
        batch = npy_files[i:i + batch_size]

        if nr_threads and nr_threads > 1:
            # Parallel processing within batch
            with Pool(processes=min(nr_threads, len(batch))) as pool:
                process_func = functools.partial(embed_single_npy_frame, data_embedder=data_embedder)
                pool.map(process_func, batch)

        else:
            # Sequential processing
            for npy_file in batch:
                embed_single_npy_frame(npy_file, data_embedder)










if __name__ == '__main__':
    data_embedding()

