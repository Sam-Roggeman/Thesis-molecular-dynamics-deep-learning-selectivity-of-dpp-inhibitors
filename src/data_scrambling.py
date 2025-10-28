import functools
from multiprocessing.pool import Pool

from tqdm import tqdm

from src.data_embedding.Scrambler import Scrambler
import numpy as np
import os
from src.utils.configParser import ConfigParser

def scramble_single_npy_frame(args, scrambler):
    filename, source_folder, target_folder, diameter = args
    source_filepath = os.path.join(source_folder, filename)
    target_filepath = os.path.join(target_folder, filename)

    coords = np.load(source_filepath)
    scrambled_coords = scrambler.scramble(coords, diameter)
    np.save(str(target_filepath), scrambled_coords)

if __name__ == '__main__':
    config_parser = ConfigParser()
    source_folder = config_parser.get("Data NPY", 'Output Folder')
    target_folder = config_parser.get("Data Scrambling", "Output Folder")
    traj_dirs = os.listdir(source_folder)
    input_files = []
    scrambler = Scrambler()
    os.makedirs(target_folder, exist_ok=True)

    for traj_dir in traj_dirs:
        _source_path = os.path.join(source_folder, traj_dir)
        _target_path = os.path.join(target_folder, traj_dir)

        # if dpp8 in source_folder name
        if _source_path.lower().find("dpp8") != -1:
            diameter = 12.327
        elif _source_path.lower().find("dpp9") != -1:
            diameter = 12.162
        else:
            raise ValueError("Source folder name must contain either 'dpp8' or 'dpp9' to determine diameter.")

        # create dir in output folder
        os.makedirs(_target_path, exist_ok=True)
        input_files += [[f, _source_path, _target_path, diameter] for f in os.listdir(_source_path) if
                      f.endswith(".npy") and not os.path.isfile(os.path.join(_target_path, f"{f[:-4]}.png"))]


    nr_threads = 16
    batch_size = 32
    # Process in batches to manage memory
    for i in tqdm(range(0, len(input_files), batch_size), desc="Processing batches"):
        batch = input_files[i:i + batch_size]

        if nr_threads and nr_threads > 1:
            # Parallel processing within batch
            with Pool(processes=min(nr_threads, len(batch))) as pool:
                process_func = functools.partial(scramble_single_npy_frame, scrambler=scrambler)
                pool.map(process_func, batch)

        else:
            # Sequential processing
            for npy_file in batch:
                scramble_single_npy_frame(npy_file, scrambler)
