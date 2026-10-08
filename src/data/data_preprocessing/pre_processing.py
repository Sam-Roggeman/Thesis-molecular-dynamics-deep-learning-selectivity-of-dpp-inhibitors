import functools

from datasets import Dataset, load_dataset, NamedSplit, concatenate_datasets, load_from_disk
from datasets.data_files import DownloadConfig
import tarfile
import os
import tempfile
from pathlib import Path


from src.training.utils import train_val_test_split
from src.utils.molecular_parsing import parse_filename
from src.config.configParser import ConfigParserWrapper
from src.data.data_preprocessing.utils import extract_coordinates, parse_pdb_streaming







def generate_dataset_from_tars(streaming_pdb_dataset_path, tar_folder, regenerate=False, num_proc=32, skip_existing=True):
    """Convert every TAR file in ``tar_folder`` into a disk-backed dataset shard.

    The output directory ends up with one subdirectory per trajectory. Each
    subdirectory contains a Hugging Face dataset saved with ``save_to_disk`` so
    the full corpus can later be reloaded and concatenated without reparsing the
    source TAR archives.
    """
    if not os.path.exists(streaming_pdb_dataset_path) or regenerate:
        print("Loading streaming dataset from TAR files...")
        # Find every decompressed trajectory archive that should be converted into
        # a dataset shard.
        tar_files = sorted(Path(tar_folder).glob('*.tar'))
        # Shards are expected to be large. This threshold is used as a coarse
        # sanity check to avoid reprocessing directories that already look complete.
        min_expected_size = 3.5 * 10**9
        counter = 0


        for tar_file in tar_files:
            filename = tar_file.stem

            # Track progress at the archive level because each TAR may contain a
            # large number of PDB frames and can take a noticeable amount of time.
            print(f"Processing file {counter}:\t\t{filename}...")
            counter += 1
            with tarfile.open(tar_file, 'r') as tar:
                # Read the archive member list once so the generator can iterate over
                # it without repeatedly querying the tar file.
                frames = tar.getmembers()

                # The filename encodes the experiment metadata that should be copied
                # into every row produced from this archive.
                dpp_class, ligand_name, binding_type, replica_id = parse_filename(filename)
                split_name = f"{dpp_class}_{binding_type}_{ligand_name}_{replica_id}"
                res_dir = os.path.join(streaming_pdb_dataset_path, split_name)
                # If a shard already exists and looks large enough, assume it was
                # fully written in a previous run and skip recomputation.
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

                # ``Dataset.from_generator`` consumes the streaming generator and
                # builds an Arrow-backed dataset from the yielded dictionaries.
                # ``split`` is set to a synthetic name so each archive can be saved
                # as an independent shard on disk.
                dataset = Dataset.from_generator(
                    func,
                    gen_kwargs={
                        'frames': frames,
                    },
                    num_proc=num_proc,
                    split = NamedSplit(split_name)
                )
                os.makedirs(res_dir, exist_ok=True)
                # Persist the shard so later runs can reload it directly without
                # re-reading the source TAR file.
                dataset.save_to_disk(res_dir, max_shard_size="4GB")
                print(f"✓ Processed {filename} and saved to {res_dir}.")
    print("✓ Streaming dataset loaded and saved to disk.")

def generate_full_dataset_from_tars(regenerate=False, num_proc=32, skip_existing=True):
    """Build the final train/validation/test dataset from all TAR shards.

    The function first materializes per-trajectory datasets on disk, then reloads
    them, concatenates them into one large dataset, shuffles the rows, and finally
    applies the project-specific train/val/test split.
    """
    # folder with tar files
    tar_folder = "/project_antwerp/dataset/compressed_dataset/"
    print("Starting data preprocessing...")
    streaming_pdb_dataset_path = f"/project_antwerp/dataset/temp/streaming_pdb_dataset/"
    # Ensure the intermediate dataset cache exists before writing any shards.
    os.makedirs(streaming_pdb_dataset_path, exist_ok=True)
    num_proc = 16
     

    # Step 1: turn each TAR archive into a saved dataset shard on disk.
    generate_dataset_from_tars(streaming_pdb_dataset_path, tar_folder, regenerate=regenerate, num_proc=num_proc, skip_existing=False)

    full_ds = []
    # Step 2: reload every shard and keep the partial datasets in memory long
    # enough to concatenate them into a single dataset object.
    for last_child in os.listdir(streaming_pdb_dataset_path):
        traj_path = os.path.join(streaming_pdb_dataset_path, last_child)

        print(f"Loading dataset for traj: {last_child}...")
        partial_ds = load_from_disk(traj_path)
        full_ds.append(partial_ds)

    # Merge all trajectory-level shards into one dataset before shuffling and
    # splitting. This gives the training pipeline a single consistent dataset view.
    full_ds = concatenate_datasets(full_ds)
    print(f"✓ Full dataset loaded with {len(full_ds)} samples.")

    # Step 3: shuffle once with a fixed seed so the downstream split is
    # deterministic and not correlated with the ordering of the source TARs.
    full_set = full_ds.shuffle(seed=42)
    
    # Create the project splits using the shared helper so the split ratios stay
    # aligned with the rest of the training codebase.
    train_val_test = train_val_test_split(full_set, train_fraction=0.7, val_fraction=0.15, seed=42)
    full_training_set = train_val_test['train']
    full_val_set = train_val_test['val']
    full_test_set = train_val_test['test']

    # Log the final split sizes so preprocessing output can be verified quickly.
    print(f"\tFinal training set size: {len(full_training_set)}")
    print(f"\tFinal validation set size: {len(full_val_set)}")
    print(f"\tFinal test set size: {len(full_test_set)}")

    # Save the final dataset hierarchy in the project output location.
    dataset_dir = os.path.dirname('/project_antwerp/dataset/full_dataset/')
    os.makedirs(dataset_dir, exist_ok=True)
    print("Saving final datasets to disk...")
    final_full_set_path = os.path.join(dataset_dir, "full_dataset")

    # Persist each split separately so training scripts can load only the subset
    # they need without reading the entire corpus every time.
    for split_name, dset in train_val_test_split(full_set, train_fraction=0.7, val_fraction=0.15, seed=42).items():
        set_path = os.path.join(final_full_set_path, split_name)
        os.makedirs(set_path, exist_ok=True)
        dset.save_to_disk(set_path, max_shard_size="4GB", num_proc=num_proc)


    print("Datasets saved.")


if __name__ == "__main__":
    # Run the full preprocessing pipeline when this module is executed directly.
    # ``regenerate`` forces shard recreation, while ``skip_existing`` can be used
    # to avoid repeating work on archives that were already processed.
    generate_full_dataset_from_tars(regenerate=False, num_proc=16, skip_existing=True)

