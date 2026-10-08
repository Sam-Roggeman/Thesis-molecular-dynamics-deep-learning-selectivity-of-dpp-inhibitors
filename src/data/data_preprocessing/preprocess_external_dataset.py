from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import argparse
import os
import tarfile
from io import StringIO
from pathlib import Path

import numpy as np
from Bio import PDB
from datasets import Dataset, NamedSplit

from src.utils.molecular_parsing import parse_filename
from src.data.data_preprocessing.utils import parse_pdb_streaming

def parse_pdb_streaming_many_tars(tar_paths):
	"""Yield frame records from a list of TAR files.

	When `tar_paths` is provided as a list in `gen_kwargs`, Hugging Face can shard
	the workload across processes.
	"""
	for tar_path in tar_paths:
		path_obj = Path(tar_path)
		stem = path_obj.name.removesuffix(".tar.gz").removesuffix(".tgz")
		dpp_class, ligand_name, binding_type, replica_id = parse_filename(stem)
		print(f"Processing {path_obj.name} with DPP class: {dpp_class}, ligand: {ligand_name}, binding type: {binding_type}, replica ID: {replica_id}...")
		yield from parse_pdb_streaming(
			tar_path=str(path_obj),
			dpp_class=dpp_class,
			ligand_name=ligand_name,
			binding_type=binding_type,
			replica_id=replica_id,
		)


def generate_unique_test_runs_dataset(
	tar_folder,
	output_root,
	num_proc=1,
	max_shard_size="4GB",
):
	"""Create one Hugging Face dataset split with all frames from all .tar.gz files."""
	tar_files = sorted(Path(tar_folder).glob("*.tar.gz"))
	tar_files.extend(sorted(Path(tar_folder).glob("*.tgz")))
	if not tar_files:
		raise FileNotFoundError(f"No .tar.gz or .tgz files found in: {tar_folder}")

	print(f"Found {len(tar_files)} compressed TAR files in {tar_folder}")
	tar_paths = [str(path) for path in tar_files]
	full_ds = Dataset.from_generator(
		parse_pdb_streaming_many_tars,
		gen_kwargs={"tar_paths": tar_paths},
		num_proc=num_proc,
		split=NamedSplit("unique_test_runs"),
	)
	print(f"Final unique_test_runs dataset size: {len(full_ds)}")

	output_root = Path(output_root)
	set_path = output_root / "unique_test_runs"
	set_path.mkdir(parents=True, exist_ok=True)
	print(f"Saving dataset to: {set_path}")
	full_ds.save_to_disk(
		str(set_path),
		max_shard_size=max_shard_size,
		num_proc=num_proc,
	)
	print("Dataset saved.")


def main():
	if not os.environ.get("HF_HOME"):
		print("Warning: HF_HOME environment variable is not set. Hugging Face datasets will be stored in the default location.")
		return -1
	
	parser = argparse.ArgumentParser(
		description="Preprocess .tar.gz PDB trajectories into a single unique_test_runs split"
	)
	parser.add_argument("--tar-folder", required=True, help="Folder containing .tar.gz/.tgz files")
	parser.add_argument("--output-root", required=True, help="Output root directory for the saved split")
	parser.add_argument("--num-proc", type=int, default=1, help="Number of processes for generation/sharding")
	parser.add_argument("--max-shard-size", default="4GB", help="HF dataset shard size")
	args = parser.parse_args()

	generate_unique_test_runs_dataset(
		tar_folder=args.tar_folder,
		output_root=args.output_root,
		num_proc=args.num_proc,
		max_shard_size=args.max_shard_size,
	)


if __name__ == "__main__":
	main()
