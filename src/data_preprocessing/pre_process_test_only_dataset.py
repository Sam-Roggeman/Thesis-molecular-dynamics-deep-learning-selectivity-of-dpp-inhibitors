import argparse
import tarfile
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from Bio import PDB
from datasets import Dataset, NamedSplit

from src.utils.utils import parse_filename


def extract_coordinates(pdb_file, pdb_id):
	"""Extract 3D coordinates from a PDB file object."""
	content = pdb_file.read().decode("utf-8")
	parser = PDB.PDBParser(QUIET=True)
	structure = parser.get_structure(pdb_id, StringIO(content))

	coords = []
	for model in structure:
		for chain in model:
			for residue in chain:
				for atom in residue:
					coords.append(atom.coord)

	return np.array(coords, dtype=np.float32)


def parse_pdb_streaming(tar_path, dpp_class, ligand_name, binding_type, replica_id):
	"""Yield frame records parsed from PDB files in a TAR archive."""
	with tarfile.open(tar_path, "r:gz") as tar:
		for frame in tar:
			if not frame.isfile() or not frame.name.endswith(".pdb"):
				continue

			pdb_file = tar.extractfile(frame)
			if pdb_file is None:
				continue

			pdb_id = Path(frame.name).stem
			try:
				coords = extract_coordinates(pdb_file, pdb_id)
				yield {
					"pdb_id": pdb_id,
					"dpp_class": dpp_class,
					"ligand_name": ligand_name,
					"binding_type": binding_type,
					"coordinates": coords,
					"num_atoms": len(coords),
					"replica_id": replica_id,
				}
			except Exception as exc:
				print(f"Error parsing {pdb_id}: {exc}")


def parse_metadata_from_filename(stem):
	"""Extract metadata from filename and fall back to unknown fields if parsing fails."""
	try:
		return parse_filename(stem)
	except Exception:
		return "unknown", "unknown", "unknown", stem


def parse_pdb_streaming_many_tars(tar_paths):
	"""Yield frame records from a list of TAR files.

	When `tar_paths` is provided as a list in `gen_kwargs`, Hugging Face can shard
	the workload across processes.
	"""
	for tar_path in tar_paths:
		path_obj = Path(tar_path)
		stem = path_obj.name.removesuffix(".tar.gz").removesuffix(".tgz")
		dpp_class, ligand_name, binding_type, replica_id = parse_metadata_from_filename(stem)

		print(f"Processing {path_obj.name}...")
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
		num_shards=max(1, num_proc),
		num_proc=num_proc,
	)
	print("Dataset saved.")


def main():
    load_dotenv()  # Load environment variables from .env file
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
