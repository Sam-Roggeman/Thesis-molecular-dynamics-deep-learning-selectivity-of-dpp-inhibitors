"""Parsing helpers for PDB content and trajectory filenames."""

from typing import TypeAlias

import torch

from src.utils.labels import get_binding_classes, ligand_to_binding_type, ligant_to_class


ParsedFilename: TypeAlias = tuple[str, str | None, str | None, str]


def parse_pdb_from_string(pdb_content: str) -> torch.Tensor:
	"""Extract atom coordinates from PDB text as an ``[N, 3]`` tensor.

	Malformed coordinate fields are skipped. If no valid atom records are found,
	the returned tensor is empty.
	"""
	coords: list[list[float]] = []
	for line in pdb_content.split("\n"):
		if line.startswith(("ATOM", "HETATM")):
			try:
				# PDB coordinates occupy fixed-width columns 30:54.
				x = float(line[30:38])
				y = float(line[38:46])
				z = float(line[46:54])
				coords.append([x, y, z])
			except (ValueError, IndexError):
				continue
	return torch.tensor(coords)


def parse_filename(filename: str) -> ParsedFilename:
	"""Parse experiment metadata encoded in a trajectory filename.

	The expected shape is ``..._<dpp_class>_<ligand>_replica<id>`` with an
	optional ``correct`` marker. The returned tuple contains DPP class, ligand,
	binding class, and replica ID.

	Raises:
		ValueError: If any encoded field is not recognized or valid.
	"""
	filename = remove_extension(filename).lower()
	parts = filename.split("_")

	if parts[5] == "correct":
		parts.pop(5)
	if len(parts) == 5:
		parts.insert(4, None)
	if len(parts) != 6:
		raise ValueError(f"Filename {filename} is not in the expected format.")

	dpp_class = parts[3]
	ligand_name = parts[4]
	replica_id = parts[5].replace("replica", "")
	binding_type = ligant_to_class(ligand_name) if ligand_name else None

	if dpp_class not in ["dpp8", "dpp9"]:
		raise ValueError(f"Filename {filename} has unknown DPP class {dpp_class}.")
	if binding_type and binding_type not in get_binding_classes():
		raise ValueError(f"Filename {filename} has unknown binding type {binding_type}.")
	if replica_id and not replica_id.isdigit():
		raise ValueError(f"Filename {filename} has invalid replica id {replica_id}.")
	if ligand_name not in ligand_to_binding_type:
		raise ValueError(f"Filename {filename} has unknown ligand name {ligand_name}.")

	return dpp_class, ligand_name, binding_type, replica_id


def remove_extension(filename: str) -> str:
	"""Remove only the final extension from a filename."""
	if "." not in filename:
		return filename
	return ".".join(filename.split(".")[:-1])


__all__ = ["parse_pdb_from_string", "parse_filename", "remove_extension"]
