import numpy as np
from Bio import PDB
from pathlib import Path


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
def parse_pdb_streaming(tar, frames, dpp_class, ligand_name, binding_type, replica_id):
    """Yield one dataset record per PDB file stored inside a TAR archive.

    The generator does not unpack the archive to disk. Instead, it walks over the
    TAR members, opens each ``.pdb`` file in memory, extracts the atomic coordinates,
    and emits a Hugging Face-style dictionary that can be consumed by
    ``Dataset.from_generator``.

    Each yielded row keeps the experiment metadata together with the parsed
    coordinates so the downstream dataset still knows which trajectory and
    ligand/binding setup the frame came from.
    """
    for frame in frames:
        if frame.name.endswith('.pdb'):
            # Open the PDB entry directly from the archive instead of extracting it
            # to a temporary file. This keeps preprocessing faster and avoids
            # creating a large intermediate directory of frame files.
            f = tar.extractfile(frame)
            pdb_id = Path(frame.name).stem

            try:
                # Parse the coordinate block for this structure. The helper returns
                # the coordinate representation used throughout the training pipeline.
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
                # A single corrupt frame should not stop the entire archive from
                # being processed, so we log the failure and continue streaming.
                print(f"Error parsing {pdb_id}: {e}")
                continue
