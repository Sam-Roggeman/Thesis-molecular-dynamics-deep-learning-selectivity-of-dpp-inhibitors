import os
import torch


# inline function to get classes
def get_binding_classes():
    return ["apo", "nonbinder", "dpp9selective", "dpp8selective", "aselective"]

ligand_to_binding_type = {
        "12i": "nonbinder", # Nonbinder
        "42":       "dpp9selective", # dpp9 selective
        "000808":   "dpp8selective", # DPP8selective
        "0003822":  "aselective",# Aselective
        "apo":      "apo",
        "0000157": "nonbinder", # Nonbinder
        "0005356": "dpp9selective", # dpp9 selective
        "0005862": "dpp8selective", # DPP8selective
        "0005362": "aselective",# Aselective
        "0000193": "nonbinder", # Nonbinder
        "0004067": "aselective", # Aselective
        "0004804": "dpp9selective", # dpp9 selective
        "0005858": "dpp8selective", # DPP8selective
    }

def ligant_to_class(ligand_name):
    """
    Map ligand name to binding type using the ligand_to_binding_type dictionary.
    :param ligand_name: str
    :return: binding type str
    """
    ligand_name = ligand_name.lower()
    if ligand_name in ligand_to_binding_type:
        return ligand_to_binding_type[ligand_name]
    elif ligand_name is None:
        return "apo"
    else:
        raise ValueError(f"Unknown ligand name {ligand_name}.")
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
    :param filename: str in format sep_prot_frames_{DPP_class}_{ligand_name}_replica{replica_id}.{extension}
    """

    filename = remove_extension(filename)
    filename = filename.lower()
    parts = filename.split('_')
    # if a part is "correct", remove it
    if parts[5] == "correct":
        parts.pop(5)

    if len(parts) == 5:
        parts.insert(4, None)  # Insert 'none' for ligand name if missing
    if len(parts) != 6:
        raise ValueError(f"Filename {filename} is not in the expected format.")
    dpp_class = parts[3]
    ligand_name = parts[4]

    replica_id = parts[5].replace('replica', '')

    binding_type = ligant_to_class(ligand_name) if ligand_name else None
    # validate    dpp_class and binding_type
    if dpp_class not in ['dpp8', 'dpp9']:
        raise ValueError(f"Filename {filename} has unknown DPP class {dpp_class}.")
    if binding_type and binding_type not in get_binding_classes():
        raise ValueError(f"Filename {filename} has unknown binding type {binding_type}.")
    if replica_id and not replica_id.isdigit():
        raise ValueError(f"Filename {filename} has invalid replica id {replica_id}.")
    if ligand_name not in ligand_to_binding_type.keys():
        raise ValueError(f"Filename {filename} has unknown ligand name {ligand_name}.")
    return dpp_class, ligand_name, binding_type, replica_id

def remove_extension(filename):
    """
    Remove the extension from a filename.
    :param filename: str
    :return:
    """
    # if no extension, return the original filename
    if '.' not in filename:
        return filename
    return '.'.join(filename.split('.')[:-1])


