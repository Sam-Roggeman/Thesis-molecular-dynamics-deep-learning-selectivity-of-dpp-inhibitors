from pathlib import Path
from src.cli.interpretability import execute_interpretability
from src.utils.utils import ligant_to_class

def main():
    root_directory = Path("/home/stijn/Sam_ModelVisualisation/input_frames/uncompressed_every100/")
    root_output_directory = Path("/home/stijn/Sam_ModelVisualisation/output")
    threshold = 0.5
    method_args = {
        "integrated_gradients":{"steps": 50},
        "occlusion": {
            "patch_size": 1,
            "perturbations_per_eval": 500,
            "shift_size": 1
        }, 
        "saliency": {}
    }
    model_dir = "/home/stijn/Sam_ModelVisualisation/Models/"
    model_checkpoints = {
         "DCNN": Path(model_dir, "dcnn", "CustomDenseNet_Randomsplit_Dataset.pth"), 
         "SCNN": Path(model_dir, "scnn", "SCNN.pth"), 
    }

    # dpp8/dpp9
    for dpp_dir in root_directory.iterdir():
        if dpp_dir.is_dir():
            dpp = dpp_dir.stem
            # ligand
            for ligand_dir in dpp_dir.iterdir():
                ligand = ligand_dir.stem
                binding_type = ligant_to_class(ligand_name=ligand) 
                if ligand_dir.is_dir():
                    for replica_dir in ligand_dir.iterdir():
                        replica = replica_dir.stem
                        # replica 
                        if replica_dir.is_dir(): 
                                for model_name, checkpoint in model_checkpoints.items():
                                    output_dir = Path(root_output_directory, dpp, ligand, replica, model_name)
                                    print(f"""pdb_directory={replica_dir}, 
                                        output_dir= {output_dir},
                                        binding_type= {binding_type}, 
                                        model_checkpoint={checkpoint}, 
                                        method_args={method_args}, 
                                        threshold={threshold}""")
                                    execute_interpretability(
                                        pdb_directory=replica_dir, 
                                        output_dir= output_dir,
                                        binding_type= binding_type, 
                                        model_checkpoint=checkpoint, 
                                        method_args=method_args, 
                                        threshold=threshold
                                    )

if __name__ == "__main__":    
    main()