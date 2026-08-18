from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import time
import torch

from src.cli.interpretability import execute_interpretability
from src.utils.utils import ligant_to_class


ROOT_DIRECTORY = Path(
    "/project_antwerp/dataset/decompressed_both_datasets/"
)
ROOT_OUTPUT_DIRECTORY = Path(
    "/project_antwerp/dataset/decompressed_both_datasets/output2/"
)

THRESHOLD = 0.5

METHOD_ARGS = {
    "integrated_gradients": {"steps": 50},
    "occlusion": {
        "patch_size": 1,
        "perturbations_per_eval": 4096,
        "shift_size": 1,
    },
    "saliency": {},
}

MODEL_CHECKPOINTS = {
    "DCNN": Path(
        "/project_antwerp/Thesis-molecular_dynamics_trajectory_embeddings/"
        "output/models/CustomDenseNet_Randomsplit_Dataset/"
        "20260323-202241/"
        "CustomDenseNet_Randomsplit_Dataset.pth"
    ),
    "SCNN": Path(
        "/project_antwerp/Thesis-molecular_dynamics_trajectory_embeddings/"
        "output/models/SCNN/"
        "20260815-030746/"
        "SCNN.pth"
    ),
}


def process_job(job):
    gpu_id, replica_dir, output_dir, binding_type, model_name, checkpoint = job

    # Each worker only sees its assigned GPU as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start_time = time.time()

    execute_interpretability(
        pdb_directory=replica_dir,
        output_dir=output_dir,
        binding_type=binding_type,
        model_checkpoint=checkpoint,
        method_args=METHOD_ARGS,
        threshold=THRESHOLD,
    )

    elapsed = time.time() - start_time

    print(
        f"GPU {gpu_id} | "
        f"{model_name} | "
        f"{replica_dir.parent.parent.name}/"
        f"{replica_dir.parent.name}/"
        f"{replica_dir.name} | "
        f"{elapsed:.1f}s",
        flush=True,
    )


def main():
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")

    jobs = []

    for dpp_dir in ROOT_DIRECTORY.iterdir():
        if not dpp_dir.is_dir():
            continue

        dpp = dpp_dir.stem
        if dpp not in ["DPP8", "DPP9"]:
            continue

        for ligand_dir in dpp_dir.iterdir():
            if not ligand_dir.is_dir():
                continue

            ligand = ligand_dir.stem
            binding_type = ligant_to_class(ligand_name=ligand)

            for replica_dir in ligand_dir.iterdir():
                if not replica_dir.is_dir():
                    continue

                replica = replica_dir.stem

                for model_name, checkpoint in MODEL_CHECKPOINTS.items():
                    output_dir = (
                        ROOT_OUTPUT_DIRECTORY
                        / dpp
                        / ligand
                        / replica
                        / model_name
                    )

                    # Optional: skip completed jobs.
                    # Adjust this depending on what execute_interpretability
                    # actually produces.
                    #
                    # if output_dir.exists():
                    #     continue

                    gpu_id = len(jobs) % num_gpus

                    jobs.append(
                        (
                            gpu_id,
                            replica_dir,
                            output_dir,
                            binding_type,
                            model_name,
                            checkpoint,
                        )
                    )

    print(f"Found {len(jobs)} jobs across {num_gpus} GPUs")

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        list(executor.map(process_job, jobs))


if __name__ == "__main__":
    main()