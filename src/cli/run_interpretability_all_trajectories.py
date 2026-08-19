import multiprocessing as mp
import os
import time
from pathlib import Path
import contextlib
from src.utils.interpretability import CaptumInterpreter, _initilize_classification_model, solve_methods
import torch

from src.cli.interpretability import execute_interpretability, save_attribution_colored_pdbs_spatial_scores, write_coloring_script
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
        "perturbations_per_eval": 2048,
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

def already_completed(output_dir: Path) -> bool:
    """
    Check if the interpretability results for a given output directory already exist.

    :param output_dir: Directory where interpretability results are expected to be saved.
    :return: True if the results already exist, False otherwise.
    """
    # already completed if the images subfolder contains 148 files
    images_dir = output_dir / "images"
    if images_dir.exists() and len(list(images_dir.glob("*"))) >= 148:
        return True
    
    return False
def worker(gpu_id, job_queue):
    # preload both models to avoid reloading them for each job


    # Each worker is permanently assigned to one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    dcnn = _initilize_classification_model(MODEL_CHECKPOINTS["DCNN"], device=torch.device("cuda"))
    scnn = _initilize_classification_model(MODEL_CHECKPOINTS["SCNN"], device=torch.device("cuda"))
    dcnn_methods = solve_methods(CaptumInterpreter(dcnn), method_args=METHOD_ARGS)
    scnn_methods = solve_methods(CaptumInterpreter(scnn), method_args=METHOD_ARGS)

    # Import CUDA-dependent code only after CUDA_VISIBLE_DEVICES is set
    # if your imports initialize CUDA.
    torch.cuda.set_device(0)
    model=None
    methods=None
    while True:
        job = job_queue.get()

        if job is None:
            break
        (
            replica_dir,
            output_dir,
            binding_type,
            model_name,
            checkpoint,
        ) = job

        start_time = time.time()

        print(
            f"[GPU {gpu_id}] START "
            f"{model_name} | {replica_dir}",
            flush=True,
        )
        
        try:
            # load the trained classification model from the provided checkpoint
            if model_name == "DCNN":
                model = dcnn
                methods = dcnn_methods
            else:
                model = scnn
                methods = scnn_methods
            # disable printing from the interpretability function to avoid cluttering the output
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                execute_interpretability(
                    pdb_directory=replica_dir,
                    output_dir=output_dir,
                    binding_type=binding_type,
                    model_checkpoint=model,
                    methods=methods,
                    threshold=THRESHOLD,
                )

            elapsed = time.time() - start_time

            print(
                f"[GPU {gpu_id}] DONE "
                f"{model_name} | {replica_dir} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

        except Exception as e:
            print(
                f"[GPU {gpu_id}] ERROR "
                f"{model_name} | {replica_dir}: {e}",
                flush=True,
            )


def main():
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    print(f"Using {num_gpus} GPUs")

    ctx = mp.get_context("spawn")

    job_queue = ctx.Queue()

    # Create jobs: (replica, model)
    jobs = []

    for dpp_dir in ROOT_DIRECTORY.iterdir():
        if not dpp_dir.is_dir():
            continue

        dpp = dpp_dir.stem
        if dpp not in ["DPP8", "DPP9"]:
            print(f"Skipping {dpp_dir} (not DPP8 or DPP9)")
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
                    if already_completed(output_dir):
                        print(
                            f"Skipping {model_name} | {replica_dir} "
                            f"(already completed)",
                            flush=True,
                        )
                        continue
                    jobs.append(
                        (
                            replica_dir,
                            output_dir,
                            binding_type,
                            model_name,
                            checkpoint,
                        )
                    )

    print(f"Total jobs: {len(jobs)}")

    # Put all jobs into the shared queue.
    for job in jobs:
        job_queue.put(job)

    # One worker per GPU.
    workers = []

    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=worker,
            args=(gpu_id, job_queue),
        )
        p.start()
        workers.append(p)

    # One sentinel per worker.
    for _ in range(num_gpus):
        job_queue.put(None)

    # Wait for workers.
    for p in workers:
        p.join()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def recover_spatial_scores(image_path):
    image = plt.imread(image_path)[..., :3]

    # Exactly black pixels: R=G=B=0
    black_pixels = np.all(image == 0.0, axis=-1)

    # Black = 1, everything else = 0
    spatial_scores = black_pixels.astype(np.float32)

    return spatial_scores

def pdb_worker(job_queue):
    threshold = 0.5
    while True:
        job = job_queue.get()

        if job is None:
            break
        (
            replica_dir,
            output_dir,
            sample
        ) = job
        # integrated_gradients, occlusion, saliency overlay
        # get the corresponding PDB file fromt he replica dir
        try:
            original_pdb_file = replica_dir / f"{sample.name.replace('_blurred_sample.png', '.pdb')}"
            
            if "integrated_gradients" in sample.name:
                method = "integrated_gradients"
                original_pdb_file = original_pdb_file.replace("_integrated_gradients", "")
            elif "occlusion" in sample.name:
                method = "occlusion"
                original_pdb_file = original_pdb_file.replace("_occlusion", "")
            elif "saliency" in sample.name:
                method = "saliency"
                original_pdb_file = original_pdb_file.replace("_saliency", "")
            # recover the spatial scores from the overlay and save them in the PDB file
            spatial_scores = recover_spatial_scores(
                image_path=sample,
            )
            spatial_scores = torch.from_numpy(
                spatial_scores.reshape(-1)
            ).float()


            colored_pdb_path = save_attribution_colored_pdbs_spatial_scores(
                pdb_file=original_pdb_file,
                output_dir=output_dir,
                spatial_scores=spatial_scores,
                method=method,
            )
            script_path = Path.joinpath(Path(colored_pdb_path).parent, f"{Path(colored_pdb_path).stem}.pml")

            write_coloring_script([colored_pdb_path], script_path, threshold=threshold)
            print(f"Processed sample {sample.name} in {output_dir}")
        except Exception as e:
            print(f"Error processing sample {sample.name} in {output_dir}: {e}")
def fix_pdbs():
    threshold = 0.5
    ctx = mp.get_context("spawn")

    job_queue = ctx.Queue()
    jobs = []

    for dpp_dir in ROOT_DIRECTORY.iterdir():
        if not dpp_dir.is_dir():
            continue

        dpp = dpp_dir.stem
        if dpp not in ["DPP8", "DPP9"]:
            print(f"Skipping {dpp_dir} (not DPP8 or DPP9)")
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
                        / "images"
                    )
                    # for each sample containing transformed sample, check if the corresponding PDB file exists in the output directory
                    for sample in output_dir.glob("*_blurred_sample.png"):
                        # only if the sample has 3 pml files (one for each method) and 3 colored PDB files (one for each method), then skip it
                        # pml_files = list(output_dir.glob(f"{sample.stem.replace('_blurred_sample', '')}_*.pml"))
                        # colored_pdb_files = list(output_dir.glob(f"{sample.stem.replace('_blurred_sample', '')}_*_bfactor.pdb"))
                        # if len(pml_files) == 3 and len(colored_pdb_files) == 3:
                        #     print(f"Skipping {sample.name} in {output_dir} (already completed)")
                        #     continue
                        jobs.append(
                            (
                                replica_dir,
                                output_dir,
                                sample
                            )
                        )
                    # handle the average sample as well

    print(f"Total jobs: {len(jobs)}")

    # Put all jobs into the shared queue.
    nr_workers = 16
    for job in jobs:
        job_queue.put(job)

    # One worker per GPU.
    workers = []

    for worker_id in range(nr_workers):
        p = ctx.Process(
            target=pdb_worker,
            args=(job_queue,),
        )
        p.start()
        workers.append(p)

    # One sentinel per worker.
    for _ in range(nr_workers):
        job_queue.put(None)

    # Wait for workers.
    for p in workers:
        p.join()          
if __name__ == "__main__":
    fix_pdbs()