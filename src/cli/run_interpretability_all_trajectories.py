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


def recover_spatial_scores(
    overlay_path,
    image_path,
):
    # Load overlay
    overlay = (
        np.asarray(Image.open(overlay_path).convert("RGB"))
        .astype(np.float32) / 255.0
    )
    sample_image = plt.imread(image_path)[..., :3]
    overlay = plt.imread(overlay_path)[..., :3]
    heatmap_rgb = (overlay - 0.6 * sample_image) / 0.4
    heatmap_rgb = np.clip(heatmap_rgb, 0, 1)

    # Recover RGB heatmap:
    #
    # overlay = 0.6 * sample_image + 0.4 * heatmap
    heatmap_rgb = (overlay - 0.6 * sample_image) / 0.4
    heatmap_rgb = np.clip(heatmap_rgb, 0, 1)

    # Invert matplotlib's "hot" colormap
    cmap = plt.get_cmap("hot")

    values = np.linspace(0, 1, 10001)
    colors = cmap(values)[:, :3]

    pixels = heatmap_rgb.reshape(-1, 3)

    # Find closest "hot" color
    distances = (
        (pixels[:, None, :] - colors[None, :, :]) ** 2
    ).sum(axis=2)

    indices = distances.argmin(axis=1)

    spatial_scores = values[indices].reshape(
        heatmap_rgb.shape[:2]
    )

    return spatial_scores
def fix_pdbs():
    threshold = 0.5
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
                    for sample in output_dir.glob("*_transformed_sample.png"):
                        # integrated_gradients, occlusion, saliency overlay
                        # get the corresponding PDB file fromt he replica dir
                        original_pdb_file = replica_dir / f"{sample.name.replace('_transformed_sample.png', '.pdb')}"
                        for overlay in ["integrated_gradients", "occlusion", "saliency"]:
                            overlay_filename = sample.name.replace("_transformed_sample.png", f"_{overlay}_overlay.png")
                            overlay_path = output_dir / overlay_filename
                            if not (overlay_path).exists():
                                print(f"Missing overlay {overlay_filename} for sample {sample.name} in {output_dir}")
                                continue
                            
                            pdb_filename = sample.name.replace("_transformed_sample.png", ".pdb")
                            # recover the spatial scores from the overlay and save them in the PDB file
                            spatial_scores = recover_spatial_scores(
                                overlay_path=overlay_path,
                                image_path=sample,
                            )


                            colored_pdb_path = save_attribution_colored_pdbs_spatial_scores(
                                pdb_file=original_pdb_file,
                                output_dir=output_dir,
                                spatial_scores=spatial_scores,
                                method=overlay,
                            )
                            script_path = Path.joinpath(Path(colored_pdb_path).parent, f"{Path(colored_pdb_path).stem}.pml")

                            write_coloring_script([colored_pdb_path], script_path, threshold=threshold)
                    # wait for input from user to continue
                    input(f"Finished fixing PDBs for {model_name} | {replica_dir} in {output_dir}. Press Enter to continue...") 
                        
if __name__ == "__main__":
    fix_pdbs()