import argparse
import os
from pathlib import Path
from typing import Callable
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.Transform.tranformators import apply_image_transform_noscramble
from src.data_preprocessing.utils import extract_coordinates
from src.model_training.LabelEncoder import LabelEncoder
from src.utils.interpretability import AttributionResult, CaptumInterpreter
from src.utils.resolvers import (
    _extract_class_name,
    _load_config_from_artifacts,
    _load_state_dict,
    _load_weights,
    _resolve_model_class,
)


def extract_pdb_file(pdb_file: str, ligand_name: str|None = None, binding_type: str|None = None, dpp_class: str|None = None, replica_id: int|None = None) -> dict:
    """
    Generate the preprocessed data sample from a PDB file.
    Preprocess the PDB file and extract relevant features
    :param pdb_file: Path to the PDB file to be processed.
    :param ligand_name: Name of the ligand. Optional
    :param binding_type: Type of binding. Optional
    :param dpp_class: DPP class. Optional
    :param replica_id: ID of the replica. Optional
    :return: Preprocessed data sample ready for interpretability analysis. 
    This is a sample containing {'coordinates': list, 'binding_type': str, 'dpp_class': str, 'ligand_name': str, 'replica_id': int, num_atoms: int}
    """
    with open(pdb_file, 'rb') as f:
        pdb_id = Path(pdb_file).stem
        coords = extract_coordinates(f, pdb_id)

    return {
        'coordinates': coords,
        'binding_type': binding_type,
        'dpp_class': dpp_class,
        'ligand_name': ligand_name,
        'replica_id': replica_id,
        'num_atoms': len(coords)
    }

def apply_transformations_to_sample(sample: dict, transform_fn: Callable) -> dict:
    """
    Apply transformations to the preprocessed data sample.
    :param sample: Preprocessed data sample.
    :param transform_fn: Callable object representing the transformations to apply.
    :return: Transformed data as a tensor.
    """
    # wrap the sample in ndarrays

    return transform_fn(
        examples_data=[sample['coordinates']], 
        examples_labels=[sample['binding_type']],
        real_nr_atoms=[sample['num_atoms']]
    )

def _extract_data_tensor(transformed_sample: dict) -> torch.Tensor:
    """Extract one sample tensor [C,H,W] from transform output dict."""
    data = transformed_sample["data"]

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Transformed data list is empty.")
        sample_tensor = torch.as_tensor(data[0], dtype=torch.float32)
    else:
        sample_tensor = torch.as_tensor(data, dtype=torch.float32)
        if sample_tensor.ndim == 4:
            sample_tensor = sample_tensor[0]

    if sample_tensor.ndim != 3:
        raise ValueError(f"Expected transformed sample shape [C,H,W], got {tuple(sample_tensor.shape)}")
    return sample_tensor


def apply_classification(sample: torch.Tensor, model: torch.nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the trained classification model to the transformed sample.
    :param sample: Transformed data sample.
    :param model: Trained classification model.
    :param device: Device used for model inference.
    :return: Model output as a tensor.
    """
    model.eval()
    with torch.no_grad():
        sample_batch = sample.unsqueeze(0).to(device).clone().detach()
        logits = model(sample_batch)
        predicted_class = torch.argmax(logits, dim=1)
    return logits, predicted_class

def solve_methods(interpreter: CaptumInterpreter, args) -> dict[str, Callable]:
    """
    Resolve a method string to the corresponding interpretability method.
    :param method_str: String identifier for the interpretability method (e.g., "integrated_gradients").
    :return: Corresponding interpretability method object.
    """
    methods = {}
    for method in args.methods:
        if method == "integrated_gradients":
            # keep argparse-driven parameters configurable per run
            methods[method] = lambda inputs, target: interpreter.integrated_gradients(
                inputs,
                target=target,
                n_steps=args.ig_steps,
            )
        elif method == "saliency":
            methods[method] = lambda inputs, target: interpreter.saliency(inputs, target=target)
        elif method == "occlusion":
            methods[method] = lambda inputs, target: interpreter.occlusion(
                inputs,
                target=target,
                patch_size=args.occlusion_patch_size,
                shift_size=args.occlusion_shift_size,
            )
        else:
            print(f"Unknown interpretability method '{method}'. Supported methods: 'integrated_gradients', 'saliency', 'occlusion'. Skipping.")
    if len(methods) == 0:
        methods["integrated_gradients"] = lambda inputs, target: interpreter.integrated_gradients(
            inputs,
            target=target,
            n_steps=args.ig_steps,
        )
    return methods
def blur_according_to_attribution_results(
    threshold: float,
    input_sample: torch.Tensor,
    insights: dict[str, AttributionResult]
) -> dict[str, torch.Tensor]:
    """Blur the input sample according to the attribution results.
    :threshold: The attribution score threshold above which pixels will be blurred.
    :input_sample: The original input sample tensor [C,H,W].
    :insights: The dictionary of attribution results per method.
    """
    blurred_samples = {}
    for method, result in insights.items():
        attributions = result.attributions
        print(f"Validating attribution results for method: {method}")
        if attributions.ndim == 4:
            attributions = attributions[0]
        if attributions.ndim != 3:
            raise ValueError(f"Expected attribution shape [C,H,W] or [1,C,H,W], got {tuple(attributions.shape)}")

        pixel_scores = attributions.detach().cpu().abs().sum(dim=0).reshape(-1)
        mask = pixel_scores > threshold
        blurred_sample = input_sample.clone()
        blurred_sample[:, mask] = 0.0
        blurred_samples[method] = blurred_sample
    return blurred_samples


def generate_interpretability_attribution(
    methods: dict[str, Callable],
    predicted_class: torch.Tensor,
    sample: torch.Tensor,
    device: torch.device,
) -> dict[str, AttributionResult]:
    """
    Generate interpretability insights using the model's output and the input sample.
    :param predicted_class: Predicted class index for the input sample.
    :param sample: Transformed input sample tensor [C,H,W].
    :param device: Device used for attribution methods.
    :return: Dictionary containing Captum AttributionResult per method.
    """
    attributions_results: dict[str, AttributionResult] = {}

    sample_batch = sample.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    target_idx = int(predicted_class.item())  # batch size is 1

    for method, method_fn in methods.items():
        print(f"Generating interpretability insights using method: {method}")
        attributions = method_fn(inputs=sample_batch, target=target_idx)
        attributions_results[method] = attributions

    return attributions_results


def _attribution_to_atom_scores(attributions: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Map attribution tensor back to first num_atoms entries of flattened 2D grid."""
    if attributions.ndim == 4:
        attributions = attributions[0]
    if attributions.ndim != 3:
        raise ValueError(f"Expected attribution shape [C,H,W] or [1,C,H,W], got {tuple(attributions.shape)}")

    # Aggregate channel-wise magnitude so each flattened pixel maps to one scalar score.
    pixel_scores = attributions.detach().cpu().abs().sum(dim=0).reshape(-1)
    if num_atoms > pixel_scores.numel():
        raise ValueError(f"num_atoms={num_atoms} exceeds available pixel scores={pixel_scores.numel()}")
    return pixel_scores[:num_atoms]


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Normalize scores to [0, 100] for PDB B-factor visualization."""
    if scores.numel() == 0:
        return scores
    min_v = scores.min()
    max_v = scores.max()
    if torch.isclose(max_v, min_v):
        return torch.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v) * 100.0


def _write_bfactor_colored_pdb(pdb_file: str, output_path: str, atom_scores: torch.Tensor) -> None:
    """Write PDB with per-atom attribution scores in B-factor column."""
    with open(pdb_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    atom_line_count = sum(1 for line in lines if line.startswith(("ATOM", "HETATM")))
    if atom_line_count != int(atom_scores.numel()):
        raise ValueError(
            "Number of ATOM/HETATM lines does not match attribution-derived atom scores. "
            f"PDB atoms={atom_line_count}, scores={atom_scores.numel()}."
        )

    score_idx = 0
    out_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            bfactor = float(atom_scores[score_idx].item())
            score_idx += 1
            padded = line.rstrip("\n")
            if len(padded) < 66:
                padded = padded.ljust(66)
            padded = padded[:60] + f"{bfactor:6.2f}" + padded[66:]
            out_lines.append(padded + "\n")
        else:
            out_lines.append(line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.writelines(out_lines)


def save_attribution_colored_pdbs(
    pdb_file: str,
    output_dir: str,
    attribution_results: dict[str, AttributionResult],
    num_atoms: int,
) -> list[str]:
    """Create one PDB per attribution method with atom scores stored in B-factor."""
    colored_pdb_paths = []
    for method, result in attribution_results.items():
        atom_scores = _attribution_to_atom_scores(result.attributions, num_atoms)
        atom_scores = _normalize_scores(atom_scores)
        # the most important atoms will have a B-factor of 100, the least important will have a B-factor of 0, and the others will be scaled in between
        # this allows for easy visualization in PyMOL using a spectrum from gray (0) to red (100)
        output_path = os.path.join(output_dir, f"{Path(pdb_file).stem}_{method}_bfactor.pdb")
        _write_bfactor_colored_pdb(pdb_file, output_path, atom_scores)
        print(f"Saved colored PDB for {method}: {output_path}")
        colored_pdb_paths.append(Path(output_path))
    return colored_pdb_paths

def _initilize_classification_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = os.path.abspath(checkpoint_path)

    config = _load_config_from_artifacts(None, checkpoint_path)
    if config is None:
        raise ValueError(
            "No training config found next to checkpoint. "
            "Expected training_config.pt or training_config.json."
        )

    model_name = _extract_class_name(config.model_class)
    if not model_name:
        raise ValueError("Could not resolve model class from training config.")

    model_class = _resolve_model_class(model_name)
    model_args = dict(config.model_args) if isinstance(config.model_args, dict) else {}

    model = model_class(**model_args).to(device)
    state_dict = _load_state_dict(checkpoint_path, device)
    _load_weights(model, state_dict)
    model.eval()
    return model
def write_coloring_script(colored_pdb_paths: list[str], script_path: str, threshold: float) -> None:
    """Write a PyMOL script to load and visualize the colored PDBs."""
    bfactor_threshold = threshold * 100.0
    with open(script_path, "w") as f:
        for pdb_path in colored_pdb_paths:
            object_name = Path(pdb_path).name
            stem_name = Path(pdb_path).stem
            f.write(f"load {object_name}, {stem_name}\n")
            f.write(f"color gray, {stem_name}\n")
            f.write(f"spectrum b, gray70 yellow orange red, {stem_name}\n")
            f.write(f"show cartoon, {stem_name}\n")
            f.write(f"select {stem_name}_high_atoms, ({stem_name} and polymer.protein and b > {bfactor_threshold:.2f})\n")
            f.write(f"select {stem_name}_high_residue, byres {stem_name}_high_atoms\n")
            f.write(f"hide cartoon, {stem_name}_high_residue\n")
            f.write(f"show sticks, {stem_name}_high_residue\n")
    print(f"PyMOL coloring script written to: {script_path}")

def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI for interpretability tools")
    parser.add_argument("--pdb_file", type=str, help="Path to a PDB file for generating interpretability insights", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save interpretability results, defaults to 'interpretability_results' next to the pdb file if not provided")
    parser.add_argument("--model_checkpoint", type=str, help="Path to the trained model checkpoint for interpretability analysis", required=True)
    parser.add_argument("--methods", nargs="+", default=["integrated_gradients"], help="List of interpretability methods to apply (e.g., 'integrated_gradients', 'saliency', 'gradient_shap').")
    parser.add_argument("--ig_steps", type=int, default=50, help="Number of steps for Integrated Gradients approximation.")
    parser.add_argument("--occlusion_patch_size", type=int, default=1, help="Patch size for occlusion attribution.")
    parser.add_argument("--occlusion_shift_size", type=int, default=1, help="Shift size for occlusion attribution.")
    parser.add_argument("--binding_type", type=str, default=None, required=True, help="Binding type of the sample. Required for proper sample construction.", choices=["apo", "dpp8selective", "dpp9selective", "aselective", 'nonbinder'])
    parser.add_argument("--open-in-pymol", action="store_true", help="Whether to automatically open the generated colored PDBs in PyMOL after processing.")
    parser.add_argument("--threshold", type=float, default=0.2, help="High attribution threshold as a fraction of the max score for PyMOL visualization (e.g., 0.8 means atoms with scores in the top 20%% will be shown as sticks).")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    if not args.pdb_file:
        print("Please provide a PDB file using the --pdb_file argument.")
        return
    if not args.output_dir:
        args.output_dir = str(Path(args.pdb_file).parent / "interpretability_results")
    return args

def probablity_string(percentages: torch.Tensor, class_labels: list[str]) -> str:
    prob_string = "Model classification choices with probabilities:\n"
    for idx, percentage in enumerate(percentages[0]):
        class_name = class_labels[idx] 
        prob_string += f"\tClass '{class_name}': {percentage.item():.2f}%\n"
    return prob_string

def main():
    """
    CLI entry point for interpretability tools.
    :param pdb_file: path to a PDB file for generating interpretability insights. If provided, the tool will process the file, feed it into the interpretability model, and output the insights. 
    :param output_dir: Directory where interpretability results will be saved. Defaults to "./interpretability_results".
    """
    args = arg_parser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_labels= LabelEncoder().get_classes()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing PDB file: {args.pdb_file}")
    sample = extract_pdb_file(args.pdb_file, binding_type=args.binding_type)

    print(f"Extracted sample from PDB file: {sample}")
    transformed_sample = apply_transformations_to_sample(sample, apply_image_transform_noscramble)
    sample_tensor = _extract_data_tensor(transformed_sample)
    print(f"Transformed sample tensor shape: {tuple(sample_tensor.shape)}")    
    model = _initilize_classification_model(args.model_checkpoint, device=device)

    logits, predicted_class = apply_classification(sample_tensor, model, device=device)
    percentages = torch.nn.functional.softmax(logits, dim=1) * 100


    prob_string = probablity_string(percentages, class_labels)
    print(prob_string)
    print(prob_string, file=open(os.path.join(args.output_dir, "classification_probabilities.txt"), "w"))
    print(f"Predicted class: {predicted_class}")
    methods = solve_methods(CaptumInterpreter(model), args)
    insights: dict[str, AttributionResult] = generate_interpretability_attribution(methods, predicted_class, sample_tensor, device=device)

    # save the insigths as image, where the attributions are mapped from gray (low attribution) to red (high attribution) on a 2d heatmap using matplotlib, one image per method
    for method, result in insights.items():
        attribution = result.attributions.detach().cpu().numpy()
        if attribution.ndim == 4:
            attribution = attribution[0]
        # sum over channels to get a single 2d map
        attribution_map = np.sum(np.abs(attribution), axis=0)
        plt.imshow(attribution_map, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Attribution heatmap for method: {method}")
        save_path = os.path.join(args.output_dir, f"{Path(args.pdb_file).stem}_{method}_heatmap.png")
        plt.savefig(save_path)
        plt.clf()
        print(f"Saved attribution heatmap for {method} to {save_path}")


        



    colored_pdb_paths = save_attribution_colored_pdbs(
        pdb_file=args.pdb_file,
        output_dir=args.output_dir,
        attribution_results=insights,
        num_atoms=sample["num_atoms"],
    )

    print(f"Interpretability analysis completed. Results saved to: {args.output_dir}")
    


    for colored_pdb in colored_pdb_paths:
        script_path = Path.joinpath(colored_pdb.parent, f"{Path(colored_pdb).stem}_{method}.pml")
        write_coloring_script([colored_pdb], script_path, threshold=args.threshold)
        # Open the script in PyMOL using the command line
        if args.open_in_pymol:
            os.system(f"pymol -c {script_path}")
    
    # validate the attribution results by blurring the pixels with an attribution score above the threshold and checking if the model's confidence in the predicted class decreases significantly
    blurred_samples = blur_according_to_attribution_results(
        args.threshold, sample_tensor, insights
    )
    print("Validating attribution results by blurring high-attribution pixels and re-evaluating the model's confidence:")
    print(f"Original predicted class: {predicted_class}, confidence: {percentages[0][predicted_class].item():.2f}%")
    print(f"Original distribution over classes:\n{prob_string}")
    for method, blurred_sample in blurred_samples.items():
        print(f"Method: {method}")
        blurred_logits, blurred_predicted_class = apply_classification(sample_tensor, model, device=device)
        blurred_percentages = torch.nn.functional.softmax(blurred_logits, dim=1) * 100
        print(f"\tBlurred predicted class: {blurred_predicted_class}, confidence: {blurred_percentages[0][blurred_predicted_class].item():.2f}%")
        
        blurred_prob_string = probablity_string(blurred_percentages, class_labels)
        print(blurred_prob_string)


    

    if args.open_in_pymol:
        print("PyMOL should now open with the colored PDB visualizations. If it does not, please check that PyMOL is installed and added to your system's PATH.")
        print("You can also manually open the generated .pml script in PyMOL to visualize the results.")
if __name__ == "__main__":    
    main()