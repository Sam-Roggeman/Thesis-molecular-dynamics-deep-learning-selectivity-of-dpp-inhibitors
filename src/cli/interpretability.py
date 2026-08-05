import argparse
from html import parser
import os
from pathlib import Path
from typing import Callable
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.data_preprocessing.utils import extract_coordinates
from src.model_training.LabelEncoder import LabelEncoder
from src.model_training.batch_preprocessing import _coords_to_rgb, _coords_to_rgb, _coords_to_tensor, prepare_model_batch
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

def apply_transformations_to_sample(sample: dict) -> dict:
    """
    Apply transformations to the preprocessed data sample.
    :param sample: Preprocessed data sample.
    :return: Transformed data as a tensor.
    """
    # the transformations should be the same as those applied during training, except for any random scrambling or augmentations that would make the sample non-deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # grab tensor of coordinates
    coords = torch.tensor(sample["coordinates"], dtype=torch.float32)  # shape [num_atoms, 3]
    coords_tensor = _coords_to_tensor([coords], device) 
    num_atoms_tensor = torch.tensor([sample["num_atoms"]], dtype=torch.int32, device=device)
    coords_tensor = _coords_to_rgb(coords_tensor, num_atoms=num_atoms_tensor)  # convert coordinates to RGB format expected by the model
    label = LabelEncoder().encode_label(sample["binding_type"])
    # remove any batch dimension if present, since we're processing one sample at a time for interpretability
    return coords_tensor[0], label
    

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
                perturbations_per_eval=args.perturbations_per_eval,
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


def _prepare_spatial_attribution_map(attributions: torch.Tensor) -> torch.Tensor:
    """Reduce attribution tensors to a single 2D spatial map [H, W]."""
    if attributions.ndim == 4:
        attributions = attributions[0]
    if attributions.ndim != 3:
        raise ValueError(f"Expected attribution shape [C,H,W] or [1,C,H,W], got {tuple(attributions.shape)}")
    return attributions.detach().cpu().abs().sum(dim=0)


def blur_top_n_pixels(n: int, input_sample: torch.Tensor, attribution_result: AttributionResult) -> torch.Tensor:
    attribution = attribution_result.attributions.detach()

    if attribution.ndim == 4:
        attribution = attribution[0]

    # same processing as your heatmap
    spatial_scores = torch.sum(torch.abs(attribution), dim=0)

    flattened_scores = spatial_scores.reshape(-1)

    k = min(int(n), flattened_scores.numel())

    top_n_indices = torch.topk(flattened_scores, k).indices

    mask = torch.zeros_like(flattened_scores, dtype=torch.bool)
    mask[top_n_indices] = True
    mask = mask.reshape(spatial_scores.shape)

    return input_sample.clone().masked_fill(
        mask.to(input_sample.device).unsqueeze(0),
        0.0
    )


def blur_according_to_attribution_results(
    threshold: float,
    input_sample: torch.Tensor,
    attribution_result: AttributionResult
) -> torch.Tensor:
    """Blur the input sample according to the attribution results.
    :threshold: The attribution score threshold above which pixels will be blurred.
    :input_sample: The original input sample tensor [C,H,W].
    :attribution_result: The attribution result for the input sample.
    """
    attributions = attribution_result.attributions
    if attributions.ndim == 4:
        attributions = attributions[0]
    if attributions.ndim != 3:
        raise ValueError(f"Expected attribution shape [C,H,W] or [1,C,H,W], got {tuple(attributions.shape)}")
    pixel_scores = attributions.detach().cpu().abs().sum(dim=0)
    mask = pixel_scores > threshold
    mask = mask.to(input_sample.device)
    blur_ratio = mask.sum().item() / mask.numel()
    print(f"\tBlur ratio (fraction of pixels above threshold {threshold}): {blur_ratio:.4f}")
    if blur_ratio > 0.5:
        print(f"\tWarning: More than 50% of pixels are above the threshold {threshold}. Consider adjusting the threshold for meaningful validation.")
    blurred_sample = input_sample.clone()
    blurred_sample = blurred_sample.masked_fill(mask.unsqueeze(0), 0.0)
    return blurred_sample


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
    attribution_result: AttributionResult,
    num_atoms: int,
    method: int
) -> list[str]:
    """Create one PDB per attribution method with atom scores stored in B-factor."""
    atom_scores = _attribution_to_atom_scores(attribution_result.attributions, num_atoms)
    atom_scores = _normalize_scores(atom_scores)
    # the most important atoms will have a B-factor of 100, the least important will have a B-factor of 0, and the others will be scaled in between
    # this allows for easy visualization in PyMOL using a spectrum from gray (0) to red (100)
    output_path = os.path.join(output_dir, f"{Path(pdb_file).stem}_{method}_bfactor.pdb")
    _write_bfactor_colored_pdb(pdb_file, output_path, atom_scores)
    print(f"Saved colored PDB for {method}: {output_path}")
    return output_path

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
    parser.add_argument("--methods", nargs="+", default=["integrated_gradients"], help="List of interpretability methods to apply (e.g., 'integrated_gradients', 'saliency', 'occlusion').")
    parser.add_argument("--ig_steps", type=int, default=50, help="Number of steps for Integrated Gradients approximation.")
    parser.add_argument("--occlusion_patch_size", type=int, default=1, help="Patch size for occlusion attribution.")
    parser.add_argument("--occlusion_shift_size", type=int, default=1, help="Shift size for occlusion attribution.")
    parser.add_argument("--binding_type", type=str, default=None, required=True, help="Binding type of the sample. Required for proper sample construction.", choices=["apo", "dpp8selective", "dpp9selective", "aselective", 'nonbinder'])
    parser.add_argument("--open_in_pymol", action="store_false", help="Whether to automatically open the generated colored PDBs in PyMOL after processing.")
    parser.add_argument("--perturbations_per_eval", type=int, default=10)

    parser.add_argument("--blur_top_n", type=int, default=0, help="Number of top attribution pixels to blur for validation. If 0, no blurring based on threshhold is performed.")
    parser.add_argument("--threshold", type=float, default=1.0, help="High attribution threshold as a fraction of the max score for PyMOL visualization (e.g., 0.8 means atoms with scores in the top 20%% will be shown as sticks).")


    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    if not args.pdb_file:
        print("Please provide a PDB file using the --pdb_file argument.")
        return
    if not args.output_dir:
        args.output_dir = str(Path(args.pdb_file).parent / "interpretability_results")
    # blur top n and threshold are mutually exclusive, so we can add a check for that later
    if args.blur_top_n > 0 and args.threshold < 1.0:
        parser.error("--blur_top_n and --threshold are mutually exclusive. Please specify only one of them.")

    return args

def probablity_string(percentages: torch.Tensor, class_labels: list[str]) -> str:
    prob_string = "Model classification probabilities:\n"
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
    blur_based_on_threshold = args.blur_top_n == 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_labels= LabelEncoder().get_classes()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing PDB file: {args.pdb_file}")
    sample = extract_pdb_file(args.pdb_file, binding_type=args.binding_type)

    sample_tensor, true_label = apply_transformations_to_sample(sample)
    print (f"Transformed sample tensor shape: {sample_tensor.shape}, true label: {class_labels[true_label]}")
    # save the transformed sample tensor as an RGB .png file for debugging purposes
    plt.imsave(os.path.join(args.output_dir, f"{Path(args.pdb_file).stem}_transformed_sample.png"), sample_tensor.detach().cpu().permute(1, 2, 0).numpy())
    print(f"Saved transformed sample tensor as image to {os.path.join(args.output_dir, f'{Path(args.pdb_file).stem}_transformed_sample.png')}")
    model = _initilize_classification_model(args.model_checkpoint, device=device)

    logits, predicted_class = apply_classification(sample_tensor, model, device=device)
    percentages = torch.nn.functional.softmax(logits, dim=1) * 100


    prob_string = probablity_string(percentages, class_labels)
    print(prob_string)
    print(prob_string, file=open(os.path.join(args.output_dir, "classification_probabilities.txt"), "w"))
    print(f"Predicted class: {predicted_class}")
    methods = solve_methods(CaptumInterpreter(model), args)
    insights: dict[str, AttributionResult] = generate_interpretability_attribution(methods, predicted_class, sample_tensor, device=device)
    predicted_class_label = class_labels[predicted_class]  # convert from tensor -> int (class label index) -> label name

    # save the insigths as image, where the attributions are mapped from gray (low attribution) to red (high attribution) on a 2d heatmap using matplotlib, one image per method
    for method, attribution_result in insights.items():
        # create subfolder for this method's results        
        method_output_dir = os.path.join(args.output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)
        print(f"Processing attribution results for method: {method}")
        
        attribution = attribution_result.attributions.detach().cpu().numpy()
        if attribution.ndim == 4:
            attribution = attribution[0]
        # sum over channels to get a single 2d map
        attribution_map = np.sum(np.abs(attribution), axis=0)
        plt.imshow(attribution_map, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title(f"Attribution heatmap\n{method}\nModel: {model.__class__.__name__}\n Predicted class: {predicted_class_label}")
        # give title some extra height to avoid overlap with colorbar
        plt.subplots_adjust(top=0.8)
        
        save_path = os.path.join(method_output_dir, f"{Path(args.pdb_file).stem}_{method}_heatmap.png")
        plt.savefig(save_path)
        plt.clf()
        print(f"\tSaved attribution heatmap for {method} to {save_path}")

        # overlay the heatmap on the transformed sample image and save it for visualization
        sample_image = sample_tensor.detach().cpu().permute(1, 2, 0).numpy()
        heatmap = plt.get_cmap("hot")(attribution_map / np.max(attribution_map))[:, :, :3]  # get RGB values from heatmap 
        overlay = (0.6 * sample_image + 0.4 * heatmap).clip(0, 1)
        overlay_save_path = os.path.join(method_output_dir, f"{Path(args.pdb_file).stem}_{method}_overlay.png")
        plt.imsave(overlay_save_path, overlay)
        print(f"Saved attribution overlay for {method} to {overlay_save_path}")

        if blur_based_on_threshold:
            # validate the attribution results by blurring the pixels with an attribution score above the threshold and checking if the model's confidence in the predicted class decreases significantly
            blurred_sample = blur_according_to_attribution_results(args.threshold, sample_tensor, attribution_result)
        else:
            blurred_sample = blur_top_n_pixels(args.blur_top_n, sample_tensor, attribution_result )




        print("\tValidating attribution results by blurring high-attribution pixels and re-evaluating the model's confidence:")
        # save the blurred sample tensor as an RGB .png file for debugging purposes
        plt.imsave(os.path.join(method_output_dir, f"{Path(args.pdb_file).stem}_{method}_blurred_sample.png"), blurred_sample.detach().cpu().permute(1, 2, 0).numpy())
        plt.clf()
        print(f"\tSaved blurred sample tensor for {method} as image to {os.path.join(method_output_dir, f'{Path(args.pdb_file).stem}_{method}_blurred_sample.png')}")
        blurred_logits, blurred_predicted_class = apply_classification(blurred_sample, model, device=device)
        blurred_percentages = torch.nn.functional.softmax(blurred_logits, dim=1) * 100
        blurred_predicted_class_label = class_labels[blurred_predicted_class]
        print(f"\tBlurred predicted class: {blurred_predicted_class_label}, confidence: {blurred_percentages[0][blurred_predicted_class].item():.2f}%")
        
        blurred_prob_string = probablity_string(blurred_percentages, class_labels)
        print(f"\tBlurred distribution over classes:\n{blurred_prob_string}")

        # create a 2x2 plot showing the original sample, the attribution heatmap according to method, the overlayed heatmap, and the blurred sample for side-by-side comparison
        fig, axs = plt.subplots(2, 2, figsize=(10, 12))
        # get suptitle with some extra height to avoid overlap with subplots
        fig.tight_layout(w_pad=0.1, h_pad=4)
        fig.subplots_adjust(top=0.90)
        fig.suptitle(f"Interpretability Analysis for {method}\nModel: {model.__class__.__name__}\nTrue label: {class_labels[true_label]}")

        axs[0, 0].imshow(sample_image)
        axs[0, 0].set_title(f"Original Sample\nPredicted: {predicted_class_label}\nConfidence: {percentages[0][predicted_class].item():.2f}%")
        axs[0, 0].axis("off")
        axs[0, 1].imshow(attribution_map, cmap="hot", interpolation="nearest")
        axs[0, 1].set_title(f"Attribution Heatmap\n{method}")
        axs[0, 1].axis("off")
        axs[1, 0].imshow(overlay)
        axs[1, 0].set_title(f"Overlay Heatmap\n{method}")
        axs[1, 0].axis("off")
        axs[1, 1].imshow(blurred_sample.detach().cpu().permute(1, 2, 0).numpy())
        axs[1, 1].set_title(f"Blurred Sample\nPredicted: {blurred_predicted_class_label}\nConfidence: {blurred_percentages[0][blurred_predicted_class].item():.2f}%")
        axs[1, 1].axis("off")
        comparison_save_path = os.path.join(method_output_dir, f"{Path(args.pdb_file).stem}_{method}_comparison.png")
        # decrease right, left and bottom margins to make the subplots larger and more visible
        
        plt.savefig(comparison_save_path)
        plt.clf()
        print(f"\tSaved comparison plot for {method} to {comparison_save_path}")
    
        colored_pdb_path = save_attribution_colored_pdbs(
            pdb_file=args.pdb_file,
            output_dir=method_output_dir,
            attribution_result=attribution_result,
            num_atoms=sample["num_atoms"],
            method=method
        )
        script_path = Path.joinpath(Path(colored_pdb_path).parent, f"{Path(colored_pdb_path).stem}.pml")
        write_coloring_script([colored_pdb_path], script_path, threshold=args.threshold)
        # Open the script in PyMOL using the command line
        if args.open_in_pymol:
            os.system(f"pymol -c {script_path}")

    print(f"Interpretability analysis completed. Results saved to: {args.output_dir}")
    




    if args.open_in_pymol:
        print("PyMOL should now open with the colored PDB visualizations. If it does not, please check that PyMOL is installed and added to your system's PATH.")
        print("You can also manually open the generated .pml script in PyMOL to visualize the results.")
if __name__ == "__main__":    
    main()