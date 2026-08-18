import argparse
from html import parser
import os
from pathlib import Path
from typing import Callable
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.data_preprocessing.utils import extract_pdb_file, extract_pdb_files_from_directory
from src.model_training.LabelEncoder import LabelEncoder
from src.model_training.batch_preprocessing import _coords_to_rgb, _coords_to_tensor, prepare_model_batch
from src.utils.interpretability import AttributionResult, CaptumInterpreter
from src.utils.resolvers import (
    _extract_class_name,
    _load_config_from_artifacts,
    _load_state_dict,
    _load_weights,
    _resolve_model_class,
)
import time




def apply_transformations_to_samples(samples: list) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply transformations to the preprocessed data samples.
    :param samples: List of preprocessed data samples.
    :return: Transformed data as a tensor [B, C, H, W] and the corresponding true labels as a tensor [B].
    """
    # the transformations should be the same as those applied during training, except for any random scrambling or augmentations that would make the sample non-deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Start with an empty tensor to hold the coordinates of all samples
    coords_list = []
    labels_list = []
    for sample in samples:
        # grab tensor of coordinates
        coords = torch.tensor(sample["coordinates"], dtype=torch.float32)  # shape [num_atoms, 3]
        coords_tensor = _coords_to_tensor([coords], device) 
        num_atoms_tensor = torch.tensor([sample["num_atoms"]], dtype=torch.int32, device=device)
        coords_tensor = _coords_to_rgb(coords_tensor, num_atoms=num_atoms_tensor)  # convert coordinates to RGB format expected by the model
        label = LabelEncoder().encode_label(sample["binding_type"])

        coords_list.append(coords_tensor[0])
        labels_list.append(label)
    # Stack the list of tensors into a single tensor for batch processing
    coords_tensor = torch.stack(coords_list, dim=0)  # shape [B, C, H, W]
    label = torch.tensor(labels_list, dtype=torch.long, device=device)  # shape [B]
    return coords_tensor, label
    
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

def solve_methods(interpreter: CaptumInterpreter, method_args) -> dict[str, Callable]:
    """
    Resolve a method string to the corresponding interpretability method.
    :param method_str: String identifier for the interpretability method (e.g., "integrated_gradients").
    :return: Corresponding interpretability method object.
    """
    methods = {}
    for method, args in method_args.items():
        if method == "integrated_gradients":
            n_steps = args["steps"]
            # keep argparse-driven parameters configurable per run
            methods[method] = lambda inputs, target: interpreter.integrated_gradients(
                inputs,
                target=target,
                n_steps=n_steps,
            )
        elif method == "saliency":
            methods[method] = lambda inputs, target: interpreter.saliency(inputs, target=target)
        elif method == "occlusion":
            patch_size = args["patch_size"]
            shift_size = args["shift_size"]
            perturbations_per_eval = args["perturbations_per_eval"]
            methods[method] = lambda inputs, target: interpreter.occlusion(
                inputs,
                target=target,
                patch_size=patch_size,
                shift_size=shift_size,
                perturbations_per_eval=perturbations_per_eval,
                
            )
        else:
            print(f"Unknown interpretability method '{method}'. Supported methods: 'integrated_gradients', 'saliency', 'occlusion'. Skipping.")
    if len(methods) == 0:
        methods["integrated_gradients"] = lambda inputs, target: interpreter.integrated_gradients(
            inputs,
            target=target,
            n_steps=method_args.ig_steps,
        )
    return methods


def _normalize_spatial_scores(spatial_scores: torch.Tensor) -> torch.Tensor:
    """Normalize spatial scores to the range [0, 1] so thresholds are comparable across methods."""
    if spatial_scores.numel() == 0:
        return spatial_scores
    max_score = spatial_scores.max()
    if not torch.isfinite(max_score) or max_score <= 0:
        return torch.zeros_like(spatial_scores)
    return spatial_scores / max_score


def remove_batch_dimension(attributions: torch.Tensor) -> torch.Tensor:
    """Remove the batch dimension from attributions if present."""
    if attributions.ndim == 4:
        return attributions[0]
    elif attributions.ndim == 3:
        return attributions
    else:
        raise ValueError(f"Expected attribution shape [C,H,W] or [1,C,H,W], got {tuple(attributions.shape)}")

def calculate_spacial_scores(attribution: torch.Tensor) -> torch.Tensor:
    """Calculate spatial scores from the attribution result.
    :attribution: The attribution tensor for the input sample.
    :return: Normalized spatial scores tensor [H,W].
    """
    spatial_scores = _normalize_spatial_scores(attribution.detach().float().abs().sum(dim=0))
    return spatial_scores

def apply_mask_to_input(input_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a binary mask to the input sample, setting masked pixels to zero.
    Args:
        input_sample (torch.Tensor): The input sample tensor [C,H,W] or [1,C,H,W].
        mask (torch.Tensor): A binary mask tensor [H,W] where True indicates pixels to be masked.

    Returns:
        torch.Tensor: The masked input sample tensor [C,H,W].
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape [H,W], got {tuple(mask.shape)}")
    if input_sample.ndim == 4:
        input_sample = input_sample[0]
    elif input_sample.ndim != 3:
        raise ValueError(f"Expected input sample shape [C,H,W] or [1,C,H,W], got {tuple(input_sample.shape)}")

    return input_sample.clone().masked_fill(mask.to(input_sample.device).unsqueeze(0), 0.0)

def blur_top_n_pixels(n: int, input_sample: torch.Tensor, attribution: torch.Tensor) -> torch.Tensor:
    """
    Blur the top n pixels in the input sample based on the attribution scores.
    Args:
        n (int): Number of top pixels to blur.
        input_sample (torch.Tensor): The input sample tensor [C,H,W] or [1,C,H,W].
        attribution (torch.Tensor): The attribution tensor for the input sample.

    Returns:
        torch.Tensor: The blurred input sample tensor [C,H,W].
    """
    spatial_scores = calculate_spacial_scores(attribution)
    flattened_scores = spatial_scores.reshape(-1)

    k = min(int(n), flattened_scores.numel())
    if k <= 0:
        return input_sample.clone()

    top_n_indices = torch.topk(flattened_scores, k).indices

    mask = torch.zeros_like(flattened_scores, dtype=torch.bool)
    mask[top_n_indices] = True
    mask = mask.reshape(spatial_scores.shape)
    print(f"\tBlurring top {k} pixels based on normalized attribution scores (actual blurred pixels: {mask.sum().item()})")
    return apply_mask_to_input(input_sample, mask)


def blur_according_to_attribution_results(
    threshold: float,
    input_sample: torch.Tensor,
    attribution: torch.Tensor
) -> torch.Tensor:
    """Blur the input sample according to the attribution results.
    :threshold: The attribution score threshold above which pixels will be blurred.
    :input_sample: The original input sample tensor [C,H,W].
    :attribution: The attribution tensor for the input sample.
    """
    spatial_scores = calculate_spacial_scores(attribution)
    mask = spatial_scores > threshold
    mask = mask.to(input_sample.device)

    blur_ratio = mask.sum().item() / mask.numel()
    print(f"\tBlur ratio (fraction of pixels above threshold {threshold}): {blur_ratio:.4f}")
    if blur_ratio > 0.5:
        print(f"\tWarning: More than 50% of pixels are above the threshold {threshold}. Consider adjusting the threshold for meaningful validation.")
    blurred_sample = input_sample.clone()
    blurred_sample = apply_mask_to_input(blurred_sample, mask)
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
    attribution: torch.Tensor,
    num_atoms: int,
    method: int
) -> list[str]:
    """Create one PDB per attribution method with atom scores stored in B-factor."""
    atom_scores = _attribution_to_atom_scores(attribution, num_atoms)
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
    parser.add_argument("--pdb_file", type=str, help="Path to a PDB file for generating interpretability insights")
    parser.add_argument("--pdb_directory", type=str, default=None, help="Path to a directory containing multiple PDB files for batch interpretability analysis. If provided, the tool will process all PDB files in the directory.")
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
    
    if not args.pdb_file and not args.pdb_directory:
        print("Please provide either a single PDB file using the --pdb_file argument or a directory of PDB files using the --pdb_directory argument.")
        return
    if args.pdb_file and args.pdb_directory:
        print("Please provide either a single PDB file using the --pdb_file argument or a directory of PDB files using the --pdb_directory argument, but not both.")
        return
    if not args.output_dir:
        if args.pdb_file:
            args.output_dir = str(Path(args.pdb_file).parent / "interpretability_results")
        else:
            args.output_dir = str(Path(args.pdb_directory).parent / "interpretability_results")
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

def execute_interpretability(pdb_file=None, pdb_directory=None, output_dir=None,binding_type=None, blur_top_n=None, model_checkpoint=None, method_args=None, threshold=None):
    """
    CLI entry point for interpretability tools.
    :param pdb_file: path to a PDB file for generating interpretability insights. If provided, the tool will process the file, feed it into the interpretability model, and output the insights. 
    :param output_dir: Directory where interpretability results will be saved. Defaults to "./interpretability_results".
    """
    start_time = time.time()

    # determine whether to blur based on threshold or top n pixels
    blur_based_on_threshold = True 
    if blur_top_n is not None and blur_top_n != 0:
        blur_based_on_threshold = False 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_labels= LabelEncoder().get_classes()
    # load the trained classification model from the provided checkpoint    

    model = _initilize_classification_model(model_checkpoint, device=device)
    methods = solve_methods(CaptumInterpreter(model), method_args=method_args)
    delta_time = time.time() - start_time
    start_time = time.time()
    print(f"loading methods and models: {delta_time}")
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # process the provided PDB file and generate interpretability insights
    # if args.pdb_directory is provided, we can extend this to process all PDB files in the directory in a loop
    if pdb_file:
        print(f"Processing PDB file: {pdb_file}")
        samples = [extract_pdb_file(pdb_file, binding_type=binding_type)]
    elif pdb_directory:
        print(f"Processing PDB files in directory: {pdb_directory}")
        samples = extract_pdb_files_from_directory(pdb_directory, binding_type=binding_type)

    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    samples_tensor, true_labels = apply_transformations_to_samples(samples)
    delta_time = time.time() - start_time
    start_time = time.time()
    print(f"transforming samples: {delta_time}")
    insights = {method: {} for method in methods.keys()}
    print(f"Transformed sample tensor shape: {samples_tensor.shape}, true label: {class_labels[true_labels[0]]}")
    for i, sample_tensor in enumerate(samples_tensor):
        sample = samples[i]
        true_label = true_labels[i].item()
        sample_prefix = sample['pdb_id']
        print(f"\nProcessing sample {i+1}/{len(samples_tensor)}: PDB ID: {sample['pdb_id']}, True label: {class_labels[true_label]}")
        # Save transformed sample tensor as RGB .png for debugging
        image_path = os.path.join(image_dir, f"{sample_prefix}_transformed_sample.png")
        plt.imsave(image_path, sample_tensor.detach().cpu().permute(1, 2, 0).numpy())
        print(f"Saved transformed sample tensor as image to {image_path}")

        logits, predicted_class = apply_classification(sample_tensor, model, device=device)
        percentages = torch.nn.functional.softmax(logits, dim=1) * 100
        prob_string = sample_prefix + ": " + probablity_string(percentages, class_labels)
        print(prob_string)
        print(prob_string, file=open(os.path.join(output_dir, "classification_probabilities.txt"), "a"))
        print(f"Predicted class: {predicted_class}")
        sample_insights: dict[str, AttributionResult] = generate_interpretability_attribution(methods, predicted_class, sample_tensor, device=device)
        predicted_class_label = class_labels[predicted_class]  # convert from tensor -> int (class label index) -> label name

        for method, attribution_result in sample_insights.items():
            insights[method][sample["pdb_id"]] = attribution_result
        delta_time = time.time() - start_time
        start_time = time.time()
        print(f"generating attribution for one frame: {delta_time}")
    delta_time = time.time() - start_time
    start_time = time.time()
    print(f"generating attribution: {delta_time}")
    # insights = {method -> {pdb_id -> AttributionResult}}
    for method, insight in insights.items():
        # take the average of the attribution scores across all samples for this method
        avg_attribution = 0
        for pdb_id, attribution_result in insight.items():
            avg_attribution += attribution_result.attributions.detach().cpu()
        avg_attribution /= len(insight)
        insights[method]["average"] = avg_attribution
    # save the insigths as image, where the attributions are mapped from gray (low attribution) to red (high attribution) on a 2d heatmap using matplotlib, one image per method
    for method, insight in insights.items():
        blurred_correct_predictions = 0
        # create subfolder for this method's results        
        print(f"Processing attribution results for method: {method}")
        for pdb_id, attribution_result in insight.items():
            if pdb_id == "average":
                print(f"\tProcessing average attribution result for method: {method}")
                attribution = avg_attribution
            else:
                attribution = attribution_result.attributions.detach()
                
            if attribution.ndim == 4:
                attribution = attribution[0]
            spatial_scores = _normalize_spatial_scores(torch.sum(torch.abs(attribution), dim=0))       
            plt.imshow(spatial_scores.cpu(), cmap="hot", interpolation="nearest")
            plt.colorbar()
            plt.title(f"Attribution heatmap\n{method}\nModel: {model.__class__.__name__}\n Predicted class: {predicted_class_label}")
            # give title some extra height to avoid overlap with colorbar
            plt.subplots_adjust(top=0.8)
            
            save_path = os.path.join(image_dir, f"{pdb_id}_{method}_heatmap.png")
            plt.savefig(save_path)
            plt.clf()
            print(f"\tSaved attribution heatmap for {method} to {save_path}")

            # overlay the heatmap on the transformed sample image and save it for visualization
            sample_image = sample_tensor.detach().cpu().permute(1, 2, 0).numpy()
            heatmap = plt.get_cmap("hot")(spatial_scores.cpu())[:, :, :3]  # get RGB values from heatmap using the normalized map
            overlay = (0.6 * sample_image + 0.4 * heatmap).clip(0, 1)
            overlay_save_path = os.path.join(image_dir, f"{pdb_id}_{method}_overlay.png")
            plt.imsave(overlay_save_path, overlay)
            print(f"Saved attribution overlay for {method} to {overlay_save_path}")

            if blur_based_on_threshold:
                # validate the attribution results by blurring the pixels with an attribution score above the threshold and checking if the model's confidence in the predicted class decreases significantly
                blurred_sample = blur_according_to_attribution_results(threshold, sample_tensor, attribution)
            else:
                blurred_sample = blur_top_n_pixels(blur_top_n, sample_tensor, attribution )




            print("\tValidating attribution results by blurring high-attribution pixels and re-evaluating the model's confidence:")
            # save the blurred sample tensor as an RGB .png file for debugging purposes
            plt.imsave(os.path.join(image_dir, f"{pdb_id}_{method}_blurred_sample.png"), blurred_sample.detach().cpu().permute(1, 2, 0).numpy())
            plt.clf()
            print(f"\tSaved blurred sample tensor for {method} as image to {os.path.join(image_dir, f'{pdb_id}_{method}_blurred_sample.png')}")
            blurred_logits, blurred_predicted_class = apply_classification(blurred_sample, model, device=device)
            blurred_percentages = torch.nn.functional.softmax(blurred_logits, dim=1) * 100
            blurred_predicted_class_label = class_labels[blurred_predicted_class]
            if blurred_predicted_class == predicted_class:
                blurred_correct_predictions += 1
            
            print(f"\tBlurred predicted class: {blurred_predicted_class_label}, confidence: {blurred_percentages[0][blurred_predicted_class].item():.2f}%")
            
            blurred_prob_string = sample_prefix + ": " + probablity_string(blurred_percentages, class_labels)
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
            axs[0, 1].imshow(spatial_scores.cpu(), cmap="hot", interpolation="nearest")
            axs[0, 1].set_title(f"Attribution Heatmap\n{method}")
            axs[0, 1].axis("off")
            axs[1, 0].imshow(overlay)
            axs[1, 0].set_title(f"Overlay Heatmap\n{method}")
            axs[1, 0].axis("off")
            axs[1, 1].imshow(blurred_sample.detach().cpu().permute(1, 2, 0).numpy())
            axs[1, 1].set_title(f"Blurred Sample\nPredicted: {blurred_predicted_class_label}\nConfidence: {blurred_percentages[0][blurred_predicted_class].item():.2f}%")
            axs[1, 1].axis("off")
            comparison_save_path = os.path.join(image_dir, f"{pdb_id}_{method}_comparison.png")
            # decrease right, left and bottom margins to make the subplots larger and more visible
            
            plt.savefig(comparison_save_path)
            plt.clf()
            print(f"\tSaved comparison plot for {method} to {comparison_save_path}")
            pdb_filepath = sample["pdb_id"]
            if pdb_directory:
                pdb_filepath = os.path.join(pdb_directory, f"{sample['pdb_id']}.pdb")
            colored_pdb_path = save_attribution_colored_pdbs(
                pdb_file=pdb_filepath,
                output_dir=image_dir,
                attribution=attribution,
                num_atoms=sample["num_atoms"],
                method=method
            )
            script_path = Path.joinpath(Path(colored_pdb_path).parent, f"{Path(colored_pdb_path).stem}.pml")
            write_coloring_script([colored_pdb_path], script_path, threshold=threshold)
        blur_accuracy = blurred_correct_predictions / len(insight) * 100
        print(f"\nBlur validation for method '{method}': {blurred_correct_predictions}/{len(insight)} samples ({blur_accuracy:.2f}%) retained the same predicted class after blurring high-attribution pixels.")
    delta_time = time.time() - start_time
    start_time = time.time()
    print(f"processing insights: {delta_time}")
    print(f"Interpretability analysis completed. Results saved to: {output_dir}")
    torch.cuda.empty_cache()

def main():
    # parse command-line arguments
    args = arg_parser()
    method_args = {
        "integrated_gradients":{"steps": args.ig_steps},
        "occlusion": {
            "patch_size": args.occlusion_patch_size,
            "perturbations_per_eval": args.perturbations_per_eval,
            "shift_size": args.occlusion_shift_size
        }, 
        "saliency": {}
    }

    execute_interpretability(
        pdb_file=args.pdb_file, 
        pdb_directory=args.pdb_directory, 
        output_dir= args.output_dir,
        binding_type= args.binding_type, 
        blur_top_n=args.blur_top_n,
        model_checkpoint=args.model_checkpoint, 
        method_args=method_args, 
        threshold=args.threshold
    )

if __name__ == "__main__":    
    main()