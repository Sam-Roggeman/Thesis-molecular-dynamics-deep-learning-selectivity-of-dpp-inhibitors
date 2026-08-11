import argparse
import ast
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv

from src.Models.DCNN import DCNN
from src.Models.LongSequenceAtomTransformer import LongSequenceAtomTransformer
from src.Models.OneLayer import OneLayerNet
from src.Models.SCNN import SCNN
from src.data_loading.HFDataloader import initialize_dataloaders, initialize_streaming_dataloader
from src.model_training.LabelEncoder import LabelEncoder
from src.utils.interpretability import CaptumInterpreter
from src.utils.training_config import TrainingConfig


# Registry to resolve a serialized model name back to an actual class.
MODEL_REGISTRY = {
    "SCNN": SCNN,
    "OneLayerNet": OneLayerNet,
    "DCNN": DCNN,
    "LongSequenceAtomTransformer": LongSequenceAtomTransformer,
}


def _extract_class_name(class_value) -> str | None:
    """Extract a clean class name from object, string, or repr-like value."""
    if class_value is None:
        return None
    if isinstance(class_value, type):
        return class_value.__name__
    if isinstance(class_value, str):
        class_pattern = r"<class '.*\.([^\.']+)'>"
        match = re.search(class_pattern, class_value)
        if match:
            return match.group(1)
        return class_value.split(".")[-1].strip()
    return None


def _load_config_from_artifacts(config_path: str | None, checkpoint_path: str) -> TrainingConfig | None:
    """Load TrainingConfig from explicit path or common files next to a checkpoint."""
    candidate_paths = []
    if config_path:
        candidate_paths.append(config_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)
    candidate_paths.append(os.path.join(checkpoint_dir, "training_config.pt"))
    candidate_paths.append(os.path.join(checkpoint_dir, "training_config.json"))

    for candidate in candidate_paths:
        if not os.path.exists(candidate):
            continue

        if candidate.endswith(".pt"):
            return TrainingConfig.load(candidate)

        # JSON payloads can contain stringified class/dict values.
        with open(candidate, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["model_class"] = _extract_class_name(payload.get("model_class"))

        model_args = payload.get("model_args")
        if isinstance(model_args, str):
            try:
                payload["model_args"] = ast.literal_eval(model_args)
            except (SyntaxError, ValueError):
                payload["model_args"] = {}

        return TrainingConfig(**payload)

    return None


def _resolve_model_class(config: TrainingConfig):
    """Resolve model class stored in config to a callable class object."""
    if isinstance(config.model_class, type):
        return config.model_class

    config_model_name = _extract_class_name(config.model_class)
    if config_model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[config_model_name]

    raise ValueError(
        "Could not resolve model class from training config. "
        f"Supported classes are: {list(MODEL_REGISTRY.keys())}"
    )


def _load_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    """Load checkpoint and return a plain state dict expected by model.load_state_dict."""
    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
        raw_checkpoint = raw_checkpoint["state_dict"]
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("Checkpoint format is unsupported. Expected a state_dict or a dict containing state_dict.")
    return raw_checkpoint


def _load_weights(model: torch.nn.Module, state_dict: dict) -> None:
    """Load model weights, including checkpoints saved from torch.compile wrappers."""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    compiled_prefix = "_orig_mod."
    stripped = {
        (k[len(compiled_prefix):] if k.startswith(compiled_prefix) else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(stripped)


def _to_display_image(sample: torch.Tensor) -> torch.Tensor:
    """Convert tensor to normalized HWC image format suitable for matplotlib."""
    if sample.ndim == 1:
        edge = int((sample.numel() / 3) ** 0.5)
        sample = sample.view(3, edge, edge)

    image = sample.detach().cpu().float()
    if image.ndim != 3:
        raise ValueError(f"Expected 3D sample tensor [C,H,W], got shape {tuple(image.shape)}")

    image = image.permute(1, 2, 0)
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min + 1e-12)


def _save_heatmaps(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    attribution_map: torch.Tensor,
    class_names: list[str],
    method: str,
    output_dir: str,
    start_index: int = 0,
    dpp_labels: list[str | None] | None = None,
) -> None:
    """Save per-sample visualizations: input, heatmap, and overlay."""
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(inputs.size(0)):
        image = _to_display_image(inputs[idx])
        heatmap = attribution_map[idx].detach().cpu().float()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image.numpy())
        axes[0].set_title("Input")
        axes[0].axis("off")

        hm = axes[1].imshow(heatmap.numpy(), cmap="hot")
        axes[1].set_title(f"{method} Heatmap")
        axes[1].axis("off")
        fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(image.numpy())
        axes[2].imshow(heatmap.numpy(), cmap="hot", alpha=0.45)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        true_idx = int(labels[idx].item())
        pred_idx = int(preds[idx].item())
        true_name = class_names[true_idx] if true_idx < len(class_names) else str(true_idx)
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        fig.suptitle(f"Method={method} | true={true_name} | pred={pred_name} | DPP={dpp_labels[idx] if dpp_labels is not None else None}", fontsize=10)

        filename = os.path.join(output_dir, f"{method}_sample_{start_index + idx:05d}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=160)
        plt.close(fig)


def _save_average_heatmap(heatmap: torch.Tensor, title: str, output_path: str) -> None:
    """Save one aggregated heatmap image (global or per-class average)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    hm = ax.imshow(heatmap.numpy(), cmap="hot")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def _resolve_device(selection: str) -> torch.device:
    """Resolve device selection from CLI and validate CUDA availability."""
    if selection == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selection == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--attribution-device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _normalize_dpp_label(raw_value) -> str | None:
    """Map dpp_class values to canonical labels ('dpp8' or 'dpp9')."""
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if "dpp8" in text or text == "8":
        return "dpp8"
    if "dpp9" in text or text == "9":
        return "dpp9"
    return None


def _prepare_inputs_for_captum(sample: torch.Tensor) -> torch.Tensor:
    """Ensure sample tensor is in the right shape and on the right device for Captum."""
    if sample.ndim == 1:
        edge = int((sample.numel() / 3) ** 0.5)
        return sample.view(3, edge, edge)
    if sample.ndim == 3:
        return sample
    raise ValueError(f"Unsupported sample shape {tuple(sample.shape)}. Expected [C,H,W] or flat [features].")


def apply_method(method, interpreter: CaptumInterpreter, inputs, target, *args):
    # target controls which class score gradients are taken from.
    # None means Captum uses its default behavior (often predicted class per sample).
    if method == "integrated_gradients":
        # Integrated Gradients compares predictions along a path from baseline -> input.
        result = interpreter.integrated_gradients(
            inputs,
            target=target,
            baselines=torch.zeros_like(inputs),
            n_steps=args.n_steps,
            internal_batch_size=args.ig_batch_size,
        )
    elif method == "saliency":
        # Saliency uses the raw input gradient magnitude as importance.
        result = interpreter.saliency(inputs, target=target)
    elif method == "guided_backprop":
        # Guided Backprop modifies backward ReLU flow for sharper maps.
        result = interpreter.guided_backprop(inputs, target=target)
    else:
        raise ValueError(f"Unsupported method requested: {method}")
    return result

def main():
    """Run Captum methods on a checkpoint and export attribution visualizations."""
    # 1) CLI options controlling methods, sample limits, and output behavior.
    parser = argparse.ArgumentParser(description="Run Captum interpretability on a trained model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth checkpoint")
    parser.add_argument("--config", default=None, help="Optional training config path (.pt or .json)")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["integrated_gradients", "saliency"],
        choices=["integrated_gradients", "saliency", "guided_backprop"],
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,   
        help="Optional override for config batch size to reduce memory usage during Captum runs",
    )
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Number of dataloader batches to process for attribution aggregation",
    )
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=None,
        help="Optional cap on total samples processed across all batches",
    )
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument(
        "--ig-batch-size",
        type=int,
        default=1,
        help="Internal batch size for Integrated Gradients (smaller uses less memory)",
    )
    parser.add_argument(
        "--attribution-device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device used for attribution computations",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help=(
            "Optional target class index for attributions. "
            "If omitted, attributions are computed with Captum's default target behavior."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Output folder for heatmaps, defaults to checkpoint_dir/interpretability/<timestamp>/")
    parser.add_argument(
        "--no-individual-plots",
        action="store_true",
        help="Disable per-sample plots and only save aggregated hotspot maps",
    )
    # add a streaming flag to force using streaming dataloader, default False
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Force using streaming dataloader. By default, the script tries to initialize the regular dataloader and falls back to streaming if it fails. Use this flag to skip the \
         regular dataloader initialization entirely.",
    )
    args = parser.parse_args()

    # Load optional env vars (for datasets, tokens, cache locations, etc.).
    load_dotenv()
    # 2) Restore training configuration and checkpoint.
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.output_dir:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(checkpoint_dir, "interpretability", run_name)
    config = _load_config_from_artifacts(args.config, args.checkpoint)
    if config is None:
        raise ValueError(
            "No training config found. Provide --config or place training_config.pt/json next to checkpoint."
        )
    # reset num_cpus in config based on current environment to avoid issues with dataloader workers in Captum runs
    config.reset_cpus()
    model_class = _resolve_model_class(config)
    model_args = dict(config.model_args) if isinstance(config.model_args, dict) else {}

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Build dataloader for only the split requested by the user.
    if args.streaming:
        print("Initializing streaming dataloader...")
        dataloaders = initialize_streaming_dataloader(config, splits=[args.split], keep_all_columns=True, shuffle=True)
    else:
        print("Initializing regular dataloader...")
        dataloaders = initialize_dataloaders(config, splits=[args.split], keep_all_columns=True)
    data_iter = iter(dataloaders[args.split])

    # Initialize model, then load learned weights.
    device = _resolve_device(args.attribution_device)
    model = model_class(**model_args).to(device)
    state_dict = _load_state_dict(args.checkpoint, device)
    _load_weights(model, state_dict)
    model.eval()

    interpreter = CaptumInterpreter(model)
    class_names = LabelEncoder().get_classes()
    nr_classes = len(class_names)

    # 3) Create output directory for this run.
    base_output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Fixed class index to explain (e.g. force class 0 across all samples).
    # Leave as None to explain each sample using Captum's default target behavior.
    target = args.target if args.target is not None else None

    # Running sums used to compute average hotspot maps later.
    global_sum: dict[str, torch.Tensor] = {}
    global_count: dict[str, int] = {method: 0 for method in args.methods}
    class_sum: dict[str, list[torch.Tensor | None]] = {
        method: [None for _ in range(nr_classes)] for method in args.methods
    }
    class_count: dict[str, list[int]] = {
        method: [0 for _ in range(nr_classes)] for method in args.methods
    }
    dpp_class_sum: dict[str, dict[str, list[torch.Tensor | None]]] = {
        method: {
            "dpp8": [None for _ in range(nr_classes)],
            "dpp9": [None for _ in range(nr_classes)],
            "combined": [None for _ in range(nr_classes)],
        }
        for method in args.methods
    }
    dpp_class_count: dict[str, dict[str, list[int]]] = {
        method: {
            "dpp8": [0 for _ in range(nr_classes)],
            "dpp9": [0 for _ in range(nr_classes)],
            "combined": [0 for _ in range(nr_classes)],
        }
        for method in args.methods
    }

    processed_samples = 0
    processed_batches = 0

    # 4) Process one or more batches and compute attributions.
    for _ in range(args.max_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        inputs = batch["data"]
        labels = batch["labels"]
        dpp_labels = batch.get("dpp_class")  # Keep raw DPP labels for per-class aggregation and display, if available.
        dpp_raw = batch.get("dpp_class")

        if args.max_samples is not None:
            inputs = inputs[: args.max_samples]
            labels = labels[: args.max_samples]
            dpp_raw = dpp_raw[: args.max_samples]

        if args.max_total_samples is not None:
            remaining = args.max_total_samples - processed_samples
            if remaining <= 0:
                break
            inputs = inputs[:remaining]
            labels = labels[:remaining]
            dpp_raw = dpp_raw[:remaining]

        if inputs.shape[0] == 0:
            break

        if dpp_raw is None:
            raise ValueError(
                "Batch is missing 'dpp_class'. Ensure dataloader keeps this column for DPP8/DPP9 averaging."
            )

        dpp_labels = [_normalize_dpp_label(value) for value in dpp_raw]

        # Captum computes gradients w.r.t. the input. This must be True for attribution methods.
        inputs = inputs.to(device).clone().detach().requires_grad_(True)
        labels = labels.to(device)

        # We only need predicted labels for reporting/plot titles here,
        # so disable gradient tracking for this forward pass.
        with torch.no_grad():
            # Logits are raw class scores before softmax (classification): shape [batch, num_classes].
            logits = model(inputs)
            # Select the class index with the highest score for each sample.
            preds = torch.argmax(logits, dim=1)

        # Run each selected attribution method on the same batch.
        for method in args.methods:
            result = apply_method(method, interpreter, inputs, target, args)
            # result.attributions is the per-input-feature importance tensor.
            # Typical shape is [batch, channels, height, width] for image-like inputs.
            heatmaps = CaptumInterpreter.summarize_attributions(
                result.attributions,
                reduce_dim=1,
                normalize=False,
            )
            # Keep non-normalized maps for averaging so relative strengths are preserved.
            heatmaps_cpu = heatmaps.detach().cpu()
            labels_cpu = labels.detach().cpu()

            # Aggregate across all samples for a global mean heatmap.
            if method not in global_sum:
                global_sum[method] = heatmaps_cpu.sum(dim=0)
            else:
                global_sum[method] += heatmaps_cpu.sum(dim=0)
            global_count[method] += int(heatmaps_cpu.shape[0])

            # Aggregate separately per class to inspect class-specific focus regions.
            for class_idx in range(nr_classes):
                class_mask = labels_cpu == class_idx
                class_nr = int(class_mask.sum().item())
                if class_nr == 0:
                    continue
                class_map_sum = heatmaps_cpu[class_mask].sum(dim=0)
                if class_sum[method][class_idx] is None:
                    class_sum[method][class_idx] = class_map_sum
                else:
                    class_sum[method][class_idx] += class_map_sum
                class_count[method][class_idx] += class_nr

                # Keep three per-class views: dpp8, dpp9, and both combined.
                for dpp_key in ["dpp8", "dpp9", "combined"]:
                    if dpp_key == "combined":
                        dpp_mask = torch.tensor([dl in {"dpp8", "dpp9"} for dl in dpp_labels], dtype=torch.bool)
                    else:
                        dpp_mask = torch.tensor([dl == dpp_key for dl in dpp_labels], dtype=torch.bool)
                    combined_mask = class_mask & dpp_mask
                    subset_nr = int(combined_mask.sum().item())
                    if subset_nr == 0:
                        continue
                    subset_sum = heatmaps_cpu[combined_mask].sum(dim=0)
                    if dpp_class_sum[method][dpp_key][class_idx] is None:
                        dpp_class_sum[method][dpp_key][class_idx] = subset_sum
                    else:
                        dpp_class_sum[method][dpp_key][class_idx] += subset_sum
                    dpp_class_count[method][dpp_key][class_idx] += subset_nr

            method_dir = os.path.join(base_output_dir, method)
            if not args.no_individual_plots:
                # Normalize for display only (better contrast in saved figures).
                display_maps = CaptumInterpreter.summarize_attributions(
                    result.attributions,
                    reduce_dim=1,
                    normalize=True,
                )
                _save_heatmaps(
                    inputs=inputs.detach().cpu(),
                    labels=labels_cpu,
                    preds=preds.detach().cpu(),
                    attribution_map=display_maps.detach().cpu(),
                    class_names=class_names,
                    method=method,
                    output_dir=method_dir,
                    start_index=processed_samples,
                    dpp_labels = dpp_labels
                )

            if result.convergence_delta is not None:
                # Captum may return a convergence delta (mainly for Integrated Gradients);
                # smaller values generally indicate better numerical approximation.
                delta_file = os.path.join(method_dir, f"convergence_delta_batch_{processed_batches:03d}.pt")
                torch.save(result.convergence_delta.detach().cpu(), delta_file)

            if device.type == "cuda":
                # Release cached blocks between methods to reduce peak memory pressure.
                torch.cuda.empty_cache()

        processed_samples += int(inputs.shape[0])
        processed_batches += 1

        if args.max_total_samples is not None and processed_samples >= args.max_total_samples:
            break

    # 5) Finalize averages and save summary plots.
    averages_dir = os.path.join(base_output_dir, "averages")
    for method in args.methods:
        if global_count[method] > 0:
            global_avg = global_sum[method] / global_count[method]
            global_avg_norm = _to_display_image(global_avg.unsqueeze(0).repeat(3, 1, 1))[:, :, 0]
            _save_average_heatmap(
                heatmap=global_avg_norm,
                title=f"Global average hotspot ({method})",
                output_path=os.path.join(averages_dir, f"global_{method}.png"),
            )

        for class_idx, class_name in enumerate(class_names):
            if class_count[method][class_idx] == 0:
                continue
            cls_sum = class_sum[method][class_idx]
            if cls_sum is None:
                continue
            cls_avg = cls_sum / class_count[method][class_idx]
            cls_avg_norm = _to_display_image(cls_avg.unsqueeze(0).repeat(3, 1, 1))[:, :, 0]
            safe_name = class_name.replace(" ", "_")
            _save_average_heatmap(
                heatmap=cls_avg_norm,
                title=f"Class average hotspot ({method}) - {class_name}",
                output_path=os.path.join(averages_dir, "per_class", method, f"{safe_name}.png"),
            )

            # Save in requested order: dpp8, dpp9, then combined.
            class_dir = os.path.join(averages_dir, "per_class", method, safe_name)
            for dpp_key, filename in [
                ("dpp8", "dpp8_heatmap.png"),
                ("dpp9", "dpp9_heatmap.png"),
                ("combined", "combined_heatmap.png"),
            ]:
                dpp_nr = dpp_class_count[method][dpp_key][class_idx]
                dpp_sum = dpp_class_sum[method][dpp_key][class_idx]
                if dpp_nr == 0 or dpp_sum is None:
                    continue
                dpp_avg = dpp_sum / dpp_nr
                dpp_avg_norm = _to_display_image(dpp_avg.unsqueeze(0).repeat(3, 1, 1))[:, :, 0]
                _save_average_heatmap(
                    heatmap=dpp_avg_norm,
                    title=f"Class average hotspot ({method}) - {class_name} - {dpp_key}",
                    output_path=os.path.join(class_dir, filename),
                )

    # Save run metadata for reproducibility.
    run_info_path = os.path.join(base_output_dir, "run_info.txt")
    with open(run_info_path, "w", encoding="utf-8") as info:
        info.write(f"checkpoint={args.checkpoint}\n")
        info.write(f"model_class={model_class.__name__}\n")
        info.write(f"methods={','.join(args.methods)}\n")
        info.write(f"processed_batches={processed_batches}\n")
        info.write(f"processed_samples={processed_samples}\n")
        info.write(f"max_samples={args.max_samples}\n")
        info.write(f"max_batches={args.max_batches}\n")
        info.write(f"max_total_samples={args.max_total_samples}\n")
        info.write(f"ig_batch_size={args.ig_batch_size}\n")
        info.write(f"attribution_device={args.attribution_device}\n")
        info.write("\n[dpp_sample_counts]\n")
        for method in args.methods:
            info.write(f"method={method}\n")
            for class_idx, class_name in enumerate(class_names):
                dpp8_count = dpp_class_count[method]["dpp8"][class_idx]
                dpp9_count = dpp_class_count[method]["dpp9"][class_idx]
                combined_count = dpp_class_count[method]["combined"][class_idx]
                info.write(
                    f"class={class_name};dpp8={dpp8_count};dpp9={dpp9_count};combined={combined_count}\n"
                )
            info.write("\n")

    print(f"Saved attribution heatmaps to: {base_output_dir}")


if __name__ == "__main__":
    # Script entry point.
    main()


