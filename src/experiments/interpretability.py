import argparse
import ast
import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv

from src.Models.DCNN import CustomDenseNet
from src.Models.OneLayer import OneLayerNet
from src.Models.SimpleCNN import SimpleCNN
from src.data_loading.HFDataloader import initialize_dataloaders
from src.model_training.LabelEncoder import LabelEncoder
from src.utils.interpretability import CaptumInterpreter
from src.utils.training_config import TrainingConfig


MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
    "OneLayerNet": OneLayerNet,
    "CustomDenseNet": CustomDenseNet,
}


def _extract_class_name(class_value) -> str | None:
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
    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
        raw_checkpoint = raw_checkpoint["state_dict"]
    if not isinstance(raw_checkpoint, dict):
        raise ValueError("Checkpoint format is unsupported. Expected a state_dict or a dict containing state_dict.")
    return raw_checkpoint


def _load_weights(model: torch.nn.Module, state_dict: dict) -> None:
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
) -> None:
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
        fig.suptitle(f"Method={method} | true={true_name} | pred={pred_name}")

        filename = os.path.join(output_dir, f"{method}_sample_{idx:03d}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=160)
        plt.close(fig)


def main():
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
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--target", type=int, default=None, help="Optional class index target")
    parser.add_argument("--output-dir", default=None, help="Output folder for heatmaps")
    args = parser.parse_args()

    load_dotenv()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

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

    dataloaders = initialize_dataloaders(config)
    batch = next(iter(dataloaders[args.split]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**model_args).to(device)
    state_dict = _load_state_dict(args.checkpoint, device)
    _load_weights(model, state_dict)
    model.eval()

    inputs = batch["data"][: args.max_samples].to(device)
    labels = batch["labels"][: args.max_samples].to(device)
    inputs = inputs.clone().detach().requires_grad_(True)

    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1)

    interpreter = CaptumInterpreter(model)
    class_names = LabelEncoder().get_classes()

    base_output_dir = args.output_dir
    if base_output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_output_dir = os.path.join(checkpoint_dir, "interpretability", run_name)
    os.makedirs(base_output_dir, exist_ok=True)

    target = args.target if args.target is not None else None

    for method in args.methods:
        if method == "integrated_gradients":
            result = interpreter.integrated_gradients(
                inputs,
                target=target,
                baselines=torch.zeros_like(inputs),
                n_steps=args.n_steps,
            )
        elif method == "saliency":
            result = interpreter.saliency(inputs, target=target)
        elif method == "guided_backprop":
            result = interpreter.guided_backprop(inputs, target=target)
        else:
            raise ValueError(f"Unsupported method requested: {method}")

        heatmaps = CaptumInterpreter.summarize_attributions(
            result.attributions,
            reduce_dim=1,
            normalize=True,
        )

        method_dir = os.path.join(base_output_dir, method)
        _save_heatmaps(
            inputs=inputs.detach().cpu(),
            labels=labels.detach().cpu(),
            preds=preds.detach().cpu(),
            attribution_map=heatmaps,
            class_names=class_names,
            method=method,
            output_dir=method_dir,
        )

        if result.convergence_delta is not None:
            delta_file = os.path.join(method_dir, "convergence_delta.pt")
            torch.save(result.convergence_delta.detach().cpu(), delta_file)

    run_info_path = os.path.join(base_output_dir, "run_info.txt")
    with open(run_info_path, "w", encoding="utf-8") as info:
        info.write(f"checkpoint={args.checkpoint}\n")
        info.write(f"model_class={model_class.__name__}\n")
        info.write(f"methods={','.join(args.methods)}\n")
        info.write(f"max_samples={args.max_samples}\n")

    print(f"Saved attribution heatmaps to: {base_output_dir}")


if __name__ == "__main__":
    main()
