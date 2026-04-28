import argparse
import ast
import json
import os
import re

import torch
from dotenv import load_dotenv
load_dotenv()
from src.Models.DCNN import CustomDenseNet
from src.Models.LinearAttentionTransformerPP import LinearAttentionTransformerPP
from src.Models.OneLayer import OneLayerNet
from src.Models.SimpleCNN import SimpleCNN
from src.data_loading.HFDataloader import initialize_dataloaders
from src.data_postprocessing.model_testing import model_testing
from src.model_training.batch_preprocessing import prepare_model_batch, prepare_sequence_batch
from src.utils.training_config import TrainingConfig
from src.model_training.utils import load_model
# logging
from src.utils.logger import init_logger, get_logger, DEBUG as LOGGING_DEBUG

from src.utils.resolvers import (
	MODEL_REGISTRY,
	BATCH_PREPARATION_REGISTRY,
	_extract_class_name,
	_load_config_from_artifacts,
	_build_test_dataloader,
	_resolve_batch_preparation_fn,
)

def _parse_model_args(model_args: str | None) -> dict:
	if not model_args:
		return {}
	parsed = ast.literal_eval(model_args)
	if not isinstance(parsed, dict):
		raise ValueError("--model-args must evaluate to a Python dict, e.g. '{\"input_size\": 168}'.")
	return parsed


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run model testing and save confusion matrix.")
	parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth/.pt).")
	parser.add_argument("--config", default=None, help="Optional training config path (.pt or .json).")
	parser.add_argument(
		"--output-dir",
		default=None,
		help="Directory where confusion_matrix.png will be saved. Defaults to 'checkpoint_dir/test_results/split'.",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=["test"],
		help="Names of the splits to test on.",
	)
	parser.add_argument(
		"--model-class",
		default=None,
		choices=sorted(MODEL_REGISTRY.keys()),
		help="Model class name. If omitted, inferred from training config.",
	)
	parser.add_argument(
		"--model-args",
		default=None,
		help="Model kwargs as a Python dict string, e.g. '{\"input_size\": 168, \"dropout_rate\": 0.2}'.",
	)
 
	parser.add_argument("--dataset-location", default=None, help="Override dataset location from config.")
	parser.add_argument("--dataset-size", type=float, default=1, help="Override dataset fraction (0, 1].")
	parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
	parser.add_argument("--num-cpus", type=int, default=None, help="Override dataloader worker count.")
	parser.add_argument("--max-batches", type=int, default=None, help="Cap number of test batches.")
	# streaming flag to force using streaming dataloader, default False
	parser.add_argument(
		"--streaming",
		action="store_true",
		help="Force using streaming dataloader. By default, the script tries to initialize the regular dataloader and falls back to streaming if it fails. Use this flag to skip the \
		 regular dataloader initialization entirely.",
	)

	parser.add_argument(
		"--device",
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="Execution device. auto selects cuda when available.",
	)
	return parser



def main() -> None:
	load_dotenv()
	args = _build_parser().parse_args()

	checkpoint_path = os.path.abspath(args.checkpoint)
	checkpoint_dir = os.path.dirname(checkpoint_path)
	logging = init_logger(model_dir=checkpoint_dir, log_mode=LOGGING_DEBUG, log_file="test_debug.log")
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	config = _load_config_from_artifacts(args.config, checkpoint_path)

	if config is None:
		if not args.model_class:
			raise ValueError(
				"No training config found. Provide --config or set --model-class/--model-args explicitly."
			)
		config = TrainingConfig(model_class=args.model_class, model_args=_parse_model_args(args.model_args))

	inferred_model_name = _extract_class_name(config.model_class)
	model_name = args.model_class or inferred_model_name
	if not model_name:
		raise ValueError("Could not determine model class. Provide --model-class.")

	if args.model_args is not None:
		config.model_args = _parse_model_args(args.model_args)
	if config.model_args is None:
		config.model_args = {}
	if args.dataset_size is not None:
		if (args.dataset_size <= 0 or args.dataset_size > 1):
			raise ValueError("Invalid dataset size. Must be between 0 and 1.")
		config.dataset_size = args.dataset_size
	if args.dataset_location is not None:
		config.dataset_location = args.dataset_location
	if args.batch_size is not None:
		config.batch_size = args.batch_size
	if args.num_cpus is not None:
		config.num_cpus = args.num_cpus
	if args.output_dir is None:
		args.output_dir = os.path.dirname(checkpoint_path) + f"/test_results/"
	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(args.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but is not available.")
	model = load_model(checkpoint_path, device=device)
	criterion = config.criterion() if config.criterion else torch.nn.CrossEntropyLoss()
	batch_preparation_fn = _resolve_batch_preparation_fn(config, model_name)
	test_loaders = _build_test_dataloader(config, splits=args.splits, streaming=args.streaming)
	logging.info(f"Evaluating checkpoint: {checkpoint_path}")
	logging.info(f"Model: {model_name} with args={config.model_args}")
	logging.info(f"Dataset: {config.dataset_location} (size={config.dataset_size})")
	logging.info(f"Batch size: {config.batch_size}, num_cpus: {config.num_cpus}")
	logging.info(f"Device: {device}")


	for split in args.splits:
		logging.info(f"Testing on split: {split}")
		test_loader = test_loaders[split]
		output_dir = os.path.join(args.output_dir, split)
		os.makedirs(output_dir, exist_ok=True)


		model_testing(
			model=model,
			testloader=test_loader,
			criterion=criterion,
			device=device,
			output_dir=output_dir,
			max_batches=args.max_batches,
			split_name=split,
			batch_preparation_fn=batch_preparation_fn,
		)

	logging.info(f"Testing complete. output saved to {args.output_dir}")


if __name__ == "__main__":
	main()