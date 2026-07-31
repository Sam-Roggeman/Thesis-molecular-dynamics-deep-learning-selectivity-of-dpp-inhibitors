import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.Models.SimpleCNN import SimpleCNN
from src.Transform.tranformators import apply_image_transform, apply_image_transform_noscramble
from src.utils.training_config import TrainingConfig
from src.utils.training_setup import train_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_output_dir(study_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("output", "tuning", "simplecnn", f"{study_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_trials(study: optuna.Study, output_dir: str) -> None:
    json_path = os.path.join(output_dir, "trials.json")
    csv_path = os.path.join(output_dir, "trials.csv")
    best_path = os.path.join(output_dir, "best_trial.json")

    trial_rows = []
    all_param_keys = set()
    for trial in study.trials:
        all_param_keys.update(trial.params.keys())

    sorted_param_keys = sorted(all_param_keys)

    for trial in study.trials:
        row = {
            "number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
        }
        for key in sorted_param_keys:
            row[key] = trial.params.get(key)
        row["run_dir"] = trial.user_attrs.get("run_dir")
        trial_rows.append(row)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(trial_rows, handle, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["number", "state", "value", "run_dir"] + sorted_param_keys
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(row)

    if study.best_trial is not None:
        best_payload = {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
            "run_dir": study.best_trial.user_attrs.get("run_dir"),
        }
        with open(best_path, "w", encoding="utf-8") as handle:
            json.dump(best_payload, handle, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SimpleCNN with Optuna.")
    parser.add_argument("--study-name", type=str, default="simplecnn_hpo")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Overall optimization timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///output/tuning/simplecnn.db")
    parser.add_argument("--dataset-size", type=float, default=0.15)
    parser.add_argument("--max-train-steps", type=int, default=4000)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--trial-time-limit", type=int, default=2 * 60 * 60, help="Per-trial time limit in seconds.")
    parser.add_argument("--dataset-location", type=str, default="Sam-Roggeman/SamRoggeman_Thesis_Dataset_full")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    output_dir = _build_output_dir(args.study_name)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 3e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.6)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        patience = trial.suggest_int("patience", 3, 10)
        minimum_delta = trial.suggest_float("minimum_delta", 1e-3, 2e-2, log=True)
        use_mixed_precision = trial.suggest_categorical("use_mixed_precision", [False, True])
        compile_model = trial.suggest_categorical("compile_model", [True, False])

        config = TrainingConfig(
            model_class=SimpleCNN,
            model_args={
                "input_size": 168,
                "dropout_rate": dropout_rate,
            },
            dataset_location=args.dataset_location,
            dataset_size=args.dataset_size,
            training_transform=apply_image_transform,
            validation_transform=apply_image_transform_noscramble,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            minimum_delta=minimum_delta,
            max_train_steps=args.max_train_steps,
            eval_every_steps=args.eval_every_steps,
            log_every_steps=args.log_every_steps,
            time_limit=args.trial_time_limit,
            use_mixed_precision=use_mixed_precision,
            compile_model=compile_model,
            stream_train_split=False,
            stream_validation_split=False,
            stream_test_split=False,
        )

        run_name = f"SimpleCNN_optuna_trial_{trial.number}"
        result = train_model(config=config, model_name=run_name, streaming=False)

        if not result or result.get("best_validation_loss") is None:
            raise RuntimeError("Trial completed without validation loss; check training logs for details.")

        trial.set_user_attr("run_dir", result.get("run_dir"))
        trial.set_user_attr("best_validation_loss", result.get("best_validation_loss"))

        return float(result["best_validation_loss"])

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    _save_trials(study, output_dir)

    print("Optimization finished")
    print(f"Study name: {study.study_name}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Saved tuning artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
