import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.Models.XGBoostImageClassifier import XGBoostImageClassifier
from src.data_loading.HFDataloader import initialize_dataloaders
from src.utils.cacheManager import cacheManager
from src.utils.training_config import TrainingConfig


def _build_output_dir(run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path("output") / "models" / "XGBoostImage" / run_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost classifier on flattened 168x168 RGB images.")
    parser.add_argument("--run-name", type=str, default="XGBoostImage_168")
    parser.add_argument("--dataset-size", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.4)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _metrics_to_dict(metrics):
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "confusion_matrix": metrics.confusion_matrix.tolist(),
    }


def main() -> None:
    load_dotenv()
    args = _parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        num_cpus=args.num_cpus,
        stream_train_split=False,
        stream_validation_split=False,
        stream_test_split=False,
    )

    cache = cacheManager(os.getenv("HF_CACHE_DIR"), os.getenv("FAST_CACHE_DIR"))
    dataloaders = initialize_dataloaders(config, cache_manager=cache, streaming=False)

    model = XGBoostImageClassifier(
        use_cuda=True,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        random_state=args.random_state,
    )

    print(f"XGBoost backend selected: {'cuda' if model.use_cuda else 'cpu'}")

    output_dir = _build_output_dir(args.run_name)

    model.fit_from_dataloader(
        dataloaders["train"],
        validationloader=dataloaders["validation"],
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
    )

    validation_metrics = model.evaluate_from_dataloader(
        dataloaders["validation"],
        max_batches=args.max_validation_batches,
    )
    test_metrics = model.evaluate_from_dataloader(
        dataloaders["test"],
        max_batches=args.max_test_batches,
    )

    model_path = model.save(output_dir / "xgboost_image_model.json")
    payload = {
        "run_name": args.run_name,
        "output_dir": str(output_dir),
        "model_path": model_path,
        "validation": _metrics_to_dict(validation_metrics),
        "test": _metrics_to_dict(test_metrics),
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "min_child_weight": args.min_child_weight,
            "gamma": args.gamma,
            "random_state": args.random_state,
        },
        "dataset_size": args.dataset_size,
        "batch_size": args.batch_size,
        "num_cpus": args.num_cpus,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()