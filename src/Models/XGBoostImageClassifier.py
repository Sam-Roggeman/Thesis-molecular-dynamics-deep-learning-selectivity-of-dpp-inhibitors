from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.model_training.batch_preprocessing import prepare_model_batch


def _resolve_xgboost():
    try:
        xgboost_module = importlib.import_module("xgboost")
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBoostImageClassifier. Install it with `pip install xgboost`."
        ) from exc
    return xgboost_module.XGBClassifier


def _get_xgboost_version() -> tuple[int, int, int]:
    xgb = importlib.import_module("xgboost")

    version_parts = []
    for part in xgb.__version__.split(".")[:3]:
        digits = []
        for char in part:
            if char.isdigit():
                digits.append(char)
            else:
                break
        version_parts.append(int("".join(digits) or 0))

    while len(version_parts) < 3:
        version_parts.append(0)

    return tuple(version_parts)

def _build_backend_params(device: str) -> dict[str, Any]:
    if device == "cpu":
        return {
            "tree_method": "hist",
            "device": "cpu",
        }

    if _get_xgboost_version() >= (2, 0, 0):
        return {
            "tree_method": "hist",
            "device": "cuda",
        }

    return {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
    }


def _flatten_images(images: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(images):
        images = torch.as_tensor(images)
    return images.detach().cpu().to(dtype=torch.float32).reshape(images.shape[0], -1).numpy()


def _collect_from_dataloader(dataloader, max_batches: int | None = None, scramble: bool = False):
    features = []
    labels = []
    device = torch.device("cpu")

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, batch_labels = prepare_model_batch(batch, device=device, scramble=scramble)
        features.append(_flatten_images(images))
        labels.append(batch_labels.detach().cpu().numpy())

    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


@dataclass
class XGBoostMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray


class XGBoostImageClassifier:
    def __init__(self, num_classes: int = 5, **xgb_params: Any):
        XGBClassifier = _resolve_xgboost()
        device = "cpu"
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. Falling back to CPU.")
        else:
            device = "cuda"

        default_params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.4,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "gamma": 0.0,
            "max_bin": 256,
            "n_jobs": max(1, (os.cpu_count() or 1) - 1),
            "random_state": 42,
            "verbosity": 1,
            "eval_metric": "mlogloss",
        }
        default_params.update(_build_backend_params(device))
        default_params.update(xgb_params)

        self.num_classes = num_classes
        self.model = XGBClassifier(**default_params)
        self.feature_shape_: tuple[int, ...] | None = None


    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None, verbose: bool = True):
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)
        self.feature_shape_ = tuple(X.shape[1:])
        return self

    def fit_from_dataloader(self, trainloader, validationloader=None, max_train_batches: int | None = None, max_validation_batches: int | None = None):
        X_train, y_train = _collect_from_dataloader(trainloader, max_batches=max_train_batches, scramble=False)
        eval_set = None
        if validationloader is not None:
            X_val, y_val = _collect_from_dataloader(validationloader, max_batches=max_validation_batches, scramble=False)
            eval_set = [(X_val, y_val)]
        self.fit(X_train, y_train, eval_set=eval_set)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_from_dataloader(self, dataloader, max_batches: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        X, y = _collect_from_dataloader(dataloader, max_batches=max_batches, scramble=False)
        return self.predict(X), y

    def evaluate_from_dataloader(self, dataloader, max_batches: int | None = None) -> XGBoostMetrics:
        predictions, labels = self.predict_from_dataloader(dataloader, max_batches=max_batches)

        return XGBoostMetrics(
            accuracy=accuracy_score(labels, predictions),
            precision=precision_score(labels, predictions, average="weighted", zero_division=0),
            recall=recall_score(labels, predictions, average="weighted", zero_division=0),
            f1_score=f1_score(labels, predictions, average="weighted", zero_division=0),
            confusion_matrix=confusion_matrix(labels, predictions, labels=list(range(self.num_classes))),
        )

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        return str(path)

    @classmethod
    def load(cls, path: str | Path, num_classes: int = 5) -> "XGBoostImageClassifier":
        instance = cls(num_classes=num_classes)
        instance.model.load_model(str(path))
        return instance