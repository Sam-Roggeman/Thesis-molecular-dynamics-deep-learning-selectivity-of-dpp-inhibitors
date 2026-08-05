import torch
from contextlib import nullcontext
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import logging
from src.model_training.batch_preprocessing import prepare_model_batch
import numpy as np

def _resolve_amp_dtype(amp_dtype):
    if isinstance(amp_dtype, torch.dtype):
        return amp_dtype
    if isinstance(amp_dtype, str):
        normalized = amp_dtype.strip().lower()
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
    return torch.bfloat16


def _autocast_context(device, use_mixed_precision=False, amp_dtype=torch.bfloat16):
    amp_dtype = _resolve_amp_dtype(amp_dtype)
    if bool(use_mixed_precision) and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    return nullcontext()

def _compute_label_prediction_statistics(labels, predictions, statistics_to_compute):
    stats = {}
    if len(labels) == 0:
        return stats

    if "accuracy" in statistics_to_compute:
        stats["accuracy"] = accuracy_score(labels, predictions)
    if "precision" in statistics_to_compute:
        stats["precision"] = precision_score(labels, predictions, average='weighted', zero_division=0)
    if "recall" in statistics_to_compute:
        stats["recall"] = recall_score(labels, predictions, average='weighted', zero_division=0)
    if "f1_score" in statistics_to_compute:
        stats["f1_score"] = f1_score(labels, predictions, average='weighted', zero_division=0)
    if "confusion_matrix" in statistics_to_compute:
        # make sure to specify labels to include all classes even if some are missing in this subset
        stats["confusion_matrix"] = confusion_matrix(labels, predictions, labels=list(range(5)))
        stats["confusion_matrix_normalized"] = confusion_matrix(labels, predictions, normalize='true', labels=list(range(5)))

    return stats

def calculate_fpr_multiclass(conf_matrix):
    """
    Calculate the False Positive Rate (FPR) for each class
    from a multiclass confusion matrix.

    Parameters
    ----------
    conf_matrix : np.ndarray
        NxN confusion matrix.

    Returns
    -------
    dict
        Dictionary mapping each class index to its FPR.
    """
    n_classes = conf_matrix.shape[0]

    if conf_matrix.shape[0] != conf_matrix.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    total = np.sum(conf_matrix)
    fpr = {}

    for i in range(n_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        tn = total - tp - fp - fn

        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    macro_fpr = np.mean(list(fpr.values()))

    return macro_fpr

def calculate_statistics(
    model,
    dataloader,
    criterion,
    device,
    max_batches: int | None = None,
    statistics_to_compute: list[str] | None = None,
    batch_preparation_fn=prepare_model_batch,
    use_mixed_precision=False,
    amp_dtype=torch.bfloat16,
) -> dict[str, float]:
    """Calculate specified statistics for a model on a given dataloader. Loop over the dataloader and compute the specified statistics for each batch, then average them over the entire dataloader.
    Args:
        model (_type_): _model to evaluate
        dataloader (_type_): dataloader to evaluate on
        criterion (_type_): loss function to use for loss calculation
        device (_type_): device to use for evaluation
        max_batches (int | None, optional): maximum number of batches to evaluate. Defaults to None.
        statistics_to_compute (list[str] | None, optional): list of statistics to compute. Defaults to None.

    Returns:
        dict[str, float]: dictionary containing the computed statistics
    """
    if statistics_to_compute is None:
        statistics_to_compute = ["accuracy", "loss", "precision", "recall", "f1_score", "confusion_matrix"]
    batch_preparation_fn = batch_preparation_fn or prepare_model_batch
    
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    num_batches = 0
    dpp_class_examples = {
        "dpp8": {"labels": [], "predictions": []},
        "dpp9": {"labels": [], "predictions": []},
    }
    has_dpp_class = False

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            logging.debug(f"Processing batch {batch_idx + 1}...")
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = batch_preparation_fn(data, device, scramble=False)
            with _autocast_context(device, use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(outputs.data, 1)
            labels_cpu = labels.cpu().numpy()
            predicted_cpu = predicted.cpu().numpy()

            all_labels.extend(labels_cpu)
            all_predictions.extend(predicted_cpu)

            if "dpp_class" in data:
                has_dpp_class = True
                dpp_values = data["dpp_class"]
                if isinstance(dpp_values, torch.Tensor):
                    dpp_values = dpp_values.cpu().numpy()
                for label, prediction, dpp_value in zip(labels_cpu, predicted_cpu, dpp_values):
                    if dpp_value is None:
                        continue
                    dpp_class_examples[dpp_value]["labels"].append(label)
                    dpp_class_examples[dpp_value]["predictions"].append(prediction)

    # statistics -> dpp_class -> accuracy, loss, precision, recall, f1_score, confusion_matrix    
    statistics = {}

    
    if "accuracy" in statistics_to_compute:
        statistics["accuracy"] = accuracy_score(all_labels, all_predictions)
    if "loss" in statistics_to_compute:
        statistics["loss"] = total_loss / max(num_batches, 1)
    if "precision" in statistics_to_compute:
        statistics["precision"] = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "recall" in statistics_to_compute:
        statistics["recall"] = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "f1_score" in statistics_to_compute:
        statistics["f1_score"] = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    if "confusion_matrix" in statistics_to_compute:
        statistics["confusion_matrix"] = confusion_matrix(all_labels, all_predictions, labels=list(range(5)))
        statistics["confusion_matrix_normalized"] = confusion_matrix(all_labels, all_predictions, normalize='true', labels=list(range(5)))
    if "fpr" in statistics_to_compute:
        statistics["fpr"] = calculate_fpr_multiclass(statistics["confusion_matrix"])

    if has_dpp_class or any(len(v["labels"]) > 0 for v in dpp_class_examples.values()):
        statistics["dpp_class"] = {}
        dpp8_count = len(dpp_class_examples["dpp8"]["labels"])
        dpp9_count = len(dpp_class_examples["dpp9"]["labels"])

        for group_name in ["dpp8", "dpp9"]:
            group_stats = _compute_label_prediction_statistics(
                dpp_class_examples[group_name]["labels"],
                dpp_class_examples[group_name]["predictions"],
                statistics_to_compute,
            )
            group_stats["num_examples"] = len(dpp_class_examples[group_name]["labels"])
            statistics["dpp_class"][group_name] = group_stats

        weighted_accuracy_numerator = 0.0
        if dpp8_count > 0:
            weighted_accuracy_numerator += statistics["dpp_class"]["dpp8"].get("accuracy", 0.0) * dpp8_count
        if dpp9_count > 0:
            weighted_accuracy_numerator += statistics["dpp_class"]["dpp9"].get("accuracy", 0.0) * dpp9_count



    return statistics

def calculate_accuracy_and_loss(
    model,
    dataloader,
    criterion,
    device,
    max_batches: int | None = None,
    batch_preparation_fn=prepare_model_batch,
    use_mixed_precision=False,
    amp_dtype=torch.bfloat16,
):
    correct = 0
    total = 0
    current_loss = 0.0
    num_batches = 0
    batch_preparation_fn = batch_preparation_fn or prepare_model_batch

    model.to(device)
    model.eval()

    with torch.no_grad():
        # iterate over streaming dataloader
        for batch_idx, data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = batch_preparation_fn(data, device, scramble=False)

            with _autocast_context(device, use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            current_loss += loss.item()
            num_batches += 1

    accuracy = correct / max(total, 1)
    current_loss /= max(num_batches, 1)

    return accuracy, current_loss

def calculate_precision_recall_f1(
    model,
    dataloader,
    device,
    max_batches: int | None = None,
    batch_preparation_fn=prepare_model_batch,
    use_mixed_precision=False,
    amp_dtype=torch.bfloat16,
):
    all_labels = []
    all_predictions = []
    batch_preparation_fn = batch_preparation_fn or prepare_model_batch
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels = batch_preparation_fn(data, device, scramble=False)
            with _autocast_context(device, use_mixed_precision=use_mixed_precision, amp_dtype=amp_dtype):
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions,normalize='true', labels=list(range(5)))

    return precision, recall, f1, conf_matrix

def all_statistics(
    model,
    dataloader,
    criterion,
    device,
    max_batches: int | None = None,
    batch_preparation_fn=prepare_model_batch,
    use_mixed_precision=False,
    amp_dtype=torch.bfloat16,
) -> dict[str, float]:
    accuracy, loss = calculate_accuracy_and_loss(
        model=model,
        dataloader=dataloader,
        device=device,
        criterion=criterion,
        max_batches=max_batches,
        batch_preparation_fn=batch_preparation_fn,
        use_mixed_precision=use_mixed_precision,
        amp_dtype=amp_dtype,
    )
    precision, recall, f1, conf_matrix = calculate_precision_recall_f1(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        batch_preparation_fn=batch_preparation_fn,
        use_mixed_precision=use_mixed_precision,
        amp_dtype=amp_dtype,
    )

    return {
        "accuracy": accuracy,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }