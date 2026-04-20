import time

import torch
from src.model_training.LabelEncoder import LabelEncoder

label_encoder = LabelEncoder()
TARGET_SIZE = 168
TARGET_PIXELS = TARGET_SIZE * TARGET_SIZE

from src.utils.logger import get_logger
logging = get_logger() 

def _labels_to_tensor(labels, device):
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long, non_blocking=True)
    else:
        raise ValueError(f"Unsupported label type: {type(labels)}. Expected tensor, list, tuple, or string.")
    if isinstance(labels, (list, tuple)):
        if len(labels) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        first = labels[0]
        if isinstance(first, str):
            encoded = label_encoder.encode_labels(labels)
            return torch.tensor(encoded, dtype=torch.long, device=device)
        return torch.as_tensor(labels, dtype=torch.long, device=device)

    # Scalar label fallback.
    if isinstance(labels, str):
        return torch.tensor([label_encoder.encode_label(labels)], dtype=torch.long, device=device)
    return torch.tensor([int(labels)], dtype=torch.long, device=device)


def _coords_to_tensor(batch_data, device):
    if torch.is_tensor(batch_data):
        return batch_data.to(device=device, dtype=torch.float32, non_blocking=True)

    if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0 and torch.is_tensor(batch_data[0]):
        return torch.stack([x.to(dtype=torch.float32) for x in batch_data], dim=0).to(device=device, non_blocking=True)

    # Fast path for pre-packed fixed-shape batches (e.g., [B, TARGET_PIXELS, 3]).
    if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0:
        try:
            packed = torch.as_tensor(batch_data, dtype=torch.float32)
            if packed.ndim == 3 and packed.shape[-1] == 3:
                return packed.to(device=device, non_blocking=True)
        except Exception:
            pass
    raise ValueError(f"Unsupported batch data format: {type(batch_data)} with element type {type(batch_data[0]) if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0 else 'N/A'}. Expected tensor or list/tuple of tensors.")
    if isinstance(batch_data, (list, tuple)):
        tensor_list = [torch.as_tensor(item, dtype=torch.float32) for item in batch_data]
        max_len = max(t.shape[0] for t in tensor_list)
        padded = torch.zeros((len(tensor_list), max_len, 3), dtype=torch.float32)
        for i, t in enumerate(tensor_list):
            padded[i, :t.shape[0]] = t
        return padded.to(device=device, non_blocking=True)

    return torch.as_tensor(batch_data, dtype=torch.float32, device=device)


def _num_atoms_to_tensor(num_atoms, coords, device):
    if num_atoms is None:
        return torch.full((coords.shape[0],), coords.shape[1], dtype=torch.long, device=device)
    if torch.is_tensor(num_atoms):
        return num_atoms.to(device=device, dtype=torch.long, non_blocking=True)
    return torch.as_tensor(num_atoms, dtype=torch.long, device=device)


def _rodrigues_rotation_matrix(vector, random_vector, device):
    v = torch.cross(vector, random_vector, dim=0)
    c = torch.dot(vector, random_vector)
    s = torch.linalg.norm(v)
    if s < 1e-8:
        return torch.eye(3, dtype=vector.dtype, device=device)
    v = v / s
    kmat = torch.tensor(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=vector.dtype,
        device=device,
    )
    kmat_sq = kmat @ kmat
    return torch.eye(3, dtype=vector.dtype, device=device) + kmat + kmat_sq * ((1.0 - c) / (s * s + 1e-12))


def _scramble_in_place(coords, num_atoms, diameter=140.0):
    radius = diameter / 2.0
    device = coords.device
    batch_size, seq_len, _ = coords.shape

    n_real = num_atoms.to(device=device, dtype=torch.long).clamp(min=1, max=seq_len)
    atom_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    valid_mask = atom_idx < n_real.unsqueeze(1)
    valid_mask_3d = valid_mask.unsqueeze(-1)

    rotated = coords
    if seq_len >= 2:
        has_two_atoms = n_real >= 2
        if bool(has_two_atoms.any()):
            vectors = coords[:, 1, :] - coords[:, 0, :]
            vectors = vectors / torch.linalg.norm(vectors, dim=1, keepdim=True).clamp_min(1e-8)

            random_vectors = torch.randn(batch_size, 3, device=device, dtype=coords.dtype)
            random_vectors = random_vectors / torch.linalg.norm(random_vectors, dim=1, keepdim=True).clamp_min(1e-8)

            v = torch.cross(vectors, random_vectors, dim=1)
            c = (vectors * random_vectors).sum(dim=1)
            s = torch.linalg.norm(v, dim=1)

            valid_rotation = has_two_atoms & (s >= 1e-8)
            axis = torch.zeros_like(v)
            axis[valid_rotation] = v[valid_rotation] / s[valid_rotation].unsqueeze(1)

            kmat = torch.zeros(batch_size, 3, 3, dtype=coords.dtype, device=device)
            kmat[:, 0, 1] = -axis[:, 2]
            kmat[:, 0, 2] = axis[:, 1]
            kmat[:, 1, 0] = axis[:, 2]
            kmat[:, 1, 2] = -axis[:, 0]
            kmat[:, 2, 0] = -axis[:, 1]
            kmat[:, 2, 1] = axis[:, 0]

            kmat_sq = torch.bmm(kmat, kmat)
            eye = torch.eye(3, dtype=coords.dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            factor = ((1.0 - c) / (s * s + 1e-12)).unsqueeze(1).unsqueeze(2)
            rotation = eye + kmat + kmat_sq * factor
            rotation = torch.where(valid_rotation.view(-1, 1, 1), rotation, eye)

            rotated = torch.bmm(coords, rotation.transpose(1, 2))
            rotated = torch.where(valid_mask_3d, rotated, coords)

    random_points = (torch.rand(batch_size, 3, device=device, dtype=coords.dtype) * 2.0 - 1.0) * radius
    denom = n_real.to(dtype=coords.dtype).unsqueeze(1)
    com = (rotated * valid_mask_3d.to(dtype=coords.dtype)).sum(dim=1) / denom
    shifts = random_points - com
    translated = rotated + shifts.unsqueeze(1)

    coords.copy_(torch.where(valid_mask_3d, translated, coords))


def _coords_to_rgb(coords, num_atoms):
    batch_size, current_len, _ = coords.shape
    device = coords.device
    out = torch.zeros((batch_size, TARGET_PIXELS, 3), dtype=coords.dtype, device=device)

    copy_len = min(current_len, TARGET_PIXELS)
    out[:, :copy_len, :] = coords[:, :copy_len, :]

    n_real = num_atoms.to(device=device, dtype=torch.long).clamp(min=1, max=copy_len)
    pixel_idx = torch.arange(TARGET_PIXELS, device=device).unsqueeze(0)
    valid_mask = pixel_idx < n_real.unsqueeze(1)
    valid_mask_3d = valid_mask.unsqueeze(-1)

    finfo = torch.finfo(out.dtype)
    masked_min = torch.where(valid_mask_3d, out, torch.full_like(out, finfo.max))
    masked_max = torch.where(valid_mask_3d, out, torch.full_like(out, finfo.min))
    coords_min = masked_min.min(dim=1).values
    coords_max = masked_max.max(dim=1).values
    coords_range = (coords_max - coords_min).clamp_min(1e-8)

    out = (out - coords_min.unsqueeze(1)) / coords_range.unsqueeze(1)
    out = torch.where(valid_mask_3d, out, torch.zeros_like(out))

    out = out.view(batch_size, TARGET_SIZE, TARGET_SIZE, 3).permute(0, 3, 1, 2).contiguous()
    return out


def prepare_model_batch(batch, device, scramble=False):
    """WITH TIMING: Measure each step"""
    
    timings = {}
    
    # Step 1: Get labels
    t0 = time.time()
    labels = _labels_to_tensor(batch["labels"], device)
    timings['labels'] = time.time() - t0
    
    # Step 2: Get data
    t0 = time.time()
    data = batch["data"]
    timings['get_data'] = time.time() - t0
    
    # Step 3: Convert coords to tensor
    t0 = time.time()
    coords = _coords_to_tensor(data, device)
    timings['coords_to_tensor'] = time.time() - t0
    
    # Step 4: Get num_atoms
    t0 = time.time()
    num_atoms = _num_atoms_to_tensor(batch.get("num_atoms"), coords, device)
    timings['num_atoms'] = time.time() - t0
    
    # Step 5: Scramble (if enabled)
    t0 = time.time()
    if scramble:
        _scramble_in_place(coords, num_atoms)
    timings['scramble'] = time.time() - t0
    
    # Step 6: Convert to RGB images (THIS IS LIKELY THE BOTTLENECK)
    t0 = time.time()
    images = _coords_to_rgb(coords, num_atoms)
    timings['coords_to_rgb'] = time.time() - t0
    
    # Print timing every N calls (e.g., every 10 batches)
    if not hasattr(prepare_model_batch, 'call_count'):
        prepare_model_batch.call_count = 0
    prepare_model_batch.call_count += 1
    
    if prepare_model_batch.call_count % 100 == 0:
        logging.debug(f"\n[Prepare Model Batch Timing - Call {prepare_model_batch.call_count}]:")
        for step, duration in timings.items():
            logging.debug(f"  {step}: {duration*1000:.2f}ms")
        logging.debug(f"  TOTAL: {sum(timings.values())*1000:.2f}ms")
    
    return images, labels