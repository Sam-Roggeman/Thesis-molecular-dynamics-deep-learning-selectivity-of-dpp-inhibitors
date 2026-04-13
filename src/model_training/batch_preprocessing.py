import torch
from src.model_training.LabelEncoder import LabelEncoder

label_encoder = LabelEncoder()
TARGET_SIZE = 168
TARGET_PIXELS = TARGET_SIZE * TARGET_SIZE



def _labels_to_tensor(labels, device):
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long, non_blocking=True)

    if isinstance(labels, (list, tuple)):
        encoded = []
        for label in labels:
            if isinstance(label, str):
                encoded.append(label_encoder.encode(label))
            else:
                encoded.append(int(label))
        return torch.tensor(encoded, dtype=torch.long, device=device)

    # Scalar label fallback.
    if isinstance(labels, str):
        return torch.tensor([label_encoder.encode(labels)], dtype=torch.long, device=device)
    return torch.tensor([int(labels)], dtype=torch.long, device=device)


def _coords_to_tensor(batch_data, device):
    if torch.is_tensor(batch_data):
        return batch_data.to(device=device, dtype=torch.float32, non_blocking=True)

    if isinstance(batch_data, (list, tuple)) and len(batch_data) > 0 and torch.is_tensor(batch_data[0]):
        return torch.stack([x.to(dtype=torch.float32) for x in batch_data], dim=0).to(device=device, non_blocking=True)

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
    for i in range(coords.shape[0]):
        n_real = int(num_atoms[i].item())
        n_real = max(1, min(n_real, coords.shape[1]))
        frame = coords[i, :n_real]

        if n_real >= 2:
            random_vector = torch.randn(3, device=device, dtype=coords.dtype)
            random_vector = random_vector / torch.linalg.norm(random_vector).clamp_min(1e-8)

            vector = frame[1] - frame[0]
            vector = vector / torch.linalg.norm(vector).clamp_min(1e-8)

            rotation_matrix = _rodrigues_rotation_matrix(vector, random_vector, device)
            coords[i, :n_real] = frame @ rotation_matrix.T

        random_point = (torch.rand(3, device=device, dtype=coords.dtype) * 2.0 - 1.0) * radius
        com = coords[i, :n_real].mean(dim=0)
        coords[i, :n_real] = coords[i, :n_real] + (random_point - com)


def _coords_to_rgb(coords, num_atoms):
    batch_size, current_len, _ = coords.shape
    device = coords.device
    out = torch.zeros((batch_size, TARGET_PIXELS, 3), dtype=coords.dtype, device=device)

    copy_len = min(current_len, TARGET_PIXELS)
    out[:, :copy_len, :] = coords[:, :copy_len, :]

    for i in range(batch_size):
        n_real = int(num_atoms[i].item())
        n_real = max(1, min(n_real, copy_len))
        real_coords = out[i, :n_real, :]
        coords_min = real_coords.min(dim=0).values
        coords_max = real_coords.max(dim=0).values
        coords_range = (coords_max - coords_min).clamp_min(1e-8)
        out[i] = (out[i] - coords_min) / coords_range
        if n_real < TARGET_PIXELS:
            out[i, n_real:, :] = 0.0

    out = out.view(batch_size, TARGET_SIZE, TARGET_SIZE, 3).permute(0, 3, 1, 2).contiguous()
    return out


def prepare_model_batch(batch, device, scramble=False):
    labels = _labels_to_tensor(batch["labels"], device)
    data = batch["data"]

    # Fast path for already image-shaped tensors.
    if torch.is_tensor(data) and data.ndim == 4 and data.shape[1] == 3:
        images = data.to(device=device, dtype=torch.float32, non_blocking=True)
        return images, labels

    coords = _coords_to_tensor(data, device)
    num_atoms = _num_atoms_to_tensor(batch.get("num_atoms"), coords, device)

    if scramble:
        _scramble_in_place(coords, num_atoms)

    images = _coords_to_rgb(coords, num_atoms)
    return images, labels
