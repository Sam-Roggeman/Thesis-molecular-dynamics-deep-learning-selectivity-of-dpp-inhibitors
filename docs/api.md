# Python API Documentation

This file is generated from docstrings in `src/`.
Documented modules: **42**.

Regenerate it with:

```bash
python3 docs/generate_api_docs.py
```

## `cli.interpretability`

Source: `src/cli/interpretability.py`

Command-line helpers for attribution maps and PDB visualizations.

### `apply_transformations_to_samples(samples: list)`

Apply transformations to the preprocessed data samples.
Args:
    samples: Samples containing coordinates, atom counts, and binding types.
Returns:
    ``(images, labels)`` with shapes ``[B, C, H, W]`` and ``[B]``.

### `apply_classification(sample: torch.Tensor, model: torch.nn.Module, device: torch.device)`

Apply the trained classification model to the transformed sample.
Args:
    sample: One transformed image with shape ``[C, H, W]``.
    model: Trained classification model.
    device: Device used for model inference.
Returns:
    ``(logits, predicted_class)`` for the single sample.

### `_normalize_spatial_scores(spatial_scores: torch.Tensor)`

Normalize spatial scores to the range [0, 1] so thresholds are comparable across methods.

### `remove_batch_dimension(attributions: torch.Tensor)`

Remove the batch dimension from attributions if present.

### `calculate_spacial_scores(attribution: torch.Tensor)`

Calculate spatial scores from the attribution result.
:attribution: The attribution tensor for the input sample.
:return: Normalized spatial scores tensor [H,W].

### `apply_mask_to_input(input_sample: torch.Tensor, mask: torch.Tensor)`

Apply a binary mask to the input sample, setting masked pixels to zero.
Args:
    input_sample (torch.Tensor): The input sample tensor [C,H,W] or [1,C,H,W].
    mask (torch.Tensor): A binary mask tensor [H,W] where True indicates pixels to be masked.

Returns:
    torch.Tensor: The masked input sample tensor [C,H,W].

### `blur_top_n_pixels(n: int, input_sample: torch.Tensor, attribution: torch.Tensor)`

Blur the top n pixels in the input sample based on the attribution scores.
Args:
    n (int): Number of top pixels to blur.
    input_sample (torch.Tensor): The input sample tensor [C,H,W] or [1,C,H,W].
    attribution (torch.Tensor): The attribution tensor for the input sample.

Returns:
    torch.Tensor: The blurred input sample tensor [C,H,W].

### `blur_according_to_attribution_results(threshold: float, input_sample: torch.Tensor, attribution: torch.Tensor)`

Blur the input sample according to the attribution results.
:threshold: The attribution score threshold above which pixels will be blurred.
:input_sample: The original input sample tensor [C,H,W].
:attribution: The attribution tensor for the input sample.

### `generate_interpretability_attribution(methods: dict[str, Callable], predicted_class: torch.Tensor, sample: torch.Tensor, device: torch.device)`

Generate interpretability insights using the model's output and the input sample.
:param predicted_class: Predicted class index for the input sample.
:param sample: Transformed input sample tensor [C,H,W].
:param device: Device used for attribution methods.
:return: Dictionary containing Captum AttributionResult per method.

### `_attribution_to_atom_scores(attributions: torch.Tensor, num_atoms: int)`

Map attribution tensor back to first num_atoms entries of flattened 2D grid.

### `_normalize_scores(scores: torch.Tensor)`

Normalize scores to [0, 100] for PDB B-factor visualization.

### `_write_bfactor_colored_pdb(pdb_file: str, output_path: str, atom_scores: torch.Tensor)`

Write PDB with per-atom attribution scores in B-factor column.

### `save_attribution_colored_pdbs_spatial_scores(pdb_file: str, output_dir: str, spatial_scores: torch.Tensor, method: int)`

Create one PDB per attribution method with atom scores stored in B-factor.

### `write_coloring_script(colored_pdb_paths: list[str], script_path: str, threshold: float)`

Write a PyMOL script to load and visualize the colored PDBs.

### `load_module(model_checkpoint: str, device: torch.device, method_args: dict)`

Load the trained classification model from the provided checkpoint and return a CaptumInterpreter instance.
:param model_checkpoint: Path to the trained model checkpoint.
:param device: Device used for model inference.
:param method_args: Dictionary of method-specific arguments for interpretability methods.
:return: CaptumInterpreter instance with the loaded model.

### `execute_interpretability(pdb_file=None, pdb_directory=None, output_dir=None, binding_type=None, blur_top_n=None, model_checkpoint=None, method_args=None, threshold=None, methods=None)`

CLI entry point for interpretability tools.
:param pdb_file: path to a PDB file for generating interpretability insights. If provided, the tool will process the file, feed it into the interpretability model, and output the insights. 
:param output_dir: Directory where interpretability results will be saved. Defaults to "./interpretability_results".

## `cli.run_interpretability_all_trajectories`

Source: `src/cli/run_interpretability_all_trajectories.py`

### `already_completed(output_dir: Path)`

Check if the interpretability results for a given output directory already exist.

:param output_dir: Directory where interpretability results are expected to be saved.
:return: True if the results already exist, False otherwise.

## `config.configParser`

Source: `src/config/configParser.py`

Small configuration-file wrapper used by preprocessing and training scripts.

### `ConfigParser`

Read project INI sections and expose them as strings or dictionaries.

#### `__init__(self, config_path='config.ini')`

Load configuration values from ``config_path`` if it exists.

#### `get(self, section, option)`

Return one option value from a named section.

#### `get_section(self, section)`

Return all options in a section as a dictionary.

### `ConfigParserWrapper`

Expose project-specific convenience accessors over ``ConfigParser``.

#### `get_raw_data_folder(self)`

Return the configured raw-data folder.

#### `get_logging(self)`

Return logging folder, file-enable, and console-enable settings.

## `config.training_config`

Source: `src/config/training_config.py`

### `calculate_num_cpus()`

Calculate the number of CPUs to use based on environment variable or default

### `TrainingConfig`

Configuration for training runs

### `TestConfig`

Configuration for validation runs

## `data.data_loading.HFDataloader`

Source: `src/data/data_loading/HFDataloader.py`

Build PyTorch dataloaders from the project’s cached safetensor splits.

### `_effective_worker_count(requested_cpus: int)`

Use physical-core-like worker count on hyperthreaded systems.

### `initialize_dataloaders(config: TrainingConfig, cache_manager: cacheManager=None, splits=None, keep_all_columns: bool=False, streaming: bool=False)`

Create one PyTorch dataloader per requested dataset split.

The cache must contain sharded safetensors with ``data``, ``labels``, and
``num_atoms`` fields. ``config.dataset_size`` limits each split, while
``config.batch_size`` and ``config.num_cpus`` control batching and worker
count. ``keep_all_columns`` and ``streaming`` are retained for API
compatibility with older loader implementations.

Returns:
    Mapping from split names such as ``train`` and ``validation`` to
    configured ``torch.utils.data.DataLoader`` instances.

### `clear_cache(remove_dataset_cache: bool=False, remove_mapped_cache: bool=True)`

Delete selected Hugging Face dataset cache directories from the environment.

## `data.data_loading.SafetensorsDataset`

Source: `src/data/data_loading/SafetensorsDataset.py`

Lazy PyTorch datasets backed by single or sharded safetensors files.

### `SafetensorsDataset`

Load ``data``, ``labels``, and ``num_atoms`` from one safetensors file.

The file is opened lazily on the first length or item request and retained in
memory for subsequent accesses.

#### `__init__(self, filepath)`

Store the file path without loading tensor data.

#### `_load(self)`

Load all tensors once when the dataset is first accessed.

#### `__getitem__(self, idx)`

Return one sample with coordinates, label, and valid-atom count.

#### `__len__(self)`

Return the number of samples in the backing file.

### `ShardedSafetensorsDataset`

Read a fraction of sharded safetensors with one-shard caching.

Each split requires ``{split}_metadata.pt`` and shard files containing the
keys ``data``, ``labels``, and ``num_atoms``. Only the shard needed for the
current item is loaded, and consecutive accesses reuse that shard.

#### `__init__(self, cache_path, split, fraction=1.0)`

Index the requested split and cap its length to ``fraction``.

#### `__getitem__(self, idx)`

Return one sample, loading and caching its source shard if needed.

#### `__len__(self)`

Return the number of samples selected from the shard set.

## `data.data_postprocessing.model_comparison`

Source: `src/data/data_postprocessing/model_comparison.py`

Load model artifacts and compare predictive and resource metrics.

### `ModelStatistics`

Collect evaluation, parameter-count, FLOP, and VRAM statistics for one model.

#### `__init__(self, model_folder, model_name, model_filename, device, criterion)`

Load a saved model artifact and calculate its test statistics.

### `ModelComparison`

Format and plot comparable statistics for several trained models.

#### `__init__(self, modelstatistics: List[ModelStatistics])`

Store the model statistics in display order.

#### `print_table(self)`

Print accuracy, loss, precision, recall, and F1 score.

#### `print_parameters_and_flops(self)`

Print parameter count, GFLOPs, and estimated VRAM usage.

#### `plot_metrics(self, path='./output/comparison_plots/')`

Save a 2x2 comparison plot of training and validation metrics.

## `data.data_preprocessing.hf_functions`

Source: `src/data/data_preprocessing/hf_functions.py`

Hugging Face dataset conversion, cleaning, merging, and upload helpers.

### `_align_dataset_to_features(ds: datasets.Dataset, target_features: datasets.Features, cpu_cores: int)`

Align dataset column types to target features (notably int-like metadata columns).

### `initialize_hf_api()`

Create an authenticated Hub API client using the ``HF_TOKEN`` variable.

### `load_dataset_from_hf(api, repo_id, cpu_cores)`

Load all repository splits from the Hugging Face Hub.

### `convert_col_to_int(dataset: datasets.Dataset, column_name: str, cpu_cores)`

Return a dataset with ``column_name`` converted to integer values.

### `remove_12i_entries(dataset, cpu_cores)`

Remove entries whose ``ligand_name`` identifies the excluded ``12i`` ligand.

### `add_replica_id_column(dataset, cpu_cores)`

Return a dataset with a constant integer ``replica_id`` column.

### `append_to_hf_dataset(dataset, new_datapath, cpu_cores)`

Append train/test/validation data saved on disk to an existing dataset.

### `append_custom_split_to_hf_dataset(new_datapath, new_split_name, repo_id, cpu_cores)`

Append a disk-backed split to a Hub dataset and push the updated dataset.
Args:
    new_datapath (_type_): Path to the new split saved on disk that belongs to the new split (e.g., "unique_test_runs") 
    new_split_name (_type_): Name of the new split (e.g., "unique_test_runs")
    repo_id (_type_): HuggingFace Hub repository ID (e.g., "username/dataset_name")
    cpu_cores (_type_): Number of CPU cores to use for processing

## `data.data_preprocessing.pre_processing`

Source: `src/data/data_preprocessing/pre_processing.py`

### `generate_dataset_from_tars(streaming_pdb_dataset_path, tar_folder, regenerate=False, num_proc=32, skip_existing=True)`

Convert every TAR file in ``tar_folder`` into a disk-backed dataset shard.

The output directory ends up with one subdirectory per trajectory. Each
subdirectory contains a Hugging Face dataset saved with ``save_to_disk`` so
the full corpus can later be reloaded and concatenated without reparsing the
source TAR archives.

### `generate_full_dataset_from_tars(regenerate=False, num_proc=32, skip_existing=True)`

Build the final train/validation/test dataset from all TAR shards.

The function first materializes per-trajectory datasets on disk, then reloads
them, concatenates them into one large dataset, shuffles the rows, and finally
applies the project-specific train/val/test split.

## `data.data_preprocessing.prepare_safetensors`

Source: `src/data/data_preprocessing/prepare_safetensors.py`

### `GracefulStopRequested`

Raised when GPULab requests a graceful shutdown (SIGUSR1).

### `_write_safetensor_shard_from_worker_split(shard_idx, n_shards, start, end, shard_path, num_workers=1)`

Materialize and write one shard inside a subprocess worker.

The parent process passes a slice of the dataset for this shard. This helper
splits that slice into smaller chunks, materializes each chunk in parallel,
concatenates the arrays back together, and then writes the shard to disk
atomically so partially written files are never treated as valid output.

### `_materialize_shard_absolute(start, end)`

Read one shard slice once, then extract arrays for all required fields.

### `_materialize_shard_arrays(dataset_split, start, end)`

Read one shard slice once, then extract arrays for all required fields.

### `_download_dataset(dataset_location, splits=None)`

Download the dataset using Hugging Face's `datasets` library.

### `calculate_sample_size(sample, prefix='sample')`

Recursively estimate sample size in bytes and print per-field breakdown.

### `_save_split_as_safetensors_memory_efficient(dataset_dict, split, cache_path, shard_size=5000, total_nr_workers=1)`

Memory-efficient saving with sharding.

shard_size: Adjust based on available RAM.
- Each shard uses ~shard_size * 168 * 3 * 4 bytes for data
- Example: 5000 samples * 168 * 168 * 3 * 32 bits / 8 bits/byte  = 1.69344 GB per shard

### `_is_host_oom_error(exc: BaseException)`

Return True for common CPU/host-memory OOM failures.

### `_is_map_worker_crash_error(exc: BaseException)`

Return True when Hugging Face map multiprocessing workers crash.

## `data.data_preprocessing.preprocess_development_dataset`

Source: `src/data/data_preprocessing/preprocess_development_dataset.py`

### `load_trajectory_fast(tar_path, progress_id=None)`

Modified to show progress for individual trajectories

### `process_single_trajectory(tar_path, progress_id)`

Modified to accept progress ID

### `process_parallel(raw_data_path, max_workers=2)`

Process individual tensors with progress bar

### `pad_and_size(tensor, target_size, pad_value=0)`

Pad tensor
:param tensor: Tensor of shape (frames, nr_atoms, 3)
:param target_size: int, target number
:param pad_value: value to use for padding
:return: padded tensor and the nr of real atoms before padding (size)

### `merge_tensors(tensor_dict)`

Merge a list of tensors into a single tensor by concatenation along the first dimension.
:param tensor_dict: dict of {class_name: [tensors]}
:return: a tensor containing the whole dataset, labelled by class_name

### `classify_tensors(trajectory_tensors, filenames)`

Classify tensors based on filename substrings.
:param trajectory_tensors:

### `create_torch_dataset_from_tensors(classified_tensors)`

Create PyTorch datasets from classified tensors.
:param classified_tensors: dict of {class_name: merged_tensor}
:return: tensor dataset

### `save_dataset(dataset, dataset_name, dataset_folder='./data/dataset/tensors', chunk_size=10000)`

Save as separate chunk files to avoid OOM

### `save_datasets(train_dataset, val_dataset, test_dataset, size_ratio)`

Save datasets to folders

### `save_classified_tensors(classified_tensors, dataset_folder='./data/dataset/tensors/raw_classified', chunk_size=10000)`

Save classified tensors separately. Uses chunking to avoid OOM.
The folder structure will be: /dataset_folder/enzyme/binding_type/chunk_0.safetensors
:param classified_tensors: dict of {enzyme:{binding_type: [tensors]}}
:param dataset_folder: folder to save datasets
:param chunk_size: chunk size for saving

## `data.data_preprocessing.preprocess_external_dataset`

Source: `src/data/data_preprocessing/preprocess_external_dataset.py`

### `parse_pdb_streaming_many_tars(tar_paths)`

Yield frame records from a list of TAR files.

When `tar_paths` is provided as a list in `gen_kwargs`, Hugging Face can shard
the workload across processes.

### `generate_unique_test_runs_dataset(tar_folder, output_root, num_proc=1, max_shard_size='4GB')`

Create one Hugging Face dataset split with all frames from all .tar.gz files.

## `data.data_preprocessing.upload_dataset`

Source: `src/data/data_preprocessing/upload_dataset.py`

Upload Dataset to Huggingface Hub

### `upload_dataset_to_huggingface(dataset_path: str, repo_name: str, hf_token: str, organization: str=None)`

Uploads a dataset stored on disk to the Huggingface Hub.

Args:
    dataset_path (str): Path to the dataset directory on disk.
    repo_name (str): Name of the repository to create on Huggingface Hub.
    hf_token (str): Huggingface authentication token.
    organization (str, optional): Organization name if uploading to an organization. Defaults to None.

## `data.data_preprocessing.utils`

Source: `src/data/data_preprocessing/utils.py`

### `extract_coordinates(pdb_file, pdb_id)`

Extract 3D coordinates from PDB file object.

### `construct_pdb_record(pdb_id, dpp_class, ligand_name, binding_type, coords, replica_id)`

Construct a Hugging Face-style dictionary for a single PDB record.

### `parse_pdb_streaming(tar, frames, dpp_class, ligand_name, binding_type, replica_id)`

Yield one dataset record per PDB file stored inside a TAR archive.

The generator does not unpack the archive to disk. Instead, it walks over the
TAR members, opens each ``.pdb`` file in memory, extracts the atomic coordinates,
and emits a Hugging Face-style dictionary that can be consumed by
``Dataset.from_generator``.

Each yielded row keeps the experiment metadata together with the parsed
coordinates so the downstream dataset still knows which trajectory and
ligand/binding setup the frame came from.

### `extract_pdb_files_from_directory(pdb_dir: str, dpp_class: str | None=None, ligand_name: str | None=None, binding_type: str | None=None, replica_id: int | None=None)`

Extract PDB files from a directory and generate preprocessed data samples.
:param pdb_dir: Path to the directory containing PDB files.
:param dpp_class: DPP class. Optional
:param ligand_name: Name of the ligand. Optional
:param binding_type: Type of binding. Optional
:param replica_id: ID of the replica. Optional
:return: List of preprocessed data samples ready for interpretability analysis.
Each sample is a dictionary containing {'coordinates': list, 'binding_type': str, 'dpp_class': str, 'ligand_name': str, 'replica_id': int, num_atoms: int}

### `extract_pdb_file(pdb_file: str, ligand_name: str | None=None, binding_type: str | None=None, dpp_class: str | None=None, replica_id: int | None=None)`

Generate the preprocessed data sample from a PDB file.
Preprocess the PDB file and extract relevant features
:param pdb_file: Path to the PDB file to be processed.
:param ligand_name: Name of the ligand. Optional
:param binding_type: Type of binding. Optional
:param dpp_class: DPP class. Optional
:param replica_id: ID of the replica. Optional
:return: Preprocessed data sample ready for interpretability analysis. 
This is a sample containing {'coordinates': list, 'binding_type': str, 'dpp_class': str, 'ligand_name': str, 'replica_id': int, num_atoms: int}

## `experiments.interpretability`

Source: `src/experiments/interpretability.py`

### `_extract_class_name(class_value)`

Extract a clean class name from object, string, or repr-like value.

### `_load_config_from_artifacts(config_path: str | None, checkpoint_path: str)`

Load TrainingConfig from explicit path or common files next to a checkpoint.

### `_resolve_model_class(config: TrainingConfig)`

Resolve model class stored in config to a callable class object.

### `_load_state_dict(checkpoint_path: str, device: torch.device)`

Load checkpoint and return a plain state dict expected by model.load_state_dict.

### `_load_weights(model: torch.nn.Module, state_dict: dict)`

Load model weights, including checkpoints saved from torch.compile wrappers.

### `_to_display_image(sample: torch.Tensor)`

Convert tensor to normalized HWC image format suitable for matplotlib.

### `_save_heatmaps(inputs: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, attribution_map: torch.Tensor, class_names: list[str], method: str, output_dir: str, start_index: int=0, dpp_labels: list[str | None] | None=None)`

Save per-sample visualizations: input, heatmap, and overlay.

### `_save_average_heatmap(heatmap: torch.Tensor, title: str, output_path: str)`

Save one aggregated heatmap image (global or per-class average).

### `_resolve_device(selection: str)`

Resolve device selection from CLI and validate CUDA availability.

### `_normalize_dpp_label(raw_value)`

Map dpp_class values to canonical labels ('dpp8' or 'dpp9').

### `_prepare_inputs_for_captum(sample: torch.Tensor)`

Ensure sample tensor is in the right shape and on the right device for Captum.

### `main()`

Run Captum methods on a checkpoint and export attribution visualizations.

## `experiments.profiler`

Source: `src/experiments/profiler.py`

Profile training steps and export a Chrome trace for a configured model.

### `profile_model(model, dataloader, criterion, optimizer)`

Warm up a model, profile ten training steps, and save the trace to disk.

## `infrastructure.logger`

Source: `src/infrastructure/logger.py`

### `init_logger(log_mode, model_dir, log_file)`

Initialize the global logger (call once at startup)

### `get_logger()`

Get the configured logger from anywhere

## `models.DCNN`

Source: `src/models/DCNN.py`

Dense convolutional classifier for RGB molecular-coordinate images.

### `_DenseLayer`

DenseNet bottleneck layer that appends ``growth_rate`` channels.

### `_DenseBlock`

Sequence of densely connected convolutional layers.

### `_Transition`

Reduce channels and spatial resolution between dense blocks.

### `DCNN`

DenseNet-style classifier accepting ``[B, 3, 168, 168]`` images.

The forward pass returns unnormalized class logits with shape
``[B, num_classes]``. ``block_config`` controls the number of layers in
each dense block and ``reduction_ratio`` controls transition compression.

#### `forward(self, x)`

Return class logits for a batch of RGB coordinate images.

#### `input_shape(self)`

Return the expected per-sample input shape ``(3, 168, 168)``.

### `create_custom_densenet(num_classes=5)`

Construct the project’s default DCNN configuration.

## `models.LongSequenceAtomTransformer`

Source: `src/models/LongSequenceAtomTransformer.py`

Transformer classifier for long sequences of molecular atom coordinates.

### `RotaryEmbedding`

Generate sinusoidal rotary embeddings for attention head positions.

### `RMSNorm`

Root-mean-square normalization used by transformer blocks.

### `SwiGLU`

Gated feed-forward projection using the SiLU activation.

### `LinearAttention`

Linear-complexity attention approximation for long sequences.

#### `forward(self, x)`

x: (B, N, D)

### `SPDAAttention`

Scaled dot-product multi-head attention using PyTorch SDPA kernels.

#### `forward(self, x)`

x:
    [B, N, D]

### `TransformerBlock`

Pre-normalized attention and SwiGLU residual block.

### `LongSequenceAtomTransformer`

Classify normalized atom sequences with masked attention pooling.

Inputs have shape ``[B, N, 3]`` and outputs have shape
``[B, num_classes]``. Zero coordinate rows are treated as padding and are
excluded from the learned pooling weights.

#### `forward(self, inputs: torch.Tensor)`

inputs:
    Tensor of shape [B, N, 3]
    where:
        B = batch size
        N = number of atoms
        3 = (x, y, z)

returns:
    Tensor of shape [B, num_classes]

## `models.OneLayer`

Source: `src/models/OneLayer.py`

### `OneLayerNet`


#### `__init__(self, input_size, nr_neurons, output_size, dropout_rate=0.5)`

Initialize the network architecture.

Args:
    input_size: Number of input features (flattened image size)
    nr_neurons: Number of neurons in the hidden layer
    output_size: Number of output classes
    dropout_rate: Probability of dropping neurons (default 0.5 = 50%)

## `models.XGBoostImageClassifier`

Source: `src/models/XGBoostImageClassifier.py`

GPU-backed XGBoost classifier for flattened molecular image features.

### `_resolve_xgboost()`

Import XGBoost lazily so the rest of the package remains importable.

### `_get_xgboost_version()`

Return the installed XGBoost version as a three-part tuple.

### `_build_backend_params()`

Select GPU parameter names compatible with the installed XGBoost version.

### `_flatten_images(images: torch.Tensor, pooled_size: int)`

Pool image batches and flatten them into CPU NumPy feature rows.

### `_collect_to_memmap(dataloader, work_dir: str | Path, split_name: str, pooled_size: int, max_batches: int | None=None, scramble: bool=False)`

Stream a dataloader into temporary feature and label memmaps.

### `XGBoostMetrics`

Classification metrics returned by dataloader evaluation.

### `XGBoostImageClassifier`

Train an XGBoost multiclass model on pooled RGB image features.

The implementation requires CUDA, uses temporary memmaps for dataloader
conversion, and supports version-specific XGBoost GPU backends.

#### `__init__(self, num_classes: int=5, pooled_size: int=32, **xgb_params: Any)`

Initialize a GPU classifier with optional XGBoost parameter overrides.

#### `fit(self, X: np.ndarray, y: np.ndarray, eval_set=None, verbose: bool=True)`

Fit the classifier on a feature matrix and integer class labels.

#### `fit_from_dataloader(self, trainloader, validationloader=None, max_train_batches: int | None=None, max_validation_batches: int | None=None)`

Convert loaders to temporary features and fit with optional validation.

#### `predict(self, X: np.ndarray)`

Predict integer class labels for a feature matrix.

#### `predict_from_dataloader(self, dataloader, max_batches: int | None=None)`

Return predictions and labels after converting a dataloader.

#### `evaluate_from_dataloader(self, dataloader, max_batches: int | None=None)`

Calculate weighted classification metrics from a dataloader.

## `models.custom_model_template`

Source: `src/models/custom_model_template.py`

### `AbstractNNModel`


#### `get_vram_usage(self)`

:return: VRAM usage of the model parameters in bytes

## `training.DataLoader`

Source: `src/training/DataLoader.py`

### `save_images_as_safetensors_separate_chunks(dataset_folder, output_folder='tensor_cache', chunk_size=10000)`

Save as separate chunk files to avoid OOM

### `MultiChunkSafeTensorDataset`

Dataset that loads from multiple safetensor chunk files

#### `_load_chunk(self, chunk_idx)`

Load a specific chunk file

### `load_dataset_from_safetensors_multichunk(tensor_folder='tensor_cache', batch_size=16, device='cpu', num_workers=4)`

Load dataset from multiple chunk files

### `load_validation_from_safetensors_multichunk(tensor_folder='tensor_cache', batch_size=16, device='cpu', num_workers=4)`

Load validation dataset from multiple chunk files

## `training.LabelEncoder`

Source: `src/training/LabelEncoder.py`

### `LabelEncoder`


#### `encode_labels(self, labels)`

Encode labels to integers. Works with lists or numpy arrays.

## `training.Metrics`

Source: `src/training/Metrics.py`

### `Metrics`


#### `save_plot(self, title, filename)`

:param metrics: dict: {"accuracy":[values], "loss":[values]}
:return:

## `training.batch_preprocessing`

Source: `src/training/batch_preprocessing.py`

Convert molecular coordinate batches into model-ready tensors.

Image models receive ``[B, 3, 168, 168]`` tensors, while sequence models
receive normalized ``[B, T, 3]`` coordinate sequences. Both paths preserve
the valid-atom count so padded coordinates do not affect normalization.

### `_labels_to_tensor(labels, device)`

Convert numeric or string labels to a ``torch.long`` device tensor.

### `_coords_to_tensor(batch_data, device)`

Convert supported collated coordinate formats to ``[B, T, 3]``.

### `_num_atoms_to_tensor(num_atoms, coords, device)`

Return one valid-atom count per sample, defaulting to sequence length.

### `_make_generator(seed, device)`

Create a device-local deterministic generator when ``seed`` is provided.

### `_rodrigues_rotation_matrix(vector, random_vector, device)`

Build a 3D rotation matrix aligning one unit vector with another.

### `_scramble_in_place(coords, num_atoms, diameter=140.0, seed=None)`

Randomly rotate and translate valid atoms while leaving padding unchanged.

### `_coords_to_rgb(coords, num_atoms)`

Normalize coordinates and reshape them into RGB image tensors.

### `_coords_to_sequence(coords, num_atoms)`

Normalize coordinates per sample while retaining ``[B, T, 3]`` layout.

### `prepare_model_batch(batch, device, scramble=False, log_every_steps=0, seed=None)`

Prepare one collated batch for image models.

Returns ``(images, labels)`` where images have shape ``[B, 3, 168, 168]``.
When enabled, ``scramble`` applies deterministic geometric augmentation
controlled by ``seed``. Optional timing is emitted every ``log_every_steps``
calls.

### `prepare_sequence_batch(batch, device, scramble=False, log_every_steps=0, seed=None)`

Prepare packed coordinates for sequence models as ``[B, T, 3]`` tensors.

## `training.metric_functions`

Source: `src/training/metric_functions.py`

### `calculate_fpr_multiclass(conf_matrix)`

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

### `calculate_statistics(model, dataloader, criterion, device, max_batches: int | None=None, statistics_to_compute: list[str] | None=None, batch_preparation_fn=prepare_model_batch, use_mixed_precision=False, amp_dtype=torch.bfloat16)`

Calculate specified statistics for a model on a given dataloader. Loop over the dataloader and compute the specified statistics for each batch, then average them over the entire dataloader.
Args:
    model (_type_): _model to evaluate
    dataloader (_type_): dataloader to evaluate on
    criterion (_type_): loss function to use for loss calculation
    device (_type_): device to use for evaluation
    max_batches (int | None, optional): maximum number of batches to evaluate. Defaults to None.
    statistics_to_compute (list[str] | None, optional): list of statistics to compute. Defaults to None.

Returns:
    dict[str, float]: dictionary containing the computed statistics

## `training.model_testing`

Source: `src/training/model_testing.py`

### `model_testing(model, testloader, criterion, device, output_dir, max_batches=None, split_name='test', batch_preparation_fn=None, use_mixed_precision=False, amp_dtype=torch.bfloat16)`

Test a model on a given test loader and compute various statistics.
Args:
    model: The PyTorch model to be tested.
    testloader: DataLoader for the test dataset.
    criterion: Loss function to compute the loss.
    device: Device to run the model on (e.g., 'cuda' or 'cpu').
    output_dir: Directory to save the results and statistics.
    max_batches: Maximum number of batches to process. If None, process all batches.
    split_name: Name of the data split (e.g., 'test', 'validation').
    batch_preparation_fn: Optional function to prepare batches before feeding them to the model.
    use_mixed_precision: Whether to use mixed precision for inference.
    amp_dtype: The data type for automatic mixed precision (e.g., torch.float16, torch.bfloat16).

## `training.train_test_split`

Source: `src/training/train_test_split.py`

Dataset splitting and image-folder export helpers.

### `split_tensor(tensor, split_ratio, shuffle)`

Split a tensor into two parts based on the split ratio
:param tensor: Input tensor
:param split_ratio: Ratio to split the tensor
:return: part1, part2 - two tensors

### `create_ligand_splits_1_train_1_valtest(classified_tensors, random_seed=42)`

Create splits such that data from 1 ligand is used for training and data from the other ligand is used for validation and testing
 (50% val, 50% test)
:param classified_tensors: {class_name: [tensors]}
:param random_seed: Random seed for reproducibility
:return: train_dataset, val_dataset, test_dataset - classified_tensors {class_name: [tensors]}

### `create_splits_randomsplit(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)`

Use PyTorch's random_split to split all frames randomly
:param dataset:pytorch Dataset containing all trajectory frames and labels
:param train_ratio: Ratio of training data
:param val_ratio: Ratio of validation data
:param test_ratio: Ratio of test data
:param random_seed: Random seed for reproducibility
:return: train_dataset, val_dataset, test_dataset - Dataset  containing the split data and labels

### `copy_dataset(dataset, split_path)`

Copy a dataset split into class-named directories under ``split_path``.

### `save_image_datasets_to_folders(train_dataset, val_dataset, test_dataset, output_dir)`

Export train, validation, and test image splits into separate folders.

### `train_val_test_split()`

Split the data into train, validation and test sets
:return:

## `training.training_setup`

Source: `src/training/training_setup.py`

### `_is_cuda_oom_error(exc: BaseException)`

Return True when exception indicates a CUDA OOM condition.

### `_cleanup_cuda_memory()`

Release cached CUDA memory between retries.

### `save_results(model_state_dict, model_dir: str, model_name: str, metrics)`

Save model, metrics, and test accuracy

### `initialize_run_directory(model_name)`

Create model directory and setup logging

### `train_model(config: TrainingConfig, model_name: str, streaming: bool=False)`

Main training function - single entry point for all models

## `training.utils`

Source: `src/training/utils.py`

Shared training, checkpoint, split, and CUDA prefetch utilities.

### `save_model(model, model_folder, model_name)`

Serialize a complete PyTorch model under ``model_folder``.

### `load_model(model_filepath, device=None, model_class=None, model_args=None)`

Load a full model or reconstruct one from a state-dict checkpoint.

If ``model_class`` is omitted, dictionary checkpoints are returned unchanged.
Reconstructed and full models are moved to ``device`` and set to evaluation
mode.

### `model_name(model_name_prefix)`

Create a timestamped model directory name from a readable prefix.

### `get_device()`

Select CUDA when available, otherwise return CPU with a warning.

### `get_subset(dataset, fraction, shuffle=True, seed=42)`

Return a deterministic fraction of a dataset, optionally shuffled.

### `clear_cache(dataset_dir)`

Remove Hugging Face dataset cache files from a disk-backed dataset.

### `train_val_test_split(dataset, train_fraction=0.7, val_fraction=0.15, seed=42)`

Split a dataset into shuffled ``train``, ``val``, and ``test`` subsets.

### `training_phase(model, trainloader, optimizer, criterion, device)`

Train over a finite dataloader and return aggregate accuracy/loss.

### `_train_single_batch(model, batch, optimizer, criterion, device)`

Train one batch and return correct predictions, sample count and loss.

### `_train_single_batch_prepared(model, inputs, labels, optimizer, criterion)`

Train one already-prepared batch and return correct predictions, sample count and loss.

### `_train_single_batch_prepared_amp(model, inputs, labels, optimizer, criterion, device, use_mixed_precision=False, amp_dtype=torch.bfloat16, grad_scaler=None)`

Train one already-prepared batch with optional CUDA AMP and return stats.

### `CUDABatchPrefetcher`

Prefetch and preprocess the next batch on a dedicated CUDA stream.

### `_safe_len(dataloader)`

Return dataloader length when available, otherwise None.

## `transform.ListScrambler`

Source: `src/transform/ListScrambler.py`

### `ListScrambler`

Optimized list-based scrambler for HuggingFace datasets using NumPy

#### `__call__(self, batch, return_numpy=False, real_nr_atoms=None)`

Apply scrambling to a batch efficiently
Args:
    batch: list of arrays or numpy array of shape (batch, n_atoms, 3)
    return_numpy: if True, return list of numpy arrays (faster); if False, return list of lists
    real_nr_atoms: optional per-sample atom counts. If provided, only the first
        ``real_nr_atoms[i]`` rows of each sample are scrambled and any padded
        rows after that are left untouched.
Returns:
    List of scrambled frames (as numpy arrays if return_numpy=True, else as lists)

#### `_scramble_single_frame(self, frame, n_real_atoms=None)`

Scramble only the real atom prefix of one frame and preserve padding rows.

#### `_oriental_scramble_single(self, frame)`

Orientation scrambling for a single frame
Args:
    frame: np.ndarray of shape (n_atoms, 3)
Returns:
    Scrambled frame

#### `_compute_rotation_matrix_single(self, vector, random_vector)`

Compute rotation matrix using Rodrigues' formula (single frame)
Args:
    vector: np.ndarray of shape (3,)
    random_vector: np.ndarray of shape (3,)
Returns:
    rotation_matrix: np.ndarray of shape (3, 3)

#### `_positional_scramble_vectorized(self, frames)`

Positional scrambling for a single frame
Args:
    frame: np.ndarray of shape (n_atoms, 3)
Returns:
    Translated frame

### `ScramblingTransform`

HuggingFace-compatible transform with caching

#### `__call__(self, batch)`

Apply scrambling to a batch from HuggingFace datasets

## `transform.Padder`

Source: `src/transform/Padder.py`

### `Padder`


#### `__call__(self, data_examples)`

Pad entries to target_size x target_size efficiently using numpy.

#### `reapply_padding(self, examples_data, real_nr_atoms)`

Reapply padding to a batch of examples based on their real number of atoms

## `transform.Scrambler`

Source: `src/transform/Scrambler.py`

### `Scrambler`


#### `scramble(self, frame: ndarray, diameter: float)`

Scramble both the position and orientation of each frame

#### `oriental_scramble(self, frame: ndarray)`

Scramble the orientation of each frame

#### `_scramble_orientation(frame, random_vector)`

scramble the orientation of each frame (randomly)
by aligning a vector defined by two arbitrarily chosen atoms (preferably on an axis connecting the intracellular and extracellular ends of the GPCR) to a random unit vector in spheri
:param frame: np.ndarray, shape=(n_atoms, 3)
    A two dimensional numpy array, with the cartesian coordinates of each atoms.
:return:

#### `positional_scramble(self, frame: ndarray, diameter: float)`

Scramble the position of each frame

#### `_scramble_position(frame, random_point: ndarray)`

scramble the position of each frame (randomly)
the positional scrambling of frames from a trajectory by moving the centers of mass
to a randomly sampled coordinate within a sphere of diameter with the size of the largest dimension of the receptor
:param frame: np.ndarray, shape=(n_atoms, 3)
    A two dimensional numpy array, with the cartesian coordinates of each atoms.
:return:

## `transform.TorchScrambler`

Source: `src/transform/TorchScrambler.py`

Torch geometric augmentation for molecular coordinate frames.

### `TorchScrambler`

Randomly rotate and translate single frames or batches of frames.

Inputs use ``[N, 3]`` or ``[B, N, 3]`` layout. Rotation aligns the vector
between the first two atoms with a random direction; translation moves the
frame center to a random point inside the configured diameter.

#### `__init__(self, diameter: float, device='cpu', seed=42)`

Configure the translation volume, device, and global random seed.

#### `__call__(self, frame: torch.Tensor, batched)`

Apply rotation and translation while preserving the input rank.
Args:
    frame: list of shape (n_atoms, 3) or (batch, n_atoms, 3)
Args:
    batched: Retained for transform-pipeline compatibility; rank is inferred
        from ``frame``.
Returns:
    Scrambled frame with the same shape as the input.

#### `oriental_scramble(self, frame: torch.Tensor)`

Scramble orientation of a single frame
Args:
    frame: torch.Tensor of shape (n_atoms, 3)

#### `oriental_scramble_batch(self, frames: torch.Tensor)`

Scramble orientation of a batch of frames
Args:
    frames: torch.Tensor of shape (batch, n_atoms, 3)

#### `_compute_rotation_matrix(self, vector: torch.Tensor, random_vector: torch.Tensor)`

Compute rotation matrix using axis-angle representation
Args:
    vector: torch.Tensor of shape (3,)
    random_vector: torch.Tensor of shape (3,)
Returns:
    rotation_matrix: torch.Tensor of shape (3, 3)

#### `_compute_rotation_matrix_batch(self, vectors: torch.Tensor, random_vectors: torch.Tensor)`

Compute rotation matrices for a batch
Args:
    vectors: torch.Tensor of shape (batch, 3)
    random_vectors: torch.Tensor of shape (batch, 3)
Returns:
    rotation_matrices: torch.Tensor of shape (batch, 3, 3)

#### `positional_scramble(self, frame: torch.Tensor)`

Scramble position of a single frame
Args:
    frame: torch.Tensor of shape (n_atoms, 3)

#### `positional_scramble_batch(self, frames: torch.Tensor)`

Scramble position of a batch of frames
Args:
    frames: torch.Tensor of shape (batch, n_atoms, 3)

### `ScramblingTransform`

Small transform wrapper exposing a configured ``TorchScrambler``.

## `transform.XYZToRGBTensor`

Source: `src/transform/XYZToRGBTensor.py`

Convert padded XYZ coordinate arrays into channel-first RGB tensors.

### `XYZToRGBTensor`

Normalize XYZ coordinates into ``[C, H, W]`` image representations.

Only real atoms participate in min/max normalization; padded atoms become
black. A list input returns a list of images, while an array input returns a
stacked NumPy array.

#### `__init__(self, target_size=168)`

Set the square image edge length used for reshaping coordinates.

#### `__call__(self, coords_batch, num_real_atoms_batch=None)`

Args:
    coords_batch: List or array of padded ``[N, 3]`` coordinate arrays.
    num_real_atoms_batch: Optional count of real atoms per sample.
Returns:
    List or array of float32 images with shape ``[3, target_size, target_size]``.

## `transform.sequence_transforms`

Source: `src/transform/sequence_transforms.py`

Preprocessing transforms for padded atom-coordinate sequences.

### `coordinate_normalization(inputs)`

Center and standardize each coordinate sequence independently.

### `_get_sequence_transforms()`

Lazily construct and reuse sequence scrambler, padder, and encoder.

### `apply_sequence_transform(examples_data, examples_labels, _real_nr_atoms)`

Pad, scramble, normalize, and encode a batch of sequence examples.

### `apply_sequence_transform_noscramble(examples_data, examples_labels, _real_nr_atoms)`

Pad and encode sequence examples without geometric augmentation.

## `transform.tranformators`

Source: `src/transform/tranformators.py`

### `_get_transforms()`

Lazy-load singleton transform instances

### `apply_image_transform(examples_data, examples_labels, real_nr_atoms)`

Apply transform to batch: pad → scramble → normalize → RGB tensor. Works with numpy arrays.

### `apply_image_transform_noscramble(examples_data, examples_labels, real_nr_atoms)`

Apply transform without scrambling: pad → normalize → RGB tensor.

## `utils.interpretability`

Source: `src/utils/interpretability.py`

### `solve_methods(interpreter: CaptumInterpreter, method_args)`

Resolve a method string to the corresponding interpretability method.
:param method_str: String identifier for the interpretability method (e.g., "integrated_gradients").
:return: Corresponding interpretability method object.

### `AttributionResult`

Container for attribution outputs returned by Captum methods.

### `CaptumInterpreter`

Utility wrapper that adds Captum interpretability methods to any PyTorch model.

#### `summarize_attributions(attributions: torch.Tensor, reduce_dim: int | tuple[int, ...] | None=1, normalize: bool=False)`

Reduce and optionally normalize attribution magnitudes for plotting or inspection.

## `utils.labels`

Source: `src/utils/labels.py`

Binding-class vocabulary and ligand-to-class mapping.

### `get_binding_classes()`

Return the binding classes in the order expected by the classifiers.

### `ligant_to_class(ligand_name: str | None)`

Map a ligand identifier to its binding class.

Args:
        ligand_name: Known ligand identifier, or ``None`` for an apo sample.

Returns:
        The canonical binding-class label.

Raises:
        ValueError: If ``ligand_name`` is not in ``ligand_to_binding_type``.

## `utils.molecular_parsing`

Source: `src/utils/molecular_parsing.py`

Parsing helpers for PDB content and trajectory filenames.

### `parse_pdb_from_string(pdb_content: str)`

Extract atom coordinates from PDB text as an ``[N, 3]`` tensor.

Malformed coordinate fields are skipped. If no valid atom records are found,
the returned tensor is empty.

### `parse_filename(filename: str)`

Parse experiment metadata encoded in a trajectory filename.

The expected shape is ``..._<dpp_class>_<ligand>_replica<id>`` with an
optional ``correct`` marker. The returned tuple contains DPP class, ligand,
binding class, and replica ID.

Raises:
        ValueError: If any encoded field is not recognized or valid.

### `remove_extension(filename: str)`

Remove only the final extension from a filename.

## `utils.utils`

Source: `src/utils/utils.py`

Backward-compatible imports for helpers moved into focused modules.
