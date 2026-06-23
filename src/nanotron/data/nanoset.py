import hashlib
import json
import os
import warnings
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datatrove.utils.dataset import DatatroveFolderDataset
from numba import jit
from tqdm import tqdm

from nanotron import logging
from nanotron.data.utils import count_dataset_indexes, normalize
from nanotron.logging import log_rank

logger = logging.get_logger(__name__)

# datatrove's DatatroveFolderDataset default shuffle seed. Token and reference-rank sidecar
# datasets must share it so their file-order permutations match index-for-index.
_DATATROVE_SHUFFLE_SEED = 42
_REFERENCE_RANK_CHECKSUM_MOD = 65521  # largest prime < 2**16; fits the uint16 sentinel slot


class SparseReferenceRankDataset:
    """Sparse offline teacher-rank sidecar keyed by Nanoset's actual_sample index."""

    def __init__(self, folder: str, sequence_length: int) -> None:
        self.folder = folder
        self.sequence_length = sequence_length
        self.width = sequence_length + 1
        meta_files = sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.startswith("_rank_sidecar.") and name.endswith(".metadata")
        )
        if not meta_files:
            raise ValueError(f"No sparse rank sidecar metadata found in {folder}")

        self.indices = {}
        self.ranks = {}
        self.num_shards = None
        for meta_file in meta_files:
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("layout") != "nanoset_sparse":
                raise ValueError(f"{meta_file} is not a nanoset_sparse rank sidecar")
            if int(meta["seq_len"]) != sequence_length:
                raise ValueError(
                    f"Sparse rank sidecar seq_len mismatch in {meta_file}: "
                    f"{meta['seq_len']} != {sequence_length}"
                )

            num_shards = int(meta["num_shards"])
            shard_index = int(meta["shard_index"])
            if self.num_shards is None:
                self.num_shards = num_shards
            elif self.num_shards != num_shards:
                raise ValueError(
                    f"Sparse rank sidecar num_shards mismatch in {meta_file}: "
                    f"{num_shards} != {self.num_shards}"
                )
            if shard_index in self.indices:
                raise ValueError(f"Duplicate sparse rank sidecar shard {shard_index} in {folder}")

            sample_indices_path = os.path.join(folder, meta["sample_indices_file"])
            ranks_path = os.path.join(folder, meta["ranks_file"])
            sample_indices = np.load(sample_indices_path, mmap_mode="r")
            num_samples = int(meta["num_samples"])
            if len(sample_indices) != num_samples:
                raise ValueError(
                    f"Sparse rank sidecar sample count mismatch in {meta_file}: "
                    f"{len(sample_indices)} != {num_samples}"
                )
            if num_samples:
                ranks = np.memmap(
                    ranks_path,
                    dtype=np.uint16,
                    mode="r",
                    shape=(num_samples, self.width),
                )
            else:
                ranks = np.empty((0, self.width), dtype=np.uint16)
            self.indices[shard_index] = sample_indices
            self.ranks[shard_index] = ranks

        if self.num_shards is None:
            raise ValueError(f"No sparse rank sidecar shards loaded from {folder}")
        missing = sorted(set(range(self.num_shards)) - set(self.indices))
        if missing:
            raise ValueError(f"Sparse rank sidecar folder {folder} is missing shards: {missing}")

    def __getitem__(self, actual_sample: int) -> np.ndarray:
        shard_index = int(actual_sample) % self.num_shards
        sample_indices = self.indices[shard_index]
        row = int(np.searchsorted(sample_indices, actual_sample))
        if row >= len(sample_indices) or int(sample_indices[row]) != int(actual_sample):
            raise KeyError(
                f"Sparse rank sidecar {self.folder} does not contain actual_sample={actual_sample} "
                f"in shard {shard_index}"
            )
        return self.ranks[shard_index][row]


def reference_rank_checksum(token_window) -> int:
    """16-bit fingerprint of a token window (input_ids, length seq_len+1).

    The offline rank-sidecar generator stores this in each window's sentinel slot
    (index 0, dropped by the collator's [:, 1:] shift); Nanoset re-validates it at read
    time to catch token/sidecar misalignment. MUST stay identical in the generator.
    """
    n = len(token_window)
    a = int(token_window[0])
    b = int(token_window[n // 2])
    c = int(token_window[-1])
    # 1000003 (a large prime) and 9176 are arbitrary fixed mixers; any fixed pair works. This is
    # an alignment tripwire, not a cryptographic hash.
    return (a * 1000003 + b * 9176 + c + n) % _REFERENCE_RANK_CHECKSUM_MOD


class Nanoset(torch.utils.data.Dataset):
    """
    The Nanoset dataset

    Args:
        dataset_folders (List[str]): List of folders with tokenized datasets
        dataset_weights (Union[List[float], None]): List with the weights for weighted datasets. If None, consume all samples from all datasets without weighting. Weights are normalized in __init__
        sequence_length (int): Sequence length of the built samples
        vocab_size (int): Vocab size of the tokenizer used to tokenize the dataset
        train_split_num_samples (int): Number of samples the dataset needs. It's the training steps * global batch size
    """

    def __init__(
        self,
        dataset_folders: List[str],
        sequence_length: int,
        token_size: int,
        train_split_num_samples: int,
        dataset_weights: Union[List[float], None] = None,
        random_seed: int = 1234,
        use_cache: bool = True,
        eos_token_id: int = None,
        return_positions: bool = True,
        rank_dataset_folders: Union[List[str], None] = None,
    ) -> None:
        # Checks
        if isinstance(dataset_folders, str):
            warnings.warn("dataset_folders should be of type List[str] but str was provided. Converting to List[str]")
            dataset_folders = [dataset_folders]

        # Init
        self.dataset_folders = dataset_folders
        self.sequence_length = sequence_length
        self.eos_token_id = eos_token_id
        self.return_positions = return_positions
        assert self.return_positions or self.eos_token_id is not None, (
            "If return_positions is True, eos_token_id must be defined"
        )
        # Number of bytes for the tokens stored in the processed dataset files. 2 for vocab sizes < 65535, 4 otherwise
        self.token_size = token_size
        self.train_split_num_samples = train_split_num_samples
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.cache_dir = "./.nanoset_cache"
        self.rank_dataset_folders = rank_dataset_folders
        self.datatrove_datasets = [
            self._build_datatrove_dataset(folder, token_size=self.token_size, return_positions=self.return_positions)
            for folder in self.dataset_folders
        ]
        # Offline reference-model top-k masking: a parallel sidecar of per-token teacher ranks
        # (uint16 .ds, token_size=2), one folder per token folder, read through the SAME builder
        # so datatrove's file-order permutation matches the token datasets index-for-index.
        self.rank_datasets = None
        self.rank_datasets_are_sparse = False
        if rank_dataset_folders is not None:
            assert len(rank_dataset_folders) == len(self.dataset_folders), (
                f"Got {len(rank_dataset_folders)} rank_dataset_folders for "
                f"{len(self.dataset_folders)} dataset_folders; they must be parallel."
            )
            sparse_flags = [self._rank_folder_is_sparse(folder) for folder in rank_dataset_folders]
            if any(sparse_flags) and not all(sparse_flags):
                raise ValueError("rank_dataset_folders must be all dense sidecars or all nanoset_sparse sidecars")
            self.rank_datasets_are_sparse = all(sparse_flags)
            if self.rank_datasets_are_sparse:
                self.rank_datasets = [
                    SparseReferenceRankDataset(folder, sequence_length=self.sequence_length)
                    for folder in rank_dataset_folders
                ]
            else:
                self.rank_datasets = [
                    self._build_datatrove_dataset(folder, token_size=2, return_positions=False)
                    for folder in rank_dataset_folders
                ]
                for i, (tok_ds, rank_ds) in enumerate(zip(self.datatrove_datasets, self.rank_datasets)):
                    # Per-file (basename, window-count) must match so [actual_sample] aligns; a total-count
                    # check alone would miss a partially-regenerated/failed shard.
                    tok_files = [(os.path.basename(f.file_path), len(f)) for f in tok_ds.files]
                    rank_files = [(os.path.basename(f.file_path), len(f)) for f in rank_ds.files]
                    assert rank_files == tok_files, (
                        f"reference-rank sidecar folder {rank_dataset_folders[i]} does not match token folder "
                        f"{self.dataset_folders[i]} file-by-file (basename, window count)."
                    )

        # Build Nanoset Index
        ## To build the index we need the length of each dataset
        self.dataset_lengths = [len(datatrove_dataset) for datatrove_dataset in self.datatrove_datasets]
        ## Set dataset weights
        if (
            dataset_weights is None
        ):  # Case of training with > 1 datasets without weighting them: Consume both datasets entirely on each epoch
            self.dataset_weights = normalize(self.dataset_lengths)
        else:
            self.dataset_weights = normalize(dataset_weights)
        assert len(dataset_folders) == len(self.dataset_weights), (
            f"Specified {len(self.dataset_weights)} weights but {len(dataset_folders)} datasets were provided."
        )
        ## Build dataset index and dataset sample index
        self.dataset_index, self.dataset_sample_index = self.build_nanoset_index()
        # self.dataset_index, self.dataset_sample_index = self.new_build_nanoset_index() # TODO: Fix this

        self.print_nanoset_info()
        # Initialize consumption tracking
        self.consumed_tokens = dict.fromkeys(range(len(self.datatrove_datasets)), 0)

    def _build_datatrove_dataset(self, folder: str, token_size: int, return_positions: bool) -> DatatroveFolderDataset:
        """Build a DatatroveFolderDataset. Token and reference-rank sidecar datasets are built
        through this single path with the SAME shuffle/seed, so datatrove's file-order
        permutation is identical and per-index reads stay aligned (see reference_rank_checksum).
        """
        return DatatroveFolderDataset(
            data_folder=folder,
            seq_len=self.sequence_length,
            token_size=token_size,
            shuffle=True,
            seed=_DATATROVE_SHUFFLE_SEED,
            return_positions=return_positions,
        )

    @staticmethod
    def _rank_folder_is_sparse(folder: str) -> bool:
        if not os.path.isdir(folder):
            return False
        for name in os.listdir(folder):
            if not (name.startswith("_rank_sidecar.") and name.endswith(".metadata")):
                continue
            with open(os.path.join(folder, name)) as f:
                meta = json.load(f)
            return meta.get("layout") == "nanoset_sparse"
        return False

    def update_consumption_metrics(self, start_idx: int, end_idx: int, sequence_length: int):
        """Update consumed samples/tokens for the current batch.

        Args:
            start_idx: Starting index of current batch for all dp ranks
            end_idx: Ending index of current batch for all dp ranks
            sequence_length: Sequence length for token calculation
        """
        if self.sequence_length is None:
            self.sequence_length = sequence_length

        # Get dataset indices for current batch
        batch_indices = self.dataset_index[start_idx:end_idx]
        unique_indices, counts = np.unique(batch_indices, return_counts=True)

        # Update consumption dictionaries
        for dataset_idx, count in zip(unique_indices, counts):
            self.consumed_tokens[dataset_idx] += int(count * sequence_length)

    def get_consumption_stats(self):
        """Get current consumption statistics for all datasets.

        Returns:
            dict: Dictionary containing samples and tokens consumed per dataset
        """
        stats = {}
        for dataset_idx, dataset in enumerate(self.datatrove_datasets):
            folder_path = dataset.folder_path
            if hasattr(folder_path, "path"):
                folder_path = folder_path.path
            stats[str(folder_path)] = {"tokens": self.consumed_tokens[dataset_idx]}
        return stats

    def __len__(self) -> int:
        """
        Returns:
            int: The number of samples of the Nanoset
        """

        return len(self.dataset_index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset using sequential access
        """
        dataset = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]

        # Get actual sample index by wrapping around dataset length
        actual_sample = sample_idx % self.dataset_lengths[dataset]

        item = self.datatrove_datasets[dataset][actual_sample]
        if self.rank_datasets is not None:
            if self.rank_datasets_are_sparse:
                rank_window = self.rank_datasets[dataset][actual_sample]  # (seq_len+1,)
            else:
                rank_window = self.rank_datasets[dataset][actual_sample]["input_ids"]  # (seq_len+1,)
            # Alignment tripwire: the sidecar's sentinel slot stores a checksum of the token
            # window. A mismatch means tokens and ranks drifted out of lockstep — fail loud.
            expected = reference_rank_checksum(item["input_ids"])
            got = int(rank_window[0])
            assert got == expected, (
                f"reference-rank sidecar misaligned (dataset {dataset}, sample {actual_sample}): "
                f"sentinel {got} != token checksum {expected}."
            )
            item["reference_ranks"] = rank_window
        return item

    def new_build_nanoset_index(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build dataset index that enables sequential reading while respecting weights.
        Uses cache if available and parameters match.
        """
        # Create a cache key based on the parameters that affect the index
        cache_params = {
            "dataset_folders": self.dataset_folders,
            "dataset_lengths": self.dataset_lengths,
            "dataset_weights": self.dataset_weights.tolist(),
            "train_split_num_samples": self.train_split_num_samples,
            "random_seed": self.random_seed,
            "token_size": self.token_size,
            "sequence_length": self.sequence_length,
        }

        # Create a deterministic cache key
        cache_key = hashlib.md5(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"index_{cache_key}.npz")

        # Try to load from cache
        if os.path.exists(cache_file):
            try:
                logger.info(f"[Nanoset] Loading index from cache: {cache_file}")
                cached_data = np.load(cache_file)
                return cached_data["dataset_index"], cached_data["dataset_sample_index"]
            except Exception as e:
                logger.warning(f"[Nanoset] Failed to load cache, rebuilding index: {e}")

        logger.info(f"[Nanoset] Building sequential Nanoset index for {len(self.dataset_folders)} datasets")

        # Original index building logic
        total_weighted_samples = np.array(self.dataset_weights) * self.train_split_num_samples
        samples_per_dataset = np.floor(total_weighted_samples).astype(np.int64)

        remaining = self.train_split_num_samples - samples_per_dataset.sum()
        if remaining > 0:
            fractional_parts = total_weighted_samples - samples_per_dataset
            indices = np.argsort(fractional_parts)[-remaining:]
            samples_per_dataset[indices] += 1

        dataset_positions = np.zeros(len(self.dataset_folders), dtype=np.int64)
        dataset_index = np.zeros(self.train_split_num_samples, dtype=np.int64)
        dataset_sample_index = np.zeros(self.train_split_num_samples, dtype=np.int64)

        dataset_order = np.repeat(np.arange(len(self.dataset_folders)), samples_per_dataset)
        rng = np.random.RandomState(self.random_seed)
        rng.shuffle(dataset_order)

        for idx, dataset_idx in tqdm(enumerate(dataset_order), desc="Building Nanoset index"):
            dataset_index[idx] = dataset_idx
            dataset_sample_index[idx] = dataset_positions[dataset_idx]
            dataset_positions[dataset_idx] += (
                1  # Read samples sequentially from each datatrove_dataset assuming they're already shuffled
            )

        # Save to cache
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.savez(cache_file, dataset_index=dataset_index, dataset_sample_index=dataset_sample_index)
            logger.info(f"[Nanoset] Saved index to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"[Nanoset] Failed to save cache: {e}")

        return dataset_index, dataset_sample_index

    def build_nanoset_index(self) -> np.ndarray:
        """
        Build dataset index and dataset sample index
        """
        # Compute samples per epoch and number of epochs
        samples_per_epoch = sum(self.dataset_lengths)
        num_epochs = int(self.train_split_num_samples / samples_per_epoch) + 1
        # Build the dataset indexes for 1 epoch
        dataset_index, dataset_sample_index = build_nanoset_index_helper(
            n_samples=samples_per_epoch, weights=self.dataset_weights, dataset_sizes=self.dataset_lengths
        )
        # Shuffle the indexes the same way
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_index)
        numpy_random_state = np.random.RandomState(self.random_seed)
        numpy_random_state.shuffle(dataset_sample_index)
        # Concatenate num_epochs the shuffled indexes
        dataset_index = np.concatenate([dataset_index for _ in range(num_epochs)])
        dataset_sample_index = np.concatenate([dataset_sample_index for _ in range(num_epochs)])
        # Just keep the necessary samples
        dataset_index = dataset_index[: self.train_split_num_samples]
        dataset_sample_index = dataset_sample_index[: self.train_split_num_samples]
        return dataset_index, dataset_sample_index

    def print_nanoset_info(self):
        log_rank(f"> Total number of samples: {len(self)}", logger=logger, level=logging.INFO, rank=0)
        log_rank(
            f"> Total number of tokens: {len(self) * self.sequence_length}", logger=logger, level=logging.INFO, rank=0
        )

        # Print samples from each dataset + weight
        dataset_sample_count = count_dataset_indexes(self.dataset_index, len(self.dataset_folders))
        for index, sample_count in enumerate(dataset_sample_count):
            log_rank(
                f">   Total number of samples from the {self.dataset_folders[index]} dataset: {sample_count} ({round(normalize(dataset_sample_count).tolist()[index], 2)})",
                logger=logger,
                level=logging.INFO,
                rank=0,
            )


@jit(nopython=True, cache=True)
def build_nanoset_index_helper(
    n_samples: int, weights: np.ndarray, dataset_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given multiple datasets and a weighting array, build samples indexes
    such that it follows those weights
    """
    # Create empty arrays for dataset indices and dataset sample indices
    dataset_index = np.empty((n_samples,), dtype="uint")
    dataset_sample_index = np.empty((n_samples,), dtype="long")  # Supports dataset with up to 2**64 samples

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros((len(weights),), dtype="long")

    # TODO: Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network

    # Iterate over all samples
    for sample_idx in range(n_samples):
        # Convert sample index to float for comparison against weights
        sample_idx_float = max(sample_idx, 1.0)

        # Find the dataset with the highest error
        errors = weights * sample_idx_float - current_samples
        max_error_index = np.argmax(errors)

        # Assign the dataset index and update the sample index
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index] % dataset_sizes[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    return dataset_index, dataset_sample_index
