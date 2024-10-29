import hashlib
import json
import time
from pathlib import Path
from typing import Generic, List, Optional, Sequence, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

from scaling.core.data.base_dataset import (
    BaseDataset,
    BaseDatasetBatchBeforeSyncGeneric,
    BaseDatasetBatchGeneric,
    BaseDatasetItemGeneric,
)
from scaling.core.data.blended_dataset_config import BlendedDatasetConfig
from scaling.core.logging import logger
from scaling.core.topology import Topology

BaseDatasetGeneric = TypeVar("BaseDatasetGeneric", bound=BaseDataset)


def weights_by_num_docs(examples: list[int], alpha: float = 0.3) -> NDArray:
    """
    Builds weights from a multinomial distribution over groups of data according to the number of
    samples in each group.

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
        |L| is the number of examples in that datapoint,
        and α is a coefficient that acts to upsample data from under-represented groups

    Hence, α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/pdf/1901.07291.pdf and https://arxiv.org/abs/1911.02116 for more details

    Args:
        examples (list): Contains the number of examples in each dataset
        alpha (float): weighting parameter (default=`0.3`)
    Returns:
        weights (np.array): weights to multiply by num docs to get dataset sample size
    """
    examples_array = np.array(examples, np.float64)
    unbiased_sample_probs = examples_array / examples_array.sum()  # p_i

    probs: np.ndarray = unbiased_sample_probs**alpha  # p_i**alpha

    # normalize
    probs = probs / probs.sum()  # q_i

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = 1 / unbiased_sample_probs  # 1/p_i
    weights: np.ndarray = probs * unbiased_sample_probs_inverse  # q_i / p_i

    # normalize
    weights = weights / weights.sum()

    return weights


def weights_examples_proportional(
    examples: list[int], temperature: float = 1.0, maximum: Optional[float] = None
) -> NDArray:
    """
    Work out the weighting of each dataset based on 'temperature' T and 'maximum' parameter K.

    l is the list of dataset sizes.

    Examples-proportional mixing sets a "limit" defined by max rate (in terms of samples).

    The sampling rate of the m'th dataset r_m is:
        r_m = min(e_m, K)/sum_n(min(e_n, K))
    where:
        limit: K,
        number of examples in N datasets: e_n,
        m'th dataset example: e_m,

    This does two things:
        - Limits all datasets larger than defined limit to a fixed equal sampling rate
        - Upsamples datasets smaller than limit K to proportionally higher rate.

    We add an option for temperature scaling (with T=1 equivalent to no scaling).
    This raises r_m to the power of 1/T, and normalizes all the weights. As T increases,
    the weights of proportionally smaller datasets increases (converges to equal sampling,
    but this case should use alpha=0 sampling instead).

    See https://arxiv.org/pdf/1910.10683.pdf (page 31) for more details.

    src: https://github.com/huggingface/datasets/issues/217#issuecomment-648115586

    Args:
        examples (list): Contains the number of examples in each dataset
        temperature (float): weighting parameter (default=`1.0`)
    Returns:
        weights (np.array): weights to multiply by num docs to get dataset sample size
    """
    assert temperature != 0, "temperature is 0, expect non-zero temperature"
    assert temperature is not None, "temperature is None, expect non-zero float"

    examples_array = np.array(examples, np.float64)  # dataset sizes

    unbiased_sample_probs = examples_array / examples_array.sum()  # normalize

    if maximum:  # only when not zero, and not None
        assert maximum > 0, f"examples-proportional sampling requires maximum limit > 0 (current max = {maximum})"
        examples_array[examples_array > maximum] = maximum  # apply limit to all dataset sizes before rate calculation

    max_limited_sample_probs = examples_array / examples_array.sum()  # normalize

    probs: np.ndarray
    if temperature != 1.0:
        probs = max_limited_sample_probs ** (1.0 / temperature)
        probs = probs / probs.sum()  # normalize again
    else:
        probs = max_limited_sample_probs

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = 1 / unbiased_sample_probs  # 1/p_i
    weights: np.ndarray = probs * unbiased_sample_probs_inverse  # q_i / p_i

    # normalize
    weights = weights / weights.sum()

    return weights


class BaseBlendedDataset(
    Generic[
        BaseDatasetItemGeneric,
        BaseDatasetBatchBeforeSyncGeneric,
        BaseDatasetBatchGeneric,
        BaseDatasetGeneric,
    ],
    BaseDataset[
        BaseDatasetItemGeneric,
        BaseDatasetBatchBeforeSyncGeneric,
        BaseDatasetBatchGeneric,
    ],
):
    """
    Torch base dataset class expected to be inherited by all datasets.
    Returns a BaseDatasetItem for each index.
    """

    def __init__(
        self,
        seed: int,
        config: BlendedDatasetConfig,
        datasets: Sequence[BaseDatasetGeneric],
    ) -> None:
        """
        seed (`int`)
            seed used to shuffle the dataset

        """
        self.config = config
        self.datasets: Sequence[BaseDatasetGeneric] = datasets
        self.num_datasets = len(self.datasets)

        # shuffling
        self.seed: Optional[int] = None
        self.set_seed(seed=seed, shuffle=True)

    def ident(self) -> str:
        prefix_str = "-".join([d.ident() for d in self.datasets])

        weight_str = "-".join([str(round(w * 100) / 100) for w in self.weights.tolist()])
        prefix_hash = hashlib.md5(prefix_str.encode("utf-8")).hexdigest()
        weight_hash = hashlib.md5(weight_str.encode("utf-8")).hexdigest()

        return f"{self.datasets[0].__class__.__name__}_prefix_{prefix_hash}_weights_{weight_hash}"

    def get_data_index_cache_filename_stem(self, seed: int) -> str:
        # compute hash combining data prefixes and weights to one short string as
        # identifier for the current settings

        assert self.config.cache_directory is not None, "cache directory is needed"
        self.config.cache_directory.mkdir(exist_ok=True, parents=True)
        cache_file = str(self.config.cache_directory / f"index_cache_blended_dataset_seed_{seed}_{self.ident()}")
        return cache_file

    def get_data_index_cache_filename_meta(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".meta.json"

    def get_data_index_cache_filename_input(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".input.json"

    def get_data_index_cache_filename_bin(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".bin"

    def __len__(self) -> int:
        # implement in child class
        return max(self.size, self.config.minimum_dataset_size)

    def __getitem__(self, index: int) -> BaseDatasetItemGeneric:
        """
        Returns a BaseDatasetItem for each index
        """
        if self.size < self.config.minimum_dataset_size:
            index %= self.size

        if self.num_datasets > 1 and self.random_index is not None:
            index = int(self.random_index[index])

        if self.num_datasets == 1:
            return self.datasets[0][index]
        else:
            dataset_index, sample_idx = self.dataset_indices[index]
            return self.datasets[int(dataset_index)][int(sample_idx)]

    def set_seed(self, seed: int, shuffle: bool = True) -> None:
        """
        Sets the seed for shuffling the dataset
        """
        # Skip if same result is to be expected
        if seed == self.seed:
            return
        self.seed = seed
        assert shuffle, "Blended datasets should always be shuffled"

        # Skip computation of data index if only one dataset is loaded
        if self.num_datasets == 1:
            self.datasets[0].set_seed(seed=seed, shuffle=shuffle)
            self.size = len(self.datasets[0])
            return

        # Create cache file
        # The cache file is created just on one rank to avoid writing to same file from multiple ranks
        # To ensure consistency and having self.data_index as indexed dataset the data index is not kept in memory
        # but always loaded from the file again.

        number_of_documents_by_dataset = list()
        for ds in self.datasets:
            ds.set_seed(seed=seed, shuffle=shuffle)
            number_of_documents_by_dataset.append(len(ds))

        # compute and normalize weights
        if self.config.weight_by_num_documents:
            # Examples proportional sampling (using num docs) only called when both flags are True
            if self.config.weight_examples_proportional:
                self.weights = weights_examples_proportional(
                    number_of_documents_by_dataset,
                    self.config.ep_temperature,
                    self.config.ep_maximum,
                )
            else:
                self.weights = weights_by_num_docs(
                    number_of_documents_by_dataset, self.config.weighted_sampler_alpha
                )  # function output is normalized
        else:
            # Normalize weights.
            assert self.config.weights is not None
            assert len(self.config.weights) == len(self.datasets)
            self.weights = np.array(self.config.weights, dtype=np.float64)
            sum_weights = np.sum(self.weights)
            assert sum_weights > 0.0
            self.weights /= sum_weights

        cache_file_stem = self.get_data_index_cache_filename_stem(seed)
        cache_file_meta = self.get_data_index_cache_filename_meta(seed)
        cache_file_input = self.get_data_index_cache_filename_input(seed)
        cache_file_bin = self.get_data_index_cache_filename_bin(seed)
        if not Path(cache_file_meta).is_file():
            # Make sure to only create the data index on one rank and read from the others
            # It is assumed that rank 0 initializes the dataset
            if (not torch.distributed.is_initialized()) or (
                torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
            ):
                start = time.time()
                logger.warning(
                    f"{self.__class__.__name__} compute_data_index start for seed {seed}",
                )

                logger.warning(
                    f"{self.__class__.__name__} cache_file_stem {cache_file_stem}",
                )

                # Compute number of data items taken from each one of the datasets
                # We return as many data items as possible, i.e.
                # edge cases: alpha 0 -> equal counts come from each datasets
                # edge cases: alpha 1 -> NOT equal counts come from each dataset;
                # each dataset is fully represented (original distribution)

                # Logic
                # We would like to intermix the different source datasets as
                # much as possible (i.e. rather evenly distributed, not clusters of single datasets)
                # To achieve this the target proportion of datasets is computed.
                #
                if (
                    self.config.weight_examples_proportional
                ):  # temp: enable rounding before int conversion for accurate sample sizes
                    # TODO: merge the rounding step in all cases, instead of just this (kind of duplicated)
                    number_to_sample_by_dataset = np.array(
                        [
                            max(1, int(round(p * n)))
                            for (n, p) in zip(
                                number_of_documents_by_dataset,
                                (self.weights / self.weights.max()),
                            )
                        ],
                        dtype=np.int64,
                    )
                else:
                    number_to_sample_by_dataset = np.array(
                        [
                            max(1, int(p * n))
                            for (n, p) in zip(
                                number_of_documents_by_dataset,
                                (self.weights / self.weights.max()),
                            )
                        ],
                        dtype=np.int64,
                    )

                try:
                    import blended_dataset_loop
                except ImportError:
                    raise ImportError(
                        "blended_dataset_loop not found. Please install the optional `training` dependencies."
                    )

                # Here we call the Rust function to compute the data index based on the number of documents per dataset.
                # This function is implemented in Rust to speed up the computation.
                # It writes the resulting files to disk.
                blended_dataset_loop.sample(
                    number_to_sample_by_dataset,
                    self.get_data_index_cache_filename_stem(seed),
                )
                logger.info(
                    f"{self.__class__.__name__} compute_data_index "
                    f"done in {time.time() - start} seconds for seed {seed}",
                )

        # Load cache files
        # Wait on all ranks until files are available
        cache_files_found = False
        attempt_count = 0
        while not cache_files_found:
            cache_files_found = (
                Path(cache_file_bin).is_file() and Path(cache_file_input).is_file() and Path(cache_file_meta).is_file()
            )
            attempt_count += 1
            if cache_files_found:
                break
            if attempt_count % 12 == 0:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
                logger.info(
                    f"BlendedDataset waiting on index for seed {seed} on rank {rank}; "
                    f"elapsed {attempt_count * 5 / 60} minutes;\n"
                    f"{cache_file_bin}: {Path(cache_file_bin).is_file()}\n"
                    f"{cache_file_input}: {Path(cache_file_input).is_file()}\n"
                    f"{cache_file_meta}: {Path(cache_file_meta).is_file()}",
                )
            time.sleep(5)

        self.dataset_meta = json.load(open(cache_file_meta, "r", encoding="UTF-8"))
        if self.config.load_dataset_indices_to_memory:
            self.dataset_indices = np.fromfile(
                cache_file_bin,
                dtype=np.dtype(self.dataset_meta["dtype"]),
            ).reshape(tuple(self.dataset_meta["shape"]))
        else:
            self.dataset_indices = np.memmap(
                cache_file_bin,
                mode="r",
                order="C",
                dtype=np.dtype(self.dataset_meta["dtype"]),
                shape=tuple(self.dataset_meta["shape"]),
            )
        self.size = self.dataset_meta["shape"][0]

        # Important for Steerable models: shuffle data_prefixes to mix datasets
        # and avoid consecutive paths being from same source (e.g. FLAN, PS)
        if self.config.load_dataset_indices_to_memory:
            if self.config.shuffle_dataset_indices and shuffle:
                np_rng = np.random.RandomState(seed=seed)
                np_rng.shuffle(self.dataset_indices)
            self.random_index = None
        else:
            random_index = np.arange(self.size)
            if self.config.shuffle_dataset_indices and shuffle:
                np_rng = np.random.RandomState(seed=seed)
                np_rng.shuffle(random_index)
            self.random_index = random_index

    def collate(self, batch: List[BaseDatasetItemGeneric]) -> BaseDatasetBatchBeforeSyncGeneric:
        return self.datasets[0].collate(batch=batch)

    @staticmethod
    def sync_batch_to_model_parallel(
        topology: Topology, batch: Optional[BaseDatasetBatchBeforeSyncGeneric]
    ) -> BaseDatasetBatchGeneric:
        # This is just defined to satisfy mypy. We can instantiate the trainer from the base dataset staticmethod
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.datasets[0].__class__.__name__}"
