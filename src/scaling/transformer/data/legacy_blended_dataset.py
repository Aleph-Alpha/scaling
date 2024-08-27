import hashlib
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from scaling.core import (
    BaseBlendedDataset,
    BlendedDatasetConfig,
)
from scaling.core.data import BaseDatasetItemGeneric

from .legacy_dataset.indexed_dataset import get_indexed_dataset_, make_builder
from .text_dataset import TextDataset
from .text_dataset_batch import TextDatasetBatch, TextDatasetBatchBeforeSync
from .text_dataset_item import TextDatasetItem


class LegacyBlendedDataset(
    BaseBlendedDataset[
        TextDatasetItem,
        TextDatasetBatchBeforeSync,
        TextDatasetBatch,
        Union[TextDataset],
    ],
    torch.utils.data.Dataset,
):
    def __init__(
        self,
        seed: int,
        config: BlendedDatasetConfig,
        datasets: list[TextDataset],
        shuffle: bool = True,
    ):
        self.config = config
        self.datasets: list[TextDataset] = datasets
        self.num_datasets = len(self.datasets)

        # initialize
        self.dataset_indices_: list[np.ndarray] = list()
        self.size: int = 0

        # instantiate so that code is kept, this is obviously missing...
        self.ep_maximum_dict: dict | None = None

        # shuffling
        self.seed = None
        self.set_seed(seed, shuffle=shuffle)

    def get_data_index_cache_filename_stem(self, seed: int) -> str:
        # compute hash combining data prefixes and weights to one short string as
        # identifier for the current settings
        assert self.config.cache_directory is not None, "cache directory is needed"
        self.config.cache_directory.mkdir(exist_ok=True, parents=True)

        prefix_str = "-".join([Path(dp.data_prefix).name for dp in self.datasets])

        weight_str = "-".join([str(round(w * 100) / 100) for w in self.weights.tolist()])
        prefix_hash = hashlib.md5(prefix_str.encode("utf-8")).hexdigest()
        weight_hash = hashlib.md5(weight_str.encode("utf-8")).hexdigest()

        cache_file = str(
            self.config.cache_directory
            / f"index_cache_blended_dataset_seed_{seed}_seq_len_{self.datasets[0].sequence_length}"
            f"_prefix_{prefix_hash}_weights_{weight_hash}"
        )
        return cache_file

    def get_data_index_cache_filename_bin(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".bin"

    def get_data_index_cache_filename_idx(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".idx"

    def get_data_index_cache_filename_done(self, seed: int) -> str:
        return self.get_data_index_cache_filename_stem(seed) + ".done"

    def set_seed(self, seed: int, shuffle: bool = True) -> None:
        """Shuffle dataset indices then save in `self.dataset_indices`

        Args:
            seed (int): for shuffling dataset indices
        """
        assert shuffle, "Blended datasets should always be shuffled"

        # Skip if same result is to be expected
        if seed == self.seed:
            return
        self.seed = seed

        # Skip computation of data index if only one dataset is loaded
        if self.num_datasets == 1:
            # no weighting necessary
            self.size = len(self.datasets[0])
            return

        # Create cache file
        # The cache file is created just on one rank to avoid writing to same file from multiple ranks
        # To ensure consistency and having self.data_index as indexed dataset the data index is not kept in memory
        # but always loaded from the file again.

        number_of_documents_by_dataset = list()
        ep_maximum_list = list()  # to store ep rate limits for each dataset, if specified
        for ds in self.datasets:
            num_docs_in_ds = len(ds)
            ds.compute_data_index(seed=seed)
            number_of_documents_by_dataset.append(num_docs_in_ds)

            if self.config.weight_examples_proportional:
                assert (self.ep_maximum_dict is not None) or (
                    self.config.ep_maximum is not None
                ), "ep_maximum should be defined when using examples_proportional sampling"
                if self.ep_maximum_dict is not None:
                    # create list of limits in same order as number_of_documents_by_dataset
                    if ds.data_prefix.stem in self.ep_maximum_dict.keys():
                        ep_maximum_list.append(self.ep_maximum_dict[ds.data_prefix.stem])  # rate limit for this dataset
                    elif self.config.ep_maximum is not None:
                        ep_maximum_list.append(self.config.ep_maximum)
                    else:  # if no ep_maximum is set
                        ep_maximum_list.append(
                            num_docs_in_ds
                        )  # equivalent to not setting limit for this dataset (= actual len)

        if len(ep_maximum_list) > 0:
            assert len(number_of_documents_by_dataset) == len(
                ep_maximum_list
            ), "list lengths of ep_maximum list, and list of docs needs to be equal"
            ep_max_entry = ep_maximum_list  # re-assign before calculating sample weights
        elif self.config.ep_maximum is not None:  # only used if self.weight_examples_proportional is True
            ep_max_entry = self.config.ep_maximum  # type: ignore[assignment]
        else:
            pass

        # compute and normalize weights
        if self.config.weight_by_num_documents:
            # Examples proportional sampling (using num docs) only called when both flags are True
            if self.config.weight_examples_proportional:
                self.weights = weights_examples_proportional(
                    number_of_documents_by_dataset,
                    self.config.ep_temperature,
                    ep_max_entry,  # ep_max_entry either an int, or created list
                )
            else:
                self.weights = weights_by_num_docs(
                    number_of_documents_by_dataset, self.config.weighted_sampler_alpha
                )  # function output is normalized
        else:
            # Normalize weights.
            self.weights = np.array(self.weights, dtype=np.float64)
            sum_weights = np.sum(self.weights)
            assert sum_weights > 0.0
            self.weights /= sum_weights

        cache_file_stem = self.get_data_index_cache_filename_stem(seed)
        cache_file_bin = self.get_data_index_cache_filename_bin(seed)
        cache_file_idx = self.get_data_index_cache_filename_idx(seed)
        cache_file_done = self.get_data_index_cache_filename_done(seed)
        if not Path(cache_file_done).is_file():
            # Make sure to only create the data index on one rank and read from the others
            # It is assumed that rank 0 initializes the dataset
            get_rank = lambda: (torch.distributed.get_rank())  # noqa: E731
            if (not torch.distributed.is_initialized()) or (torch.distributed.is_initialized() and get_rank() == 0):
                start = time.time()
                print(
                    f"{self.__class__.__name__} compute_data_index start for seed {seed}",
                    flush=True,
                )

                # Indexed dataset builder
                builder = make_builder(
                    cache_file_bin,
                    impl="mmap",
                )

                # Compute number of data items taken from each one of the datasets
                # We return as many data items as possible, i.e.
                # edge cases: alpha 0 -> equal counts come from each datasets
                # edge cases: alpha 1 -> NOT equal counts come from each dataset; each dataset is fully represented
                # (original distribution)

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
                number_sampled_by_dataset = np.zeros((len(self.datasets)), dtype=np.int64)
                pbar = tqdm(
                    total=int(number_to_sample_by_dataset.sum()),
                    desc=f"{self.__class__.__name__} creating index",
                )

                while (number_sampled_by_dataset < number_to_sample_by_dataset).all():
                    # The next sample comes from the dataset that is furthest off the target
                    proportion_sampled_by_dataset = number_sampled_by_dataset / number_to_sample_by_dataset
                    dataset_index_to_sample = int(proportion_sampled_by_dataset.argmin())
                    index_in_dataset = number_sampled_by_dataset[dataset_index_to_sample]
                    number_sampled_by_dataset[dataset_index_to_sample] += (
                        1  # increment is enough because the source dataset is already shuffled
                    )
                    builder.add_item(torch.IntTensor([dataset_index_to_sample, index_in_dataset]))
                    pbar.update(1)

                builder.finalize(cache_file_idx)
                with open(cache_file_done, "w") as f:
                    f.write("True")

                pbar.close()
                print(
                    f"{self.__class__.__name__} compute_data_index done in {time.time() - start} "
                    f"seconds for seed {seed}",
                    flush=True,
                )

        # Load cache files
        # Wait on all ranks until files are available
        cache_files_found = False
        attempt_count = 0
        while not cache_files_found:
            cache_files_found = (
                Path(cache_file_bin).is_file() and Path(cache_file_idx).is_file() and Path(cache_file_done).is_file()
            )
            attempt_count += 1
            if cache_files_found:
                break
            if attempt_count % 12 == 0:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
                print(
                    f"BlendedDataset waiting on index for seed {seed} on rank {rank}; elapsed "
                    f"{attempt_count * 5 / 60} minutes;\n{cache_file_bin}: {Path(cache_file_bin).is_file()}\n"
                    f"{cache_file_idx}: {Path(cache_file_idx).is_file()}\n{cache_file_done}: "
                    f"{Path(cache_file_done).is_file()}",
                    flush=True,
                )
            time.sleep(5)

        self.dataset_indices_ = get_indexed_dataset_(  # type: ignore[assignment]
            cache_file_stem,
            "mmap",
            True,
        )
        self.size = len(self.dataset_indices_)

    def __len__(self) -> int:
        return max(self.size, self.config.minimum_dataset_size)

    def __getitem__(self, idx: int) -> BaseDatasetItemGeneric:  # type: ignore[type-var]
        if self.size < self.config.minimum_dataset_size:
            idx %= self.size

        if self.num_datasets == 1:
            return self.datasets[0][idx]  # type: ignore[return-value]
        else:
            dataset_index, sample_idx = self.dataset_indices_[idx].tolist()
            return self.datasets[dataset_index][sample_idx]  # type: ignore[return-value]


def weights_by_num_docs(number_of_examples_by_dataset: list[int], alpha: float = 0.3) -> np.ndarray:
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
        number_of_examples_by_dataset (list): Contains the number of examples in each dataset
        alpha (float): weighting parameter (default=`0.3`)
    Returns:
        weights (np.array): weights to multiply by num docs to get dataset sample size
    """
    l_examples = np.array(number_of_examples_by_dataset, np.float64)
    unbiased_sample_probs = l_examples / l_examples.sum()  # p_i

    probs = unbiased_sample_probs**alpha  # p_i**alpha

    # normalize
    probs = probs / probs.sum()  # q_i

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = 1 / unbiased_sample_probs  # 1/p_i
    weights = probs * unbiased_sample_probs_inverse  # q_i / p_i

    # normalize
    weights = weights / weights.sum()

    return weights


def weights_examples_proportional(
    number_of_examples_by_dataset: list[int], temperature: float = 1.0, maximum: list[int] | int | None = None
) -> np.ndarray:
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
        number_of_examples_by_dataset (list): Contains the number of examples in each dataset
        temperature (float): weighting parameter (default=`1.0`)
        maximum (Union[int, list]): rate limit, either as single value, or list for each document
    Returns:
        weights (np.array): weights to multiply by num docs to get dataset sample size
    """
    assert temperature != 0, "temperature is 0, expect non-zero temperature"
    assert temperature is not None, "temperature is None, expect non-zero float"

    l_examples = np.array(number_of_examples_by_dataset, np.float64)  # dataset sizes

    unbiased_sample_probs = l_examples / l_examples.sum()  # normalize

    if maximum:  # only when not zero, and not None
        if isinstance(maximum, list):
            max_array = np.array(maximum, np.float64)
            idx_l_above_limit = l_examples > max_array
            l_examples[idx_l_above_limit] = np.array(maximum, np.float64)[idx_l_above_limit]
        else:
            assert maximum > 0, f"examples-proportional sampling requires maximum limit > 0 (current max = {maximum})"
            l_examples[l_examples > maximum] = maximum  # apply limit to all dataset sizes before rate calculation

    max_limited_sample_probs = l_examples / l_examples.sum()  # normalize

    if temperature != 1.0:
        probs = max_limited_sample_probs ** (1.0 / temperature)
        probs = probs / probs.sum()  # normalize again
    else:
        probs = max_limited_sample_probs

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = 1 / unbiased_sample_probs  # 1/p_i
    weights = probs * unbiased_sample_probs_inverse  # q_i / p_i

    # normalize
    weights = weights / weights.sum()

    return weights
