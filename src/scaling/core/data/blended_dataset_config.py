# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from EleutherAI
# Copyright (c) 2024, EleutherAI
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List, Optional

from pydantic import Field

from scaling.core.config import BaseConfig


class BlendedDatasetConfig(BaseConfig):
    weight_by_num_documents: bool = Field(
        True,
        description="If weight_by_num_documents is True, Builds dataset weights from a multinomial "
        "distribution over groups of data according to the number of documents in each group. "
        "WARNING: setting this to True will override any user provided weights",
    )

    weighted_sampler_alpha: float = Field(
        1.0,
        description="""
    Alpha value for `weight_by_num_documents`. Only has an effect if `weight_by_num_documents` = True.

  when alpha = 1, the probability of sampling from a given group = n_samples / total_samples
  as alpha -> 0, the probability of sampling from all groups becomes equal, and number of documents has no effect
  as alpha -> inf, the probability of sampling from the groups with *the most samples* -> 1
    """,
    )

    weights: Optional[List[float]] = Field(
        None,
        description="weights of singular datasets. "
        "The list needs to have the same length and order as the datasets provided",
    )

    weight_examples_proportional: bool = Field(
        False,
        description="""If True (with weight_by_num_documents set to True),
    this uses a modified method to build dataset weights

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
    """,
    )

    ep_maximum: Optional[int] = Field(
        None,
        description="If set, rate limit K used in 'weight_examples_proportional'. "
        "Only has an effect if `weight_examples_proportional` = True.",
    )

    ep_temperature: float = Field(
        1.0,
        description="Temperature value for `weight_examples_proportional`. "
        "Only has an effect if `weight_examples_proportional` = True. "
        "Temperature is inverse of alpha (as in weighted_sampler_alpha)",
    )

    minimum_dataset_size: int = Field(
        0,
        description="Minimal size of the dataset.",
    )

    cache_directory: Optional[Path] = Field(
        None,
        description="directory to cache blended dataset index. "
        "this only needs to be set if more than one dataset is provided.",
    )

    shuffle_dataset_indices: bool = Field(
        False,
        description="if True, shuffles after blended data index creation.",
    )

    load_dataset_indices_to_memory: bool = Field(
        False,
        description="if True, loads dataset indices to memory rather than using mmap",
    )
