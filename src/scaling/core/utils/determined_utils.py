# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Determined.ai, Inc.
# Copyright 2017-2024 Determined.ai, Inc.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import faulthandler
from typing import Iterator

import determined.profiler  # type: ignore
from determined import ClusterInfo  # type: ignore
from determined._experiment_config import ExperimentConfig as DeterminedExperimentConfig  # type: ignore
from determined.core._context import Context as DeterminedContext  # type: ignore


@contextlib.contextmanager
def maybe_periodic_stacktraces(debug_enabled: bool) -> Iterator[None]:
    if debug_enabled:
        faulthandler.dump_traceback_later(30, repeat=True)
    try:
        yield
    finally:
        if debug_enabled:
            faulthandler.cancel_dump_traceback_later()


def determined_profiler_from_ctx(
    ctx: DeterminedContext,
    config_determined: DeterminedExperimentConfig,
    info: ClusterInfo,
) -> "determined.profiler.ProfilerAgent":
    begin_on_batch, end_after_batch = config_determined.profiling_interval()
    return determined.profiler.ProfilerAgent(
        trial_id=str(ctx.train._trial_id),
        agent_id=info.agent_id,
        master_url=info.master_url,
        profiling_is_enabled=config_determined.profiling_enabled(),
        global_rank=ctx.distributed.get_rank(),
        local_rank=ctx.distributed.get_local_rank(),
        begin_on_batch=begin_on_batch,
        end_after_batch=end_after_batch,
        sync_timings=config_determined.profiling_sync_timings(),
    )
