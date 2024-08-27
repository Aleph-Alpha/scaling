# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Determined.ai, Inc.
# Copyright 2017-2024 Determined.ai, Inc.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
from typing import Any

import determined as det  # type: ignore


def make_default_exp_config(
    hparams: dict[str, Any],
    checkpoint_dir: str | None = None,
) -> dict:
    return {
        "resources": {"native_parallel": False, "slots_per_trial": 1},
        "hyperparameters": hparams,
        "data_layer": {"type": "shared_fs"},
        "checkpoint_storage": {
            "type": "shared_fs",
            "host_path": checkpoint_dir or "/tmp",
        },
    }


def make_mock_cluster_info(num_slots: int, latest_checkpoint: str | None = None) -> det.ClusterInfo:
    config = make_default_exp_config({}, None)
    trial_info_mock = det.TrialInfo(
        trial_id=1,
        experiment_id=1,
        trial_seed=0,
        hparams={},
        config=config,
        steps_completed=0,
        trial_run_id=0,
        debug=False,
        inter_node_network_interface=None,
    )
    rendezvous_info_mock = det.RendezvousInfo(
        container_addrs=["localhost"],
        container_rank=0,
        container_slot_counts=list(range(num_slots)),
    )
    cluster_info_mock = det.ClusterInfo(
        master_url="localhost",
        cluster_id="clusterId",
        agent_id="agentId",
        slot_ids=list(range(num_slots)),
        task_id="taskId",
        allocation_id="allocationId",
        session_token="sessionToken",
        task_type="TRIAL",
        rendezvous_info=rendezvous_info_mock,
        trial_info=trial_info_mock,
        latest_checkpoint=latest_checkpoint,
    )
    return cluster_info_mock


@contextlib.contextmanager
def get_determined_context(checkpoint_dir):
    try:
        os.environ["LOCAL_SIZE"] = os.environ["WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = os.environ["LOCAL_SLOT"]
        os.environ["CROSS_RANK"] = "0"
        os.environ["CROSS_SIZE"] = "1"
        os.environ["DET_CHIEF_IP"] = "localhost"
        os.environ["DETERMINED_TEST"] = "True"
        distributed = det.core.DistributedContext.from_deepspeed()
        with det.core._dummy_init(
            distributed=distributed,
            checkpoint_storage=str(checkpoint_dir),
        ) as determined_context:
            yield determined_context
    finally:
        del os.environ["LOCAL_SIZE"]
        del os.environ["CROSS_RANK"]
        del os.environ["CROSS_SIZE"]
        del os.environ["DET_CHIEF_IP"]
        pass
