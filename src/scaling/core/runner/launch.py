# Copyright (c) 2024, IPAI Aleph Alpha Research GmbH
# Open Aleph License 1.0
#
# This file also contains code from Microsoft Corporation
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import collections
import logging
import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from typing import Any

from .launch_config import decode_base64


def parse_args() -> Namespace:
    parser = ArgumentParser(description="process launch")

    # Optional arguments for the launch helper
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The rank of the node for multi-node distributed training",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        help="Master node (rank 0)'s address, should be either"
        " the IP address or the hostname of node 0, for"
        " single node multi-proc training, the"
        " --master_addr can simply be 127.0.0.1",
    )
    parser.add_argument(
        "--master_port",
        default=29500,
        type=int,
        help="Master node (rank 0)'s free port that needs to "
        "be used for communication during distributed "
        "training",
    )
    parser.add_argument(
        "--resource_pool",
        required=True,
        type=str,
        help="world info base64 encoded dictionary",
    )

    # positional
    parser.add_argument(
        "script",
        type=str,
        help="The full path to the script for a single process (usually corresponding to one gpu",
    )

    # rest from the training program
    parser.add_argument(
        "--payload",
        type=str,
        required=False,
        default=None,
        help="payload base64 encoded dictionary",
    )
    return parser.parse_args()


def main() -> None:
    # get args
    args = parse_args()

    # decode resource pool
    resource_pool = decode_base64(args.resource_pool)

    # get local slots
    local_node_name = resource_pool["nodes"][args.node_rank]["name"]
    local_slots = resource_pool["nodes"][args.node_rank]["slots"]

    # compute global ranks
    # assert consistent world size
    curr_global_rank = 0
    global_rank_mapping = collections.defaultdict(list)
    for node in resource_pool["nodes"]:
        for _ in node["slots"]:
            global_rank_mapping[node["name"]].append(curr_global_rank)
            curr_global_rank += 1
    local_global_ranks = global_rank_mapping[local_node_name]
    assert len(local_slots) == len(local_global_ranks), "number of local slots differs from local global ranks"

    world_size = resource_pool["world_size"]
    assert world_size == curr_global_rank, "derived world size differs from communicated world size"

    # collect environment
    # set PyTorch distributed related environmental variables
    # It would be possible to use argparse arguments again
    # We nevertheless stick to the pytorch convention
    current_env = os.environ.copy()

    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(world_size)

    processes = []
    for local_slot, local_global_rank in zip(local_slots, local_global_ranks):
        # each process's rank
        current_env["RANK"] = str(local_global_rank)
        current_env["LOCAL_SLOT"] = str(local_slot)

        # spawn the processes
        cmd = [sys.executable, "-u"]
        cmd.append(args.script)
        cmd += ["--payload", args.payload]

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None

    def sigkill_handler(signum: int, frame: Any) -> None:
        for process in processes:
            logging.info(f"Killing subprocess {process.pid}")
            try:
                process.kill()
            except Exception:
                pass
        if last_return_code is not None:
            logging.error(f"exits with return code = {last_return_code}")
            sys.exit(last_return_code)
        if signum in sig_names:
            logging.info(f"Main process received {sig_names[signum]}, exiting")

        sys.exit(1)

    # pass SIGINT/SIGTERM to children if the parent is being terminated
    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

    alive_processes = set(processes)
    while len(alive_processes):
        finished_processes = []
        for process in alive_processes:
            if process.poll() is None:
                # the process is still running
                continue
            else:
                if process.returncode != 0:
                    last_return_code = process.returncode  # for sigkill_handler
                    sigkill_handler(signal.SIGTERM, None)  # not coming back
                else:
                    # exited cleanly
                    logging.info(f"Process {process.pid} exits successfully.")
                    finished_processes.append(process)
        alive_processes = set(alive_processes) - set(finished_processes)

        time.sleep(1)


if __name__ == "__main__":
    main()
