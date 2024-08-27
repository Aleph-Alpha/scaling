import base64
import collections
import json
import os
import subprocess
import sys
from shlex import quote
from typing import Any

import torch

from .runner_config import RunnerConfig, RunnerType

EXPORT_ENVS = ["NCCL", "PYTHON", "MV2", "UCX"]
PDSH_MAX_FAN_OUT = 1024
ENVIRONMENT_PATH = os.path.join(os.path.expanduser("~"), ".deepspeed_env")


class PDSHRunner:
    def __init__(
        self,
        master_addr: str,
        config: RunnerConfig,
        resource_pool: dict[str, int | list[dict[str, str | list[int]]]],
        payload: dict[Any, Any],
        exports: dict[str, str],
        environment: dict[str, str],
    ):
        self.master_addr = master_addr
        self.config = config
        self.resource_pool = resource_pool
        self.payload = payload
        self.exports = exports
        self.environment = environment
        self.environment["PDSH_RCMD_TYPE"] = "ssh"

    @property
    def name(self) -> str:
        return "pdsh"

    def get_cmd(self) -> list[str]:
        command = []

        use_pdsh = not (
            len(self.resource_pool["nodes"]) == 1  # type: ignore
            and self.resource_pool["nodes"][0]["name"] in ["localhost", "127.0.0.1"]  # type: ignore
        )
        if use_pdsh:
            active_workers = ",".join([node["name"] for node in self.resource_pool["nodes"]])  # type: ignore
            # PDSH flags for max node fan out and specific hosts to launch on
            # See https://linux.die.net/man/1/pdsh for flag details
            command = ["pdsh", "-f", str(PDSH_MAX_FAN_OUT), "-w", active_workers]

        if self.config.runner_type == RunnerType.PDSH_DOCKER:
            command += ["sudo"] if self.config.docker_config.docker_sudo else []
            command += ["docker", "run", "--rm", "--privileged"]

            for key, val in self.exports.items():
                if key.lower().startswith("python"):
                    continue
                command += ["--env", f"{key}={quote(val)}"]

            if self.config.docker_config.docker_mounts is not None:
                for (
                    mnt_dir_host,
                    mnt_dir_container,
                ) in self.config.docker_config.docker_mounts:
                    command += ["-v", f"{mnt_dir_host}:{mnt_dir_container}"]

            command += [
                # f'--workdir="{os.path.abspath(".")}"',
                "--network=host",
                "--gpus",
                "all",
                "--ipc=host",
                "--name=scaling.core",
                "--ulimit",
                "memlock=-1",
                "--ulimit",
                "stack=67108864",
                self.config.docker_config.docker_container,  # type: ignore
            ]

        if use_pdsh and self.config.runner_type != RunnerType.PDSH_DOCKER:
            exports = ""
            for key, val in self.exports.items():
                exports += f"export {key}={quote(val)}; "
            command += [exports]

            command += [f"cd {os.path.abspath('')};"]

        # https://linux.die.net/man/1/pdsh
        # %n will be replaced by pdsh command
        command += [
            ("python" if self.config.runner_type == RunnerType.PDSH_DOCKER else sys.executable),
            "-u",
            "-m",
            "scaling.core.runner.launch",
            f"--resource_pool={encode_base64(self.resource_pool)}",
        ]

        if use_pdsh:
            command += ["--node_rank=%n"]

        command += [
            f"--master_addr={self.master_addr}",
            f"--master_port={self.config.master_port}",
        ]

        command += [
            str(self.config.script.absolute()),  # type: ignore
            "--payload",
            encode_base64(self.payload),
        ]
        return command


def parse_host(config: RunnerConfig, host_str: str) -> tuple[str, list[int]]:
    split = host_str.split(" ")
    if len(split) == 1:
        host_name = split[0]
        slots = list(range(config.default_gpu_count))
    elif len(split) == 2:
        host_name = split[0]
        slots_str = split[1]
        assert slots_str.startswith("slots="), "second part of host should start with 'slots'; e.g. 'nodename slots=8'"
        slot_data = slots_str.split("=")
        assert len(slot_data) == 2, "second part of host should start with 'slots'; e.g. 'nodename slots=8'"
        assert len(slot_data[1]) > 0, f"no slots defined in '{host_str}'"

        if "," in slot_data[1]:
            slots = [int(slot) for slot in slot_data[1].split(",") if len(slot) > 0]
        else:
            slots = list(range(int(slot_data[1])))
    else:
        raise ValueError(f"cannot parse host '{host_str}'")

    return host_name, slots


def hosts_str_to_resource_pool(config: RunnerConfig, hosts: list[str]) -> dict[str, list[int]]:
    resource_pool = collections.OrderedDict()

    # read from hosts
    for host in hosts:
        host_name, slots = parse_host(config=config, host_str=host)
        if host_name in resource_pool:
            raise ValueError(f"host '{host_name}' defined more than once")
        resource_pool[host_name] = slots

    # fill localhost
    if len(resource_pool) == 0:
        device_count = torch.cuda.device_count()
        assert device_count > 0, "would use local resources for training without hosts defined; no local gpus found"
        resource_pool["localhost"] = list(range(device_count))

    return resource_pool


def get_resource_pool(config: RunnerConfig) -> dict[str, Any]:
    resource_pool = None

    # get resources from either hostsfile or hosts
    if config.hostsfile is not None and config.hosts is None:
        # read from hostsfile
        if not config.hostsfile.is_file():
            raise ValueError(f"Hostsfile not found: {config.hostsfile}")
        hosts = []
        with open(config.hostsfile, "r") as file_handle:
            for line in file_handle.readlines():
                line = line.strip()

                #  skip empty lines
                if line == "":
                    continue

                hosts.append(line)

        resource_pool = hosts_str_to_resource_pool(config=config, hosts=hosts)

    elif config.hostsfile is None and config.hosts is not None:
        # read from hosts
        resource_pool = hosts_str_to_resource_pool(config=config, hosts=config.hosts)
    elif config.hostsfile is not None and config.hosts is not None:
        raise ValueError("Both hostsfile and hosts cannot be defined at the same time.")
    else:
        resource_pool = hosts_str_to_resource_pool(config=config, hosts=[])

    # convert to list to guarantee order

    world_size = sum([len(slots) for slots in resource_pool.values()])
    nodes_list = list()
    for name, slots in resource_pool.items():
        nodes_list.append({"name": name, "slots": slots})

    return {"nodes": nodes_list, "world_size": world_size}


def encode_base64(d: dict) -> str:
    d_str = json.dumps(d).encode("utf-8")
    result = base64.urlsafe_b64encode(d_str).decode("utf-8")
    return result


def runner_main(config: RunnerConfig, payload: dict[str, Any]) -> int:
    assert config.script is not None, "must provide a script to run"

    # get resource pool and validate
    resource_pool = get_resource_pool(config=config)
    assert len(resource_pool) > 0, "could not identify resources"

    # get master address
    if config.master_addr is None:
        first_host = resource_pool["nodes"][0]["name"]
        if first_host in ["localhost", "127.0.0.1"]:
            master_addr = "localhost"
        else:
            hostname_cmd = [f"ssh {first_host} hostname -I"]
            result = subprocess.check_output(hostname_cmd, shell=True)
            master_addr = result.decode("utf-8").split()[0]
    else:
        master_addr = config.master_addr

    # collect env
    env = os.environ.copy()
    curr_path = os.path.abspath("")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = curr_path + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = curr_path

    # collect exports
    exports = dict()
    for var_name, var_value in env.items():
        if any([var_name.startswith(name) for name in EXPORT_ENVS]):
            exports[var_name.strip()] = var_value.strip()

    environ_file = ENVIRONMENT_PATH
    if os.path.isfile(environ_file):
        with open(environ_file, "r") as fd:
            for var in fd.readlines():
                key, val = var.split("=", maxsplit=1)
                exports[key] = val

    runner = PDSHRunner(
        master_addr=master_addr,
        config=config,
        resource_pool=resource_pool,
        payload=payload,
        exports=exports,
        environment=env,
    )
    command = runner.get_cmd()
    process = subprocess.Popen(command, env=env, stdout=None)

    process.wait()

    returncode = process.returncode

    # In case of failure must propagate the error-condition back to the caller (usually shell). The
    # actual error and traceback should have been printed in the subprocess, so in order to avoid
    # unnecessary noise we just quietly exit here with the same code as the subprocess
    if returncode > 0:
        sys.exit(returncode)

    return returncode
